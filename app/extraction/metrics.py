from dataclasses import dataclass

import torch
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

from app.extraction.enums import IGNORE_INDEX
from app.extraction.word_metrics import ValidationMetricsResult, WordMetrics


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Store the mean loss of a training epoch.

    Attributes:
        loss: Mean loss weighted by supervised-token count.
    """

    loss: float


class LossMetrics:
    """Average batch losses by their supervised-token counts.

    Attributes:
        ignore_index: Target ID excluded from the token count.
        loss_sum: Sum of batch mean losses multiplied by supervised-token counts.
        token_count: Number of supervised tokens accumulated.
    """

    def __init__(self, ignore_index: int = IGNORE_INDEX):
        """Create an empty loss accumulator.

        Args:
            ignore_index: Target ID excluded from loss aggregation.
        """
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        """Clear loss totals and supervised-token counts."""
        self.loss_sum = 0.0
        self.token_count = 0

    def update(
        self, logits: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor
    ) -> None:
        """Add a batch loss weighted by the number of supervised tokens.

        Fully ignored batches contribute nothing, even if their loss is NaN.

        Args:
            logits: Model scores; accepted for the accumulator contract and unused here.
            labels: Target IDs used to count supervised tokens.
            loss: Scalar mean loss over supervised tokens.
        """
        count = int((labels != self.ignore_index).sum().item())
        if count:
            self.loss_sum += loss.detach().item() * count
            self.token_count += count

    def compute(self) -> TrainResult:
        """Return the token-weighted epoch loss.

        Returns:
            Training result with a native Python float.

        Raises:
            ValueError: No supervised tokens have been accumulated.
        """
        if self.token_count == 0:
            raise ValueError("Training loader contains no supervised tokens.")
        return TrainResult(self.loss_sum / self.token_count)


@dataclass(frozen=True)
class EntityReport:
    """Store entity scores with JSON-compatible numeric values.

    Attributes:
        precision: Micro-averaged entity precision.
        recall: Micro-averaged entity recall.
        f1: Micro-averaged entity F1.
        per_entity: Seqeval report including per-entity and aggregate rows.
    """

    precision: float
    recall: float
    f1: float
    per_entity: dict[str, dict[str, float | int]]


class BioEntityScorer:
    """Compute span metrics using an explicit seqeval interpretation.

    Attributes:
        options: Extra classification_report options, such as strict mode and IOB2 scheme.
    """

    def __init__(self, **options):
        """Configure BIO interpretation without changing model predictions.

        Args:
            **options: Seqeval options other than output_dict and zero_division.
        """
        self.options = options

    def score(
        self, actual: list[list[str]], predicted: list[list[str]]
    ) -> EntityReport:
        """Score matching sequences of BIO labels.

        Uses zero for undefined scores. Default options preserve seqeval decoding;
        pass strict mode and a scheme when invalid BIO boundaries must be rejected.

        Args:
            actual: Gold label sequences, one per evaluation unit.
            predicted: Predicted sequences aligned with the gold labels.

        Returns:
            Entity report with float scores and integer support counts.
        """
        report = classification_report(
            actual, predicted, output_dict=True, zero_division=0, **self.options
        )
        values = {
            entity: {
                key: int(value) if key == "support" else float(value)
                for key, value in metrics.items()
            }
            for entity, metrics in report.items()
        }
        micro = values["micro avg"]
        return EntityReport(
            float(micro["precision"]),
            float(micro["recall"]),
            float(micro["f1-score"]),
            values,
        )


class WindowMetrics:
    """Score first-subtoken word labels and strict IOB2 entities per window.

    Each input row is one evaluation sequence. Mask continuation subtokens,
    special tokens, and padding with ignore_index before calling update.
    Field metrics merge B/I labels and exclude O. Predictions are not repaired.
    """

    def __init__(self, id2label: dict[int, str], ignore_index: int = IGNORE_INDEX):
        self.id2label = dict(id2label)
        self.ignore_index = ignore_index
        labels = [self.id2label[index] for index in range(len(self.id2label))]
        fields = [label[2:] for label in labels if label.startswith("B-")]
        expected = ["O"] + [
            f"{prefix}-{field}" for field in fields for prefix in ("B", "I")
        ]
        if labels != expected:
            raise ValueError("Labels must contain O followed by paired B/I labels")
        self.words = WordMetrics(labels, fields, ignore_index)
        self.scorer = BioEntityScorer(mode="strict", scheme=IOB2)
        self.reset()

    def reset(self) -> None:
        self.words.reset()
        self.actual_sequences: list[list[str]] = []
        self.predicted_sequences: list[list[str]] = []

    @torch.inference_mode()
    def update(self, logits, targets, loss, *, predictions=None) -> None:
        if not (targets != self.ignore_index).any():
            return
        if predictions is None:
            predictions = logits.argmax(dim=-1)
        self.words.update_predictions(predictions, targets, loss)
        for actual, predicted in zip(
            targets.detach().cpu(), predictions.detach().cpu(), strict=True
        ):
            mask = actual != self.ignore_index
            if mask.any():
                self.actual_sequences.append(
                    [self.id2label[i] for i in actual[mask].tolist()]
                )
                self.predicted_sequences.append(
                    [self.id2label[i] for i in predicted[mask].tolist()]
                )

    def compute(self) -> ValidationMetricsResult:
        if not self.actual_sequences:
            raise ValueError("Evaluation loader contains no supervised tokens.")
        report = self.scorer.score(self.actual_sequences, self.predicted_sequences)
        return ValidationMetricsResult(
            word_metrics=self.words.compute(),
            entity_precision=report.precision,
            entity_recall=report.recall,
            entity_f1=report.f1,
            entity_report=report.per_entity,
        )
