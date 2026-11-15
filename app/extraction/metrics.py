from dataclasses import dataclass

import torch
from seqeval.metrics import classification_report

from app.extraction.enums import IGNORE_INDEX


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
