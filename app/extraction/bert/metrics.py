from dataclasses import dataclass

import torch
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from app.extraction.enums import IGNORE_INDEX
from app.extraction.metrics import BioEntityScorer


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Store BERT token metrics and window-level entity scores.

    Attributes:
        loss: Mean supervised-token loss.
        precision: Micro entity precision.
        recall: Micro entity recall.
        f1: Micro entity F1 used for best-weight selection.
        token_accuracy: Accuracy over supervised token positions.
        token_macro_f1: Macro F1 over configured token labels.
        per_entity: Seqeval per-entity and aggregate report.
    """

    loss: float
    precision: float
    recall: float
    f1: float
    token_accuracy: float
    token_macro_f1: float
    per_entity: dict[str, dict[str, float | int]]

    @property
    def score(self) -> float:
        """Return the checkpoint selection score.

        Returns:
            Micro entity F1.
        """
        return self.f1

    @property
    def scalars(self) -> dict[str, float]:
        """Return numeric history and publication values.

        Returns:
            Scalar metric names mapped to Python floats; excludes the nested report.
        """
        return {
            "loss": self.loss,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "token_accuracy": self.token_accuracy,
            "token_macro_f1": self.token_macro_f1,
        }


class EvalMetrics:
    """Accumulate supervised token metrics and BIO entities per window.

    Decoded predictions affect token and entity scores. Loss always comes from
    the original model logits. Window boundaries remain entity boundaries.

    Attributes:
        id2label: Copied mapping from integer IDs to BIO labels.
        ignore_index: Target label excluded from every metric.
        entity_scorer: Scorer using seqeval default BIO interpretation.
    """

    def __init__(self, id2label: dict[int, str], ignore_index: int = IGNORE_INDEX):
        """Create token metrics and an empty BIO sequence accumulator.

        Args:
            id2label: Complete mapping of token IDs to BIO labels.
            ignore_index: Target ID excluded from metrics and loss aggregation.
        """
        self.entity_scorer = BioEntityScorer()
        self.id2label = dict(id2label)
        self.ignore_index = ignore_index
        self.accuracy: MulticlassAccuracy = MulticlassAccuracy(
            num_classes=len(id2label), average="micro"
        )
        self.macro_f1: MulticlassF1Score = MulticlassF1Score(
            num_classes=len(id2label), average="macro"
        )
        self.reset()

    def reset(self) -> None:
        """Clear token metrics, weighted loss totals, and BIO sequences."""
        self.accuracy.reset()
        self.macro_f1.reset()
        self.loss_sum = 0.0
        self.token_count = 0
        self.actual_sequences: list[list[str]] = []
        self.predicted_sequences: list[list[str]] = []

    @torch.inference_mode()
    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss: torch.Tensor,
        *,
        predictions: torch.Tensor | None = None,
    ) -> None:
        """Accumulate supervised predictions and per-window BIO sequences.

        Fully ignored batches are skipped. Scoring tensors are detached and moved
        to the CPU so metric state does not retain the model computation graph.

        Args:
            logits: Scores with shape [batch, tokens, labels].
            targets: Gold IDs with shape [batch, tokens].
            loss: Mean supervised-token loss for this batch.
            predictions: Optional decoded IDs; None uses the logits argmax.
        """
        valid = targets != self.ignore_index
        count = int(valid.sum().item())
        if count == 0:
            return

        if predictions is None:
            predictions = logits.argmax(dim=-1)

        targets = targets.detach().cpu()
        predictions = predictions.detach().cpu()
        valid = valid.cpu()
        # TorchMetrics wraps these methods at runtime; ty misreads their binding.
        self.accuracy.update(predictions[valid], targets[valid])  # ty: ignore[invalid-argument-type]
        self.macro_f1.update(predictions[valid], targets[valid])  # ty: ignore[invalid-argument-type]
        self.loss_sum += loss.detach().item() * count
        self.token_count += count

        for truth, predicted, mask in zip(targets, predictions, valid):
            if not mask.any():
                continue
            self.actual_sequences.append(
                [self.id2label[index] for index in truth[mask].tolist()]
            )
            self.predicted_sequences.append(
                [self.id2label[index] for index in predicted[mask].tolist()]
            )

    def compute(self) -> EvalResult:
        """Compute token metrics and default BIO entity scores.

        Returns:
            Evaluation result with token-weighted loss and JSON-compatible report values.

        Raises:
            ValueError: No supervised tokens have been accumulated.
        """
        if self.token_count == 0:
            raise ValueError("Evaluation loader contains no supervised tokens.")

        report = self.entity_scorer.score(
            self.actual_sequences, self.predicted_sequences
        )
        return EvalResult(
            loss=self.loss_sum / self.token_count,
            precision=report.precision,
            recall=report.recall,
            f1=report.f1,
            token_accuracy=self.accuracy.compute().item(),  # ty: ignore[missing-argument]
            token_macro_f1=self.macro_f1.compute().item(),  # ty: ignore[missing-argument]
            per_entity=report.per_entity,
        )
