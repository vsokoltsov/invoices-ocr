from collections.abc import Sequence
from dataclasses import dataclass

import torch

from app.extraction.enums import IGNORE_INDEX


@dataclass(frozen=True, slots=True)
class FieldMetrics:
    """Store word-level scores for one field, merging B and I labels.

    Attributes:
        precision: Correct field words divided by predicted field words.
        recall: Correct field words divided by gold field words.
        f1: Harmonic mean of field-word precision and recall.
        support: Number of gold words for this field.
    """

    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True, slots=True)
class EpochMetricsResult:
    """Store supervised word metrics for a complete epoch.

    Attributes:
        loss: Word-count-weighted mean loss.
        bio_word_accuracy: Accuracy with B and I treated as distinct labels.
        field_word_micro_precision: Micro precision after merging each field B/I pair.
        field_word_micro_recall: Micro recall after merging each field B/I pair.
        field_word_micro_f1: Micro field-word F1.
        field_word_macro_f1: Mean field-word F1 for fields present in gold or predictions.
        per_field: Scores and gold word support for each configured field.
    """

    loss: float
    bio_word_accuracy: float
    field_word_micro_precision: float
    field_word_micro_recall: float
    field_word_micro_f1: float
    field_word_macro_f1: float
    per_field: dict[str, FieldMetrics]

    @property
    def scalars(self) -> dict[str, float]:
        """Return flat word metrics for history rows.

        Returns:
            Numeric word metrics without the nested per-field report.
        """
        return {
            "loss": self.loss,
            "bio_word_accuracy": self.bio_word_accuracy,
            "field_word_micro_precision": self.field_word_micro_precision,
            "field_word_micro_recall": self.field_word_micro_recall,
            "field_word_micro_f1": self.field_word_micro_f1,
            "field_word_macro_f1": self.field_word_macro_f1,
        }


@dataclass(frozen=True, slots=True)
class ValidationMetricsResult:
    """Combine word metrics with strict page-level entity evaluation.

    Attributes:
        word_metrics: Word classification scores and mean loss.
        entity_precision: Strict entity micro precision.
        entity_recall: Strict entity micro recall.
        entity_f1: Strict entity micro F1 used for checkpoint selection.
        entity_report: Strict seqeval per-entity and aggregate report.
    """

    word_metrics: EpochMetricsResult
    entity_precision: float
    entity_recall: float
    entity_f1: float
    entity_report: dict[str, dict[str, float | int]]

    @property
    def loss(self) -> float:
        """Return the supervised-word validation loss.

        Returns:
            Loss from the word metric result.
        """
        return self.word_metrics.loss

    @property
    def score(self) -> float:
        """Return the checkpoint selection score.

        Returns:
            Strict page-level entity micro F1.
        """
        return self.entity_f1

    @property
    def scalars(self) -> dict[str, float]:
        """Combine flat word and entity metrics for history.

        Returns:
            Numeric validation values without nested reports.
        """
        return {
            **self.word_metrics.scalars,
            "entity_precision": self.entity_precision,
            "entity_recall": self.entity_recall,
            "entity_f1": self.entity_f1,
        }


class EpochMetrics:
    """Accumulate word statistics without retaining model outputs.

    Attributes:
        labels: Ordered O label followed by a B/I pair per field.
        fields: Field names in the same order as the B/I pairs.
        ignore_index: Target ID excluded from supervision.
        confusion: CPU confusion matrix with gold rows and predicted columns.
        loss_sum: Loss weighted by supervised-word counts.
        word_count: Number of supervised words accumulated.
    """

    def __init__(
        self, labels: Sequence[str], fields: Sequence[str], ignore_index=IGNORE_INDEX
    ):
        """Create an empty confusion matrix for the configured BIO labels.

        Args:
            labels: O followed by one B/I pair for each field.
            fields: Fields in the same order as their label pairs.
            ignore_index: Target ID excluded from metrics.

        Raises:
            ValueError: The number of labels does not match the number of fields.
        """
        if len(labels) != 1 + 2 * len(fields):
            raise ValueError("Labels must contain O and one B/I pair per field")
        self.confusion = torch.zeros(
            len(labels),
            len(labels),
            dtype=torch.long,
        )
        self.labels = labels
        self.fields = fields
        self.loss_sum = 0.0
        self.word_count = 0
        self.ignore_index = ignore_index

    def reset(self) -> None:
        """Clear confusion counts and weighted loss totals."""
        self.confusion.zero_()
        self.loss_sum = 0.0
        self.word_count = 0

    def update(self, logits, labels, loss):
        """Add predictions for supervised word-start tokens.

        Args:
            logits: Scores with shape [batch, tokens, labels].
            labels: Gold IDs; continuation, padding, and special tokens must be ignored.
            loss: Scalar mean loss for this batch.
        """
        valid = labels != self.ignore_index

        truth = labels[valid].detach().cpu()
        predicted = logits.argmax(dim=-1)[valid].detach().cpu()

        size = len(self.labels)
        counts = torch.bincount(
            truth * size + predicted,
            minlength=size * size,
        )

        self.confusion += counts.reshape(size, size)
        self.loss_sum += loss.detach().item() * truth.numel()
        self.word_count += truth.numel()

    def compute(self) -> EpochMetricsResult:
        """Compute BIO accuracy and field-word scores from confusion counts.

        Field scores merge B/I labels and exclude O. Macro F1 includes only fields
        with gold or predicted words; per-field support counts gold words.

        Returns:
            Word-count-weighted loss, aggregate scores, and per-field results.

        Raises:
            ValueError: No supervised words have been accumulated.
        """
        if self.word_count == 0:
            raise ValueError("No supervised words")

        # Map O -> 0, B/I of each field -> one field ID.
        mapping = [0] + [
            field_id for field_id in range(1, len(self.fields) + 1) for _ in range(2)
        ]

        matrix = torch.zeros(
            len(self.fields) + 1,
            len(self.fields) + 1,
            dtype=torch.float64,
        )

        for truth_id, truth_field in enumerate(mapping):
            for pred_id, pred_field in enumerate(mapping):
                matrix[truth_field, pred_field] += self.confusion[truth_id, pred_id]

        tp = matrix.diag()[1:]
        fp = matrix.sum(dim=0)[1:] - tp
        fn = matrix.sum(dim=1)[1:] - tp

        precision = tp / (tp + fp).clamp_min(1)
        recall = tp / (tp + fn).clamp_min(1)
        f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1)

        active = (tp + fp + fn) > 0

        return EpochMetricsResult(
            loss=self.loss_sum / self.word_count,
            bio_word_accuracy=(self.confusion.diag().sum().item() / self.word_count),
            field_word_micro_precision=(tp.sum() / (tp + fp).sum().clamp_min(1)).item(),
            field_word_micro_recall=(tp.sum() / (tp + fn).sum().clamp_min(1)).item(),
            field_word_micro_f1=(
                2 * tp.sum() / (2 * tp + fp + fn).sum().clamp_min(1)
            ).item(),
            field_word_macro_f1=(f1[active].mean().item() if active.any() else 0.0),
            per_field={
                field: FieldMetrics(
                    precision=precision[index].item(),
                    recall=recall[index].item(),
                    f1=f1[index].item(),
                    support=int((tp + fn)[index].item()),
                )
                for index, field in enumerate(self.fields)
            },
        )
