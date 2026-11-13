from collections.abc import Sequence
from dataclasses import dataclass

import torch

from app.extraction.layoutlm.enums import IGNORE_INDEX


@dataclass(frozen=True, slots=True)
class FieldMetrics:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True, slots=True)
class EpochMetricsResult:
    loss: float
    bio_word_accuracy: float
    field_word_micro_precision: float
    field_word_micro_recall: float
    field_word_micro_f1: float
    field_word_macro_f1: float
    per_field: dict[str, FieldMetrics]


@dataclass(frozen=True, slots=True)
class ValidationMetricsResult:
    word_metrics: EpochMetricsResult
    entity_precision: float
    entity_recall: float
    entity_f1: float
    entity_report: dict[str, dict[str, float | int]]


class EpochMetrics:
    """Accumulate word metrics without retaining model outputs."""

    def __init__(
        self, labels: Sequence[str], fields: Sequence[str], ignore_index=IGNORE_INDEX
    ):
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
        self.confusion.zero_()
        self.loss_sum = 0.0
        self.word_count = 0

    def update(self, logits, labels, loss):
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
