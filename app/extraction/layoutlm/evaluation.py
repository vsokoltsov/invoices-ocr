from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from app.extraction.batches import TensorBatchAdapter
from app.extraction.evaluation import TokenClassificationEvaluation
from app.extraction.metrics import WindowMetrics
from app.extraction.schema import InvoiceLabelSchema
from app.extraction.word_metrics import ValidationMetricsResult


@dataclass
class LayoutLMEvaluation(TokenClassificationEvaluation[ValidationMetricsResult]):
    """Score first-subtoken word predictions with shared window metrics."""

    metrics: WindowMetrics
    schema: InvoiceLabelSchema

    batch_adapter = TensorBatchAdapter(
        ("input_ids", "attention_mask", "bbox", "pixel_values")
    )

    @property
    def ignore_index(self) -> int:
        return self.metrics.ignore_index

    def reset(self) -> None:
        self.metrics.reset()

    def update(
        self,
        batch: dict,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss: torch.Tensor,
    ) -> None:
        self.metrics.update(logits, labels, loss)

    def compute(self, loader: DataLoader) -> ValidationMetricsResult:
        return self.metrics.compute()
