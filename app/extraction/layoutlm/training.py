from dataclasses import dataclass
from typing import cast

from torch.optim.lr_scheduler import LRScheduler
from transformers import PreTrainedModel

from app.extraction.batches import TensorBatchAdapter
from app.extraction.layoutlm.artifacts import LayoutLMArtifacts
from app.extraction.layoutlm.checkpoint import LayoutLMCheckpoint
from app.extraction.layoutlm.evaluation import LayoutLMEvaluation
from app.extraction.layoutlm.metrics import (
    EpochMetrics,
    EpochMetricsResult,
    ValidationMetricsResult,
)
from app.extraction.schema import InvoiceLabelSchema
from app.extraction.training import (
    BestCheckpointWriter,
    HistoryWriter,
    TokenClassificationTraining,
    TrainingCallback,
)


@dataclass
class OCRInvoiceTraining(
    TokenClassificationTraining[EpochMetricsResult, ValidationMetricsResult]
):
    """Configure shared training for image, text, and layout inputs.

    Model, loaders, optimizer, criterion, device, and epoch count come from the
    shared trainer. Evaluation reconstructs pages and uses strict IOB2 scoring.

    Attributes:
        metrics: Word-level accumulator with the same labels as the schema.
        scheduler: Learning-rate schedule advanced after each optimizer update.
        schema: Field mapping and deterministic BIO IDs.
        artifacts: Storage used for epoch history and best.pt.
    """

    metrics: EpochMetrics
    scheduler: LRScheduler
    schema: InvoiceLabelSchema
    artifacts: LayoutLMArtifacts

    batch_adapter = TensorBatchAdapter(
        ("input_ids", "attention_mask", "bbox", "pixel_values")
    )

    def __post_init__(self) -> None:
        """Validate epoch count and metric label alignment.

        Raises:
            ValueError: Epochs are invalid or metric labels differ from the schema.
        """
        super().__post_init__()
        if tuple(self.metrics.labels) != self.schema.labels:
            raise ValueError("Metric labels must match the training schema")

    @property
    def training_metrics(self) -> EpochMetrics:
        """Return the word-level training accumulator.

        Returns:
            Configured LayoutLM epoch metrics.
        """
        return self.metrics

    @property
    def schedule(self) -> LRScheduler:
        """Return the configured optimizer-step schedule.

        Returns:
            Learning-rate scheduler supplied by the pipeline.
        """
        return self.scheduler

    def make_evaluator(self) -> LayoutLMEvaluation:
        """Build strict page-level evaluation for the current model.

        Returns:
            Evaluator sharing this model, schema, criterion, and metrics.
        """
        return LayoutLMEvaluation(
            model=self.model,
            metrics=self.metrics,
            schema=self.schema,
            device=self.device,
            criterion=self.criterion,
        )

    def callbacks(self) -> tuple[TrainingCallback, ...]:
        """Configure epoch history and improved-checkpoint persistence.

        Reads token length from the first training sample; the training dataset
        must contain at least one sample.

        Returns:
            History writer followed by the LayoutLM checkpoint writer.
        """
        return (
            HistoryWriter(self.artifacts),
            BestCheckpointWriter(
                LayoutLMCheckpoint(
                    artifacts=self.artifacts,
                    config=cast(PreTrainedModel, self.model.backbone).config,
                    schema=self.schema,
                    fields=list(self.metrics.fields),
                    max_length=len(self.train_loader.dataset[0]["input_ids"]),
                ),
            ),
        )
