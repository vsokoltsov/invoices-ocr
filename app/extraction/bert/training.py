from dataclasses import dataclass

import pandas as pd

from app.extraction.batches import TensorBatchAdapter
from app.extraction.bert.artifacts import BertArtifacts
from app.extraction.bert.decoding import OCRWordDecoder
from app.extraction.bert.evaluation import BertEvaluation
from app.extraction.bert.inspector import EntityErrorInspector
from app.extraction.bert.metrics import EvalMetrics, EvalResult
from app.extraction.metrics import LossMetrics, TrainResult
from app.extraction.protocols import ValidationResult
from app.extraction.training import (
    BestCheckpointWriter,
    ConstantSchedule,
    HistoryWriter,
    TokenClassificationTraining,
    TrainingCallback,
)


@dataclass
class DiagnosticsWriter(TrainingCallback):
    """Write BERT entity diagnostics after each validation epoch.

    Attributes:
        inspector: Inspector using the current model and decoder.
        artifacts: Run storage from which epoch report paths are derived.
        records: Validation records aligned with the inspected documents.
    """

    inspector: EntityErrorInspector
    artifacts: BertArtifacts
    records: list[dict]

    def on_epoch(
        self, epoch: int, validation: ValidationResult, history: pd.DataFrame
    ) -> None:
        """Inspect validation records and save reports for this epoch.

        Args:
            epoch: One-based epoch used in the diagnostic directory name.
            validation: Validation metrics; unused by this callback.
            history: Epoch history; unused by this callback.
        """
        self.inspector.run(
            self.records, artifacts=self.artifacts.diagnostics(epoch), epoch=epoch
        )


@dataclass
class OCRInvoiceTraining(TokenClassificationTraining[TrainResult, EvalResult]):
    """Configure shared training for BERT token windows.

    Uses a constant learning rate and seqeval default BIO scoring. Model, loaders,
    optimizer, criterion, device, and epoch count are inherited from the shared
    trainer. A checkpoint callback packages weights only on score improvement.

    Attributes:
        id2label: Mapping used by token and BIO metrics.
        inspector: Optional entity-error diagnostic component.
        artifacts: Optional run storage for per-epoch history and diagnostics.
        validation_records: Records required when diagnostics are enabled.
        word_decoder: Optional OCR-word decoder; otherwise use token argmax.
        checkpoint_writer: Optional callback that saves improved model weights.
    """

    id2label: dict[int, str]
    inspector: EntityErrorInspector | None = None
    artifacts: BertArtifacts | None = None
    validation_records: list[dict] | None = None
    word_decoder: OCRWordDecoder | None = None
    checkpoint_writer: BestCheckpointWriter | None = None

    batch_adapter = TensorBatchAdapter(("input_ids", "attention_mask"))

    def __post_init__(self) -> None:
        """Initialize BERT metrics and validate enabled diagnostic inputs.

        Raises:
            ValueError: Epochs are invalid or enabled diagnostics lack validation records.
        """
        super().__post_init__()
        self.metrics = EvalMetrics(self.id2label)
        self._training_metrics = LossMetrics()
        self._schedule = ConstantSchedule()
        self.callbacks()

    @property
    def training_metrics(self) -> LossMetrics:
        """Return the supervised-token loss accumulator.

        Returns:
            BERT training loss accumulator.
        """
        return self._training_metrics

    @property
    def schedule(self) -> ConstantSchedule:
        """Return the constant learning-rate policy.

        Returns:
            Policy that leaves the optimizer learning rate unchanged.
        """
        return self._schedule

    def make_evaluator(self) -> BertEvaluation:
        """Build BERT evaluation using the current decoder configuration.

        Returns:
            Evaluator sharing this model, criterion, and validation metrics.
        """
        return BertEvaluation(
            model=self.model,
            metrics=self.metrics,
            device=self.device,
            criterion=self.criterion,
            word_decoder=self.word_decoder,
        )

    def callbacks(self) -> tuple[TrainingCallback, ...]:
        """Configure history writing and optional validation diagnostics.

        Returns:
            Ordered callbacks enabled by the artifact and inspector configuration.

        Raises:
            ValueError: Diagnostics are enabled without validation records.
        """
        checkpoint = (self.checkpoint_writer,) if self.checkpoint_writer else ()
        if self.artifacts is None:
            return checkpoint
        history = HistoryWriter(self.artifacts)
        if self.inspector is None or self.artifacts.diagnostics_dir is None:
            return (*checkpoint, history)
        if self.validation_records is None:
            raise ValueError("validation_records are required for diagnostics")
        return (
            *checkpoint,
            history,
            DiagnosticsWriter(self.inspector, self.artifacts, self.validation_records),
        )
