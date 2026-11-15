from dataclasses import dataclass

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers.tokenization_utils_tokenizers import TokenizersBackend

from app.extraction.bert.artifacts import BertArtifacts
from app.extraction.bert.checkpoint import BertEvaluationRecords, BertPytorchCheckpoint
from app.extraction.bert.decoding import OCRWordDecoder
from app.extraction.bert.inspector import EntityErrorInspector
from app.extraction.bert.model import BertFieldExtractor
from app.extraction.bert.training import OCRInvoiceTraining
from app.extraction.enums import IGNORE_INDEX
from app.extraction.results import TrainingResult
from app.extraction.schema import InvoiceLabelSchema
from app.extraction.training import BestCheckpointWriter
from app.pipelines.bert.builder import BuildResult
from app.pipelines.bert.split import DatasetsSplit


@dataclass
class BertTraining:
    """Configure BERT optimization and package the selected best model.

    Attributes:
        model: BERT token classifier.
        schema: Shared invoice label schema.
        tokenizer: Fast tokenizer saved with the checkpoint.
        device: Execution device.
        epochs: Number of training epochs.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        artifacts: Checkpoint, history, and diagnostic storage.
        max_length_tokens: Token window length saved for later evaluation.
        inspector: Optional validation diagnostics.
        word_decoder: Optional OCR-word decoder used for validation.
    """

    model: BertFieldExtractor
    schema: InvoiceLabelSchema
    tokenizer: TokenizersBackend
    device: torch.device
    epochs: int
    learning_rate: float
    weight_decay: float
    artifacts: BertArtifacts
    max_length_tokens: int = 512
    inspector: EntityErrorInspector | None = None
    word_decoder: OCRWordDecoder | None = None

    def train(
        self,
        dataloaders: dict[str, DataLoader],
        build_result: BuildResult,
        split: DatasetsSplit,
    ) -> TrainingResult:
        """Train BERT and save checkpoints on validation score improvements.

        Each improved validation F1 saves a checkpoint before the next epoch.
        The model is moved to the CPU when training or serialization exits.

        Args:
            dataloaders: Training and validation loaders.
            build_result: Prepared split records, including validation and test inputs.
            split: Source frames used to retain original OCR offsets.

        Returns:
            Checkpoint directory and per-epoch history path.
        """
        validation_records = build_result.records.val.to_dict(orient="records")
        trainer = OCRInvoiceTraining(
            model=self.model,
            optimizer=torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            ),
            criterion=nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX),
            train_loader=dataloaders["train"],
            val_loader=dataloaders["val"],
            device=self.device,
            epochs=self.epochs,
            id2label=self.schema.id2label,
            inspector=self.inspector,
            artifacts=self.artifacts,
            validation_records=validation_records,
            word_decoder=self.word_decoder,
            checkpoint_writer=BestCheckpointWriter(
                BertPytorchCheckpoint(
                    artifacts=self.artifacts,
                    encoder_config=self.model.bert.config,
                    tokenizer=self.tokenizer,
                    evaluation_records=BertEvaluationRecords(
                        artifacts=self.artifacts,
                        schema=self.schema,
                        records={
                            "validation": validation_records,
                            "test": build_result.records.test.to_dict(orient="records"),
                        },
                        frame=pd.concat((split.train, split.val, split.test)),
                        max_length=self.max_length_tokens,
                        decoder="ocr_word"
                        if self.word_decoder is not None
                        else "token_argmax",
                    ),
                ),
            ),
        )
        try:
            trainer.train()
        finally:
            self.model.to("cpu")
        return TrainingResult(
            self.artifacts.checkpoint_dir, self.artifacts.history_path
        )
