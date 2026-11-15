from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers.tokenization_utils_tokenizers import TokenizersBackend

from app.extraction.bert.artifacts import BertArtifacts
from app.extraction.bert.checkpoint import BertPytorchCheckpoint
from app.extraction.bert.decoding import OCRWordDecoder
from app.extraction.bert.inspector import EntityErrorInspector
from app.extraction.bert.model import BertFieldExtractor
from app.extraction.bert.schema import InvoiceLabelSchema
from app.extraction.bert.training import OCRInvoiceTraining
from app.extraction.enums import IGNORE_INDEX
from app.pipelines.bert.builder import BuildResult
from app.pipelines.bert.split import DatasetsSplit


@dataclass(frozen=True)
class TrainingResult:
    """Paths to the checkpoint directory and scalar training history."""

    checkpoint_path: Path
    history_path: Path


@dataclass
class BertTraining:
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
        )
        try:
            history = trainer.train()
            self.artifacts.save_history(history)
            best_epoch = int(history.loc[history["validation_f1"].idxmax(), "epoch"])
            BertPytorchCheckpoint(self.artifacts).save(
                self.model,
                self.tokenizer,
                self.schema,
                validation_records,
                pd.concat((split.train, split.val, split.test)),
                epoch=best_epoch,
                max_length=self.max_length_tokens,
                test_records=build_result.records.test.to_dict(orient="records"),
                decoder="ocr_word" if self.word_decoder is not None else "token_argmax",
            )
        finally:
            self.model.to("cpu")
        return TrainingResult(
            self.artifacts.checkpoint_dir, self.artifacts.history_path
        )
