import json
import random
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, set_seed

from app.aggregates.bert.params import BertPytorchParams, DatasetName
from app.extraction.bert.pytorch.builder import InvoiceRecordBuilder
from app.extraction.bert.pytorch.checkpoint import BertPytorchCheckpoint
from app.extraction.bert.pytorch.collation import InvoiceDataCollator
from app.extraction.bert.pytorch.datasets import InvoiceDataset
from app.extraction.bert.pytorch.decoding import OCRWordDecoder
from app.extraction.bert.pytorch.docile.builder import DocileRecordBuilder
from app.extraction.bert.pytorch.docile.item_builder import DocileItemRecordBuilder
from app.extraction.bert.pytorch.inspector import EntityErrorInspector
from app.extraction.bert.pytorch.model import BertFieldExtractor
from app.extraction.bert.pytorch.schema import InvoiceLabelSchema
from app.extraction.bert.pytorch.spans import (
    Normalizer,
    NumericField,
    OCRSpanMatcher,
    OCRSummarySpan,
)
from app.extraction.bert.pytorch.training import OCRInvoiceTraining
from app.extraction.enums import InvoiceBIOLabel, InvoiceField


@dataclass(frozen=True)
class TrainingResult:
    checkpoint_path: Path
    history_path: Path

@dataclass
class BertTrainingPipeline:
    model: BertFieldExtractor
    schema: InvoiceLabelSchema
    inspector: EntityErrorInspector
    device: torch.device
    epochs: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    output_dir: Path
    checkpoint_path: Path
    diagnostc_dir: Path | None = None
    validation_records: list[dict] | None = None
    word_decoder: OCRWordDecoder | None = None

    def run(self, dataloaders: dict[str, DataLoader]):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        trainer = OCRInvoiceTraining(
            model=self.model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            device=self.device,
            epochs=self.epochs,
            id2label=self.schema.id2label,
            inspector=self.inspector,
            diagnostic_dir=self.diagnostc_dir,
            validation_records=self.validation_records,
            word_decoder=self.word_decoder,
        )
        self.model.to("cpu")
        return TrainingResult(
            checkpoint_path=self.checkpoint_path,
            history_path=self.output_dir / "history.csv",
        )
        