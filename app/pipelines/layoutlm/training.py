from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from app.extraction.enums import IGNORE_INDEX
from app.extraction.layoutlm.artifacts import LayoutLMArtifacts
from app.extraction.layoutlm.metrics import EpochMetrics
from app.extraction.layoutlm.model import LayoutLMFieldExtractor
from app.extraction.layoutlm.schema import InvoiceLabelSchema
from app.extraction.layoutlm.training import OCRInvoiceTraining


@dataclass(frozen=True)
class TrainingResult:
    checkpoint_path: Path
    history_path: Path


@dataclass
class LayoutLMTraining:
    model: LayoutLMFieldExtractor
    schema: InvoiceLabelSchema
    device: torch.device
    epochs: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    artifacts: LayoutLMArtifacts

    def train(self, dataloaders: dict[str, DataLoader]) -> TrainingResult:
        decay, no_decay = [], []
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                (no_decay if parameter.ndim < 2 else decay).append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.learning_rate,
        )
        total_steps = self.epochs * len(dataloaders["train"])
        warmup_steps = int(self.warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

        trainer = OCRInvoiceTraining(
            model=self.model,
            metrics=EpochMetrics(self.schema.labels, list(self.schema.field_to_entity)),
            optimizer=optimizer,
            criterion=criterion,
            train_loader=dataloaders["train"],
            val_loader=dataloaders["val"],
            device=self.device,
            scheduler=scheduler,
            schema=self.schema,
            epochs=self.epochs,
            artifacts=self.artifacts,
        )
        try:
            trainer.train()
        finally:
            self.model.to("cpu")
        return TrainingResult(
            checkpoint_path=self.artifacts.checkpoint_path,
            history_path=self.artifacts.history_path,
        )
