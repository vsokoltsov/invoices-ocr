from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from app.extraction.enums import IGNORE_INDEX
from app.extraction.layoutlm.artifacts import LayoutLMArtifacts
from app.extraction.layoutlm.metrics import EpochMetrics
from app.extraction.layoutlm.model import LayoutLMFieldExtractor
from app.extraction.layoutlm.training import OCRInvoiceTraining
from app.extraction.results import TrainingResult
from app.extraction.schema import InvoiceLabelSchema
from app.pipelines.training import TrainingRun


@dataclass
class LayoutLMTraining:
    """Configure LayoutLM optimization, scheduling, and run storage.

    Attributes:
        model: LayoutLM token classifier.
        schema: Shared invoice label schema.
        device: Execution device.
        epochs: Number of training epochs.
        learning_rate: AdamW learning rate.
        weight_decay: Decay applied to trainable parameters with two or more dimensions.
        warmup_ratio: Fraction of planned optimizer steps used for linear warmup.
        artifacts: Storage for history and best.pt.
    """

    model: LayoutLMFieldExtractor
    schema: InvoiceLabelSchema
    device: torch.device
    epochs: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    artifacts: LayoutLMArtifacts

    def train(self, dataloaders: dict[str, DataLoader]) -> TrainingResult:
        """Train LayoutLM with AdamW and linear warmup/decay.

        Separates parameters by dimensionality to exclude biases and normalization
        weights from decay. The trainer saves improved weights through its callback.
        The model moves to the CPU on success or failure.

        Args:
            dataloaders: Training and validation loaders.

        Returns:
            Best-checkpoint and per-epoch history paths.
        """
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
        return TrainingRun(trainer, self.model, self.artifacts).run()
