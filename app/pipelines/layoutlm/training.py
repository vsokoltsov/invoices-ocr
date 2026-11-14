import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from app.aggregates.layoutlm.params import LayoutLMTrainingParams
from app.extraction.layoutlm.enums import IGNORE_INDEX
from app.extraction.layoutlm.metrics import EpochMetrics
from app.extraction.layoutlm.model import LayoutLMFieldExtractor
from app.extraction.layoutlm.schema import InvoiceLabelSchema
from app.extraction.layoutlm.training import OCRInvoiceTraining


@dataclass
class LayoutLMTraining:
    model: LayoutLMFieldExtractor
    metrics: EpochMetrics
    schema: InvoiceLabelSchema
    params: LayoutLMTrainingParams
    device: torch.device

    def train(self, dataloaders: dict[str, DataLoader]) -> pd.DataFrame:
        decay, no_decay = [], []
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                (no_decay if parameter.ndim < 2 else decay).append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.params.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.params.learning_rate,
        )
        total_steps = self.params.num_epoch * len(dataloaders["train"])
        warmup_steps = int(self.params.warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        output_dir = Path(self.params.output_dir)

        trainer = OCRInvoiceTraining(
            model=self.model,
            metrics=self.metrics,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=dataloaders["train"],
            val_loader=dataloaders["val"],
            device=self.device,
            scheduler=scheduler,
            schema=self.schema,
            epochs=self.params.num_epoch,
            output_dir=output_dir,
            checkpoint_path=Path(self.params.checkpoint_path),
        )
        history = trainer.train()
        test_metrics = trainer.eval_one_epoch(
            epoch=self.params.num_epoch, external_loader=dataloaders["test"]
        )
        (output_dir / "test_metrics.json").write_text(
            json.dumps(asdict(test_metrics), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        return history
