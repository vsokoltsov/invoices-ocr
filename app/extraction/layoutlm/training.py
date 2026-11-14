from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import pandas as pd
import structlog
import torch
from seqeval.metrics import (
    classification_report,
)
from seqeval.scheme import IOB2
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from app.extraction.layoutlm.dataset import InvoiceDataset
from app.extraction.layoutlm.metrics import (
    EpochMetrics,
    EpochMetricsResult,
    ValidationMetricsResult,
)
from app.extraction.layoutlm.schema import InvoiceLabelSchema

logger = structlog.get_logger()


def move_inputs(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: batch[key].to(device, non_blocking=True)
        for key in ("input_ids", "attention_mask", "bbox", "pixel_values")
    }


@dataclass
class OCRInvoiceTraining:
    """Train token labels, evaluate complete pages, and restore the best entity F1."""

    model: nn.Module
    metrics: EpochMetrics
    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    train_loader: DataLoader
    val_loader: DataLoader
    device: torch.device
    scheduler: LRScheduler
    schema: InvoiceLabelSchema
    epochs: int
    output_dir: Path
    checkpoint_path: Path

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if tuple(self.metrics.labels) != self.schema.labels:
            raise ValueError("Metric labels must match the training schema")
        self.output_dir = Path(self.output_dir)
        self.checkpoint_path = Path(self.checkpoint_path)

    def train_one_epoch(self, epoch: int) -> EpochMetricsResult:
        self.model.train()
        self.metrics.reset()

        progress_bar = tqdm(
            self.train_loader, position=1, desc=f"Train epoch {epoch}", leave=False
        )
        for batch in progress_bar:
            inputs = move_inputs(batch, self.device)
            labels = batch["labels"].to(self.device, non_blocking=True)

            if not (labels != self.metrics.ignore_index).any():
                continue

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(**inputs)
            loss = self.criterion(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss is not finite")

            loss.backward()

            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0,
                error_if_nonfinite=True,
            )

            self.optimizer.step()
            self.scheduler.step()
            self.metrics.update(logits, labels, loss)

        return self.metrics.compute()

    @torch.inference_mode()
    def eval_one_epoch(
        self, epoch: int, external_loader: DataLoader | None = None
    ) -> ValidationMetricsResult:
        self.model.eval()
        self.metrics.reset()

        word_predictions = {}

        loader = external_loader if external_loader is not None else self.val_loader
        progress_bar = tqdm(loader, position=1, desc=f"Eval epoch {epoch}", leave=False)
        for batch in progress_bar:
            inputs = move_inputs(batch, self.device)
            labels = batch["labels"].to(self.device, non_blocking=True)

            logits = self.model(**inputs)
            loss = self.criterion(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
            )

            if not torch.isfinite(loss):
                raise FloatingPointError("Evaluation loss is not finite")

            self.metrics.update(logits, labels, loss)
            predicted = logits.argmax(dim=-1).cpu()

            for batch_index, page_index in enumerate(batch["page_index"].tolist()):
                for token_index, word_index in enumerate(
                    batch["word_ids"][batch_index].tolist()
                ):
                    if word_index < 0:
                        continue

                    key = (page_index, word_index)

                    if key in word_predictions:
                        raise ValueError("A word was evaluated more than once")

                    word_predictions[key] = predicted[batch_index, token_index].item()

        # Restore complete pages before measuring entity boundaries.
        true_sequences = []
        pred_sequences = []

        dataset = cast(InvoiceDataset, loader.dataset)
        for page_index, page in enumerate(dataset.pages):
            true_sequences.append(
                [self.schema.id2label[label] for label in page["labels"]]
            )

            pred_sequences.append(
                [
                    self.schema.id2label[word_predictions[(page_index, word_index)]]
                    for word_index in range(len(page["words"]))
                ]
            )

        report = classification_report(
            true_sequences,
            pred_sequences,
            mode="strict",
            scheme=IOB2,
            output_dict=True,
            zero_division=0,
        )
        report = {
            entity: {
                key: int(value) if key == "support" else float(value)
                for key, value in values.items()
            }
            for entity, values in report.items()
        }
        return ValidationMetricsResult(
            word_metrics=self.metrics.compute(),
            entity_precision=float(report["micro avg"]["precision"]),
            entity_recall=float(report["micro avg"]["recall"]),
            entity_f1=float(report["micro avg"]["f1-score"]),
            entity_report=report,
        )

    def train(self) -> pd.DataFrame:
        """Write history each epoch and save best weights with schema metadata."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.to(self.device)
        best_state = None
        history = []
        best_f1 = -1.0

        progress_bar = tqdm(range(1, self.epochs + 1), position=0, leave=True)
        for epoch in progress_bar:
            progress_bar.set_description(f"Train model {epoch}/{self.epochs}")
            train_metrics = self.train_one_epoch(epoch=epoch)
            val_metrics = self.eval_one_epoch(epoch=epoch)
            validation_scalars = {
                key: value
                for key, value in asdict(val_metrics.word_metrics).items()
                if key != "per_field"
            }
            validation_scalars.update(
                entity_precision=val_metrics.entity_precision,
                entity_recall=val_metrics.entity_recall,
                entity_f1=val_metrics.entity_f1,
            )
            epoch_metrics = {
                "epoch": epoch,
                "train_loss": train_metrics.loss,
                **{
                    f"validation_{key}": value
                    for key, value in validation_scalars.items()
                },
            }

            history.append(epoch_metrics)
            pd.DataFrame(history).to_csv(
                self.output_dir / "history.csv",
                index=False,
            )
            logger.info(
                "Training results",
                epoch=f"{epoch}/{self.epochs}",
                train_loss=f"{train_metrics.loss:.4f}",
                val_loss=f"{val_metrics.word_metrics.loss:.4f}",
                val_entity_F1=f"{val_metrics.entity_f1:.4f}",
            )
            if val_metrics.entity_f1 > best_f1:
                best_f1 = float(val_metrics.entity_f1)
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }

                backbone_config = getattr(
                    getattr(self.model, "backbone", None), "config", None
                )
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "model_name": getattr(backbone_config, "_name_or_path", None),
                        "max_length": len(self.train_loader.dataset[0]["input_ids"]),
                        "labels": list(self.schema.labels),
                        "fields": list(self.metrics.fields),
                        "field_to_entity": dict(self.schema.field_to_entity),
                        "epoch": epoch,
                        "val_entity_f1": best_f1,
                    },
                    self.checkpoint_path,
                )
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return pd.DataFrame(history)
