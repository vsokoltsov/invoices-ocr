from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import structlog
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from app.extraction.bert.pytorch.decoding import OCRWordDecoder
from app.extraction.bert.pytorch.evaluation import BertEvaluation
from app.extraction.bert.pytorch.inspector import EntityErrorInspector
from app.extraction.bert.pytorch.metrics import EvalMetrics, EvalResult
from app.extraction.enums import IGNORE_INDEX

logger = structlog.get_logger()


@dataclass
class TrainResult:
    loss: float


@dataclass
class OCRInvoiceTraining:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    train_loader: DataLoader
    val_loader: DataLoader
    device: torch.device
    epochs: int
    id2label: dict[int, str]
    inspector: EntityErrorInspector | None = None
    diagnostic_dir: str | Path | None = None
    validation_records: list[dict] | None = None
    word_decoder: OCRWordDecoder | None = None

    def __post_init__(self):
        if self.epochs < 1:
            raise ValueError("epochs must be positive.")
        self.metrics = EvalMetrics(self.id2label)
        self.diagnostic_dir = (
            Path(self.diagnostic_dir).expanduser().resolve()
            if self.diagnostic_dir
            else None
        )
        if self.inspector is not None and self.diagnostic_dir is not None:
            if self.validation_records is None:
                raise ValueError("validation_records are required for diagnostics")

    def train_one_epoch(self, epoch: int) -> TrainResult:
        self.model.train()

        total_loss = 0.0
        total_tokens = 0

        progress_bar = tqdm(
            self.train_loader, position=1, desc=f"Train epoch {epoch}", leave=False
        )
        for batch in progress_bar:
            batch = {key: value.to(self.device) for key, value in batch.items()}

            valid_tokens = int((batch["labels"] != IGNORE_INDEX).sum().item())
            if valid_tokens == 0:
                continue

            self.optimizer.zero_grad(set_to_none=True)

            logits = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            loss = self.criterion(
                logits.reshape(-1, logits.shape[-1]),
                batch["labels"].reshape(-1),
            )

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Weight the epoch loss by the number of supervised tokens.
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

        if total_tokens == 0:
            raise ValueError("Training loader contains no supervised tokens.")

        return TrainResult(loss=total_loss / total_tokens)

    def eval_one_epoch(
        self, epoch: int, external_loader: DataLoader | None = None
    ) -> EvalResult:
        loader = external_loader if external_loader is not None else self.val_loader
        return BertEvaluation(
            model=self.model,
            metrics=self.metrics,
            device=self.device,
            criterion=self.criterion,
            word_decoder=self.word_decoder,
        ).evaluate(loader, epoch)

    def train(self) -> pd.DataFrame:
        self.model.to(self.device)

        best_f1 = -1.0
        best_state = None
        history = []

        progress_bar = tqdm(range(1, self.epochs + 1), position=0, leave=True)
        for epoch in progress_bar:
            progress_bar.set_description(f"Train model {epoch}/{self.epochs}")
            train_metrics = self.train_one_epoch(epoch=epoch)

            validation_metrics = self.eval_one_epoch(epoch=epoch)

            epoch_metrics = {
                "epoch": epoch,
                "train_loss": train_metrics.loss,
                **{
                    f"validation_{key}": value
                    for key, value in asdict(validation_metrics).items()
                    if key != "per_entity"
                },
            }

            history.append(epoch_metrics)

            logger.info(
                "Train / eval results",
                epoch=f"{epoch}/{self.epochs}",
                train_loss=train_metrics.loss,
                val_loss=validation_metrics.loss,
                val_f1=validation_metrics.f1,
            )
            if self.inspector is not None and self.diagnostic_dir is not None:
                self.inspector.run(
                    self.validation_records,
                    output_dir=Path(self.diagnostic_dir) / f"epoch_{epoch:03d}",
                    epoch=epoch,
                )

            if validation_metrics.f1 > best_f1:
                best_f1 = validation_metrics.f1

                # Copy the tensors; state_dict() alone shares model storage.
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }

        if best_state is not None:
            self.model.load_state_dict(best_state)
        history_df = pd.DataFrame(history)

        return history_df
