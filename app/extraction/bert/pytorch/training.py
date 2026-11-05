from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import structlog
import torch
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
)
from tqdm.auto import tqdm

from app.extraction.bert.pytorch.decoding import OCRWordDecoder
from app.extraction.bert.pytorch.inspector import EntityErrorInspector

logger = structlog.get_logger()


@dataclass
class TrainResult:
    loss: float


@dataclass
class EvalResult:
    loss: float
    precision: float
    recall: float
    f1: float
    token_accuracy: float
    token_macro_f1: float
    per_entity: dict[str, dict[str, float]]


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

            valid_tokens = int((batch["labels"] != -100).sum().item())
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

    @torch.inference_mode()
    def eval_one_epoch(self, epoch: int) -> EvalResult:
        self.model.eval()

        num_labels = len(self.id2label)

        accuracy = MulticlassAccuracy(
            num_classes=num_labels,
            average="micro",
        ).to(self.device)

        macro_f1 = MulticlassF1Score(
            num_classes=num_labels,
            average="macro",
        ).to(self.device)

        actual_sequences = []
        predicted_sequences = []

        total_loss = 0.0
        total_tokens = 0

        progress_bar = tqdm(
            self.val_loader, position=1, desc=f"Eval epoch {epoch}", leave=False
        )
        for batch in progress_bar:
            metadata = batch.get("decoder_metadata")
            batch = {
                key: value for key, value in batch.items() if key != "decoder_metadata"
            }
            batch = {key: value.to(self.device) for key, value in batch.items()}
            targets = batch["labels"]
            valid = targets != -100
            valid_tokens = int(valid.sum().item())
            if valid_tokens == 0:
                continue

            logits = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            loss = self.criterion(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )

            predictions = logits.argmax(dim=-1)
            if self.word_decoder is not None:
                if metadata is None:
                    raise ValueError("OCR-word evaluation requires decoding metadata")
                for index, item in enumerate(metadata):
                    length = len(item["offsets"])
                    predictions[index, :length] = torch.tensor(
                        self.word_decoder.decode(
                            logits[index, :length], item["offsets"], item["words"]
                        ),
                        device=self.device,
                    )

            accuracy.update(predictions[valid], targets[valid])
            macro_f1.update(predictions[valid], targets[valid])

            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

            for predicted_row, target_row in zip(
                predictions.cpu().tolist(),
                targets.cpu().tolist(),
            ):
                actual_sequences.append(
                    [self.id2label[target] for target in target_row if target != -100]
                )

                predicted_sequences.append(
                    [
                        self.id2label[predicted]
                        for predicted, target in zip(predicted_row, target_row)
                        if target != -100
                    ]
                )

        if total_tokens == 0:
            raise ValueError("Evaluation loader contains no supervised tokens.")

        # seqeval accepts integer zero_division values despite its str annotation.
        # TorchMetrics wraps compute() at runtime; ty misidentifies its binding.
        return EvalResult(
            loss=total_loss / total_tokens,
            precision=precision_score(
                actual_sequences,
                predicted_sequences,
                zero_division=0,  # ty: ignore[invalid-argument-type]
            ),
            recall=recall_score(
                actual_sequences,
                predicted_sequences,
                zero_division=0,  # ty: ignore[invalid-argument-type]
            ),
            f1=f1_score(
                actual_sequences,
                predicted_sequences,
                zero_division=0,  # ty: ignore[invalid-argument-type]
            ),
            token_accuracy=accuracy.compute().item(),  # ty: ignore[missing-argument]
            token_macro_f1=macro_f1.compute().item(),  # ty: ignore[missing-argument]
            per_entity=classification_report(
                actual_sequences,
                predicted_sequences,
                output_dict=True,
                zero_division=0,
            ),
        )

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
