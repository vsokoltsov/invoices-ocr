from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from app.extraction.bert.pytorch.decoding import OCRWordDecoder
from app.extraction.bert.pytorch.metrics import EvalMetrics, EvalResult


@dataclass
class BertEvaluation:
    """Evaluate BIO token windows with optional OCR-word decoding."""

    model: nn.Module
    metrics: EvalMetrics
    device: torch.device
    criterion: nn.Module
    word_decoder: OCRWordDecoder | None = None

    @torch.inference_mode()
    def evaluate(self, loader: DataLoader, epoch: int) -> EvalResult:
        self.model.eval()

        self.metrics.reset()

        progress_bar = tqdm(loader, position=1, desc=f"Eval epoch {epoch}", leave=False)
        for batch in progress_bar:
            metadata = batch.get("decoder_metadata")
            batch = {
                key: value for key, value in batch.items() if key != "decoder_metadata"
            }
            batch = {key: value.to(self.device) for key, value in batch.items()}
            targets = batch["labels"]
            valid = targets != self.metrics.ignore_index
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

            if not torch.isfinite(loss):
                raise FloatingPointError("Evaluation loss is not finite")

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

            self.metrics.update(logits, targets, loss, predictions=predictions)

        return self.metrics.compute()
