from dataclasses import dataclass
from typing import cast

import torch
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from app.extraction.layoutlm.dataset import InvoiceDataset
from app.extraction.layoutlm.metrics import EpochMetrics, ValidationMetricsResult
from app.extraction.layoutlm.schema import InvoiceLabelSchema


def move_inputs(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: batch[key].to(device, non_blocking=True)
        for key in ("input_ids", "attention_mask", "bbox", "pixel_values")
    }


@dataclass
class LayoutLMEvaluation:
    """Evaluate word labels and strict BIO entities after restoring complete pages."""

    model: nn.Module
    metrics: EpochMetrics
    schema: InvoiceLabelSchema
    device: torch.device
    criterion: nn.Module

    @torch.inference_mode()
    def evaluate(self, loader: DataLoader, epoch: int) -> ValidationMetricsResult:
        self.model.eval()
        self.metrics.reset()

        word_predictions = {}

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
