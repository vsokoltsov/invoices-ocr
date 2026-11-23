"""Inspect LayoutLM validation errors using the training window contract."""

import json
from collections import Counter
from dataclasses import asdict, dataclass
from typing import ClassVar, cast

import pandas as pd
import structlog
import torch
from seqeval.scheme import IOB2, Entities
from torch.utils.data import DataLoader

from app.extraction.batches import TensorBatchAdapter, token_loss
from app.extraction.layoutlm.artifacts import LayoutLMArtifacts
from app.extraction.layoutlm.converter import AttachmentToImageConverter
from app.extraction.layoutlm.dataset import InvoiceDataset
from app.extraction.layoutlm.visualization import ErrorPageRenderer
from app.extraction.metrics import WindowMetrics
from app.extraction.schema import InvoiceLabelSchema

logger = structlog.get_logger()


@dataclass
class EntityErrorInspector:
    """Export validation predictions, scores, and strict entity errors.

    Attributes:
        model: Current token classifier on the configured device.
        label_schema: Field names and BIO label IDs used in training.
        device: Device used for the additional validation inference pass.
        converter: Optional page loader used to draw error examples.
        max_visualizations: Maximum number of error pages saved per epoch.
    """

    model: torch.nn.Module
    label_schema: InvoiceLabelSchema
    device: torch.device
    converter: AttachmentToImageConverter | None = None
    max_visualizations: int = 10

    batch_adapter: ClassVar[TensorBatchAdapter] = TensorBatchAdapter(
        ("input_ids", "attention_mask", "bbox", "pixel_values")
    )
    error_columns: ClassVar[list[str]] = [
        "document_id",
        "page",
        "window_index",
        "status",
        "error_type",
        "label",
        "start",
        "end",
        "text",
        "bbox",
        "overlapping_entities",
        "context",
        "epoch",
    ]

    @staticmethod
    def _entities(labels: list[str], word_ids: list[int], page: dict) -> dict:
        """Map strict window spans to page word indices and integer boxes."""
        result = {}
        for entity in Entities([labels], IOB2).entities[0]:
            indices = word_ids[entity.start : entity.end]
            boxes = [page["boxes"][index] for index in indices]
            result[(entity.tag, entity.start, entity.end)] = {
                "label": entity.tag,
                "start": indices[0],
                "end": indices[-1] + 1,
                "text": " ".join(page["words"][index] for index in indices),
                "bbox": [
                    min(box[0] for box in boxes),
                    min(box[1] for box in boxes),
                    max(box[2] for box in boxes),
                    max(box[3] for box in boxes),
                ],
            }
        return result

    @staticmethod
    def _invalid_transition(label: str, previous: str) -> bool:
        """Identify I-tags without a preceding B/I-tag of the same field."""
        return label.startswith("I-") and previous not in {
            f"B-{label[2:]}",
            f"I-{label[2:]}",
        }

    def _word_rows(self, page, word_ids, actual, predicted, confidence) -> list[dict]:
        """Keep one row per supervised OCR word with raw prediction labels."""
        rows = []
        previous = "O"
        for word_id, gold, guess, probability in zip(
            word_ids, actual, predicted, confidence, strict=True
        ):
            rows.append(
                {
                    "word_index": word_id,
                    "text": page["words"][word_id],
                    "bbox": json.dumps(page["boxes"][word_id]),
                    "gold_label": gold,
                    "predicted_label": guess,
                    "confidence": probability,
                    "correct": gold == guess,
                    "invalid_bio": self._invalid_transition(guess, previous),
                }
            )
            previous = guess
        return rows

    def _entity_errors(self, actual, predicted, word_ids, page) -> list[dict]:
        """Separate strict span errors into boundary, field, and missing/extra cases."""
        gold = self._entities(actual, word_ids, page)
        guessed = self._entities(predicted, word_ids, page)
        errors = []
        for status, source, reference in (
            ("false_positive", guessed, gold),
            ("false_negative", gold, guessed),
        ):
            for key in sorted(source.keys() - reference.keys()):
                entity = source[key]
                overlapping = [
                    value
                    for value in reference.values()
                    if entity["start"] < value["end"] and entity["end"] > value["start"]
                ]
                error_type = "extra" if status == "false_positive" else "missed"
                if overlapping:
                    error_type = (
                        "boundary"
                        if any(
                            value["label"] == entity["label"] for value in overlapping
                        )
                        else "wrong_field"
                    )
                errors.append(
                    {
                        "status": status,
                        "error_type": error_type,
                        **entity,
                        "bbox": json.dumps(entity["bbox"]),
                        "overlapping_entities": json.dumps(
                            overlapping, ensure_ascii=False
                        ),
                        "context": " ".join(
                            page["words"][
                                max(0, entity["start"] - 5) : entity["end"] + 5
                            ]
                        ),
                    }
                )
        return errors

    def _collect(self, loader: DataLoader) -> tuple[WindowMetrics, list, list]:
        """Run the existing validation windows with the same supervision mask."""
        metrics = WindowMetrics(self.label_schema.id2label)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=metrics.ignore_index)
        predictions, errors = [], []
        dataset = cast(InvoiceDataset, loader.dataset)
        window_index = 0
        for batch in loader:
            labels = batch["labels"].to(self.device)
            if not (labels != metrics.ignore_index).any():
                window_index += len(labels)
                continue
            logits = self.model(**self.batch_adapter(batch, self.device))
            loss = token_loss(logits, labels, criterion, "Inspection")
            guesses = logits.argmax(dim=-1)
            metrics.update(logits, labels, loss, predictions=guesses)
            probabilities = (
                logits.softmax(dim=-1).gather(-1, guesses.unsqueeze(-1)).squeeze(-1)
            )
            for index in range(len(labels)):
                valid = batch["labels"][index] != metrics.ignore_index
                if not valid.any():
                    window_index += 1
                    continue
                page = dataset.pages[int(batch["page_index"][index])]
                word_ids = batch["word_ids"][index][valid].tolist()
                actual = [
                    self.label_schema.id2label[value]
                    for value in batch["labels"][index][valid].tolist()
                ]
                predicted = [
                    self.label_schema.id2label[value]
                    for value in guesses[index].detach().cpu()[valid].tolist()
                ]
                confidence = probabilities[index].detach().cpu()[valid].tolist()
                metadata = {
                    "document_id": page["document_id"],
                    "page": page["page"],
                    "window_index": window_index,
                }
                predictions.extend(
                    {**metadata, **row}
                    for row in self._word_rows(
                        page, word_ids, actual, predicted, confidence
                    )
                )
                errors.extend(
                    {**metadata, **row}
                    for row in self._entity_errors(actual, predicted, word_ids, page)
                )
                window_index += 1
        return metrics, predictions, errors

    @torch.inference_mode()
    def run(self, loader: DataLoader, artifacts: LayoutLMArtifacts, *, epoch: int):
        """Save diagnostics after one complete validation epoch.

        Args:
            loader: Validation loader with page_index and source word_ids metadata.
            artifacts: Output storage for this epoch.
            epoch: One-based epoch included in report rows.

        Returns:
            Per-entity scores and false-positive/false-negative records.

        Notes:
            Performs an extra inference pass. Uses strict IOB2 per window and
            ignores padding, special tokens, and continuation subtokens. Restores
            the model's original training mode, including when inspection fails.
        """
        was_training = self.model.training
        self.model.eval()
        try:
            # A second loader pass must not change the next training shuffle.
            with torch.random.fork_rng(devices=[]):
                metrics, words, errors = self._collect(loader)
        finally:
            self.model.train(was_training)
        result = metrics.compute()
        report = pd.DataFrame(result.entity_report).T.assign(epoch=epoch)
        error_frame = (
            pd.DataFrame(errors).reindex(columns=self.error_columns).assign(epoch=epoch)
        )
        word_frame = pd.DataFrame(words).assign(epoch=epoch)
        artifacts.save_diagnostics(report, error_frame)
        fields = pd.DataFrame(
            {
                name: asdict(value)
                for name, value in result.word_metrics.per_field.items()
            }
        ).T
        fields["support"] = fields["support"].astype(int)
        artifacts.write_csv(
            artifacts.output_dir / "per_field.csv",
            fields.assign(epoch=epoch),
            index_label="field",
        )
        artifacts.write_csv(artifacts.output_dir / "word_predictions.csv", word_frame)
        confusion = pd.DataFrame(
            metrics.words.confusion.numpy(),
            index=pd.Index(self.label_schema.labels),
            columns=pd.Index(self.label_schema.labels),
        )
        artifacts.write_csv(
            artifacts.output_dir / "confusion.csv", confusion, index_label="gold_label"
        )
        invalid_count = int(word_frame.invalid_bio.sum())
        artifacts.write_json(
            artifacts.output_dir / "summary.json",
            {
                "epoch": epoch,
                "metric_contract": result.metric_contract,
                **result.scalars,
                "windows": len(metrics.actual_sequences),
                "words": len(words),
                "incorrect_bio_words": int((~word_frame.correct).sum()),
                "invalid_bio_transitions": invalid_count,
                "invalid_bio_rate": invalid_count / len(words),
                "entity_errors": dict(Counter(error_frame.error_type)),
            },
        )
        if self.converter is not None:
            ErrorPageRenderer(self.converter, self.max_visualizations).save(
                cast(InvoiceDataset, loader.dataset).pages,
                word_frame,
                error_frame,
                artifacts.output_dir,
            )
        logger.info(
            "Validation diagnostics saved",
            epoch=epoch,
            output_dir=str(artifacts.output_dir),
            entity_errors=len(error_frame),
            invalid_bio_transitions=invalid_count,
        )
        return report, error_frame
