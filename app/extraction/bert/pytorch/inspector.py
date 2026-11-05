import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import structlog
import torch
from seqeval.metrics import classification_report
from seqeval.metrics.sequence_labeling import get_entities
from transformers import PreTrainedTokenizerBase

from app.extraction.bert.pytorch.datasets import InvoiceDataset
from app.extraction.bert.pytorch.schema import InvoiceLabelSchema

logger = structlog.get_logger()


@dataclass
class EntityErrorInspector:
    model: torch.nn.Module
    tokenizer: PreTrainedTokenizerBase
    label_schema: InvoiceLabelSchema
    device: torch.device
    max_length: int = 512
    debug_document_id: str | None = "657e3c28c1d74e698a0cd884"
    debug_start: int = 1991
    debug_end: int = 2034
    training_records: list[dict] | None = None

    def __post_init__(self):
        if (
            self.debug_document_id is not None
            and not 0 <= self.debug_start < self.debug_end
        ):
            raise ValueError("Token-debug offsets must satisfy 0 <= start < end")

    def _save_token_debug(
        self,
        record,
        example,
        offsets,
        logits,
        output_dir,
        *,
        epoch,
        window_index,
        record_split="validation",
    ):
        """Retain all window tokens, including ignored and special tokens."""
        probabilities = logits.softmax(dim=-1)[0].cpu()
        predictions = logits.argmax(dim=-1)[0].cpu().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(example["input_ids"])
        address_probabilities = {}
        for entity in ("CLIENT_ADDRESS", "SELLER_ADDRESS"):
            ids = [
                self.label_schema.label2id[label]
                for prefix in ("B", "I")
                if (label := f"{prefix}-{entity}") in self.label_schema.label2id
            ]
            address_probabilities[entity] = (
                probabilities[:, ids].sum(dim=-1).tolist()
                if ids
                else [None] * len(tokens)
            )

        rows = []
        for index, (token, (start, end), actual, predicted) in enumerate(
            zip(
                tokens,
                offsets,
                example["labels"],
                predictions,
                strict=True,
            )
        ):
            rows.append(
                {
                    "epoch": epoch,
                    "document_id": record["document_id"],
                    "record_split": record_split,
                    "window_index": window_index,
                    "token_index": index,
                    "token": token,
                    "ocr_text": record["text"][start:end],
                    "start": start,
                    "end": end,
                    "actual": "IGNORED"
                    if actual == -100
                    else self.label_schema.id2label[actual],
                    "predicted": self.label_schema.id2label[predicted],
                    "predicted_probability": probabilities[index, predicted].item(),
                    "actual_probability": None
                    if actual == -100
                    else probabilities[index, actual].item(),
                    "client_address_probability": address_probabilities[
                        "CLIENT_ADDRESS"
                    ][index],
                    "seller_address_probability": address_probabilities[
                        "SELLER_ADDRESS"
                    ][index],
                }
            )
        path = (
            output_dir / f"tokens_{record['document_id']}_window_{window_index:03d}.csv"
        )
        pd.DataFrame(rows).to_csv(path, index=False)
        logger.info("Saved token diagnostics", path=str(path.resolve()), epoch=epoch)

    def _encode_record(self, record):
        dataset = InvoiceDataset(
            [record],
            self.tokenizer,
            self.label_schema.label2id,
            self.label_schema.field_to_entity,
            max_length=self.max_length,
        )
        encoded = self.tokenizer(
            record["text"],
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            stride=0,
        )
        return dataset, encoded["offset_mapping"]

    def _export_training_debug(self, record, output_dir, epoch):
        """Inspect selected training windows without adding them to evaluation."""
        dataset, offsets = self._encode_record(record)
        exported = 0
        for window_index, window_offsets in enumerate(offsets):
            if not any(
                start < self.debug_end and end > self.debug_start
                for start, end in window_offsets
            ):
                continue
            example = dataset[window_index]
            logits = self.model(
                **{
                    key: torch.tensor(
                        [example[key]], dtype=torch.long, device=self.device
                    )
                    for key in ("input_ids", "attention_mask")
                }
            )
            self._save_token_debug(
                record,
                example,
                window_offsets,
                logits,
                output_dir,
                epoch=epoch,
                window_index=window_index,
                record_split="train",
            )
            exported += 1
        return exported

    @staticmethod
    def _entities(labels, offsets, text):
        """Map BIO entity positions to original OCR character offsets."""
        entities = {}

        # seqeval returns an inclusive last-token index.
        for label, first, last in get_entities(labels):
            start = offsets[first][0]
            end = offsets[last][1]

            entities[(label, first, last)] = {
                "label": label,
                "start": start,
                "end": end,
                "text": text[start:end],
            }

        return entities

    @torch.inference_mode()
    def run(self, records, output_dir, *, epoch: int):
        """Export entity errors and token details for the configured debug region.

        Debug windows are selected by OCR character offsets. If a region crosses
        a window boundary, export each intersecting window. Set debug_document_id
        to None to disable the token export. A debug target found only in
        training_records produces token files marked train, without contributing
        to entity metrics or error tables.
        """
        self.model.eval()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        debug_document_found = False
        debug_windows = 0
        debug_split = None

        errors = []
        actual_sequences = []
        predicted_sequences = []

        for record_index, record in enumerate(records):
            text = record["text"]
            is_debug_document = (
                self.debug_document_id is not None
                and record.get("document_id") == self.debug_document_id
            )
            debug_document_found |= is_debug_document
            if is_debug_document:
                debug_split = "validation"

            dataset, record_offsets = self._encode_record(record)

            for window_index in range(len(dataset)):
                example = dataset[window_index]
                inputs = {
                    key: torch.tensor(
                        [example[key]],
                        dtype=torch.long,
                        device=self.device,
                    )
                    for key in ("input_ids", "attention_mask")
                }

                logits = self.model(**inputs)
                predictions = logits.argmax(dim=-1)[0].cpu().tolist()
                window_offsets = record_offsets[window_index]
                if is_debug_document and any(
                    start < self.debug_end and end > self.debug_start
                    for start, end in window_offsets
                ):
                    self._save_token_debug(
                        record,
                        example,
                        window_offsets,
                        logits,
                        output_dir,
                        epoch=epoch,
                        window_index=window_index,
                    )
                    debug_windows += 1

                # Apply exactly the same -100 mask as eval_one_epoch().
                valid_indices = [
                    index
                    for index, target in enumerate(example["labels"])
                    if target != -100
                ]
                if not valid_indices:
                    continue

                actual = [
                    self.label_schema.id2label[example["labels"][index]]
                    for index in valid_indices
                ]
                predicted = [
                    self.label_schema.id2label[predictions[index]]
                    for index in valid_indices
                ]
                offsets = [
                    record_offsets[window_index][index] for index in valid_indices
                ]

                actual_sequences.append(actual)
                predicted_sequences.append(predicted)

                gold = self._entities(actual, offsets, text)
                guessed = self._entities(predicted, offsets, text)

                comparisons = (
                    ("false_positive", guessed, gold),
                    ("false_negative", gold, guessed),
                )

                for status, source, reference in comparisons:
                    for key in sorted(source.keys() - reference.keys()):
                        entity = source[key]

                        overlapping = [
                            candidate
                            for candidate in reference.values()
                            if entity["start"] < candidate["end"]
                            and entity["end"] > candidate["start"]
                        ]

                        errors.append(
                            {
                                "document_id": record.get(
                                    "document_id",
                                    record.get("row_index", record_index),
                                ),
                                "window_index": window_index,
                                "status": status,
                                **entity,
                                "overlapping_entities": json.dumps(
                                    overlapping,
                                    ensure_ascii=False,
                                ),
                                "context": text[
                                    max(0, entity["start"] - 100) : min(
                                        len(text), entity["end"] + 100
                                    )
                                ],
                            }
                        )

        if self.debug_document_id is not None and not debug_document_found:
            debug_record = next(
                (
                    record
                    for record in self.training_records or []
                    if record.get("document_id") == self.debug_document_id
                ),
                None,
            )
            if debug_record is not None:
                debug_document_found = True
                debug_split = "train"
                debug_windows = self._export_training_debug(
                    debug_record, output_dir, epoch
                )

        report = classification_report(
            actual_sequences,
            predicted_sequences,
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report).T

        errors_df = pd.DataFrame(
            errors,
            columns=pd.Index(
                [
                    "document_id",
                    "window_index",
                    "status",
                    "label",
                    "start",
                    "end",
                    "text",
                    "overlapping_entities",
                    "context",
                ]
            ),
        )

        if self.debug_document_id is not None:
            logger.info(
                "Token debug selection",
                document_id=self.debug_document_id,
                document_found=debug_document_found,
                record_split=debug_split,
                exported_windows=debug_windows,
                start=self.debug_start,
                end=self.debug_end,
                epoch=epoch,
            )
        report_df["epoch"] = epoch
        errors_df["epoch"] = epoch
        report_df.to_csv(
            output_dir / "per_entity.csv",
            index_label="entity",
        )
        errors_df.to_csv(output_dir / "entity_errors.csv", index=False)

        for status in ("false_positive", "false_negative"):
            errors_df.loc[errors_df["status"] == status].to_csv(
                output_dir / f"{status}.csv",
                index=False,
            )

        return report_df, errors_df
