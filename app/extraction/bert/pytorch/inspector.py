import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from seqeval.metrics import classification_report
from seqeval.metrics.sequence_labeling import get_entities
from transformers import PreTrainedTokenizerBase

from app.extraction.bert.pytorch.datasets import InvoiceDataset
from app.extraction.bert.pytorch.decoding import OCRWordDecoder
from app.extraction.bert.pytorch.schema import InvoiceLabelSchema


@dataclass
class EntityErrorInspector:
    model: torch.nn.Module
    tokenizer: PreTrainedTokenizerBase
    label_schema: InvoiceLabelSchema
    device: torch.device
    max_length: int = 512
    word_decoder: OCRWordDecoder | None = None

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
        """Export validation entity metrics and errors with the evaluation mask."""
        self.model.eval()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        errors = []
        actual_sequences = []
        predicted_sequences = []

        for record_index, record in enumerate(records):
            text = record["text"]
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
                if self.word_decoder is not None:
                    predictions = self.word_decoder.decode(
                        logits[0],
                        record_offsets[window_index],
                        record["text_with_boxes"],
                    )
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
