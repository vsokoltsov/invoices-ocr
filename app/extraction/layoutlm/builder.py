import json
from collections import defaultdict
from collections.abc import Mapping

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from app.extraction.layoutlm.schema import InvoiceLabelSchema


class InvoiceBuilder:
    """Build page records from OCR words and annotation boxes."""

    def __init__(self, fields, label2id=None, min_coverage=0.5):
        """Accept field-to-entity paths or invoice field names and explicit label IDs."""
        if not 0 < min_coverage <= 1:
            raise ValueError("min_coverage must be greater than zero and at most one")
        if isinstance(fields, Mapping):
            mapping = dict(fields)
        else:
            mapping = {}
            for field in fields:
                path = field if "." in field else f"invoice.{field}"
                mapping[path] = fields.get(path, field)
        schema = InvoiceLabelSchema(mapping)
        self.field_to_entity = schema.field_to_entity
        self.label2id = dict(schema.label2id if label2id is None else label2id)
        if set(schema.labels) - self.label2id.keys():
            raise ValueError("label2id must include O and B/I labels for every field")
        self.min_coverage = min_coverage

    @staticmethod
    def validate_box(value):
        box = np.asarray(value, dtype=np.float64)

        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError(f"Invalid box: {value}")

        x0, y0, x1, y1 = box

        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError(f"Invalid normalized coordinates: {value}")

        return box

    @staticmethod
    def word_coverage(word_box, annotation_box):
        top_left = np.maximum(word_box[:2], annotation_box[:2])
        bottom_right = np.minimum(word_box[2:], annotation_box[2:])

        intersection = np.maximum(bottom_right - top_left, 0).prod()
        word_area = np.prod(word_box[2:] - word_box[:2])

        return float(intersection / word_area)

    def prepare_document(self, row):
        json_data = json.loads(row.json_data)
        metadata = json_data["metadata"]
        page_count = int(metadata["page_count"])
        annotations = []

        for field, entity in self.field_to_entity.items():
            annotation = json_data
            for key in field.split("."):
                annotation = annotation.get(key)
                if annotation is None:
                    break

            if annotation is None:
                continue

            if len(annotation["pages"]) != len(annotation["bbox"]):
                raise ValueError(f"Annotation lengths differ: {field}")

            for page, box in zip(annotation["pages"], annotation["bbox"]):
                page = int(page)

                if not 0 <= page < page_count:
                    raise ValueError(f"Invalid annotation page: {page}")

                annotations.append(
                    {
                        "field": entity,
                        "page": page,
                        "box": self.validate_box(box),
                    }
                )

        words_by_page = defaultdict(list)
        matched_annotations = set()

        # Preserve the OCR order used by the parquet text offsets.
        words = sorted(row.text_with_boxes, key=lambda word: word["start"])

        for word in words:
            if not word["text"].strip():
                continue

            page = int(word["page"])

            if not 0 <= page < page_count:
                raise ValueError(f"Invalid OCR page: {page}")

            box = self.validate_box(word["bbox"])
            candidates = []

            for annotation_id, annotation in enumerate(annotations):
                if annotation["page"] != page:
                    continue

                coverage = self.word_coverage(box, annotation["box"])

                if coverage >= self.min_coverage:
                    candidates.append(annotation_id)

            if len(candidates) > 1:
                raise ValueError(f"Ambiguous word: {word['text']!r}, page={page}")

            owner = candidates[0] if candidates else None

            if owner is not None:
                matched_annotations.add(owner)

            words_by_page[page].append(
                {
                    "text": word["text"],
                    "box": (box * 1000).astype(np.int64).tolist(),
                    "owner": owner,
                }
            )

        missing = [
            annotation["field"]
            for index, annotation in enumerate(annotations)
            if index not in matched_annotations
        ]

        if missing:
            raise ValueError(f"Annotations without OCR matches: {missing}")

        if not words_by_page:
            raise ValueError("Document has no OCR words")

        records = []

        for page, page_words in sorted(words_by_page.items()):
            labels = []
            previous_owner = None

            for word in page_words:
                owner = word["owner"]

                if owner is None:
                    label = "O"
                else:
                    prefix = "I" if owner == previous_owner else "B"
                    field = annotations[owner]["field"]
                    label = f"{prefix}-{field}"

                labels.append(self.label2id[label])
                previous_owner = owner

            records.append(
                {
                    "document_id": row.document_id,
                    "pdf_path": str(row.pdf_path),
                    "page": page,
                    "words": [word["text"] for word in page_words],
                    "boxes": [word["box"] for word in page_words],
                    "labels": labels,
                }
            )

        return records

    def build(self, frame):
        records = []
        rejected = []

        for row in tqdm(
            frame.itertuples(index=False),
            total=len(frame),
            desc="Prepare documents",
        ):
            try:
                document_records = self.prepare_document(row)
            except ValueError as error:
                rejected.append(
                    {
                        "document_id": row.document_id,
                        "reason": str(error),
                    }
                )
                continue

            records.extend(document_records)

        return records, pd.DataFrame(
            rejected,
            columns=pd.Index(["document_id", "reason"]),
        )
