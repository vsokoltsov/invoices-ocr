from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.extraction.schema import InvoiceLabelSchema
from app.pipelines.preprocess import DataFramePipeline, DecodeDocuments, DocumentStep


class BoxGeometry:
    """Validate normalized boxes and measure OCR word coverage."""

    @staticmethod
    def validate_box(value):
        """Validate one normalized box.

        Args:
            value: Four coordinates in x0, y0, x1, y1 order.

        Returns:
            A finite NumPy box with positive width and height.

        Raises:
            ValueError: Coordinates are invalid or outside the page.
        """
        box = np.asarray(value, dtype=np.float64)
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError(f"Invalid box: {value}")
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError(f"Invalid normalized coordinates: {value}")
        return box

    @staticmethod
    def box_area(box):
        """Calculate the area of a validated normalized box.

        Args:
            box: Valid coordinates in x0, y0, x1, y1 order.

        Returns:
            Box width multiplied by box height.
        """
        return float((box[2] - box[0]) * (box[3] - box[1]))

    @staticmethod
    def word_coverage(word_box, annotation_box):
        """Calculate the fraction of a word box covered by an annotation.

        Args:
            word_box: Valid OCR word box.
            annotation_box: Valid reference annotation box.

        Returns:
            Intersection area divided by word area.
        """
        top_left = np.maximum(word_box[:2], annotation_box[:2])
        bottom_right = np.minimum(word_box[2:], annotation_box[2:])
        intersection = np.maximum(bottom_right - top_left, 0).prod()
        return float(intersection / np.prod(word_box[2:] - word_box[:2]))


@dataclass
class CollectAnnotations(DocumentStep):
    """Select and validate annotation occurrences for the configured fields.

    Attributes:
        fields: Mapping from invoice field paths to entity names.
    """

    fields: Mapping[str, str]
    output_column = "annotations"

    def process(self, row: pd.Series) -> list[dict]:
        """Read annotation boxes without changing their contents.

        Args:
            row: A document with decoded annotations and page metadata.

        Returns:
            Selected field occurrences with validated boxes and page indices.

        Raises:
            ValueError: An annotation page, box, or occurrence count is invalid.
        """
        page_count = int(row.json_data["metadata"]["page_count"])
        annotations = []
        for field, entity in self.fields.items():
            annotation = row.json_data
            for key in field.split("."):
                annotation = annotation.get(key)
                if annotation is None:
                    break
            if annotation is None:
                continue
            if len(annotation["pages"]) != len(annotation["bbox"]):
                raise ValueError(f"Annotation lengths differ: {field}")
            for page, box in zip(annotation["pages"], annotation["bbox"], strict=True):
                page = int(page)
                if not 0 <= page < page_count:
                    raise ValueError(f"Invalid annotation page: {page}")
                annotations.append(
                    {
                        "field": entity,
                        "page": page,
                        "box": BoxGeometry.validate_box(box),
                    }
                )
        return annotations


@dataclass
class AssignWords(DocumentStep):
    """Assign words by rounded coverage, then prefer smaller annotation boxes.

    Attributes:
        min_coverage: Minimum fraction of a word covered by an annotation.
    """

    min_coverage: float
    output_column = "assignment"

    def process(self, row: pd.Series) -> tuple[dict, set]:
        """Select one annotation for each word with sufficient box coverage.

        Args:
            row: Document with selected annotations and OCR words.

        Returns:
            Words grouped by page and IDs of annotations matched by OCR.

        Raises:
            ValueError: A word has invalid coordinates or an invalid page.
        """
        words_by_page = defaultdict(list)
        matched = set()
        page_count = int(row.json_data["metadata"]["page_count"])
        for word in sorted(row.text_with_boxes, key=lambda word: word["start"]):
            if not word["text"].strip():
                continue
            page = int(word["page"])
            if not 0 <= page < page_count:
                raise ValueError(f"Invalid OCR page: {page}")
            box = BoxGeometry.validate_box(word["bbox"])
            matches = []
            for annotation_id, annotation in enumerate(row["annotations"]):
                if annotation["page"] != page:
                    continue

                coverage = BoxGeometry.word_coverage(box, annotation["box"])
                if coverage >= self.min_coverage:
                    matches.append(
                        {
                            "annotation_id": annotation_id,
                            "coverage": coverage,
                            "area": BoxGeometry.box_area(annotation["box"]),
                        }
                    )

            owner: int | None = None
            if matches:
                best = min(
                    matches,
                    key=lambda match: (
                        -round(match["coverage"], 1),
                        match["area"],
                        match["annotation_id"],
                    ),
                )
                owner = best["annotation_id"]
                matched.add(owner)

            words_by_page[page].append(
                {
                    "text": word["text"],
                    "box": (box * 1000).astype(np.int64).tolist(),
                    "owner": owner,
                }
            )
        return dict(words_by_page), matched


class ValidateMatches(DocumentStep):
    """Reject unmatched annotations and documents without OCR words."""

    output_column = "matched_pages"

    def process(self, row: pd.Series) -> dict:
        """Reject annotations that received no words after owner selection.

        An unmatched box that contains a selected box is reported as superseded.
        It still rejects the document.

        Args:
            row: Document with selected annotations and word assignments.

        Returns:
            Validated words grouped by page.

        Raises:
            ValueError: An annotation has no OCR match, or no words exist.
        """
        words_by_page, matched = row.assignment
        if not words_by_page:
            raise ValueError("Document has no OCR words")

        issues = []
        for annotation_id, annotation in enumerate(row["annotations"]):
            if annotation_id in matched:
                continue

            is_superseded = False

            for matched_id in matched:
                selected = row["annotations"][matched_id]

                if selected["page"] != annotation["page"]:
                    continue

                # Measure how much of the selected box lies in this box.
                containment = BoxGeometry.word_coverage(
                    selected["box"],
                    annotation["box"],
                )

                if containment > 0.8:
                    is_superseded = True
                    break

            reason = (
                "superseded_by_nested_annotation"
                if is_superseded
                else "annotation_without_assigned_words"
            )
            issues.append(
                f"{reason}: {annotation['field']} (annotation_id={annotation_id})"
            )

        if issues:
            raise ValueError("; ".join(issues))

        return words_by_page


@dataclass
class CreatePages(DocumentStep):
    """Convert validated word assignments into LayoutLM page records.

    Attributes:
        label2id: Mapping from BIO label names to integer targets.
    """

    label2id: dict[str, int]
    output_column = "pages"

    def process(self, row: pd.Series) -> list[dict]:
        """Create page records with OCR words, integer boxes, and BIO labels.

        Args:
            row: Document with validated page words and source annotations.

        Returns:
            One training record per page with OCR words.
        """
        records = []
        for page, words in sorted(row.matched_pages.items()):
            labels = []
            previous_owner = None
            for word in words:
                owner = word["owner"]
                if owner is None:
                    label = "O"
                else:
                    prefix = "I" if owner == previous_owner else "B"
                    label = f"{prefix}-{row.annotations[owner]['field']}"
                labels.append(self.label2id[label])
                previous_owner = owner
            records.append(
                {
                    "document_id": row.document_id,
                    "pdf_path": str(row.pdf_path),
                    "page": page,
                    "words": [word["text"] for word in words],
                    "boxes": [word["box"] for word in words],
                    "labels": labels,
                }
            )
        return records


class InvoicePreprocessor(BoxGeometry):
    """Run annotation matching and validation steps to create page labels."""

    def __init__(self, fields, label2id=None, min_coverage=0.5):
        """Configure field selection and word matching.

        Args:
            fields: Mapping from invoice field paths to entity names.
            label2id: Optional BIO label IDs; None uses the field schema.
            min_coverage: Required annotation coverage of each OCR word.

        Raises:
            ValueError: Coverage or label mappings are invalid.
        """
        if not 0 < min_coverage <= 1:
            raise ValueError("min_coverage must be greater than zero and at most one")
        schema = InvoiceLabelSchema(fields)
        self.field_to_entity = schema.field_to_entity
        self.label2id = dict(schema.label2id if label2id is None else label2id)
        if set(schema.labels) - self.label2id.keys():
            raise ValueError("label2id must include O and B/I labels for every field")
        self.min_coverage = min_coverage

    def pipeline(self) -> DataFramePipeline:
        """Compose independent annotation, assignment, validation, and page steps.

        Returns:
            An ordered preprocessing pipeline executed by PipeFunc.
        """
        return DataFramePipeline(
            (
                DecodeDocuments(),
                CollectAnnotations(self.field_to_entity),
                AssignWords(self.min_coverage),
                ValidateMatches(),
                CreatePages(self.label2id),
            )
        )

    def prepare_document(self, row) -> list[dict]:
        """Preprocess one document and raise its validation error directly.

        Args:
            row: A Series or namespace containing one source document.

        Returns:
            LayoutLM page records for the document.

        Raises:
            ValueError: The document fails preprocessing validation.
        """
        values = row.to_dict() if isinstance(row, pd.Series) else vars(row)
        prepared = self.pipeline().run(pd.DataFrame([values])).iloc[0]
        if prepared.reason:
            raise ValueError(prepared.reason)
        return prepared.pages

    def run(self, frame: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
        """Preprocess a partition and separate page records from rejections.

        Args:
            frame: Documents with annotations, OCR words, and PDF paths.

        Returns:
            Accepted page records and per-document rejection reasons.
        """
        prepared = self.pipeline().run(frame)
        records = [
            page
            for pages in prepared.loc[prepared.reason.eq(""), "pages"]
            for page in pages
        ]
        rejected = prepared.loc[
            prepared.reason.ne(""), ["document_id", "reason"]
        ].reset_index(drop=True)
        return records, rejected
