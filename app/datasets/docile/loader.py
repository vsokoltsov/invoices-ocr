import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from tqdm.std import tqdm

from app.datasets.docile.converter import DocileAnnotationConverter
from app.datasets.docile.corrections import DocileAnnotationCorrections
from app.datasets.docile.enums import DocileSplit
from app.datasets.docile.models import BBox, DocileRecord, OCRWord
from app.datasets.docile.order import OCRReadingOrder


@dataclass(init=False)
class DocileDatasetLoader:
    """Load OCR and optionally correct field types in memory before conversion."""

    root: Path
    use_snapped_boxes: bool = True
    corrections: DocileAnnotationCorrections | None = None

    def __init__(
        self,
        root: str | Path,
        use_snapped_boxes: bool = True,
        *,
        corrections_path: str | Path | None = None,
    ) -> None:
        self.root = Path(root).expanduser()
        self.use_snapped_boxes = use_snapped_boxes
        self.corrections = (
            DocileAnnotationCorrections.from_file(corrections_path)
            if corrections_path
            else None
        )

    @staticmethod
    def _parse_box(geometry: list) -> BBox:
        """Convert OCR corners to the annotation box format."""
        if len(geometry) != 2:
            raise ValueError(f"Expected two box corners: {geometry!r}")

        (left, top), (right, bottom) = geometry

        return (
            float(left),
            float(top),
            float(right),
            float(bottom),
        )

    def document_ids(
        self,
        split: DocileSplit | str = DocileSplit.TRAINVAL,
    ) -> list[str]:
        split = DocileSplit(split)
        path = self.root / f"{split}.json"

        with path.open(encoding="utf-8") as file:
            document_ids = json.load(file)

        if not isinstance(document_ids, list) or not all(
            isinstance(document_id, str) for document_id in document_ids
        ):
            raise ValueError(f"{path} must contain a list of document IDs.")
        if len(document_ids) != len(set(document_ids)):
            raise ValueError(f"{path} contains duplicate document IDs.")

        return document_ids

    def load(self, document_id: str) -> DocileRecord:
        """Load OCR and annotations while preserving OCR text and offsets.

        Words are ordered by page and visual row, joined by spaces, and given
        character offsets into the resulting text. Annotations are converted
        to invoice fields while preserving their boxes and pages.
        """
        if (
            not document_id
            or document_id in {".", ".."}
            or "/" in document_id
            or "\\" in document_id
        ):
            raise ValueError("document_id must be a file stem.")

        annotation_path = self.root / "annotations" / f"{document_id}.json"
        ocr_path = self.root / "ocr" / f"{document_id}.json"

        with annotation_path.open(encoding="utf-8") as file:
            json_data = json.load(file)
        if self.corrections is not None:
            json_data = self.corrections.apply(document_id, json_data)

        with ocr_path.open(encoding="utf-8") as file:
            ocr_data = json.load(file)

        parts = []
        words = []
        position = 0

        for page_position, page in enumerate(ocr_data["pages"]):
            page_index = int(page.get("page_idx", page_position))

            for block_index, block in enumerate(page["blocks"]):
                for line_index, line in enumerate(block["lines"]):
                    line_words = [
                        (word_index, word)
                        for word_index, word in enumerate(line["words"])
                        if word["value"].strip()
                    ]

                    if not line_words:
                        continue

                    if parts:
                        separator = " "
                        parts.append(separator)
                        position += len(separator)

                    for index_in_line, (word_index, word) in enumerate(line_words):
                        if index_in_line:
                            parts.append(" ")
                            position += 1

                        value = word["value"]

                        geometry = word["geometry"]
                        if self.use_snapped_boxes:
                            geometry = word.get("snapped_geometry") or geometry

                        start = position
                        parts.append(value)
                        position += len(value)

                        words.append(
                            OCRWord(
                                text=value,
                                bbox=self._parse_box(geometry),
                                page=page_index,
                                start=start,
                                end=position,
                                confidence=word.get("confidence"),
                                block_index=block_index,
                                line_index=line_index,
                                word_index=word_index,
                            )
                        )

        ocr_text = "".join(parts)

        ocr_text, words = OCRReadingOrder().build(words)
        json_data = DocileAnnotationConverter().convert(json_data)

        return DocileRecord(
            document_id=document_id,
            ocr_text=ocr_text,
            text_with_boxes=words,
            json_data=json_data,
        )

    def iter_split(
        self,
        split: DocileSplit | str = DocileSplit.TRAIN,
    ) -> Iterator[DocileRecord]:
        """Load documents one at a time rather than preloading the split."""
        for document_id in self.document_ids(split):
            yield self.load(document_id)

    def to_dataframe(
        self,
        split: DocileSplit | str = DocileSplit.TRAINVAL,
        *,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Load a split, preserving nested annotations and OCR word metadata."""
        document_ids = self.document_ids(split)
        rows = []

        for document_id in tqdm(
            document_ids,
            desc=f"Loading {split}",
            unit="document",
            disable=not show_progress,
        ):
            try:
                record = self.load(document_id)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load document {document_id}: {exc}"
                ) from exc

            rows.append(
                {
                    "document_id": record.document_id,
                    "ocr_text": record.ocr_text,
                    "text_with_boxes": [
                        asdict(word) for word in record.text_with_boxes
                    ],
                    "json_data": record.json_data,
                    "pdf_path": str(self.root / "pdfs" / f"{document_id}.pdf"),
                }
            )
            if self.corrections is not None:
                rows[-1]["annotation_corrections"] = [
                    rule.model_dump(mode="json")
                    for rule in self.corrections.for_document(document_id)
                ]

        return pd.DataFrame(rows)
