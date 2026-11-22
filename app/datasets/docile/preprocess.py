import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.datasets.docile.converter import DocileAnnotationConverter
from app.datasets.docile.corrections import DocileAnnotationCorrections
from app.datasets.docile.models import OCRWord


@dataclass
class ApplyCorrections:
    """Apply reviewed annotation corrections before field conversion.

    Attributes:
        corrections: Validated correction manifest; None keeps the source labels.
    """

    corrections: DocileAnnotationCorrections | None

    def __call__(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return corrected annotations with per-document audit records.

        Args:
            frame: Original DocILE document rows.

        Returns:
            A copy with optional corrected annotations and audit records.

        Raises:
            RuntimeError: A correction does not match the selected source document.
        """
        result = frame.copy()
        if self.corrections is None:
            return result
        values, audits = [], []
        for _, row in frame.iterrows():
            try:
                values.append(self.corrections.apply(row.document_id, row.json_data))
            except ValueError as error:
                raise RuntimeError(
                    f"Failed to preprocess document {row.document_id}: {error}"
                ) from error
            audits.append(
                [
                    rule.model_dump(mode="json")
                    for rule in self.corrections.for_document(row.document_id)
                ]
            )
        result["json_data"] = values
        result["annotation_corrections"] = audits
        return result


@dataclass
class ExtractOCRWords:
    """Convert original OCR blocks into word records without ordering them.

    Attributes:
        use_snapped_boxes: Select snapped word geometry when available.
    """

    use_snapped_boxes: bool = True

    def __call__(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Flatten OCR hierarchy and preserve source word metadata.

        Args:
            frame: Original rows containing ocr_data.

        Returns:
            A copy with text_with_boxes. Character offsets are assigned later.

        Raises:
            ValueError: OCR geometry does not contain two box corners.
        """
        result = frame.copy()
        documents = []
        for _, row in frame.iterrows():
            words = []
            for page_position, page in enumerate(row.ocr_data["pages"]):
                page_index = int(page.get("page_idx", page_position))
                for block_index, block in enumerate(page["blocks"]):
                    for line_index, line in enumerate(block["lines"]):
                        for word_index, word in enumerate(line["words"]):
                            if not word["value"].strip():
                                continue
                            geometry = (
                                word.get("snapped_geometry")
                                if self.use_snapped_boxes
                                else None
                            )
                            geometry = geometry or word["geometry"]
                            if len(geometry) != 2:
                                raise ValueError(
                                    f"Invalid OCR geometry in {row.document_id}: {geometry!r}"
                                )
                            (left, top), (right, bottom) = geometry
                            words.append(
                                OCRWord(
                                    text=word["value"],
                                    bbox=(
                                        float(left),
                                        float(top),
                                        float(right),
                                        float(bottom),
                                    ),
                                    page=page_index,
                                    start=0,
                                    end=0,
                                    confidence=word.get("confidence"),
                                    block_index=block_index,
                                    line_index=line_index,
                                    word_index=word_index,
                                )
                            )
            documents.append(words)
        result["text_with_boxes"] = documents
        return result


class ConvertAnnotations:
    """Map DocILE field types and line items to the invoice schema."""

    def __call__(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Convert annotations without changing their boxes or source values.

        Args:
            frame: Documents with original or corrected DocILE annotations.

        Returns:
            A copy containing invoice-schema annotations.
        """
        result = frame.copy()
        result["json_data"] = result["json_data"].map(
            DocileAnnotationConverter().convert
        )
        return result.drop(columns=["ocr_data"])


@dataclass
class DocileSource:
    """Read raw DocILE Parquet and adapt its original OCR and annotations.

    Attributes:
        dataset_path: Raw Parquet written by DocileDatasetLoader.save.
        use_snapped_boxes: Whether to select snapped OCR geometry.
        corrections_path: Optional reviewed annotation correction manifest.
    """

    dataset_path: Path
    use_snapped_boxes: bool = True
    corrections_path: str | Path | None = None

    def load(self) -> pd.DataFrame:
        """Read original records from the raw snapshot without accessing JSON files.

        Returns:
            Raw document records with decoded OCR and annotation containers.

        Raises:
            ValueError: The input is not a valid raw DocILE Parquet dataset.
        """
        path = Path(self.dataset_path).expanduser().resolve()
        if path.suffix.lower() != ".parquet":
            raise ValueError("Raw DocILE input must be a .parquet file")
        frame = pd.read_parquet(path)
        required = {"document_id", "ocr_data", "json_data", "pdf_path"}
        if missing := required - set(frame.columns):
            raise ValueError(
                f"Raw DocILE Parquet is missing columns: {sorted(missing)}. Run datasets load docile first."
            )
        if frame.empty:
            raise ValueError("Raw DocILE Parquet has no documents")
        if frame[list(required)].isna().any().any():
            raise ValueError("Raw DocILE documents must not contain missing values")
        if frame["document_id"].duplicated().any():
            raise ValueError("Raw DocILE document IDs must be unique")
        for column in ("ocr_data", "json_data"):
            frame[column] = frame[column].map(
                lambda value: json.loads(value) if isinstance(value, str) else value
            )
        return frame

    def normalization_steps(self):
        """Configure transformations specific to the DocILE source format.

        Returns:
            Ordered steps for corrections, OCR conversion, and annotation conversion.
        """
        corrections = (
            DocileAnnotationCorrections.from_file(self.corrections_path)
            if self.corrections_path
            else None
        )
        return (
            ApplyCorrections(corrections),
            ExtractOCRWords(self.use_snapped_boxes),
            ConvertAnnotations(),
        )
