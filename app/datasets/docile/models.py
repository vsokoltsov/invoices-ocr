from dataclasses import dataclass
from typing import Any

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class OCRWord:
    text: str
    bbox: BBox
    page: int
    start: int
    end: int
    confidence: float | None
    block_index: int
    line_index: int
    word_index: int


@dataclass
class DocileRecord:
    """Store original DocILE data without applying preprocessing rules.

    Attributes:
        document_id: Document ID from the official split manifest.
        ocr_data: Original OCR pages, blocks, lines, and words.
        json_data: Original annotation JSON.
        pdf_path: Location of the source PDF.
    """

    document_id: str
    ocr_data: dict[str, Any]
    json_data: dict[str, Any]
    pdf_path: str
