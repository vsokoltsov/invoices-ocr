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
    document_id: str
    ocr_text: str
    text_with_boxes: list[OCRWord]
    json_data: dict[str, Any]
