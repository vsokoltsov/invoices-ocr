from typing import NotRequired, TypedDict


class CandidateLabel(TypedDict):
    label: str
    occurrence: int
    annotation_id: int
    line_item_id: NotRequired[int]


class CandidateWord(TypedDict):
    """Typed assignment fields added to the original OCR word metadata."""

    ocr_index: int
    candidates: list[CandidateLabel]
