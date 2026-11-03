from dataclasses import dataclass


@dataclass
class OCRSpanEntity:
    start: int
    end: int
    label: str
    text: str
    needs_review: bool | None = None
    score: float | None = None
    match_status: str | None = None
    expected: str | None = None


@dataclass
class OCRSpanResult:
    text: str
    entities: list[OCRSpanEntity]


@dataclass
class OCRSummarySpanEntity:
    start: int
    end: int
    label: str
    text: str
    score: float
    match_status: str
    needs_review: bool
