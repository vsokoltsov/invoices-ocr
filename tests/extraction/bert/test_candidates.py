from collections.abc import Callable
from typing import Protocol

import pytest

from app.extraction.bert.candidates import CandidatesSelector
from app.extraction.bert.enums import EntityType
from app.extraction.bert.spans import CandidateSpan


class SpanFor(Protocol):
    def __call__(
        self,
        text: str,
        value: str,
        occurrence: int = 1,
    ) -> CandidateSpan: ...


@pytest.fixture
def selector() -> CandidatesSelector:
    return CandidatesSelector()


@pytest.fixture
def candidate_factory() -> Callable[[int, int], CandidateSpan]:
    def create(start: int, end: int) -> CandidateSpan:
        return CandidateSpan(start=start, end=end)

    return create


@pytest.fixture
def span_for() -> SpanFor:
    def create(text: str, value: str, occurrence: int = 1) -> CandidateSpan:
        start = -1

        for _ in range(occurrence):
            start = text.index(value, start + 1)

        return CandidateSpan(start=start, end=start + len(value))

    return create


@pytest.fixture
def unique_candidate(candidate_factory: Callable[[int, int], CandidateSpan]) -> CandidateSpan:
    return candidate_factory(12, 20)


@pytest.fixture
def financial_text() -> str:
    return "Line total 25.00 Tax 2.50 FINAL TOTAL 27.50"


@pytest.fixture
def financial_candidates(
    financial_text: str,
    span_for: SpanFor,
) -> list[CandidateSpan]:
    return [
        span_for(financial_text, "25.00"),
        span_for(financial_text, "2.50"),
        span_for(financial_text, "27.50"),
    ]


def test_select_returns_unique_candidate_for_non_financial_entity(
    selector: CandidatesSelector,
    unique_candidate: CandidateSpan,
) -> None:
    """Select the only unambiguous match for a regular entity."""

    result = selector.select(
        text="Invoice no: 12345678",
        entity_type=EntityType.INVOICE_NUMBER,
        candidates=[unique_candidate],
    )

    assert result == unique_candidate


def test_select_returns_none_when_non_financial_entity_has_no_candidates(
    selector: CandidatesSelector,
) -> None:
    """Leave a missing regular entity unaligned instead of guessing a span."""

    result = selector.select(
        text="Invoice number is missing",
        entity_type=EntityType.INVOICE_NUMBER,
        candidates=[],
    )

    assert result is None


def test_select_returns_none_when_non_financial_entity_is_ambiguous(
    selector: CandidatesSelector,
    candidate_factory: Callable[[int, int], CandidateSpan],
) -> None:
    """Reject multiple regular-entity matches because they are ambiguous."""

    result = selector.select(
        text="Invoice numbers: 12345678 and 87654321",
        entity_type=EntityType.INVOICE_NUMBER,
        candidates=[
            candidate_factory(17, 25),
            candidate_factory(30, 38),
        ],
    )

    assert result is None


@pytest.mark.parametrize(
    "entity_type",
    [EntityType.TAX, EntityType.TOTAL],
)
def test_select_financial_entity_uses_last_candidate_after_final_total_anchor(
    selector: CandidatesSelector,
    financial_text: str,
    financial_candidates: list[CandidateSpan],
    entity_type: EntityType,
) -> None:
    """Prefer financial values belonging to the final summary row."""

    result = selector.select(
        text=financial_text,
        entity_type=entity_type,
        candidates=financial_candidates,
    )

    assert result == financial_candidates[-1]


def test_select_financial_entity_matches_total_anchor_case_insensitively(
    selector: CandidatesSelector,
    span_for: SpanFor,
) -> None:
    """Detect the Total anchor regardless of OCR capitalization."""

    text = "Tax 2.50 ToTaL 27.50"
    expected = span_for(text, "27.50")

    result = selector.select(
        text=text,
        entity_type=EntityType.TOTAL,
        candidates=[span_for(text, "2.50"), expected],
    )

    assert result == expected


def test_select_financial_entity_falls_back_to_last_candidate_without_total_anchor(
    selector: CandidatesSelector,
    span_for: SpanFor,
) -> None:
    """Use the final financial match when the Total anchor is absent."""

    text = "Net 25.00 Tax 2.50 Gross 27.50"
    candidates = [
        span_for(text, "25.00"),
        span_for(text, "2.50"),
        span_for(text, "27.50"),
    ]

    result = selector.select(
        text=text,
        entity_type=EntityType.TAX,
        candidates=candidates,
    )

    assert result == candidates[-1]


def test_select_financial_entity_falls_back_when_all_candidates_precede_final_anchor(
    selector: CandidatesSelector,
    span_for: SpanFor,
) -> None:
    """Keep the last usable match when no candidate follows the final anchor."""

    text = "Net 25.00 Tax 2.50 Total unavailable"
    candidates = [
        span_for(text, "25.00"),
        span_for(text, "2.50"),
    ]

    result = selector.select(
        text=text,
        entity_type=EntityType.TOTAL,
        candidates=candidates,
    )

    assert result == candidates[-1]


def test_select_financial_entity_returns_none_without_candidates(
    selector: CandidatesSelector,
) -> None:
    """Return no financial match when the candidate collection is empty."""

    result = selector.select(
        text="Total unavailable",
        entity_type=EntityType.TOTAL,
        candidates=[],
    )

    assert result is None
