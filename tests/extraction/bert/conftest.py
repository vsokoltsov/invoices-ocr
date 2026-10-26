from collections.abc import Callable
from typing import Protocol

import pytest

from app.extraction.bert.spans import CandidateSpan


class _SpanFor(Protocol):
    def __call__(
        self,
        text: str,
        value: str,
        occurrence: int = 1,
    ) -> CandidateSpan: ...


@pytest.fixture
def candidate_factory() -> Callable[[int, int], CandidateSpan]:
    def create(start: int, end: int) -> CandidateSpan:
        return CandidateSpan(start=start, end=end)

    return create


@pytest.fixture
def span_for() -> _SpanFor:
    def create(text: str, value: str, occurrence: int = 1) -> CandidateSpan:
        start = -1

        for _ in range(occurrence):
            start = text.index(value, start + 1)

        return CandidateSpan(start=start, end=start + len(value))

    return create
