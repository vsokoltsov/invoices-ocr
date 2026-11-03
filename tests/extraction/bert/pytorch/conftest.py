import pytest

from app.extraction.bert.pytorch.builder import InvoiceRecordBuilder
from app.extraction.bert.pytorch.spans import (
    Normalizer,
    NumericField,
    OCRSpanMatcher,
    OCRSummarySpan,
)


@pytest.fixture
def normalizer():
    return Normalizer()


@pytest.fixture
def span_matcher(normalizer):
    return OCRSpanMatcher(
        normalizer=normalizer,
        numeric=NumericField(normalizer),
        summary_span=OCRSummarySpan(normalizer),
    )


@pytest.fixture
def record_builder(span_matcher, normalizer):
    return InvoiceRecordBuilder(
        span_matcher=span_matcher,
        numeric=span_matcher.numeric,
        normalizer=normalizer,
    )
