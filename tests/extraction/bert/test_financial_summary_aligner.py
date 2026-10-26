from typing import Protocol

import pytest

from app.extraction.bert.alignment import FinancialSummaryAligner
from app.extraction.bert.spans import CandidateSpan, FinancialSummarySpans


class SpanFor(Protocol):
    def __call__(
        self,
        text: str,
        value: str,
        occurrence: int = 1,
    ) -> CandidateSpan: ...


@pytest.fixture
def aligner() -> FinancialSummaryAligner:
    return FinancialSummaryAligner()


@pytest.fixture
def invoice_summary_text() -> str:
    return (
        "SUMMARY VAT [%] Net worth VAT Gross worth "
        "10% 452,92 45,29 498,21 "
        "Total $ 452,92 $ 45,29 $ 498,21"
    )


@pytest.fixture
def expected_summary_spans(
    invoice_summary_text: str,
    span_for: SpanFor,
) -> FinancialSummarySpans:
    return FinancialSummarySpans(
        net=span_for(invoice_summary_text, "452,92", occurrence=2),
        tax=span_for(invoice_summary_text, "45,29", occurrence=2),
        total=span_for(invoice_summary_text, "498,21", occurrence=2),
    )


def test_find_returns_absolute_spans_from_final_total_row(
    aligner: FinancialSummaryAligner,
    invoice_summary_text: str,
    expected_summary_spans: FinancialSummarySpans,
) -> None:
    """Return absolute OCR spans for net, tax, and total in the final row."""

    result = aligner.find(invoice_summary_text)

    assert result == expected_summary_spans
    assert invoice_summary_text[result.net.start : result.net.end] == "452,92"
    assert invoice_summary_text[result.tax.start : result.tax.end] == "45,29"
    assert invoice_summary_text[result.total.start : result.total.end] == "498,21"


def test_find_prefers_the_final_total_anchor(
    aligner: FinancialSummaryAligner,
    span_for: SpanFor,
) -> None:
    """Ignore earlier Total rows when a later complete summary row is available."""

    text = "Total 10.00 1.00 11.00 footer TOTAL 200.00 20.00 220.00"

    result = aligner.find(text)

    assert result == FinancialSummarySpans(
        net=span_for(text, "200.00"),
        tax=span_for(text, "20.00"),
        total=span_for(text, "220.00"),
    )


def test_find_uses_earlier_anchor_when_final_anchor_is_incomplete(
    aligner: FinancialSummaryAligner,
    span_for: SpanFor,
) -> None:
    """Fall back to an earlier complete Total row when the final anchor has no values."""

    text = "Total 100.00 10.00 110.00 footer Total unavailable"

    result = aligner.find(text)

    assert result == FinancialSummarySpans(
        net=span_for(text, "100.00"),
        tax=span_for(text, "10.00"),
        total=span_for(text, "110.00"),
    )


def test_find_returns_none_without_total_anchor(
    aligner: FinancialSummaryAligner,
) -> None:
    """Do not infer a summary from monetary values without a Total anchor."""

    result = aligner.find("Net 100.00 Tax 10.00 Gross 110.00")

    assert result is None


def test_find_does_not_treat_subtotal_as_total_anchor(
    aligner: FinancialSummaryAligner,
) -> None:
    """Require Total as a complete word instead of matching it inside Subtotal."""

    result = aligner.find("Subtotal 100.00 10.00 110.00")

    assert result is None


def test_find_returns_none_when_total_row_has_fewer_than_three_amounts(
    aligner: FinancialSummaryAligner,
) -> None:
    """Reject a Total row that cannot provide net, tax, and total values."""

    result = aligner.find("Total 100.00 10.00")

    assert result is None


def test_find_respects_configured_lookahead() -> None:
    """Ignore financial values positioned beyond the configured search window."""

    aligner = FinancialSummaryAligner(lookahead=20)
    text = f"Total 100.00 {'x' * 20} 10.00 110.00"

    result = aligner.find(text)

    assert result is None


@pytest.mark.parametrize(
    ("net", "tax", "total"),
    [
        ("1 234,56", "123,45", "1 358,01"),
        ("1\u00a0234,56", "123,45", "1\u00a0358,01"),
        ("1'234.56", "123.45", "1'358.01"),
        ("1.234,56", "123,45", "1.358,01"),
    ],
)
def test_find_supports_expected_ocr_money_formats(
    aligner: FinancialSummaryAligner,
    span_for: SpanFor,
    net: str,
    tax: str,
    total: str,
) -> None:
    """Recognize common decimal and thousands separators produced by OCR."""

    text = f"Total $ {net} $ {tax} $ {total}"

    result = aligner.find(text)

    assert result == FinancialSummarySpans(
        net=span_for(text, net),
        tax=span_for(text, tax),
        total=span_for(text, total),
    )
