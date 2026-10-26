from collections.abc import Callable
from typing import Protocol

import pandas as pd
import pytest

from app.extraction.bert.alignment import DocumentAligner, FinancialSummaryAligner
from app.extraction.bert.candidates import CandidatesSelector
from app.extraction.bert.models import InvoiceTarget
from app.extraction.bert.reporting import AlignmentReport
from app.extraction.bert.spans import CandidateSpan


class _SpanFor(Protocol):
    def __call__(
        self,
        text: str,
        value: str,
        occurrence: int = 1,
    ) -> CandidateSpan: ...


class _TargetFactory(Protocol):
    def __call__(
        self,
        *,
        invoice: dict[str, object] | None = None,
        subtotal: dict[str, object] | None = None,
        payment_instructions: dict[str, object] | None = None,
    ) -> InvoiceTarget: ...


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


@pytest.fixture
def document_aligner() -> DocumentAligner:
    return DocumentAligner(
        financial_summary_aligner=FinancialSummaryAligner(),
        selector=CandidatesSelector(),
    )


@pytest.fixture
def target_factory() -> _TargetFactory:
    def create(
        *,
        invoice: dict[str, object] | None = None,
        subtotal: dict[str, object] | None = None,
        payment_instructions: dict[str, object] | None = None,
    ) -> InvoiceTarget:
        return InvoiceTarget.model_validate(
            {
                "invoice": invoice or {},
                "subtotal": subtotal or {},
                "payment_instructions": payment_instructions or {},
                "items": [],
            }
        )

    return create


@pytest.fixture
def alignment_frame(target_factory: _TargetFactory) -> pd.DataFrame:
    complete_target = target_factory(
        invoice={
            "invoice_number": "1001",
            "seller_name": "Acme",
        },
        subtotal={
            "tax": "10.00",
            "total": "100.00",
        },
    )
    partial_target = target_factory(
        invoice={
            "invoice_number": "2002",
            "seller_name": "Missing Corp",
        },
        subtotal={
            "tax": "5.00",
            "total": "55.00",
        },
    )

    return pd.DataFrame(
        [
            {
                "file_name": "complete.jpg",
                "ocred_text": (
                    "Invoice no: 1001 Seller: Acme "
                    "Total $ 90.00 $ 10.00 $ 100.00"
                ),
                "target": complete_target,
            },
            {
                "file_name": "partial.jpg",
                "ocred_text": (
                    "Invoice no: 2002 "
                    "Total $ 45.00 $ 5.00 $ 50.00"
                ),
                "target": partial_target,
            },
        ],
        index=pd.Index([10, 20]),
    )


@pytest.fixture
def alignment_report(
    alignment_frame: pd.DataFrame,
    document_aligner: DocumentAligner,
) -> AlignmentReport:
    return AlignmentReport.build(
        frame=alignment_frame,
        aligner=document_aligner,
    )
