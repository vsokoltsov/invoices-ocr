from typing import Protocol

import pandas as pd
import pytest

from app.extraction.bert.alignment import DocumentAligner
from app.extraction.bert.enums import EntityType
from app.extraction.bert.inspection import AlignmentInspector
from app.extraction.bert.models import InvoiceTarget
from app.extraction.bert.reporting import AlignmentReport


class TargetFactory(Protocol):
    def __call__(
        self,
        *,
        invoice: dict[str, object] | None = None,
        subtotal: dict[str, object] | None = None,
        payment_instructions: dict[str, object] | None = None,
    ) -> InvoiceTarget: ...


@pytest.fixture
def inspector(alignment_report: AlignmentReport) -> AlignmentInspector:
    return AlignmentInspector(report=alignment_report)


@pytest.fixture
def edge_case_inspector(
    document_aligner: DocumentAligner,
    target_factory: TargetFactory,
) -> AlignmentInspector:
    frame = pd.DataFrame(
        [
            {
                "file_name": "ambiguous.jpg",
                "ocred_text": "Seller Acme Client Acme",
                "target": target_factory(
                    invoice={"seller_name": "Acme"},
                ),
            },
            {
                "file_name": "partial-address.jpg",
                "ocred_text": "Seller address: 123 Main Street",
                "target": target_factory(
                    invoice={
                        "seller_address": (
                            "123 Main Street\n"
                            "Missing City, NY 10001"
                        )
                    },
                ),
            },
        ],
        index=pd.Index([30, 40]),
    )

    return AlignmentInspector(
        report=AlignmentReport.build(
            frame=frame,
            aligner=document_aligner,
        )
    )


def test_document_returns_sorted_details_for_requested_row(
    inspector: AlignmentInspector,
) -> None:
    """Return only one document's detail rows sorted by entity name."""

    result = inspector.document(row_index=20)

    assert set(result["row_index"]) == {20}
    assert list(result["entity"]) == sorted(result["entity"])
    assert list(result.index) == list(range(len(result)))


def test_document_returns_empty_dataframe_for_unknown_row(
    inspector: AlignmentInspector,
) -> None:
    """Return an empty view when the requested dataframe index is unknown."""

    result = inspector.document(row_index=999)

    assert result.empty


def test_missing_returns_only_unaligned_entities(
    inspector: AlignmentInspector,
) -> None:
    """Expose expected entities for which the aligner selected no OCR span."""

    result = inspector.missing()

    assert len(result) == 1
    assert result.iloc[0]["row_index"] == 20
    assert result.iloc[0]["file_name"] == "partial.jpg"
    assert result.iloc[0]["entity"] == EntityType.SELLER_NAME.value
    assert not result.iloc[0]["aligned"]


def test_ambiguous_returns_entities_with_multiple_raw_candidates(
    edge_case_inspector: AlignmentInspector,
) -> None:
    """Expose expected values that occur in multiple OCR positions."""

    result = edge_case_inspector.ambiguous()

    assert len(result) == 1
    assert result.iloc[0]["row_index"] == 30
    assert result.iloc[0]["entity"] == EntityType.SELLER_NAME.value
    assert result.iloc[0]["candidate_counts"] == (2,)
    assert result.iloc[0]["ambiguous"]


def test_partial_matches_returns_partially_aligned_multiline_fields(
    edge_case_inspector: AlignmentInspector,
) -> None:
    """Expose multiline fields for which only some expected parts occur in OCR."""

    result = edge_case_inspector.partial_matches()

    assert len(result) == 1
    assert result.iloc[0]["row_index"] == 40
    assert result.iloc[0]["entity"] == EntityType.SELLER_ADDRESS.value
    assert result.iloc[0]["observed"] == "123 Main Street"
    assert result.iloc[0]["parts"] == 2
    assert result.iloc[0]["matched_parts"] == 1
    assert result.iloc[0]["partially_raw_matched"]


def test_worst_documents_delegates_to_report_coverage_ordering(
    inspector: AlignmentInspector,
) -> None:
    """Return the requested number of documents with the lowest coverage first."""

    result = inspector.worst_documents(limit=1)

    assert len(result) == 1
    assert result.iloc[0]["row_index"] == 20
    assert result.iloc[0]["alignment_coverage"] == pytest.approx(0.75)


def test_financial_mismatches_returns_only_values_different_from_json(
    inspector: AlignmentInspector,
) -> None:
    """Expose aligned TAX or TOTAL values whose digits differ from JSON labels."""

    result = inspector.financial_mismatches()

    assert len(result) == 1
    assert result.iloc[0]["row_index"] == 20
    assert result.iloc[0]["entity"] == EntityType.TOTAL.value
    assert result.iloc[0]["expected"] == "55.00"
    assert result.iloc[0]["observed"] == "50.00"
    assert not result.iloc[0]["value_matches_json"]


def test_inspection_methods_handle_empty_report(
    document_aligner: DocumentAligner,
) -> None:
    """Return empty inspection views when the alignment report has no documents."""

    report = AlignmentReport.build(
        frame=pd.DataFrame(
            columns=pd.Index(["ocred_text", "target"]),
        ),
        aligner=document_aligner,
    )
    inspector = AlignmentInspector(report=report)

    assert inspector.document(row_index=1).empty
    assert inspector.missing().empty
    assert inspector.ambiguous().empty
    assert inspector.partial_matches().empty
    assert inspector.worst_documents().empty
    assert inspector.financial_mismatches().empty
