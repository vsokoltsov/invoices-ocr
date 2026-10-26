from typing import Protocol

import pandas as pd
import pytest

from app.extraction.bert.alignment import DocumentAligner
from app.extraction.bert.enums import EntityType
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


def test_build_creates_document_level_coverage_rows(
    alignment_report: AlignmentReport,
) -> None:
    """Summarize expected, aligned, and missing entities for every document."""

    documents = alignment_report.documents.set_index("row_index")

    assert list(documents.index) == [10, 20]
    assert documents.loc[10, "file_name"] == "complete.jpg"
    assert documents.loc[10, "expected_entities"] == 4
    assert documents.loc[10, "matched_entities"] == 4
    assert documents.loc[10, "alignment_coverage"] == pytest.approx(1.0)
    assert documents.loc[10, "missing_entities"] == ()

    assert documents.loc[20, "file_name"] == "partial.jpg"
    assert documents.loc[20, "expected_entities"] == 4
    assert documents.loc[20, "matched_entities"] == 3
    assert documents.loc[20, "alignment_coverage"] == pytest.approx(0.75)
    assert documents.loc[20, "missing_entities"] == (
        EntityType.SELLER_NAME.value,
    )


def test_build_creates_one_detail_row_per_expected_entity(
    alignment_report: AlignmentReport,
) -> None:
    """Preserve alignment evidence for each expected field in each document."""

    details = alignment_report.details

    assert len(details) == 8

    missing_seller = details[
        (details["row_index"] == 20)
        & (details["entity"] == EntityType.SELLER_NAME.value)
    ].iloc[0]

    assert missing_seller["expected"] == "Missing Corp"
    assert missing_seller["observed"] is None
    assert not missing_seller["aligned"]
    assert missing_seller["span_count"] == 0
    assert missing_seller["candidate_counts"] == (0,)

    mismatched_total = details[
        (details["row_index"] == 20)
        & (details["entity"] == EntityType.TOTAL.value)
    ].iloc[0]

    assert mismatched_total["expected"] == "55.00"
    assert mismatched_total["observed"] == "50.00"
    assert mismatched_total["aligned"]
    assert not mismatched_total["fully_raw_matched"]


def test_build_aggregates_entity_alignment_metrics(
    alignment_report: AlignmentReport,
) -> None:
    """Aggregate document alignment and raw-match rates for every entity type."""

    entities = alignment_report.entities.set_index("entity")

    assert entities.loc[EntityType.SELLER_NAME.value, "expected_documents"] == 2
    assert entities.loc[EntityType.SELLER_NAME.value, "aligned_documents"] == 1
    assert entities.loc[
        EntityType.SELLER_NAME.value,
        "alignment_coverage",
    ] == pytest.approx(0.5)

    for entity in (
        EntityType.INVOICE_NUMBER,
        EntityType.TAX,
        EntityType.TOTAL,
    ):
        assert entities.loc[entity.value, "expected_documents"] == 2
        assert entities.loc[entity.value, "aligned_documents"] == 2
        assert entities.loc[entity.value, "alignment_coverage"] == pytest.approx(1.0)

    assert entities.loc[EntityType.TOTAL.value, "fully_raw_matched"] == 1
    assert entities.loc[EntityType.TOTAL.value, "raw_match_rate"] == pytest.approx(
        0.5
    )


def test_build_reports_financial_span_and_value_accuracy_separately(
    alignment_report: AlignmentReport,
) -> None:
    """Separate locating financial spans from matching their expected JSON values."""

    financial = alignment_report.financial.set_index("entity")

    assert financial.loc[EntityType.TAX.value, "documents"] == 2
    assert financial.loc[EntityType.TAX.value, "spans_found"] == 2
    assert financial.loc[EntityType.TAX.value, "values_matching_json"] == 2
    assert financial.loc[EntityType.TAX.value, "span_coverage"] == pytest.approx(1.0)
    assert financial.loc[EntityType.TAX.value, "ocr_value_accuracy"] == pytest.approx(
        1.0
    )

    assert financial.loc[EntityType.TOTAL.value, "documents"] == 2
    assert financial.loc[EntityType.TOTAL.value, "spans_found"] == 2
    assert financial.loc[EntityType.TOTAL.value, "values_matching_json"] == 1
    assert financial.loc[EntityType.TOTAL.value, "span_coverage"] == pytest.approx(
        1.0
    )
    assert financial.loc[
        EntityType.TOTAL.value,
        "ocr_value_accuracy",
    ] == pytest.approx(0.5)


def test_missing_counts_and_low_coverage_expose_problem_documents(
    alignment_report: AlignmentReport,
) -> None:
    """Provide concise views of missing entities and lowest-coverage documents."""

    missing = alignment_report.missing_counts()
    low_coverage = alignment_report.low_coverage(limit=1)

    assert missing.loc[EntityType.SELLER_NAME.value, "missing_count"] == 1
    assert len(low_coverage) == 1
    assert low_coverage.iloc[0]["row_index"] == 20
    assert low_coverage.iloc[0]["alignment_coverage"] == pytest.approx(0.75)


def test_build_accepts_model_dictionary_and_json_targets(
    document_aligner: DocumentAligner,
    target_factory: TargetFactory,
) -> None:
    """Load targets from each representation supported by dataframe rows."""

    target = target_factory(
        invoice={"invoice_number": "1001"},
    )
    rows = [
        {"target": target},
        {"target": target.model_dump()},
        {"json_data": target.model_dump_json()},
    ]

    for row in rows:
        frame = pd.DataFrame(
            [
                {
                    "ocred_text": "Invoice no: 1001",
                    **row,
                }
            ],
            index=pd.Index([42]),
        )

        report = AlignmentReport.build(
            frame=frame,
            aligner=document_aligner,
        )

        document = report.documents.iloc[0]
        assert document["row_index"] == 42
        assert document["file_name"] == "42"
        assert document["alignment_coverage"] == pytest.approx(1.0)


def test_build_handles_empty_dataframe(
    document_aligner: DocumentAligner,
) -> None:
    """Return empty report views when the source dataframe has no documents."""

    frame = pd.DataFrame(
        columns=pd.Index(["ocred_text", "target"]),
    )

    report = AlignmentReport.build(
        frame=frame,
        aligner=document_aligner,
    )

    assert report.documents.empty
    assert report.details.empty
    assert report.entities.empty
    assert report.financial.empty
    assert report.missing_counts().empty
    assert report.low_coverage().empty
