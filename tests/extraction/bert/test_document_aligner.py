from typing import Protocol

import pytest

from app.extraction.bert.alignment import DocumentAligner
from app.extraction.bert.enums import EntityType
from app.extraction.bert.models import InvoiceTarget


class TargetFactory(Protocol):
    def __call__(
        self,
        *,
        invoice: dict[str, object] | None = None,
        subtotal: dict[str, object] | None = None,
        payment_instructions: dict[str, object] | None = None,
    ) -> InvoiceTarget: ...


@pytest.fixture
def invoice_text() -> str:
    return (
        "Invoice no: 40108666 Date of issue: 02/07/2020 "
        "Seller: Client: Bailey, Murray and Lewis Hester Inc "
        "33771 Powell Pike Suite 054 4000 Tonya Rapid Suite 963 "
        "East Lisastad, RI 21938 South Marissaport, MN 39537 "
        "Total $ 452,92 $ 45,29 $ 498,21"
    )


@pytest.fixture
def invoice_target(target_factory: TargetFactory) -> InvoiceTarget:
    return target_factory(
        invoice={
            "invoice_number": "40108666",
            "invoice_date": "02/07/2020",
            "seller_name": "Bailey, Murray and Lewis",
            "seller_address": (
                "33771 Powell Pike Suite 054\n"
                "East Lisastad, RI 21938"
            ),
            "client_name": "Hester Inc",
            "client_address": (
                "4000 Tonya Rapid Suite 963\n"
                "South Marissaport, MN 39537"
            ),
        },
        subtotal={
            "tax": "45.29",
            "total": "498.21",
        },
    )


def test_find_spans_aligns_document_fields_and_returns_them_in_text_order(
    document_aligner: DocumentAligner,
    invoice_text: str,
    invoice_target: InvoiceTarget,
) -> None:
    """Align regular, address, and financial fields in their OCR text order."""

    spans = document_aligner.find_spans(
        text=invoice_text,
        target=invoice_target,
    )

    assert [span.entity for span in spans] == [
        EntityType.INVOICE_NUMBER,
        EntityType.INVOICE_DATE,
        EntityType.SELLER_NAME,
        EntityType.CLIENT_NAME,
        EntityType.SELLER_ADDRESS,
        EntityType.CLIENT_ADDRESS,
        EntityType.SELLER_ADDRESS,
        EntityType.CLIENT_ADDRESS,
        EntityType.TAX,
        EntityType.TOTAL,
    ]
    assert [span.value for span in spans] == [
        "40108666",
        "02/07/2020",
        "Bailey, Murray and Lewis",
        "Hester Inc",
        "33771 Powell Pike Suite 054",
        "4000 Tonya Rapid Suite 963",
        "East Lisastad, RI 21938",
        "South Marissaport, MN 39537",
        "45,29",
        "498,21",
    ]
    assert spans == sorted(spans, key=lambda span: span.start)
    assert all(
        span.value == invoice_text[span.start : span.end]
        for span in spans
    )


def test_find_spans_omits_regular_entity_with_ambiguous_matches(
    document_aligner: DocumentAligner,
    target_factory: TargetFactory,
) -> None:
    """Omit a regular entity when its expected value appears more than once."""

    target = target_factory(
        invoice={"seller_name": "Acme"},
    )

    spans = document_aligner.find_spans(
        text="Seller Acme Client Acme",
        target=target,
    )

    assert spans == []


def test_find_spans_does_not_assign_one_text_range_to_multiple_entities(
    document_aligner: DocumentAligner,
    target_factory: TargetFactory,
) -> None:
    """Prevent two target fields from claiming the same OCR character range."""

    target = target_factory(
        invoice={
            "seller_name": "Acme",
            "client_name": "Acme",
        },
    )

    spans = document_aligner.find_spans(
        text="Company Acme",
        target=target,
    )

    assert len(spans) == 1
    assert spans[0].entity is EntityType.SELLER_NAME
    assert spans[0].value == "Acme"


def test_find_spans_uses_financial_summary_instead_of_earlier_matching_amount(
    document_aligner: DocumentAligner,
    target_factory: TargetFactory,
) -> None:
    """Align tax and total to the final summary row rather than item-level amounts."""

    text = "Item tax 45,29 Total $ 452,92 $ 45,29 $ 498,21"
    target = target_factory(
        subtotal={
            "tax": "45.29",
            "total": "498.21",
        },
    )

    spans = document_aligner.find_spans(
        text=text,
        target=target,
    )

    assert [(span.entity, span.value) for span in spans] == [
        (EntityType.TAX, "45,29"),
        (EntityType.TOTAL, "498,21"),
    ]
    assert spans[0].start == text.rindex("45,29")


def test_find_spans_falls_back_to_field_matching_without_financial_summary(
    document_aligner: DocumentAligner,
    target_factory: TargetFactory,
) -> None:
    """Use ordinary money matching when no complete summary row is available."""

    target = target_factory(
        subtotal={"tax": "45.29"},
    )

    spans = document_aligner.find_spans(
        text="Tax due 45,29",
        target=target,
    )

    assert len(spans) == 1
    assert spans[0].entity is EntityType.TAX
    assert spans[0].value == "45,29"


def test_find_spans_preserves_ocr_observed_identifier_formatting(
    document_aligner: DocumentAligner,
    target_factory: TargetFactory,
) -> None:
    """Return the OCR substring rather than copying the normalized target value."""

    target = target_factory(
        invoice={"invoice_number": "40108666"},
    )

    spans = document_aligner.find_spans(
        text="Invoice no: 4010 8666",
        target=target,
    )

    assert len(spans) == 1
    assert spans[0].entity is EntityType.INVOICE_NUMBER
    assert spans[0].value == "4010 8666"


def test_find_spans_returns_empty_list_when_no_target_values_are_present(
    document_aligner: DocumentAligner,
    target_factory: TargetFactory,
) -> None:
    """Return an empty collection when the target contains no extractable fields."""

    spans = document_aligner.find_spans(
        text="Unstructured OCR text",
        target=target_factory(),
    )

    assert spans == []
