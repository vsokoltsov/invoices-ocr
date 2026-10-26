from typing import Any

import pytest

from app.extraction.bert.json_evaluation import (
    ACCOUNT_NUMBER_PATH,
    document_is_exact,
    document_to_dict,
    evaluate_json_records,
    flatten_json,
    normalize_money,
    normalize_value,
    optional_text,
)
from app.extraction.bert.models import DocumentOutput, InvoiceData


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("€ 1.234,50", "1234.5"),
        ("$1,234.50", "1234.5"),
        ("1 234,50", "1234.5"),
        ("not money", ""),
    ],
)
def test_normalize_money_handles_common_ocr_number_formats(
    value: str,
    expected: str,
) -> None:
    """Normalize decimal and thousands separators before numeric comparison."""

    assert normalize_money(value) == expected


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("subtotal.tax", "€ 1.234,50", "1234.5"),
        ("invoice.invoice_number", " inv-001 ", "INV001"),
        ("invoice.seller_name", "  ACME\n  Ltd ", "acme ltd"),
        ("invoice.seller_address", "Apt: 10,  Berlin", "apt.10,berlin"),
        ("invoice.seller_name", None, None),
    ],
)
def test_normalize_value_uses_field_specific_rules(
    field: str,
    value: object,
    expected: str | None,
) -> None:
    """Use money, identifier, and ordinary-text rules according to the field path."""

    assert normalize_value(field, value) == expected


def test_document_to_dict_and_flatten_json_handle_models_and_mappings() -> None:
    """Convert documents to flat field paths while intentionally excluding lists."""

    model = DocumentOutput(
        invoice=InvoiceData(invoice_number="1001"),
        items=[{"description": "ignored item"}],
    )
    mapping = {
        "invoice": {"seller_name": "Acme"},
        "items": [{"description": "ignored item"}],
    }

    assert document_to_dict(model) == {
        "invoice": {"invoice_number": "1001"},
        "subtotal": {},
        "payment_instructions": {},
        "items": [{"description": "ignored item"}],
    }
    assert document_to_dict(mapping) == mapping
    assert flatten_json(model) == {"invoice.invoice_number": "1001"}
    assert flatten_json(mapping) == {"invoice.seller_name": "Acme"}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("  ", None),
        (" value ", "value"),
        (123, "123"),
    ],
)
def test_optional_text_normalizes_presence(value: object, expected: str | None) -> None:
    """Convert meaningful values to stripped text and treat blanks as absent."""

    assert optional_text(value) == expected


@pytest.fixture
def evaluation_records() -> list[dict[str, Any]]:
    """Provide correct, missing, and unknown-account output comparison records."""

    return [
        {
            "actual": {
                "invoice": {"invoice_number": "INV-001"},
                "subtotal": {"tax": "1,234.50"},
                "payment_instructions": {
                    "account_number": "GB12ABCD12345678901234"
                },
            },
            "predicted": {
                "invoice": {"invoice_number": "inv 001"},
                "subtotal": {"tax": "1 234,50"},
                "payment_instructions": {
                    "account_number": "GB12ABCD12345678901234"
                },
            },
        },
        {
            "actual": {
                "invoice": {"invoice_number": "MISS-002"},
                "payment_instructions": {},
            },
            "predicted": {
                "payment_instructions": {
                    "account_number": "GB99UNKNOWN123456789012"
                },
            },
        },
    ]


def test_evaluate_json_records_uses_normalized_field_comparison(
    evaluation_records: list[dict[str, Any]],
) -> None:
    """Calculate metrics after normalization and skip unknown account annotations."""

    result = evaluate_json_records(evaluation_records).set_index("field")

    invoice_number = result.loc["invoice.invoice_number"]
    assert invoice_number["support"] == 2
    assert invoice_number["exact_matches"] == 1
    assert invoice_number["precision"] == pytest.approx(1.0)
    assert invoice_number["recall"] == pytest.approx(0.5)
    assert invoice_number["f1"] == pytest.approx(2 / 3)

    tax = result.loc["subtotal.tax"]
    assert tax["support"] == 1
    assert tax["exact_match_rate"] == pytest.approx(1.0)

    account_number = result.loc[ACCOUNT_NUMBER_PATH]
    assert account_number["support"] == 1
    assert account_number["exact_matches"] == 1


def test_document_is_exact_can_optionally_penalize_extra_predictions() -> None:
    """Distinguish required-value correctness from strict equality with extra fields."""

    record = {
        "actual": {
            "invoice": {"invoice_number": "INV-001"},
            "subtotal": {"tax": ""},
        },
        "predicted": {
            "invoice": {
                "invoice_number": "inv 001",
                "seller_name": "Acme",
            },
            "subtotal": {"tax": "999.99"},
        },
    }

    assert document_is_exact(record)
    assert not document_is_exact(record, penalize_extra_fields=True)
