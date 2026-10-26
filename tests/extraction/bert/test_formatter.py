import json
from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest

from app.extraction.bert.enums import EntityType
from app.extraction.bert.formatter import (
    ACCOUNT_NUMBER_PATH,
    DocumentOutputFormatter,
    account_values_equal,
    build_account_evaluation,
    document_is_exact,
    document_to_dict,
    evaluate_json_records,
    extract_anchored_account_number,
    flatten_json,
    normalize_money,
    normalize_value,
    optional_text,
)
from app.extraction.bert.models import DocumentOutput, InvoiceData, PredictedEntity


@pytest.fixture
def output_formatter() -> DocumentOutputFormatter:
    """Provide the stateless formatter used to construct prediction output."""

    return DocumentOutputFormatter()


@pytest.fixture
def formatted_entities(
    predicted_entity_factory: Callable[..., PredictedEntity],
) -> list[PredictedEntity]:
    """Provide shuffled entities covering first, last, and multiline field rules."""

    return [
        predicted_entity_factory(
            entity=EntityType.TOTAL,
            start=280,
            text=" 125.00 ",
        ),
        predicted_entity_factory(
            entity=EntityType.SELLER_ADDRESS,
            start=100,
            text=" North Street ",
        ),
        predicted_entity_factory(
            entity=EntityType.INVOICE_NUMBER,
            start=10,
            text=" INV-100 ",
        ),
        predicted_entity_factory(
            entity=EntityType.SELLER_ADDRESS,
            start=80,
            text=" 42 South Road ",
        ),
        predicted_entity_factory(
            entity=EntityType.TAX,
            start=250,
            text=" 10.00 ",
        ),
        predicted_entity_factory(
            entity=EntityType.TAX,
            start=260,
            text=" 12.50 ",
        ),
        predicted_entity_factory(
            entity=EntityType.TOTAL,
            start=270,
            text=" 100.00 ",
        ),
        predicted_entity_factory(
            entity=EntityType.CLIENT_ADDRESS,
            start=150,
            text=" Client Street ",
        ),
        predicted_entity_factory(
            entity=EntityType.ACCOUNT_NUMBER,
            start=220,
            text=" GB12ABCD12345678901234 ",
        ),
        predicted_entity_factory(
            entity=EntityType.SELLER_NAME,
            start=50,
            text=" Acme Ltd ",
        ),
        predicted_entity_factory(
            entity=EntityType.CLIENT_NAME,
            start=70,
            text=" Buyer Inc ",
        ),
        predicted_entity_factory(
            entity=EntityType.INVOICE_DATE,
            start=30,
            text=" 2026-10-26 ",
        ),
    ]


def test_formatter_builds_nested_output_from_shuffled_entities(
    output_formatter: DocumentOutputFormatter,
    formatted_entities: list[PredictedEntity],
) -> None:
    """Sort entities by position and apply first, last, and multiline field rules."""

    output = output_formatter.format(formatted_entities)

    assert output.invoice.invoice_number == "INV-100"
    assert output.invoice.invoice_date == "2026-10-26"
    assert output.invoice.seller_name == "Acme Ltd"
    assert output.invoice.client_name == "Buyer Inc"
    assert output.invoice.seller_address == "42 South Road\nNorth Street"
    assert output.invoice.client_address == "Client Street"
    assert output.subtotal.tax == "12.50"
    assert output.subtotal.total == "125.00"
    assert (
        output.payment_instructions.account_number
        == "GB12ABCD12345678901234"
    )
    assert output.items is None


def test_formatter_omits_absent_entities(
    output_formatter: DocumentOutputFormatter,
    predicted_entity_factory: Callable[..., PredictedEntity],
) -> None:
    """Leave fields as None when the model did not predict their entities."""

    output = output_formatter.format(
        [
            predicted_entity_factory(
                entity=EntityType.INVOICE_NUMBER,
                start=0,
                text="1001",
            )
        ]
    )

    assert output.invoice.invoice_number == "1001"
    assert output.invoice.seller_name is None
    assert output.subtotal.tax is None
    assert output.payment_instructions.account_number is None


def test_formatter_serializes_only_present_values(
    output_formatter: DocumentOutputFormatter,
    predicted_entity_factory: Callable[..., PredictedEntity],
) -> None:
    """Exclude None fields from JSON output while honouring the requested indentation."""

    rendered = output_formatter.format_json(
        [
            predicted_entity_factory(
                entity=EntityType.INVOICE_NUMBER,
                start=0,
                text="1001",
            )
        ],
        indent=None,
    )

    assert "\n" not in rendered
    assert json.loads(rendered) == {
        "invoice": {"invoice_number": "1001"},
        "subtotal": {},
        "payment_instructions": {},
    }


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
    """Use money, identifier, and ordinary-text normalization according to field path."""

    assert normalize_value(field, value) == expected


def test_document_to_dict_and_flatten_json_handle_models_and_mappings() -> None:
    """Convert supported document inputs into flat field paths while skipping lists."""

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
    """Calculate field metrics after normalizing values and skipping unknown accounts."""

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
    """Distinguish required-field correctness from strict equality with extra fields."""

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


def test_extract_anchored_account_number_returns_value_after_known_alias() -> None:
    """Extract only an account number that is introduced by an accepted OCR alias."""

    assert (
        extract_anchored_account_number(
            "Payment details. IBAN: GB12ABCD12345678901234"
        )
        == "GB12ABCD12345678901234"
    )
    assert extract_anchored_account_number("Reference GB12ABCD12345678901234") is None


def test_account_values_equal_compares_compacted_identifiers() -> None:
    """Ignore casing and separators but reject absent account values."""

    assert account_values_equal("GB12-AB34", "gb12ab34")
    assert not account_values_equal("GB12AB34", "GB12AB35")
    assert not account_values_equal("GB12AB34", None)


@pytest.fixture
def account_evaluation_records() -> list[dict[str, Any]]:
    """Provide exact, OCR-equivalent, and unannotated account prediction records."""

    exact = "GB12ABCD12345678901234"
    expected_equivalent = "GB55XDOT36152114286113"
    observed_equivalent = "GBSSXDOT36152114286113"
    unannotated = "GB99ABCD12345678901234"

    return [
        {
            "row_index": 1,
            "file_name": "exact.jpg",
            "ocred_text": f"IBAN: {exact}",
            "actual": {
                "payment_instructions": {"account_number": exact}
            },
            "predicted": {
                "payment_instructions": {"account_number": exact}
            },
        },
        {
            "row_index": 2,
            "file_name": "equivalent.jpg",
            "ocred_text": f"IBAN: {observed_equivalent}",
            "actual": {
                "payment_instructions": {
                    "account_number": expected_equivalent
                }
            },
            "predicted": {
                "payment_instructions": {
                    "account_number": observed_equivalent
                }
            },
        },
        {
            "row_index": 3,
            "file_name": "unknown.jpg",
            "ocred_text": f"IBAN: {unannotated}",
            "actual": {"payment_instructions": {}},
            "predicted": {
                "payment_instructions": {"account_number": unannotated}
            },
        },
    ]


def test_build_account_evaluation_separates_json_and_ocr_quality(
    account_evaluation_records: list[dict[str, Any]],
) -> None:
    """Report exact JSON, OCR-equivalent JSON, and OCR-anchor accuracy separately."""

    details, metrics = build_account_evaluation(account_evaluation_records)
    details = details.set_index("row_index")
    metrics = metrics.set_index("metric")

    assert details.loc[1, "json_match_type"] == "exact"
    assert details.loc[2, "json_match_type"] == "ocr_equivalent"
    assert bool(details.loc[2, "json_match_accepted"])
    assert pd.isna(details.loc[3, "json_match_accepted"])
    assert bool(details["matches_ocr"].all())

    assert metrics.loc["OCR account detected", "accuracy"] == pytest.approx(1.0)
    assert metrics.loc["Prediction matches OCR", "accuracy"] == pytest.approx(1.0)
    assert metrics.loc["Exact JSON match", "accuracy"] == pytest.approx(0.5)
    assert (
        metrics.loc[
            "Exact or OCR-equivalent JSON match",
            "accuracy",
        ]
        == pytest.approx(1.0)
    )
