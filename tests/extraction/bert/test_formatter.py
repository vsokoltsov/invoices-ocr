import json
from collections.abc import Callable

import pytest

from app.extraction.bert.enums import EntityType
from app.extraction.bert.formatter import DocumentOutputFormatter
from app.extraction.bert.models import PredictedEntity


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
