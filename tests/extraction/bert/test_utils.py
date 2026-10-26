from collections.abc import Iterable

import pytest

from app.extraction.bert.enums import EntityType
from app.extraction.bert.spans import EntitySpan
from app.extraction.bert.utils import (
    group_spans_by_entity,
    money_digits,
    normalize_with_mapping,
)


@pytest.mark.parametrize(
    ("text", "expected_normalized", "expected_positions"),
    [
        (
            "A--  B\nC",
            "a b c",
            [0, 1, 5, 6, 7],
        ),
        (
            "  Invoice\t\t123  ",
            "invoice 123",
            [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13],
        ),
        (
            "Ä 42",
            "ä 42",
            [0, 1, 2, 3],
        ),
        (
            "---",
            "",
            [],
        ),
    ],
)
def test_normalize_with_mapping_collapses_separators_and_keeps_positions(
    text: str,
    expected_normalized: str,
    expected_positions: list[int],
) -> None:
    """Normalize OCR text while retaining an original index for each output character."""

    normalized, positions = normalize_with_mapping(text)

    assert normalized == expected_normalized
    assert positions == expected_positions
    assert len(normalized) == len(positions)


def test_normalize_with_mapping_maps_normalized_matches_to_original_text() -> None:
    """Allow normalized substring matches to be converted back to source boundaries."""

    text = "IBAN: GB17 KVLY"
    normalized, positions = normalize_with_mapping(text)
    start = normalized.index("gb17")
    end = start + len("gb17")

    original_start = positions[start]
    original_end = positions[end - 1] + 1

    assert normalized == "iban gb17 kvly"
    assert text[original_start:original_end] == "GB17"


def test_group_spans_by_entity_accepts_any_iterable_and_sorts_each_group() -> None:
    """Group spans by entity and return immutable, position-ordered tuples."""

    spans: Iterable[EntitySpan] = (
        span
        for span in [
            EntitySpan(
                start=20,
                end=25,
                entity=EntityType.SELLER_NAME,
                value="second",
            ),
            EntitySpan(
                start=5,
                end=10,
                entity=EntityType.TAX,
                value="10,00",
            ),
            EntitySpan(
                start=10,
                end=15,
                entity=EntityType.SELLER_NAME,
                value="first",
            ),
            EntitySpan(
                start=30,
                end=36,
                entity=EntityType.TAX,
                value="100,00",
            ),
        ]
    )

    grouped = group_spans_by_entity(spans)

    assert grouped[EntityType.SELLER_NAME] == (
        EntitySpan(
            start=10,
            end=15,
            entity=EntityType.SELLER_NAME,
            value="first",
        ),
        EntitySpan(
            start=20,
            end=25,
            entity=EntityType.SELLER_NAME,
            value="second",
        ),
    )
    assert [span.value for span in grouped[EntityType.TAX]] == [
        "10,00",
        "100,00",
    ]
    assert all(isinstance(value, tuple) for value in grouped.values())


def test_group_spans_by_entity_returns_empty_mapping_for_no_spans() -> None:
    """Return an empty mapping when a document has no aligned entity spans."""

    assert group_spans_by_entity([]) == {}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("€ 1.234,50", "123450"),
        ("-12,05", "1205"),
        (123.45, "12345"),
        (None, ""),
        ("no numeric characters", ""),
    ],
)
def test_money_digits_extracts_only_numeric_characters(
    value: object,
    expected: str,
) -> None:
    """Provide a separator-independent digit representation for money comparisons."""

    assert money_digits(value) == expected
