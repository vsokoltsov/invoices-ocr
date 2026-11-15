from decimal import Decimal

import pytest


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("$1,234.50", "1234.50"),
        ("1.234,50", "1234.50"),
        ("2 731,78", "2731.78"),
        ("-12.00", "-12.00"),
        ("1,000", None),
        ("1.2.34", None),
        ("USD 12.00", None),
        ("12.00 13.00", None),
        ("", None),
    ],
)
def test_normalize_amounts_without_merging_invalid_numbers(normalizer, value, expected):
    assert normalizer.normalize_number(value) == (
        Decimal(expected) if expected is not None else None
    )


def test_normalization_preserves_offsets_and_identifier_digits(normalizer):
    text = "  Straße—ＡＣＭＥ! "
    normalized, offsets = normalizer.normalize_with_offsets(text)
    assert normalized == "strasse acme"
    assert len(normalized) == len(offsets)
    assert offsets == sorted(offsets)
    assert offsets[4] == offsets[5] == text.index("ß")
    assert (
        normalizer.normalize("GB12 / AB-34", "payment_instructions.account_number")
        == "gb12ab34"
    )
    assert normalizer.normalize_number("10%", percentage=True) == Decimal("10")
    assert normalizer.normalize_number("10", percentage=True) is None
