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


def test_numeric_pattern_keeps_decimal_boundary_and_percentage(span_matcher):
    numeric = span_matcher.numeric
    pattern = numeric.number_pattern("1234.50")
    assert pattern.fullmatch("1 234,50")
    assert not pattern.fullmatch("12345,0")
    assert numeric.number_pattern("10%").fullmatch("10 %")
    assert numeric.is_numeric_field("items[2].quantity")
    assert not numeric.is_numeric_field("items[2].description")
    with pytest.raises(ValueError, match="Invalid or ambiguous"):
        numeric.number_pattern("1,000")


@pytest.mark.parametrize(
    "text",
    [
        "no summary",
        "Total 10,00 11,00",
        "Total 10,00 10,00 20,00",
    ],
)
def test_summary_rejects_missing_or_ambiguous_amounts(span_matcher, text):
    result = span_matcher.summary_span.find(text, {"subtotal": {"tax": "10.00"}})
    assert result == {}


def test_summary_uses_last_total_and_original_offsets(span_matcher):
    text = "Total 1,00 2,00 3,00 SUMMARY 10% Total $ 100,00 $ 10,00 $ 110,00"
    result = span_matcher.summary_span.find(
        text, {"subtotal": {"tax": "10.00", "total": "110.00"}}
    )
    assert {key: value.text for key, value in result.items()} == {
        "subtotal.tax": "10,00",
        "subtotal.total": "110,00",
    }
    for entity in result.values():
        assert text[entity.start : entity.end] == entity.text
        assert entity.start > text.index("SUMMARY")
        assert entity.needs_review is False
    percentages = span_matcher.summary_span.find(text, {"subtotal": {"tax": "10%"}})
    assert percentages["subtotal.tax"].text == "10%"


def test_names_addresses_and_items_use_expected_alignment_policy(span_matcher):
    text = "Seller Acme Corporation 12 Main Street Town IBAN: GB12ABCD12345678901234"
    result = span_matcher.find_spans(
        text,
        {
            "invoice": {
                "seller_name": "Acme Corporation",
                "seller_address": "12 Main Street\nTown",
            },
            "items": [{"description": "Acme Corporation", "quantity": "1.00"}],
        },
    )
    assert [entity.text for entity in result.entities] == [
        "Acme Corporation",
        "12 Main Street",
        "Town",
    ]
    assert all(text[e.start : e.end] == e.text for e in result.entities)
    assert all(not e.needs_review for e in result.entities)


def test_partial_names_exclude_neighboring_unrelated_words(span_matcher):
    entity = span_matcher.find_name_span(
        "Parker Thompson, and Johnson",
        "Thompson, Terry and Johnson",
        "invoice.client_name",
    )
    assert entity is not None
    assert entity.text == "Thompson, and Johnson"
    assert entity.match_status == "partial"
    assert entity.needs_review is False


def test_description_spans_have_exact_exclusive_offsets(span_matcher):
    text = "Acme Corporation!"
    spans, unmatched = span_matcher.find_description_spans(
        text, "Acme Corporation", "items.description"
    )
    assert not unmatched
    assert len(spans) == 1
    assert text[spans[0].start : spans[0].end] == spans[0].text


@pytest.mark.parametrize("text", ["", "!!!"])
def test_empty_normalized_text_does_not_create_invalid_spans(span_matcher, text):
    assert (
        span_matcher.find_normalized_span(text, "Acme", "invoice.seller_name") is None
    )
    result = span_matcher.find_spans(
        text,
        {
            "invoice": {"invoice_number": "123", "seller_address": "12 Main"},
            "payment_instructions": {"account_number": "GB12"},
        },
    )
    assert result.entities == []
