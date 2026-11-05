from copy import deepcopy
from dataclasses import replace

import pytest

from app.datasets.docile.converter import DocileAnnotationConverter
from app.datasets.docile.models import OCRWord
from app.datasets.docile.order import OCRReadingOrder


def test_converter_retains_repeated_locations_fragments_and_unknown_fields():
    def field(name, text, page=0):
        return {
            "fieldtype": name,
            "text": text,
            "page": page,
            "bbox": [0.1, 0.1, 0.2, 0.2],
        }

    raw = {
        "field_extractions": [
            field("vendor_name", "Acme"),
            field("vendor_name", "ACME", 1),
            field("iban", "GB123"),
        ],
        "line_item_extractions": [
            {**field("line_item_description", "first"), "line_item_id": 7},
            {**field("line_item_description", "second"), "line_item_id": 7},
            {**field("line_item_quantity", "2"), "line_item_id": 8},
        ],
        "metadata": {"currency": "usd"},
    }
    original = deepcopy(raw)
    result = DocileAnnotationConverter().convert(raw)
    assert raw == original
    seller = result["invoice"]["seller_name"]
    assert seller["pages"] == [0, 1]
    assert seller["source_values"] == ["Acme", "ACME"]
    assert len(seller["bbox"]) == 2
    assert result["other_fields"]["iban"]["value"] == "GB123"
    assert result["items"][0]["fields"]["description"]["value"] == "first\nsecond"
    assert [item["line_item_id"] for item in result["items"]] == [7, 8]
    assert result["metadata"] == {"currency": "usd"}


def test_reading_order_sorts_pages_rows_and_rebuilds_offsets():
    first = OCRWord(
        text="first",
        bbox=(0.1, 0.1, 0.2, 0.2),
        page=0,
        start=100,
        end=105,
        confidence=1.0,
        block_index=0,
        line_index=0,
        word_index=0,
    )
    second = replace(first, text="second", bbox=(0.3, 0.11, 0.4, 0.21))
    third = replace(first, text="third", bbox=(0.1, 0.4, 0.2, 0.5))
    last = replace(first, text="last", page=1)
    text, words = OCRReadingOrder().build([last, third, second, first])
    assert text == "first second third last"
    assert [word.page for word in words] == [0, 0, 0, 1]
    assert all(text[w.start : w.end] == w.text for w in words)
    with pytest.raises(ValueError, match="word heights"):
        OCRReadingOrder().build([replace(first, bbox=(0, 0, 1, 0))])
