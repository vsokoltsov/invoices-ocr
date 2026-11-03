import json

import pandas as pd
import pytest

from app.extraction.bert.pytorch.annotations import OCRSpanEntity, OCRSpanResult


def test_builder_returns_json_serializable_records_and_rejections(record_builder):
    frame = pd.DataFrame(
        [
            {
                "ocred_text": "Seller Acme Corporation IBAN: GB12ABCD12345678901234",
                "json_data": json.dumps(
                    {
                        "invoice": {"seller_name": "Acme Corporation"},
                        "payment_instructions": {"account_number": ""},
                    }
                ),
            },
            {
                "ocred_text": "Total 10,00 1,00 11,00",
                "json_data": {
                    "subtotal": {"tax": "1.00", "total": "12.00"},
                },
            },
        ],
        index=pd.Index([10, 20]),
    )
    records, rejected = record_builder.build(frame)
    assert [r["row_index"] for r in records] == [10]
    assert records[0]["entities"][0]["label"] == "invoice.seller_name"
    region = records[0]["ignored_regions"][0]
    assert (
        records[0]["text"][region["start"] : region["end"]] == "GB12ABCD12345678901234"
    )
    assert rejected[0]["row_index"] == 20
    assert any(issue["status"] == "missing" for issue in rejected[0]["issues"])
    json.dumps((records, rejected))


@pytest.mark.parametrize(
    ("case", "status"),
    [
        ("review", "needs_review"),
        ("overlap", "overlap"),
        ("address", "partial_address"),
    ],
)
def test_builder_rejects_review_overlap_and_partial_addresses(
    record_builder, monkeypatch, case, status
):
    text = "Acme Corporation"
    target = {}
    entities = [OCRSpanEntity(0, 4, "invoice.seller_name", "Acme")]
    if case == "review":
        entities[0].needs_review = True
    elif case == "overlap":
        entities.append(OCRSpanEntity(0, 4, "invoice.client_name", "Acme"))
    else:
        target = {"invoice": {"seller_address": "Acme Corporation"}}
        entities = [OCRSpanEntity(0, 4, "invoice.seller_address", "Acme")]
    monkeypatch.setattr(record_builder, "find_unmatched_fields", lambda row: [])
    monkeypatch.setattr(
        record_builder.span_matcher,
        "find_spans",
        lambda *args: OCRSpanResult(text, entities),
    )
    records, rejected = record_builder.build(
        pd.DataFrame([{"ocred_text": text, "json_data": target}])
    )
    assert not records
    assert status in {issue["status"] for issue in rejected[0]["issues"]}


def test_builder_raises_for_invalid_offsets(record_builder, monkeypatch):
    monkeypatch.setattr(record_builder, "find_unmatched_fields", lambda row: [])
    monkeypatch.setattr(
        record_builder.span_matcher,
        "find_spans",
        lambda *args: OCRSpanResult(
            "Acme", [OCRSpanEntity(0, 99, "invoice.seller_name", "Acme")]
        ),
    )
    with pytest.raises(ValueError, match="Invalid span"):
        record_builder.build(pd.DataFrame([{"ocred_text": "Acme", "json_data": {}}]))


def test_diagnostics_compare_numeric_values_and_serialize_review_spans(
    record_builder, monkeypatch
):
    entities = [
        OCRSpanEntity(0, 4, "subtotal.tax", "2,00"),
        OCRSpanEntity(5, 9, "invoice.seller_name", "Acme", needs_review=True),
    ]
    monkeypatch.setattr(
        record_builder.span_matcher,
        "find_spans",
        lambda *args: OCRSpanResult("2,00 Acme", entities),
    )
    issues = record_builder.find_unmatched_fields(
        {
            "ocred_text": "2,00 Acme",
            "json_data": {
                "subtotal": {"tax": "1.00"},
                "invoice": {"seller_name": "Acme Corp"},
            },
        }
    )
    assert {issue["status"] for issue in issues} == {"numeric_mismatch", "needs_review"}
    json.dumps(issues)


def test_account_regions_are_ignored_only_without_expected_account(record_builder):
    text = "IBAN: GB12ABCD"
    assert record_builder.find_ignored_account_regions(text, {}) == [
        {"start": 6, "end": 14}
    ]
    assert (
        record_builder.find_ignored_account_regions(
            text,
            {
                "payment_instructions": {"account_number": "GB12ABCD"},
            },
        )
        == []
    )


def test_flatten_fields_preserves_item_indices_and_skips_empty_values(record_builder):
    assert record_builder._flatten_field_paths(
        {
            "invoice": {"seller_name": "Acme", "due_date": " "},
            "items": [{"quantity": "1.00"}],
        }
    ) == [{"invoice.seller_name": "Acme"}, {"items[0].quantity": "1.00"}]
