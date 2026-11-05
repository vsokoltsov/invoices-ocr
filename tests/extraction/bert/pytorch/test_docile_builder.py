import re

import pandas as pd
import pytest

from app.extraction.bert.pytorch.datasets import InvoiceDataset
from app.extraction.bert.pytorch.docile.builder import (
    DocileRecordBuilder,
    prepare_docile_records,
)
from app.extraction.bert.pytorch.schema import InvoiceLabelSchema


def docile_frame(text, fields):
    matches = list(re.finditer(r"\S+", text))
    words = [
        {
            "text": match.group(),
            "start": match.start(),
            "end": match.end(),
            "page": 0,
            "bbox": [index / 10, 0.1, (index + 0.8) / 10, 0.2],
        }
        for index, match in enumerate(matches)
    ]
    data = {}
    for path, (first, last) in fields.items():
        section, name = path.split(".")
        value = text[matches[first].start() : matches[last].end()]
        data.setdefault(section, {})[name] = {
            "value": value,
            "source_values": [value],
            "pages": [0],
            "bbox": [[first / 10, 0.1, (last + 0.8) / 10, 0.2]],
        }
    return pd.DataFrame(
        [
            {
                "document_id": "doc-1",
                "ocr_text": text,
                "text_with_boxes": words,
                "json_data": data,
            }
        ]
    )


def test_docile_records_use_shared_dataset_contract(fast_tokenizer):
    schema = InvoiceLabelSchema()
    frame = docile_frame(
        "Acme Corporation prefix suffix",
        {
            "invoice.seller_name": (0, 1),
            "invoice.seller_address": (0, 3),
            "invoice.due_date": (2, 2),
            "payment_instructions.payment_terms": (3, 3),
        },
    )
    records, rejected = DocileRecordBuilder(schema).build(frame)
    assert not rejected
    assert records[0]["text"] == frame.iloc[0]["ocr_text"]
    assert [entity["text"] for entity in records[0]["entities"]] == [
        "Acme Corporation",
        "prefix suffix",
    ]
    assert records[0]["ignored_regions"] == []
    dataset = InvoiceDataset(
        records,
        fast_tokenizer,
        schema.label2id,
        schema.field_to_entity,
    )
    assert dataset[0]["labels"] == [
        -100,
        schema.label2id["B-SELLER_NAME"],
        schema.label2id["I-SELLER_NAME"],
        schema.label2id["B-SELLER_ADDRESS"],
        schema.label2id["I-SELLER_ADDRESS"],
        -100,
    ]


def test_custom_schema_supports_docile_specific_fields(fast_tokenizer):
    schema = InvoiceLabelSchema(
        {
            "invoice.due_date": "DUE_DATE",
            "payment_instructions.payment_terms": "PAYMENT_TERMS",
        }
    )
    frame = docile_frame(
        "middle suffix",
        {
            "invoice.due_date": (0, 0),
            "payment_instructions.payment_terms": (1, 1),
        },
    )
    records, rejected = DocileRecordBuilder(schema).build(frame)
    assert not rejected
    dataset = InvoiceDataset(
        records,
        fast_tokenizer,
        schema.label2id,
        schema.field_to_entity,
    )
    assert dataset[0]["labels"] == [
        -100,
        schema.label2id["B-DUE_DATE"],
        schema.label2id["B-PAYMENT_TERMS"],
        -100,
    ]


@pytest.mark.parametrize(
    "fields,status",
    [
        ({"invoice.client_name": (0, 0), "invoice.seller_name": (0, 0)}, "overlap"),
        (
            {"invoice.client_name": (0, 0), "invoice.client_address": (0, 0)},
            "no_supervised_words",
        ),
        ({"invoice.due_date": (0, 0)}, "no_selected_annotations"),
    ],
)
def test_builder_rejects_only_issues_in_selected_annotations(fields, status):
    frame = docile_frame("Acme", fields)
    records, rejected = DocileRecordBuilder().build(frame)
    assert not records
    assert status in {issue["status"] for issue in rejected[0]["issues"]}


def test_preparation_function_uses_default_schema():
    frame = docile_frame("Acme", {"invoice.client_name": (0, 0)})
    assert prepare_docile_records(frame) == DocileRecordBuilder().build(frame)


@pytest.mark.parametrize("failure", ["missing_span", "invalid_offsets"])
def test_builder_rejects_missing_words_and_invalid_character_offsets(failure):
    frame = docile_frame("Acme", {"invoice.client_name": (0, 0)})
    if failure == "missing_span":
        frame.iloc[0]["json_data"]["invoice"]["client_name"]["pages"] = [3]
    else:
        frame.iloc[0]["text_with_boxes"][0]["end"] = 100
    records, rejected = DocileRecordBuilder().build(frame)
    assert not records
    assert rejected[0]["document_id"] == "doc-1"
    assert failure in {issue["status"] for issue in rejected[0]["issues"]}
