import re
from copy import deepcopy

import pandas as pd
import pytest

from app.aggregates.bert.params import DatasetName
from app.extraction.bert.pytorch.datasets import InvoiceDataset
from app.extraction.bert.pytorch.docile.item_builder import DocileItemRecordBuilder
from app.extraction.bert.pytorch.pipeline import BertPytorchPipeline


def item_frame():
    text = "Acme Corporation prefix suffix"
    words = [
        {
            "text": match.group(),
            "start": match.start(),
            "end": match.end(),
            "page": 0,
            "bbox": [index / 10, 0.1, (index + 0.8) / 10, 0.2],
        }
        for index, match in enumerate(re.finditer(r"\S+", text))
    ]

    def field(index):
        return {
            "value": words[index]["text"],
            "source_values": [words[index]["text"]],
            "pages": [0],
            "bbox": [words[index]["bbox"]],
        }

    return pd.DataFrame(
        [
            {
                "document_id": "items",
                "ocr_text": text,
                "text_with_boxes": words,
                "json_data": {
                    "invoice": {"seller_name": field(0)},
                    "line_item_headers": {"description": field(0)},
                    "items": [
                        {
                            "line_item_id": 7,
                            "fields": {
                                "description": field(0),
                                "quantity": field(2),
                                "unit_price_net": field(0),
                            },
                        },
                        {
                            "line_item_id": 9,
                            "fields": {
                                "description": field(1),
                                "total_price": field(3),
                            },
                        },
                    ],
                },
            }
        ]
    )


def test_item_only_records_keep_membership_and_use_shared_token_dataset(fast_tokenizer):
    records, rejected, schema = BertPytorchPipeline.prepare_records(
        item_frame(), DatasetName.DOCILE, line_items_only=True
    )
    assert not rejected
    assert schema.num_labels == 9
    entities = records[0]["entities"]
    assert [(e["text"], e["label"], e["line_item_id"]) for e in entities] == [
        ("Acme", "items.description", 7),
        ("Corporation", "items.description", 9),
        ("prefix", "items.quantity", 7),
        ("suffix", "items.total_price", 9),
    ]
    assert all(e["sources"][0]["line_item_id"] == e["line_item_id"] for e in entities)
    dataset = InvoiceDataset(
        records, fast_tokenizer, schema.label2id, schema.field_to_entity
    )
    labels = [schema.id2label[label] for label in dataset[0]["labels"] if label != -100]
    assert labels == [
        "B-ITEM_DESCRIPTION",
        "B-ITEM_DESCRIPTION",
        "B-ITEM_QUANTITY",
        "B-ITEM_AMOUNT_GROSS",
    ]


@pytest.mark.parametrize(
    "failure", ["overlap", "missing_span", "invalid_offsets", "no_selected_annotations"]
)
def test_item_builder_reports_conflicts_missing_words_and_invalid_offsets(failure):
    frame = item_frame()
    data = frame.iloc[0]["json_data"]
    if failure == "overlap":
        # Identical labels from different items must not collapse into one label.
        data["items"][1]["fields"]["description"] = deepcopy(
            data["items"][0]["fields"]["description"]
        )
    elif failure == "missing_span":
        data["items"][0]["fields"]["description"]["pages"] = [9]
    elif failure == "invalid_offsets":
        frame.iloc[0]["text_with_boxes"][0]["end"] = 999
    else:
        data["items"] = []
    records, rejected = DocileItemRecordBuilder().build(frame)
    assert not records
    assert failure in {issue["status"] for issue in rejected[0]["issues"]}
    if failure == "overlap":
        issue = next(
            issue for issue in rejected[0]["issues"] if issue["status"] == "overlap"
        )
        assert issue["line_item_ids"] == [7, 9]


def test_item_description_fragments_keep_id_across_pages():
    frame = item_frame()
    data = frame.iloc[0]["json_data"]
    description = data["items"][0]["fields"]["description"]
    other = data["items"][1]["fields"].pop("description")
    description["bbox"].extend(other["bbox"])
    description["pages"].append(1)
    description["source_values"].extend(other["source_values"])
    frame.iloc[0]["text_with_boxes"][1]["page"] = 1
    records, rejected = DocileItemRecordBuilder().build(frame)
    assert not rejected
    descriptions = [
        e for e in records[0]["entities"] if e["label"] == "items.description"
    ]
    assert [(e["text"], e["page"], e["line_item_id"]) for e in descriptions] == [
        ("Acme", 0, 7),
        ("Corporation", 1, 7),
    ]


def test_item_only_mode_does_not_accept_kaggle():
    with pytest.raises(ValueError, match="requires --dataset-name docile"):
        BertPytorchPipeline.prepare_records(
            pd.DataFrame(), DatasetName.KAGGLE_INVOICES, line_items_only=True
        )
