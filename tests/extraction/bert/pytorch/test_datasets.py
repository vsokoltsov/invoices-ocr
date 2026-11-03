from app.extraction.bert.pytorch.datasets import InvoiceDataset
from app.extraction.bert.pytorch.schema import InvoiceLabelSchema


def make_dataset(fast_tokenizer, text, entities, ignored_regions=(), max_length=32):
    schema = InvoiceLabelSchema()
    dataset = InvoiceDataset(
        [{"text": text, "entities": entities, "ignored_regions": ignored_regions}],
        fast_tokenizer,
        schema.label2id,
        schema.field_to_entity,
        max_length=max_length,
    )
    return dataset, schema


def test_dataset_aligns_bio_labels_and_masks_special_tokens(fast_tokenizer):
    dataset, schema = make_dataset(
        fast_tokenizer,
        "Acme Corporation suffix",
        [
            {"start": 0, "end": 16, "label": "invoice.seller_name"},
        ],
    )
    assert dataset[0]["labels"] == [
        -100,
        schema.label2id["B-SELLER_NAME"],
        schema.label2id["I-SELLER_NAME"],
        schema.label2id["O"],
        -100,
    ]
    assert "offset_mapping" not in dataset[0]


def test_dataset_masks_ignored_conflicting_and_boundary_crossing_tokens(fast_tokenizer):
    dataset, _ = make_dataset(
        fast_tokenizer,
        "Acme Corporation suffix",
        [
            {"start": 0, "end": 4, "label": "invoice.seller_name"},
            {"start": 0, "end": 4, "label": "invoice.client_name"},
            {"start": 6, "end": 16, "label": "invoice.client_address"},
        ],
        ignored_regions=[{"start": 17, "end": 23}],
    )
    assert dataset[0]["labels"] == [-100] * 5


def test_dataset_retains_all_overflow_windows(fast_tokenizer):
    dataset, schema = make_dataset(
        fast_tokenizer,
        "Acme " * 7,
        [
            {"start": 0, "end": 34, "label": "invoice.seller_name"},
        ],
        max_length=5,
    )
    assert len(dataset) == 3
    assert sum(len(e["input_ids"]) - 2 for e in dataset.examples) == 7
    assert all(
        e["labels"][1] == schema.label2id["B-SELLER_NAME"] for e in dataset.examples
    )
