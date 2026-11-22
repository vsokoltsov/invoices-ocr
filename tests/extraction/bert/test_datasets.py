from app.extraction.bert.datasets import InvoiceDataset
from app.extraction.schema import InvoiceLabelSchema


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


def test_ocr_words_are_scored_once_even_when_split_into_multiple_tokens(fast_tokenizer):
    schema = InvoiceLabelSchema({"invoice.seller_name": "NAME"})
    # Whitespace tokenization splits punctuation. OCR supplies one word for A-B.
    record = {
        "text": "A-B Acme",
        "entities": [{"start": 0, "end": 8, "label": "invoice.seller_name"}],
        "text_with_boxes": [
            {"text": "A-B", "start": 0, "end": 3},
            {"text": "Acme", "start": 4, "end": 8},
        ],
    }
    dataset = InvoiceDataset(
        [record], fast_tokenizer, schema.label2id, schema.field_to_entity, max_length=4
    )
    labels = [
        label for sample in dataset for label in sample["labels"] if label != -100
    ]
    assert labels == [schema.label2id["B-NAME"], schema.label2id["B-NAME"]]
