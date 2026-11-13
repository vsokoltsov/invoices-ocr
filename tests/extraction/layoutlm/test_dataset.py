from copy import deepcopy
from unittest.mock import Mock

import pytest
import torch
from torch.utils.data import DataLoader

from app.extraction.layoutlm.builder import InvoiceBuilder
from app.extraction.layoutlm.converter import AttachmentToImageConverter
from app.extraction.layoutlm.dataset import InvoiceDataset
from app.extraction.layoutlm.enums import FIELD_TO_ENTITY


def test_encoding_preserves_word_alignment_and_caches_float_images(
    document_row,
    layout_tokenizer,
    image_processor,
    tmp_path,
):
    pages = InvoiceBuilder(FIELD_TO_ENTITY).prepare_document(document_row)
    converter = AttachmentToImageConverter(tmp_path / "images")
    processor = Mock(wraps=image_processor)
    dataset = InvoiceDataset(
        pages, layout_tokenizer, processor, converter, max_length=8
    )
    assert len(dataset) == 3
    assert processor.call_count == len(pages)
    assert layout_tokenizer.convert_ids_to_tokens(dataset[0]["input_ids"].tolist()) == [
        "<s>",
        "Ġ",
        "A",
        "Ġ",
        "B",
        "B",
        "</s>",
        "<pad>",
    ]
    assert dataset[0]["word_ids"].tolist() == [-1, 0, -1, 1, -1, -1, -1, -1]
    assert dataset[0]["attention_mask"].tolist() == [1, 1, 1, 1, 1, 1, 1, 0]
    seen = []
    for index in range(len(dataset)):
        sample = dataset[index]
        valid = sample["word_ids"] >= 0
        page = pages[int(sample["page_index"])]
        assert sample["labels"][valid].tolist() == [
            page["labels"][word] for word in sample["word_ids"][valid].tolist()
        ]
        assert sample["bbox"][valid].tolist() == [
            page["boxes"][word] for word in sample["word_ids"][valid].tolist()
        ]
        assert sample["pixel_values"].dtype == torch.float32
        assert sample["bbox"].shape == (8, 4)
        assert sample["input_ids"].dtype == torch.long
        valid = sample["labels"] != -100
        assert torch.equal(valid, sample["word_ids"] >= 0)
        seen.extend(
            (sample["page_index"].item(), word)
            for word in sample["word_ids"][valid].tolist()
        )
    assert seen == [(0, 0), (0, 1), (0, 2), (1, 0)]
    first_page = [sample for sample in dataset if sample["page_index"] == 0]
    assert len(first_page) > 1
    assert first_page[0]["pixel_values"] is first_page[1]["pixel_values"]
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    assert batch["pixel_values"].shape == (2, 3, 16, 16)


@pytest.mark.parametrize("max_length", [3, 513])
def test_invalid_window_length_is_rejected(
    max_length, layout_tokenizer, image_processor, tmp_path
):
    with pytest.raises(ValueError, match="max_length"):
        InvoiceDataset(
            [],
            layout_tokenizer,
            image_processor,
            AttachmentToImageConverter(tmp_path),
            max_length=max_length,
        )


@pytest.mark.parametrize("failure", ["empty", "mismatched", "overlong"])
def test_invalid_words_fail_before_training(
    document_row, layout_tokenizer, image_processor, tmp_path, failure
):
    page = deepcopy(InvoiceBuilder(FIELD_TO_ENTITY).prepare_document(document_row)[0])
    if failure == "empty":
        page.update(words=[], boxes=[], labels=[])
    elif failure == "mismatched":
        page["labels"].pop()
    else:
        page["words"][0] = "A" * 20
    with pytest.raises(ValueError):
        InvoiceDataset(
            [page],
            layout_tokenizer,
            image_processor,
            AttachmentToImageConverter(tmp_path),
            max_length=8,
        )


def test_custom_ignore_index_masks_special_padding_and_continuation_tokens(
    document_row, layout_tokenizer, image_processor, tmp_path
):
    pages = InvoiceBuilder(FIELD_TO_ENTITY).prepare_document(document_row)
    dataset = InvoiceDataset(
        pages,
        layout_tokenizer,
        image_processor,
        AttachmentToImageConverter(tmp_path),
        ignore_index=-7,
        max_length=8,
    )
    for sample in dataset:
        assert torch.equal(sample["labels"] == -7, sample["word_ids"] == -1)
