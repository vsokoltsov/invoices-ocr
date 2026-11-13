import json
from types import SimpleNamespace

import pymupdf
import pytest
from tokenizers.pre_tokenizers import ByteLevel
from transformers import (
    AutoModel,
    LayoutLMv3Config,
    LayoutLMv3ImageProcessor,
    LayoutLMv3Model,
    LayoutLMv3Tokenizer,
)


@pytest.fixture
def document_row(tmp_path):
    pdf_path = tmp_path / "invoice.pdf"
    with pymupdf.open() as document:
        for index in range(2):
            page = document.new_page(width=144, height=144)
            page.insert_text((20, 40), f"Invoice page {index + 1}")
        document.save(pdf_path)
    return SimpleNamespace(
        document_id="doc-1",
        pdf_path=str(pdf_path),
        json_data=json.dumps(
            {
                "metadata": {"page_count": 2},
                "invoice": {
                    "seller_name": {
                        "pages": [0],
                        "bbox": [[0.1, 0.1, 0.6, 0.2]],
                    }
                },
            }
        ),
        text_with_boxes=[
            {"text": "BB", "start": 2, "page": 0, "bbox": [0.3, 0.1, 0.5, 0.2]},
            {"text": "C", "start": 5, "page": 0, "bbox": [0.1, 0.4, 0.2, 0.5]},
            {"text": "A", "start": 0, "page": 0, "bbox": [0.1, 0.1, 0.2, 0.2]},
            {"text": "Other", "start": 7, "page": 1, "bbox": [0.1, 0.1, 0.4, 0.2]},
        ],
    )


@pytest.fixture
def layout_tokenizer():
    tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    tokens.extend(sorted(ByteLevel.alphabet()))
    return LayoutLMv3Tokenizer(
        vocab={token: index for index, token in enumerate(tokens)}, merges=[]
    )


@pytest.fixture
def image_processor():
    return LayoutLMv3ImageProcessor(apply_ocr=False, size={"height": 16, "width": 16})


@pytest.fixture
def layout_backbone(layout_tokenizer, monkeypatch):
    config = LayoutLMv3Config(
        vocab_size=len(layout_tokenizer),
        hidden_size=24,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        coordinate_size=4,
        shape_size=4,
        input_size=16,
        patch_size=8,
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
    )
    config._name_or_path = "local-tiny"
    backbone = LayoutLMv3Model(config)
    monkeypatch.setattr(AutoModel, "from_pretrained", lambda *args, **kwargs: backbone)
    return backbone
