import json

import pandas as pd
import pytest
import torch
from transformers import BertConfig, BertModel

from app.aggregates.bert.params import BertPytorchDebugParams
from app.extraction.bert.pytorch.checkpoint import BertPytorchCheckpoint
from app.extraction.bert.pytorch.debug import BertPytorchDebugPipeline
from app.extraction.bert.pytorch.model import BertFieldExtractor
from app.extraction.bert.pytorch.schema import InvoiceLabelSchema


def test_checkpoint_reloads_weights_tokenizer_and_validation_without_network(
    tmp_path, fast_tokenizer
):
    schema = InvoiceLabelSchema({"subtotal.total": "TOTAL"})
    model = BertFieldExtractor(
        "",
        schema.num_labels,
        encoder=BertModel(
            BertConfig(
                vocab_size=len(fast_tokenizer),
                hidden_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=16,
            )
        ),
    ).eval()
    records = [
        {
            "document_id": "doc",
            "text": "Acme Corporation",
            "entities": [
                {
                    "start": 0,
                    "end": 16,
                    "label": "subtotal.total",
                    "text": "Acme Corporation",
                }
            ],
            "ignored_regions": [],
        }
    ]
    words = [
        {"start": 0, "end": 4, "text": "Acme"},
        {"start": 5, "end": 16, "text": "Corporation"},
    ]
    checkpoint = BertPytorchCheckpoint(tmp_path / "checkpoint")
    checkpoint.save(
        model,
        fast_tokenizer,
        schema,
        records,
        pd.DataFrame([{"document_id": "doc", "text_with_boxes": words}]),
        epoch=19,
        max_length=32,
    )
    restored, tokenizer, restored_schema, payload = checkpoint.load(torch.device("cpu"))
    assert restored_schema.labels == schema.labels
    assert payload["epoch"] == 19
    assert payload["records"][0]["text_with_boxes"] == words
    inputs = fast_tokenizer(records[0]["text"], return_tensors="pt")
    restored_inputs = tokenizer(records[0]["text"], return_tensors="pt")
    assert inputs["input_ids"].tolist() == restored_inputs["input_ids"].tolist()
    with torch.inference_mode():
        torch.testing.assert_close(
            model(inputs["input_ids"], inputs["attention_mask"]),
            restored(restored_inputs["input_ids"], restored_inputs["attention_mask"]),
        )
    output = BertPytorchDebugPipeline().run(
        BertPytorchDebugParams(
            str(checkpoint.directory), str(tmp_path / "debug"), "cpu"
        )
    )
    assert set(pd.read_csv(output)["decoder"]) == {"token_argmax", "ocr_word"}
    # A checkpoint without OCR words cannot silently use whitespace as OCR words.
    path = checkpoint.directory / "validation.json"
    del payload["records"][0]["text_with_boxes"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="DocILE validation words"):
        BertPytorchDebugPipeline().run(
            BertPytorchDebugParams(
                str(checkpoint.directory), str(tmp_path / "invalid"), "cpu"
            )
        )


def test_debug_fails_for_missing_artifacts_and_invalid_configuration(tmp_path):
    pipeline = BertPytorchDebugPipeline()
    with pytest.raises(FileNotFoundError, match="Missing checkpoint artifact"):
        pipeline.run(
            BertPytorchDebugParams(str(tmp_path / "missing"), str(tmp_path / "out"))
        )
    with pytest.raises(ValueError, match="outside the checkpoint"):
        pipeline.run(BertPytorchDebugParams(str(tmp_path), str(tmp_path / "out")))
    with pytest.raises(ValueError, match="Device must"):
        pipeline.run(
            BertPytorchDebugParams(
                str(tmp_path / "model"), str(tmp_path / "out"), "invalid"
            )
        )
