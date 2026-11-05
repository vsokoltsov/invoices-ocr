import pandas as pd
import pytest
import torch
from torch import nn

from app.extraction.bert.pytorch.inspector import EntityErrorInspector
from app.extraction.bert.pytorch.schema import InvoiceLabelSchema


class AddressTagger(nn.Module):
    scores: torch.Tensor

    def __init__(self, schema):
        super().__init__()
        logits = torch.zeros(schema.num_labels)
        logits[schema.label2id["B-SELLER_ADDRESS"]] = 2
        logits[schema.label2id["I-SELLER_ADDRESS"]] = 1
        self.register_buffer("scores", logits)

    def forward(self, input_ids, attention_mask):  # ty: ignore[invalid-mutable-override]
        return self.scores.expand(*input_ids.shape, -1)


@pytest.mark.parametrize("prefix_count,windows", [(12, [2]), (11, [1, 2])])
def test_token_exports_select_character_region_and_include_role_probabilities(
    tmp_path,
    fast_tokenizer,
    prefix_count,
    windows,
):
    schema = InvoiceLabelSchema()
    model = AddressTagger(schema)
    text = "prefix " * prefix_count + "Acme Corporation" + " suffix" * 12
    start = text.index("Acme")
    end = start + len("Acme Corporation")
    record = {
        "document_id": "debug-doc",
        "text": text,
        "entities": [
            {
                "start": start,
                "end": end,
                "label": "invoice.client_address",
                "text": text[start:end],
            }
        ],
    }
    inspector = EntityErrorInspector(
        model,
        fast_tokenizer,
        schema,
        torch.device("cpu"),
        max_length=8,
        debug_document_id="debug-doc",
        debug_start=start,
        debug_end=end,
    )
    inspector.run([record], tmp_path, epoch=7)
    paths = sorted(tmp_path.glob("tokens_*.csv"))
    assert [p.name for p in paths] == [
        f"tokens_debug-doc_window_{window:03d}.csv" for window in windows
    ]
    frames = [pd.read_csv(path, keep_default_na=False) for path in paths]
    for frame, window in zip(frames, windows, strict=True):
        assert frame["epoch"].unique().tolist() == [7]
        assert frame["record_split"].unique().tolist() == ["validation"]
        assert frame["window_index"].unique().tolist() == [window]
        assert (frame["actual"] == "IGNORED").sum() == 2
        assert frame["predicted"].eq("B-SELLER_ADDRESS").all()
        expected = model.scores.softmax(dim=-1)
        for entity in ("CLIENT_ADDRESS", "SELLER_ADDRESS"):
            probability = sum(
                expected[schema.label2id[f"{prefix}-{entity}"]].item()
                for prefix in ("B", "I")
            )
            assert frame[f"{entity.lower()}_probability"].tolist() == pytest.approx(
                [probability] * len(frame)
            )
        assert (
            frame["seller_address_probability"] > frame["client_address_probability"]
        ).all()
    address_rows = pd.concat(frames).loc[lambda f: (f.start < end) & (f.end > start)]
    assert address_rows["ocr_text"].tolist() == ["Acme", "Corporation"]
    assert address_rows["actual"].str.endswith("CLIENT_ADDRESS").all()
    assert (tmp_path / "per_entity.csv").is_file()


def test_unmatched_debug_document_does_not_export_another_record(
    tmp_path, fast_tokenizer, capsys
):
    schema = InvoiceLabelSchema()
    inspector = EntityErrorInspector(
        AddressTagger(schema),
        fast_tokenizer,
        schema,
        torch.device("cpu"),
        debug_document_id="missing",
        debug_start=0,
        debug_end=4,
    )
    inspector.run(
        [
            {
                "document_id": "another",
                "text": "Acme",
                "entities": [
                    {
                        "start": 0,
                        "end": 4,
                        "label": "invoice.client_address",
                        "text": "Acme",
                    }
                ],
            }
        ],
        tmp_path,
        epoch=1,
    )
    assert not list(tmp_path.glob("tokens_*.csv"))
    assert "document_found=False" in capsys.readouterr().out


def test_training_debug_target_is_exported_without_changing_validation_metrics(
    tmp_path,
    fast_tokenizer,
):
    schema = InvoiceLabelSchema()
    validation = {
        "document_id": "validation-doc",
        "text": "Acme",
        "entities": [
            {"start": 0, "end": 4, "label": "invoice.client_address", "text": "Acme"}
        ],
    }
    text = "prefix " * 12 + "Acme Corporation"
    start = text.index("Acme")
    training = {
        "document_id": "train-debug",
        "text": text,
        "entities": [
            {
                "start": start,
                "end": len(text),
                "label": "invoice.client_address",
                "text": text[start:],
            }
        ],
    }
    inspector = EntityErrorInspector(
        AddressTagger(schema),
        fast_tokenizer,
        schema,
        torch.device("cpu"),
        max_length=8,
        debug_document_id="train-debug",
        debug_start=start,
        debug_end=len(text),
        training_records=[training],
    )
    report, errors = inspector.run([validation], tmp_path, epoch=1)
    tokens = pd.read_csv(tmp_path / "tokens_train-debug_window_002.csv")
    assert tokens["record_split"].unique().tolist() == ["train"]
    assert report.loc["CLIENT_ADDRESS", "support"] == 1
    assert set(errors["document_id"]) == {"validation-doc"}
