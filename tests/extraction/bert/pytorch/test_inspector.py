import pandas as pd
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


def test_entity_errors_keep_offsets_epoch_and_document_identity(
    tmp_path, fast_tokenizer
):
    schema = InvoiceLabelSchema()
    text = "Acme Corporation"
    record = {
        "document_id": "validation-doc",
        "text": text,
        "entities": [
            {
                "start": 0,
                "end": len(text),
                "label": "invoice.client_address",
                "text": text,
            }
        ],
    }
    inspector = EntityErrorInspector(
        AddressTagger(schema), fast_tokenizer, schema, torch.device("cpu")
    )
    report, errors = inspector.run([record], tmp_path, epoch=7)

    assert report.loc["CLIENT_ADDRESS", "support"] == 1
    assert report.loc["CLIENT_ADDRESS", "recall"] == 0
    assert set(errors["document_id"]) == {"validation-doc"}
    assert set(errors["epoch"]) == {7}
    for row in errors.itertuples():
        assert row.text == text[row.start : row.end]
    assert errors.loc[errors.status.eq("false_negative"), "text"].tolist() == [text]
    for filename in ("per_entity", "entity_errors", "false_positive", "false_negative"):
        saved = pd.read_csv(tmp_path / f"{filename}.csv")
        assert saved["epoch"].eq(7).all()
    assert not list(tmp_path.glob("tokens_*.csv"))
