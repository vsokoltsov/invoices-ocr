import json
from unittest.mock import Mock

import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from app.extraction.layoutlm.artifacts import LayoutLMArtifacts
from app.extraction.layoutlm.evaluation import LayoutLMEvaluation
from app.extraction.layoutlm.inspector import EntityErrorInspector
from app.extraction.metrics import WindowMetrics
from app.extraction.schema import InvoiceLabelSchema


class DiagnosticDataset(Dataset):
    def __init__(self):
        self.pages = [
            {
                "document_id": "doc-a",
                "page": 0,
                "words": ["Acme", "Ltd", "Other"],
                "boxes": [
                    [100, 100, 200, 200],
                    [300, 100, 500, 200],
                    [100, 400, 200, 500],
                ],
            },
            {
                "document_id": "doc-a",
                "page": 1,
                "words": ["123"],
                "boxes": [[100, 100, 300, 200]],
            },
        ]
        self.samples = []
        for page, tokens, labels, word_ids in (
            (0, [0, 1, 2, 0, 0], [-100, 1, -100, 2, -100], [-1, 0, -1, 1, -1]),
            (0, [0, 3, 0, 0, 0], [-100, 0, -100, -100, -100], [-1, 2, -1, -1, -1]),
            (1, [0, 4, 0, 0, 0], [-100, 3, -100, -100, -100], [-1, 0, -1, -1, -1]),
            (0, [0] * 5, [-100] * 5, [-1] * 5),
        ):
            self.samples.append(
                {
                    "input_ids": torch.tensor(tokens),
                    "attention_mask": torch.ones(5, dtype=torch.long),
                    "bbox": torch.zeros(5, 4, dtype=torch.long),
                    "pixel_values": torch.zeros(3, 4, 4),
                    "labels": torch.tensor(labels),
                    "word_ids": torch.tensor(word_ids),
                    "page_index": torch.tensor(page),
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class DiagnosticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(5, 5)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.eye(5) * 5)

    def forward(self, input_ids, attention_mask, bbox, pixel_values):
        assert bbox.dtype == torch.long
        assert pixel_values.dtype == torch.float32
        return self.embedding(input_ids)


def test_inspector_matches_validation_and_preserves_window_word_coordinates(tmp_path):
    schema = InvoiceLabelSchema(
        {
            "invoice.seller_name": "SELLER_NAME",
            "invoice.invoice_number": "INVOICE_NUMBER",
        }
    )
    model = DiagnosticModel()
    loader = DataLoader(DiagnosticDataset(), batch_size=2)
    evaluator = LayoutLMEvaluation(
        model=model,
        metrics=WindowMetrics(schema.id2label),
        schema=schema,
        device=torch.device("cpu"),
        criterion=nn.CrossEntropyLoss(ignore_index=-100),
    )
    expected = evaluator.evaluate(loader, 1)
    model.train()
    random_state = torch.random.get_rng_state()
    image = Image.new("RGB", (200, 300), "white")
    converter = Mock()
    converter.load_page_image.return_value = image
    inspector = EntityErrorInspector(
        model, schema, torch.device("cpu"), converter, max_visualizations=1
    )
    artifacts = LayoutLMArtifacts(tmp_path, diagnostics_dir=tmp_path / "diagnostics")
    epoch = artifacts.diagnostics(1)
    report, errors = inspector.run(loader, epoch, epoch=1)
    assert model.training
    assert torch.equal(random_state, torch.random.get_rng_state())
    summary = json.loads((epoch.output_dir / "summary.json").read_text())
    for key, value in expected.scalars.items():
        assert summary[key] == pytest.approx(value)
    assert summary["words"] == 4
    assert summary["windows"] == 3
    assert summary["invalid_bio_transitions"] == 1
    assert summary["entity_errors"] == {"boundary": 2, "extra": 1, "missed": 1}
    assert report.loc["SELLER_NAME", "support"] == 1
    assert report.loc["INVOICE_NUMBER", "support"] == 1
    words = pd.read_csv(epoch.output_dir / "word_predictions.csv")
    assert words.text.tolist() == ["Acme", "Ltd", "Other", "123"]
    assert words.word_index.tolist() == [0, 1, 2, 0]
    assert words.window_index.tolist() == [0, 0, 1, 2]
    assert words.page.tolist() == [0, 0, 0, 1]
    assert words.confidence.between(0, 1).all()
    assert len(errors) == 4
    assert errors.loc[errors.status.eq("false_negative"), "text"].tolist() == [
        "Acme Ltd",
        "123",
    ]
    assert pd.read_csv(epoch.output_dir / "false_positive.csv").shape[0] == 2
    assert pd.read_csv(epoch.output_dir / "false_negative.csv").shape[0] == 2
    confusion = pd.read_csv(epoch.output_dir / "confusion.csv", index_col=0)
    assert confusion.to_numpy().sum() == 4
    assert confusion.loc["B-INVOICE_NUMBER", "I-INVOICE_NUMBER"] == 1
    fields = pd.read_csv(epoch.output_dir / "per_field.csv", index_col=0)
    assert fields.loc["SELLER_NAME", "support"] == 2
    overlays = pd.read_csv(epoch.output_dir / "overlays" / "index.csv")
    assert overlays.document_id.tolist() == ["doc-a"]
    assert overlays.page.tolist() == [0]
    with Image.open(epoch.output_dir / "overlays" / overlays.image.iloc[0]) as overlay:
        assert overlay.size == (400, 332)
        assert overlay.getpixel((60, 62)) == (0, 128, 0)
        assert overlay.getpixel((260, 62)) == (255, 0, 0)
    assert image.getpixel((60, 30)) == (255, 255, 255)

    # A later perfect epoch must retain the earlier reports and write empty error CSVs.
    before = (epoch.output_dir / "entity_errors.csv").read_bytes()
    with torch.no_grad():
        for token, label in [(0, 2), (3, 0), (4, 3)]:
            model.embedding.weight[token].zero_()
            model.embedding.weight[token, label] = 5
    later = artifacts.diagnostics(2)
    _, errors = inspector.run(loader, later, epoch=2)
    assert errors.empty
    assert pd.read_csv(later.output_dir / "entity_errors.csv").empty
    assert not (later.output_dir / "overlays").exists()
    assert json.loads((later.output_dir / "summary.json").read_text())["entity_f1"] == 1
    assert (epoch.output_dir / "entity_errors.csv").read_bytes() == before


def test_inspection_failure_restores_model_mode_and_does_not_save_reports(tmp_path):
    schema = InvoiceLabelSchema(
        {
            "invoice.seller_name": "SELLER_NAME",
            "invoice.invoice_number": "INVOICE_NUMBER",
        }
    )
    model = DiagnosticModel()
    with torch.no_grad():
        model.embedding.weight.fill_(float("nan"))
    inspector = EntityErrorInspector(model, schema, torch.device("cpu"))
    with pytest.raises(FloatingPointError, match="Inspection loss is not finite"):
        inspector.run(
            DataLoader(DiagnosticDataset(), batch_size=2),
            LayoutLMArtifacts(tmp_path),
            epoch=1,
        )
    assert model.training
    assert not (tmp_path / "summary.json").exists()
