import json
from unittest.mock import Mock

import pandas as pd
import pytest

from app.config import Settings
from app.datasets.docile.loader import DocileDatasetLoader
from app.extraction.bert.preprocess import InvoicePreprocessor as BertPreprocessor
from app.extraction.layoutlm.preprocess import (
    InvoicePreprocessor as LayoutLMPreprocessor,
)
from app.extraction.prepared import ModelPreparation
from app.extraction.schema import InvoiceLabelSchema
from app.pipelines.bert.preprocess import BertPreprocess
from app.pipelines.layoutlm.preprocess import LayoutLMPreprocess
from commands import cli


@pytest.mark.parametrize("model", ["bert", "layoutlm"])
def test_cli_exports_model_records_and_rejections_then_training_reuses_them(
    model, docile_archive, tmp_path, monkeypatch, capsys
):
    raw = DocileDatasetLoader(docile_archive.parent).save(
        tmp_path / "raw.parquet", "trainval"
    )
    frame = pd.read_parquet(raw)
    annotation = json.loads(frame.at[0, "json_data"])
    annotation["field_extractions"][0]["bbox"] = [0.7, 0.7, 0.9, 0.9]
    frame.at[0, "json_data"] = json.dumps(annotation)
    frame.to_parquet(raw, index=False)
    original = raw.read_bytes()
    settings = Settings(_env_file=None)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    output = tmp_path / f"{model}.parquet"
    cli.main(
        ["preprocess", model, "--source-path", str(raw), "--output-path", str(output)]
    )
    messages = capsys.readouterr().out
    assert "12 documents; accepted 11; rejected 1 (8.3%)" in messages
    assert "Model rejection checks are not run here" not in messages
    assert raw.read_bytes() == original
    report = pd.read_csv(tmp_path / f"{model}_rejected.csv")
    assert report.document_id.tolist() == ["doc-0"]
    if model == "bert":
        assert json.loads(report.issues.iloc[0])
        preprocessor = BertPreprocessor()
        step = BertPreprocess(preprocessor, Mock())
    else:
        assert "annotation_without_assigned_words" in report.reason.iloc[0]
        preprocessor = LayoutLMPreprocessor(InvoiceLabelSchema().field_to_entity)
        step = LayoutLMPreprocess(preprocessor, Mock())
    exported = pd.read_parquet(output)
    assert len(exported) == 11
    assert "doc-0" not in set(exported.document_id)
    expected, rejected = preprocessor.run(exported)
    assert len(rejected) == 0
    # A different machine must supply its own resolved PDF paths.
    exported["pdf_path"] = "/new-cache/invoice.pdf"
    monkeypatch.setattr(
        preprocessor,
        "run",
        Mock(side_effect=AssertionError("must reuse saved records")),
    )
    records, rejected = step.prepare_partition(exported)
    assert rejected.empty
    if model == "bert":
        assert records.entities.tolist() == [record["entities"] for record in expected]
        assert records.text_with_boxes.map(len).eq(2).all()
    else:
        assert records.labels.tolist() == [record["labels"] for record in expected]
        assert records.pdf_path.eq("/new-cache/invoice.pdf").all()
        assert (
            "pdf_path"
            not in json.loads(exported.prepared_records.iloc[0])["records"][0]
        )
    # A prepared file cannot be reused with a different model or label schema.
    other = LayoutLMPreprocessor(InvoiceLabelSchema().field_to_entity, min_coverage=0.8)
    with pytest.raises(ValueError, match="settings differ"):
        ModelPreparation("layoutlm", other).load(exported)


def test_all_rejected_writes_report_and_stops_before_export(tmp_path, capsys):
    frame = pd.DataFrame(
        {
            "document_id": ["bad"],
            "json_data": [{"metadata": {"page_count": 1}}],
            "text_with_boxes": [[]],
        }
    )
    path = tmp_path / "rejected.csv"
    step = ModelPreparation(
        "layoutlm", LayoutLMPreprocessor(InvoiceLabelSchema().field_to_entity), path
    )
    with pytest.raises(ValueError, match="No accepted documents"):
        step(frame)
    assert pd.read_csv(path).document_id.tolist() == ["bad"]
    assert "rejected 1 (100.0%)" in capsys.readouterr().out
