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


@pytest.mark.parametrize(
    "model,options,expected_text,uses_xy_cut",
    [
        ("bert", ["--no-line-items-only"], "Alpha Beta 10 20", True),
        ("bert", ["--line-items-only"], "Alpha 10 Beta 20", False),
        ("bert", ["--no-line-items-only", "--no-xy-cut"], "Alpha 10 Beta 20", False),
        ("layoutlm", [], "Alpha Beta 10 20", True),
    ],
)
def test_preprocess_selects_reading_order_before_creating_model_records(
    tmp_path, monkeypatch, capsys, model, options, expected_text, uses_xy_cut
):
    words = [
        {"value": text, "geometry": [[x, y], [x + 0.1, y + 0.02]]}
        for text, x, y in (
            ("Alpha", 0.1, 0.1),
            ("10", 0.7, 0.1),
            ("Beta", 0.1, 0.15),
            ("20", 0.7, 0.15),
        )
    ]
    annotations = {
        "metadata": {"page_count": 1},
        "field_extractions": [
            {
                "fieldtype": "vendor_name",
                "page": 0,
                "bbox": [0.09, 0.09, 0.21, 0.18],
                "text": "Alpha Beta",
            }
        ],
        "line_item_extractions": [
            {
                "fieldtype": "line_item_description",
                "line_item_id": index,
                "page": 0,
                "bbox": [0.09, y - 0.01, 0.81, y + 0.03],
                "text": text,
            }
            for index, y, text in [(1, 0.1, "Alpha 10"), (2, 0.15, "Beta 20")]
        ],
    }
    raw = tmp_path / "raw.parquet"
    pd.DataFrame(
        [
            {
                "document_id": "two-columns",
                "pdf_path": "unused.pdf",
                "ocr_data": json.dumps(
                    {
                        "pages": [
                            {"page_idx": 0, "blocks": [{"lines": [{"words": words}]}]}
                        ]
                    }
                ),
                "json_data": json.dumps(annotations),
            }
        ]
    ).to_parquet(raw, index=False)
    original = raw.read_bytes()
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None))
    output = tmp_path / "prepared.parquet"
    cli.main(
        [
            "preprocess",
            model,
            "--source-path",
            str(raw),
            "--output-path",
            str(output),
            *options,
        ]
    )
    messages = capsys.readouterr().out
    assert ("Running XYCut..." in messages) == uses_xy_cut
    row = pd.read_parquet(output).iloc[0]
    assert row.ocr_text == expected_text
    assert [word["text"] for word in row.text_with_boxes] == expected_text.split()
    for word in row.text_with_boxes:
        assert row.ocr_text[word["start"] : word["end"]] == word["text"]
    records = json.loads(row.prepared_records)["records"]
    if model == "bert":
        assert records[0]["text"] == expected_text
        for entity in records[0]["entities"]:
            assert expected_text[entity["start"] : entity["end"]] == entity["text"]
        if "--line-items-only" in options:
            assert [entity["text"] for entity in records[0]["entities"]] == [
                "Alpha 10",
                "Beta 20",
            ]
            assert [entity["line_item_id"] for entity in records[0]["entities"]] == [
                1,
                2,
            ]
        elif uses_xy_cut:
            assert [entity["text"] for entity in records[0]["entities"]] == [
                "Alpha Beta"
            ]
    else:
        assert records[0]["words"] == expected_text.split()
    assert raw.read_bytes() == original
