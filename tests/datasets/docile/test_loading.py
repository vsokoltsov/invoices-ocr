import json
from dataclasses import asdict
from unittest.mock import Mock
from zipfile import ZipFile

import pandas as pd
import pytest

from app.config import DocileInvoicesConfig
from app.datasets.docile.downloader import DocileDatasetDownloader
from app.datasets.docile.enums import DocileSplit
from app.datasets.docile.loader import DocileDatasetLoader
from app.datasets.docile.preprocess import DocileSource
from app.datasets.split import DatasetsSplit
from app.extraction.bert.preprocess import InvoicePreprocessor
from app.extraction.order import OCRReadingOrder
from app.extraction.preprocess import InvoicePreprocess, OrderOCRWords


def test_loader_reads_original_data_and_preprocessing_preserves_provenance(
    docile_archive,
):
    with ZipFile(docile_archive) as archive:
        archive.extractall(docile_archive.parent)
    loader = DocileDatasetLoader(str(docile_archive.parent))
    record = loader.load("doc-0")
    assert (
        record.ocr_data["pages"][0]["blocks"][0]["lines"][0]["words"][0]["value"]
        == "Corporation"
    )
    assert record.json_data["field_extractions"][0]["fieldtype"] == "vendor_name"
    assert "invoice" not in record.json_data
    source = pd.DataFrame([asdict(record)])
    original = json.dumps(source.iloc[0].to_dict(), sort_keys=True)
    from app.datasets.docile.preprocess import (
        ConvertAnnotations,
        ExtractOCRWords,
    )
    from app.pipelines.preprocess import DataFramePipeline

    def prepare(snapped):
        return (
            DataFramePipeline(
                (ExtractOCRWords(snapped), OrderOCRWords(), ConvertAnnotations())
            )
            .run(source)
            .iloc[0]
        )

    prepared = prepare(True)
    assert prepared.ocr_text == "Acme Corporation"
    assert all(
        prepared.ocr_text[word["start"] : word["end"]] == word["text"]
        for word in prepared.text_with_boxes
    )
    assert prepared.text_with_boxes[1]["bbox"] == (0.31, 0.11, 0.59, 0.19)
    assert prepare(False).text_with_boxes[1]["bbox"] == (0.3, 0.1, 0.6, 0.2)
    assert prepared.json_data["invoice"]["seller_name"]["source_values"] == [
        "Acme Corporation"
    ]
    assert prepared.json_data["items"][0]["fields"]["description"]["value"] == "prefix"
    assert prepared.json_data["line_item_headers"]["description"]["value"] == "prefix"
    assert json.dumps(source.iloc[0].to_dict(), sort_keys=True) == original
    assert len(list(loader.iter_split("val"))) == 2
    assert len(loader.to_dataframe(show_progress=False)) == 12
    assert OCRReadingOrder().build([]) == ("", [])
    with pytest.raises(ValueError, match="file stem"):
        loader.load("../outside")
    (loader.root / "val.json").write_text('["doc-0", "doc-0"]')
    with pytest.raises(ValueError, match="duplicate"):
        loader.document_ids("val")
    (loader.root / "train.json").write_text('["missing"]')
    with pytest.raises(RuntimeError, match="missing"):
        loader.to_dataframe("train", show_progress=False)


@pytest.mark.parametrize("extension", ["csv", "parquet"])
def test_preparation_roundtrip_and_pipeline_selection(
    docile_archive, tmp_path, monkeypatch, extension
):
    output = tmp_path / f"prepared.{extension}"
    raw_path = tmp_path / "raw.parquet"
    loader = DocileDatasetLoader(docile_archive.parent)
    assert loader.save(raw_path) == raw_path.resolve()
    original_raw = raw_path.read_bytes()
    raw = pd.read_parquet(raw_path)
    first = raw.iloc[0]
    assert json.loads(first.ocr_data) == json.loads(
        (docile_archive.parent / f"ocr/{first.document_id}.json").read_text()
    )
    assert json.loads(first.json_data) == json.loads(
        (docile_archive.parent / f"annotations/{first.document_id}.json").read_text()
    )
    assert "ocr_text" not in raw and "text_with_boxes" not in raw
    preparation = InvoicePreprocess(DocileSource(raw_path), output)
    assert preparation.run() == output.resolve()
    assert docile_archive.is_file()
    monkeypatch.setattr(
        DocileDatasetDownloader,
        "download",
        Mock(side_effect=AssertionError("Already extracted")),
    )
    assert preparation.run() == output.resolve()
    if extension == "parquet":
        frame = pd.read_parquet(output)
        frame["json_data"] = frame["json_data"].map(json.loads)
        records, rejected = InvoicePreprocessor().run(frame)
        assert len(records) == 12 and not rejected
        partitions = DatasetsSplit.create(frame, 42)
        assert (len(partitions.train), len(partitions.val), len(partitions.test)) == (
            9,
            1,
            2,
        )
        assert not set(partitions.train.document_id) & set(partitions.val.document_id)
    assert (
        len(pd.read_parquet(output) if extension == "parquet" else pd.read_csv(output))
        == 12
    )
    (docile_archive.parent / "val.json").write_text(json.dumps(["doc-0"]))
    assert preparation.run() == output.resolve()
    assert raw_path.read_bytes() == original_raw
    with pytest.raises(ValueError, match="both train and val"):
        loader.save(tmp_path / "invalid.parquet")


def test_invalid_output_fails_before_download(tmp_path):
    preparation = InvoicePreprocess(
        DocileSource(tmp_path / "raw"), tmp_path / "data.txt"
    )
    with pytest.raises(ValueError, match="extension"):
        preparation.run()


@pytest.mark.parametrize("split", list(DocileSplit))
def test_config_accepts_all_split_values(split):
    config = DocileInvoicesConfig(_env_file=None, SPLIT=split)
    assert config.SPLIT is split


def test_test_split_is_loaded_and_keeps_its_assignment(docile_archive, tmp_path):
    with ZipFile(docile_archive) as archive:
        archive.extractall(docile_archive.parent)
    root = docile_archive.parent
    (root / "test.json").write_text('["test-doc"]')
    for directory in ("ocr", "annotations"):
        (root / directory / "test-doc.json").write_bytes(
            (root / directory / "doc-0.json").read_bytes()
        )
    raw_path = DocileDatasetLoader(root).save(
        tmp_path / "test-raw.parquet", split=DocileSplit.TEST
    )
    output = InvoicePreprocess(
        DocileSource(raw_path),
        tmp_path / "test.csv",
    ).run()
    frame = pd.read_csv(output)
    assert frame["document_id"].tolist() == ["test-doc"]
    assert frame["split"].tolist() == ["test"]


def test_preprocessing_reads_snapshot_without_source_json(docile_archive, tmp_path):
    raw = DocileDatasetLoader(docile_archive.parent).save(tmp_path / "raw.parquet")
    original = raw.read_bytes()
    # Hide the source files to prove that preparation uses only the snapshot.
    for name in ("ocr", "annotations"):
        directory = docile_archive.parent / name
        directory.rename(directory.with_name(f"{name}-hidden"))
    output = InvoicePreprocess(DocileSource(raw), tmp_path / "prepared.parquet").run()
    frame = pd.read_parquet(output)
    assert len(frame) == 12
    assert frame.iloc[0].ocr_text == "Acme Corporation"
    assert frame["split"].value_counts().to_dict() == {"train": 10, "val": 2}
    assert raw.read_bytes() == original
    with pytest.raises(ValueError, match="different paths"):
        InvoicePreprocess(DocileSource(raw), raw).run()
    assert raw.read_bytes() == original
    with pytest.raises(ValueError, match="missing columns"):
        DocileSource(output).load()


def test_raw_loader_requires_parquet_output(tmp_path):
    with pytest.raises(ValueError, match="parquet extension"):
        DocileDatasetLoader(tmp_path).save(tmp_path / "raw.csv")
    with pytest.raises(ValueError, match="parquet file"):
        DocileSource(tmp_path / "raw.csv").load()


def test_export_reports_progress_and_counts_to_stdout(docile_archive, tmp_path, capsys):
    raw = DocileDatasetLoader(docile_archive.parent).save(tmp_path / "raw.parquet")
    capsys.readouterr()
    output = InvoicePreprocess(DocileSource(raw), tmp_path / "prepared.parquet").run()
    messages = capsys.readouterr().out
    assert "Loading raw Parquet..." in messages
    assert "Loaded 12 documents" in messages
    assert "Running ExtractOCRWords..." in messages
    assert "Running OrderOCRWords..." in messages
    assert "Running XYCut..." in messages
    assert messages.index("OrderOCRWords: 12 documents") < messages.index(
        "Running XYCut..."
    )
    frame = pd.read_parquet(output)
    words = sum(len(value) for value in frame.text_with_boxes)
    assert (
        f"Exported 12 documents; {words:,} OCR words; 0 documents without OCR words (0.0%)"
        in messages
    )
    assert "Model rejection checks are not run here" in messages
    assert "Finished in" in messages


def test_export_reports_failed_step_without_success_message(tmp_path, capsys):
    from app.extraction.preprocess import PreprocessProgress

    step = PreprocessProgress("BrokenStep", Mock(side_effect=ValueError("bad input")))
    with pytest.raises(ValueError, match="bad input"):
        step(pd.DataFrame({"document_id": ["doc"]}))
    messages = capsys.readouterr().out
    assert "Running BrokenStep..." in messages
    assert "BrokenStep failed after" in messages
    assert "documents in" not in messages
