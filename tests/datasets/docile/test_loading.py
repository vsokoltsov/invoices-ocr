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
from app.datasets.docile.order import OCRReadingOrder
from app.datasets.docile.preprocess import DocilePreprocess
from app.datasets.split import DatasetsSplit
from app.extraction.bert.preprocess import InvoicePreprocessor


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
        OrderOCRWords,
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
    preparation = DocilePreprocess(docile_archive.parent, output)
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
    with pytest.raises(ValueError, match="both train and val"):
        preparation.run()


def test_invalid_output_fails_before_download(tmp_path):
    preparation = DocilePreprocess(tmp_path / "raw", tmp_path / "data.txt")
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
    output = DocilePreprocess(
        root,
        tmp_path / "test.csv",
        split=DocileSplit.TEST,
    ).run()
    frame = pd.read_csv(output)
    assert frame["document_id"].tolist() == ["test-doc"]
    assert frame["split"].tolist() == ["test"]
