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
from app.datasets.docile.preparation import DocileInvoicesPreparation
from app.datasets.split import DatasetsSplit
from app.extraction.bert.builder import InvoiceRecordBuilder


def test_loader_preserves_boxes_items_offsets_and_provenance(docile_archive):
    with ZipFile(docile_archive) as archive:
        archive.extractall(docile_archive.parent)
    loader = DocileDatasetLoader(str(docile_archive.parent))
    record = loader.load("doc-0")
    assert record.ocr_text == "Acme Corporation"
    assert all(
        record.ocr_text[w.start : w.end] == w.text for w in record.text_with_boxes
    )
    assert record.text_with_boxes[1].bbox == (0.31, 0.11, 0.59, 0.19)
    assert DocileDatasetLoader(loader.root, False).load("doc-0").text_with_boxes[
        1
    ].bbox == (0.3, 0.1, 0.6, 0.2)
    assert record.json_data["invoice"]["seller_name"]["source_values"] == [
        "Acme Corporation"
    ]
    assert record.json_data["items"][0]["fields"]["description"]["value"] == "prefix"
    assert record.json_data["line_item_headers"]["description"]["value"] == "prefix"
    assert len(list(loader.iter_split("val"))) == 2
    assert len(loader.to_dataframe(show_progress=False)) == 12
    assert OCRReadingOrder().build([]) == ("", [])
    original = asdict(record.text_with_boxes[0])
    OCRReadingOrder().build(record.text_with_boxes[::-1])
    assert asdict(record.text_with_boxes[0]) == original
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
    downloader = DocileDatasetDownloader(
        "dummy", target_dir=docile_archive.parent, unzip=True, minimum_archive_bytes=0
    )
    output = tmp_path / f"prepared.{extension}"
    preparation = DocileInvoicesPreparation(downloader, output)
    assert preparation.run() == output.resolve()
    monkeypatch.setattr(
        DocileDatasetDownloader,
        "download",
        Mock(side_effect=AssertionError("Already extracted")),
    )
    assert preparation.run() == output.resolve()
    if extension == "parquet":
        frame = pd.read_parquet(output)
        frame["json_data"] = frame["json_data"].map(json.loads)
        records, rejected = InvoiceRecordBuilder().build(frame)
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
    (downloader.target_dir / "val.json").write_text(json.dumps(["doc-0"]))
    with pytest.raises(ValueError, match="both train and val"):
        preparation.run()


def test_invalid_output_fails_before_download(tmp_path):
    preparation = DocileInvoicesPreparation(
        DocileDatasetDownloader("dummy"), tmp_path / "data.txt"
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
    downloader = DocileDatasetDownloader("dummy", target_dir=root)
    output = DocileInvoicesPreparation(
        downloader,
        tmp_path / "test.csv",
        split=DocileSplit.TEST,
    ).run()
    frame = pd.read_csv(output)
    assert frame["document_id"].tolist() == ["test-doc"]
    assert frame["split"].tolist() == ["test"]
