import json
import shutil
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pymupdf
import pytest

from app.config import Settings
from app.datasets.clearml import (
    ClearMLDatasetDownload,
    DocileImagesDatasetUpload,
    PreprocessedDocileDatasetUpload,
    RawDocileDatasetUpload,
)
from app.datasets.docile.images import DocileImageCache
from app.datasets.docile.loader import DocileDatasetLoader
from app.datasets.docile.preprocess import DocileSource
from app.extraction.preprocess import InvoicePreprocess
from commands import cli


@pytest.fixture
def storage(tmp_path, monkeypatch):
    """Keep SDK calls local while exercising the uploaded file layout."""
    datasets = {}
    api = Mock()

    def create(**options):
        dataset = Mock()
        dataset.id = str(len(datasets) + 1)
        root = tmp_path / "server" / dataset.id
        root.mkdir(parents=True)
        dataset.get_local_copy.return_value = str(root)
        dataset.list_files.side_effect = lambda: [
            p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
        ]
        dataset.is_final.return_value = False
        for parent in options.get("parent_datasets", []):
            shutil.copytree(datasets[parent].get_local_copy(), root, dirs_exist_ok=True)

        def add_files(path, wildcard="*", dataset_path=""):
            source = Path(path)
            for file in source.rglob(wildcard):
                if file.is_file():
                    target = root / dataset_path / file.relative_to(source)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(file, target)

        def finalize():
            dataset.is_final.return_value = True

        def download(target_folder, overwrite):
            assert not overwrite
            shutil.copytree(root, target_folder)
            return target_folder

        dataset.add_files.side_effect = add_files
        dataset.finalize.side_effect = finalize
        dataset.get_mutable_local_copy.side_effect = download
        datasets[dataset.id] = dataset
        return dataset

    api.create.side_effect = create
    api.get.side_effect = lambda dataset_id: datasets[dataset_id]
    monkeypatch.setattr("app.datasets.clearml.Dataset", api)
    return api, datasets


@pytest.fixture
def raw_dataset(docile_archive, tmp_path):
    raw = DocileDatasetLoader(docile_archive.parent).save(tmp_path / "source.parquet")
    for path in pd.read_parquet(raw).pdf_path:
        pdf = Path(path)
        pdf.parent.mkdir(parents=True, exist_ok=True)
        with pymupdf.open() as document:
            document.new_page(width=72, height=72)
            document.new_page(width=72, height=72)
            document.save(pdf)
    return raw


def test_cli_roundtrip_raw_parent_and_image_child(
    raw_dataset, storage, tmp_path, monkeypatch, capsys
):
    api, datasets = storage
    settings = Settings(_env_file=None)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    source_bytes = raw_dataset.read_bytes()
    cli.main(
        [
            "datasets",
            "upload",
            "docile",
            "--dataset-path",
            str(raw_dataset),
            "--project",
            "invoices",
            "--name",
            "raw",
            "--output-uri",
            "s3://bucket/data",
        ]
    )
    assert capsys.readouterr().out.strip() == "1"
    parent = datasets["1"]
    parent.upload.assert_called_once_with(output_url="s3://bucket/data")
    parent.finalize.assert_called_once_with()
    remote = Path(parent.get_local_copy())
    original = pd.read_parquet(raw_dataset)
    portable = pd.read_parquet(remote / "raw.parquet")
    assert portable.pdf_path.tolist() == [
        f"pdfs/{key}.pdf" for key in original.document_id
    ]
    pd.testing.assert_frame_equal(
        original.drop(columns="pdf_path"), portable.drop(columns="pdf_path")
    )
    assert json.loads(portable.iloc[0].ocr_data)["pages"]
    assert raw_dataset.read_bytes() == source_bytes

    local = tmp_path / "machine-two"
    cli.main(
        [
            "datasets",
            "download",
            "clearml",
            "--dataset-id",
            "1",
            "--destination-path",
            str(local),
        ]
    )
    assert capsys.readouterr().out.strip() == str(local)
    # The relative references must work regardless of the working directory.
    monkeypatch.chdir(tmp_path)
    prepared = InvoicePreprocess(
        DocileSource(local / "raw.parquet"), tmp_path / "prepared.parquet"
    ).run()
    assert Path(pd.read_parquet(prepared).iloc[0].pdf_path).is_file()
    images = tmp_path / "cache"
    cli.main(
        [
            "datasets",
            "cache-images",
            "docile",
            "--dataset-path",
            str(local / "raw.parquet"),
            "--images-path",
            str(images),
        ]
    )
    capsys.readouterr()
    cli.main(
        [
            "datasets",
            "upload",
            "docile-images",
            "--parent-id",
            "1",
            "--images-path",
            str(images),
            "--project",
            "invoices",
            "--name",
            "images",
        ]
    )
    assert capsys.readouterr().out.strip() == "2"
    assert api.create.call_args.kwargs == {
        "dataset_project": "invoices",
        "dataset_name": "images",
        "parent_datasets": ["1"],
    }
    child = datasets["2"]
    assert child.add_files.call_count == 1
    assert child.add_files.call_args.kwargs["dataset_path"] == "images/dpi_150"
    combined = tmp_path / "combined"
    cli.main(
        [
            "datasets",
            "download",
            "clearml",
            "--dataset-id",
            "2",
            "--destination-path",
            str(combined),
        ]
    )
    downloaded = pd.read_parquet(combined / "raw.parquet")
    assert all(
        Path(value).is_absolute() and Path(value).is_file()
        for value in downloaded.pdf_path
    )
    pd.testing.assert_frame_equal(
        downloaded.drop(columns="pdf_path"), portable.drop(columns="pdf_path")
    )
    assert (
        pd.read_parquet(remote / "raw.parquet").pdf_path.tolist()
        == portable.pdf_path.tolist()
    )
    assert len(list((combined / "pdfs").glob("*.pdf"))) == 12
    assert len(list((combined / "images").rglob("*.png"))) == 24
    assert not (remote / "images").exists()
    assert Path(
        DocileSource(combined / "raw.parquet").load().iloc[0].pdf_path
    ).is_file()


@pytest.mark.parametrize("failure", ["missing", "invalid-id"])
def test_invalid_raw_files_fail_before_remote_creation(raw_dataset, storage, failure):
    frame = pd.read_parquet(raw_dataset)
    if failure == "missing":
        Path(frame.iloc[0].pdf_path).unlink()
    else:
        frame.loc[0, "document_id"] = "../outside"
        frame.to_parquet(raw_dataset, index=False)
    with pytest.raises((ValueError, FileNotFoundError)):
        RawDocileDatasetUpload(raw_dataset, "ocr", "raw").run()
    storage[0].create.assert_not_called()


def test_incomplete_cache_and_unfinalized_parent_are_rejected(
    raw_dataset, storage, tmp_path
):
    api, datasets = storage
    parent_id = RawDocileDatasetUpload(raw_dataset, "ocr", "raw").run()
    uploader = DocileImagesDatasetUpload(parent_id, tmp_path / "empty", "ocr", "images")
    with pytest.raises(ValueError, match="24 missing"):
        uploader.run()
    assert api.create.call_count == 1
    datasets[parent_id].is_final.return_value = False
    with pytest.raises(ValueError, match="finalized"):
        uploader.run()
    with pytest.raises(ValueError, match="finalized"):
        ClearMLDatasetDownload(parent_id, tmp_path / "download").run()


def test_download_keeps_existing_files(storage, tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(ValueError, match="new or empty"):
        ClearMLDatasetDownload("id", destination).run()
    assert marker.read_text() == "keep"
    storage[0].get.assert_not_called()


def test_preprocessed_upload_uses_existing_parquet_without_preprocessing(
    raw_dataset, storage, tmp_path, monkeypatch, capsys
):
    api, datasets = storage
    raw_id = RawDocileDatasetUpload(raw_dataset, "ocr", "raw").run()
    prepared = InvoicePreprocess(
        DocileSource(raw_dataset), tmp_path / "prepared.parquet"
    ).run()
    original = prepared.read_bytes()
    frame = pd.read_parquet(prepared)
    monkeypatch.setattr(
        InvoicePreprocess,
        "run",
        Mock(side_effect=AssertionError("Storage must not run preprocessing")),
    )
    settings = Settings(_env_file=None)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    cli.main(
        [
            "datasets",
            "upload",
            "docile-preprocessed",
            "--dataset-path",
            str(prepared),
            "--parent-id",
            raw_id,
            "--project",
            "ocr",
            "--name",
            "prepared",
        ]
    )
    processed_id = capsys.readouterr().out.strip()
    assert processed_id == "2"
    assert api.create.call_args.kwargs["parent_datasets"] == [raw_id]
    remote = Path(datasets[processed_id].get_local_copy())
    uploaded = pd.read_parquet(remote / "preprocessed.parquet")
    pd.testing.assert_frame_equal(
        uploaded.drop(columns="pdf_path"), frame.drop(columns="pdf_path")
    )
    assert uploaded.pdf_path.tolist() == [
        f"pdfs/{value}.pdf" for value in frame.document_id
    ]
    assert prepared.read_bytes() == original
    assert (remote / "raw.parquet").is_file()
    cache = DocileImageCache(raw_dataset, tmp_path / "images").run()
    images_id = DocileImagesDatasetUpload(processed_id, cache, "ocr", "images").run()
    assert api.create.call_args.kwargs["parent_datasets"] == [processed_id]
    destination = ClearMLDatasetDownload(images_id, tmp_path / "retrieved").run()
    assert len(list((destination / "images").rglob("*.png"))) == 24

    restored = pd.read_parquet(destination / "preprocessed.parquet")
    assert all(
        Path(value).is_file() and Path(value).is_absolute()
        for value in restored.pdf_path
    )
    assert (
        uploaded.pdf_path.tolist()
        == pd.read_parquet(remote / "preprocessed.parquet").pdf_path.tolist()
    )


def test_processed_upload_rejects_documents_missing_from_parent(
    raw_dataset, storage, tmp_path
):
    api, _ = storage
    parent_id = RawDocileDatasetUpload(raw_dataset, "ocr", "raw").run()
    prepared = InvoicePreprocess(
        DocileSource(raw_dataset), tmp_path / "prepared.parquet"
    ).run()
    frame = pd.read_parquet(prepared)
    frame.loc[0, "document_id"] = "unrelated"
    frame.to_parquet(prepared, index=False)
    with pytest.raises(ValueError, match="missing document PDFs"):
        PreprocessedDocileDatasetUpload(prepared, parent_id, "ocr", "processed").run()
    assert api.create.call_count == 1
