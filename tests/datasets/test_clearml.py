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
    monkeypatch.setattr("app.datasets.source.Dataset", api)
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


def test_remote_raw_preprocessing_and_separate_processed_and_image_inputs(
    raw_dataset, storage, tmp_path, monkeypatch
):
    from app.datasets.source import DatasetSource
    from app.extraction.layoutlm.artifacts import LayoutLMArtifacts
    from app.extraction.layoutlm.converter import AttachmentToImageConverter
    from app.pipelines.bert.pipeline import BertTrainingPipeline
    from app.pipelines.layoutlm.pipeline import LayoutLMTrainingPipeline
    from commands.cli import bert_params, layoutlm_params

    api, datasets = storage
    raw_id = RawDocileDatasetUpload(raw_dataset, "ocr", "raw").run()
    raw_root = Path(datasets[raw_id].get_local_copy())
    original = (raw_root / "raw.parquet").read_bytes()
    raw_reference = f"clearml://{raw_id}/raw.parquet"
    settings = Settings(_env_file=None)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    prepared = tmp_path / "prepared.parquet"
    cli.main(
        [
            "preprocess",
            "bert",
            "--source-path",
            raw_reference,
            "--output-path",
            str(prepared),
        ]
    )
    assert (raw_root / "raw.parquet").read_bytes() == original
    assert Path(pd.read_parquet(prepared).iloc[0].pdf_path).is_file()
    selected = datasets[raw_id].get_local_copy.call_args.kwargs["files"]
    assert "raw.parquet" in selected
    assert all(path == "raw.parquet" or path.startswith("pdfs/") for path in selected)

    processed_id = PreprocessedDocileDatasetUpload(
        prepared, raw_id, "ocr", "processed"
    ).run()
    image_cache = tmp_path / "images"
    cli.main(
        [
            "datasets",
            "cache-images",
            "docile",
            "--dataset-path",
            raw_reference,
            "--images-path",
            str(image_cache),
        ]
    )
    images_id = DocileImagesDatasetUpload(raw_id, image_cache, "ocr", "images").run()
    parser = cli.build_parser(settings)
    processed_reference = f"clearml://{processed_id}/preprocessed.parquet"
    images_reference = f"clearml://{images_id}/images"
    args = parser.parse_args(
        [
            "train",
            "layoutlm",
            "--dataset-path",
            processed_reference,
            "--images-path",
            images_reference,
            "--output-dir",
            str(tmp_path / "run"),
            "--checkpoint-path",
            str(tmp_path / "run" / "best.pt"),
        ]
    )
    params = layoutlm_params(args)
    loaded = LayoutLMTrainingPipeline(params).load_source()
    assert len(loaded) == 12
    assert all(Path(path).is_file() for path in loaded.pdf_path)
    artifacts = LayoutLMArtifacts.from_params(params)
    artifacts.prepare()
    assert (
        artifacts.source_images_path
        == Path(datasets[images_id].get_local_copy()) / "images"
    )
    assert artifacts.images_path == tmp_path / "run" / "images"
    monkeypatch.setattr(
        pymupdf, "open", Mock(side_effect=AssertionError("Use cached image"))
    )
    with AttachmentToImageConverter(artifacts).load_page_image(
        {
            "document_id": loaded.iloc[0].document_id,
            "page": 0,
            "pdf_path": "missing.pdf",
        }
    ) as image:
        assert image.size == (150, 150)
    assert not list(artifacts.images_path.rglob("*.png"))
    selected = datasets[images_id].get_local_copy.call_args_list[-2].kwargs["files"]
    assert all(path.startswith("images/dpi_150/") for path in selected)
    args = parser.parse_args(["train", "bert", "--dataset-path", processed_reference])
    assert len(BertTrainingPipeline(bert_params(args)).load_source()) == 12
    assert (
        DatasetSource(f"clearml://{processed_id}")
        .resolve("preprocessed.parquet")
        .path.name
        == "preprocessed.parquet"
    )


@pytest.mark.parametrize(
    "reference",
    [
        "clearml:/bad/raw.parquet",
        "clearml:///raw.parquet",
        "clearml://id/../raw.parquet",
        "clearml://id/raw.parquet?x=1",
        "clearml://user:secret@id/raw.parquet",
        "clearml://id/images",
        "clearml://id/%2e%2e/raw.parquet",
    ],
)
def test_invalid_remote_references_fail_before_server_access(storage, reference):
    from app.datasets.source import DatasetSource

    with pytest.raises(ValueError, match="clearml://"):
        DatasetSource(reference).resolve("raw.parquet")
    storage[0].get.assert_not_called()


def test_remote_sources_require_final_datasets_and_expected_members(
    raw_dataset, storage
):
    from app.datasets.source import DatasetSource

    _, datasets = storage
    dataset_id = RawDocileDatasetUpload(raw_dataset, "ocr", "raw").run()
    with pytest.raises(FileNotFoundError, match="preprocessed.parquet"):
        DatasetSource(f"clearml://{dataset_id}").resolve("preprocessed.parquet")
    with pytest.raises(FileNotFoundError, match="images"):
        DatasetSource(f"clearml://{dataset_id}/images").resolve("images")
    datasets[dataset_id].is_final.return_value = False
    with pytest.raises(ValueError, match="finalized"):
        DatasetSource(f"clearml://{dataset_id}").resolve("raw.parquet")


def test_local_source_does_not_access_clearml(storage, tmp_path):
    from app.datasets.source import DatasetSource

    local = tmp_path / "records.parquet"
    pd.DataFrame({"value": [1]}).to_parquet(local)
    source = DatasetSource(local).resolve("preprocessed.parquet")
    assert source.read_parquet().value.tolist() == [1]
    assert source.root is None
    storage[0].get.assert_not_called()


@pytest.mark.parametrize("failure", ["unsafe-reference", "missing-pdf"])
def test_remote_parquet_checks_pdf_references(raw_dataset, storage, failure):
    from app.datasets.source import DatasetSource

    _, datasets = storage
    dataset_id = RawDocileDatasetUpload(raw_dataset, "ocr", "raw").run()
    root = Path(datasets[dataset_id].get_local_copy())
    path = root / "raw.parquet"
    if failure == "unsafe-reference":
        frame = pd.read_parquet(path)
        frame.loc[0, "pdf_path"] = "../outside.pdf"
        frame.to_parquet(path, index=False)
    else:
        pdf = next((root / "pdfs").glob("*.pdf"))
        pdf.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        DatasetSource(f"clearml://{dataset_id}").resolve("raw.parquet").read_parquet()
