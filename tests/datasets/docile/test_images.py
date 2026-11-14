from unittest.mock import Mock

import pandas as pd
import pymupdf
import pytest
from PIL import Image

from app.config import DocileInvoicesConfig, LayoutLMConfig, Settings
from app.datasets.docile.images import DocileImageCache
from commands import cli


@pytest.fixture
def pdf_path(tmp_path):
    path = tmp_path / "invoice.pdf"
    with pymupdf.open() as document:
        for index in range(2):
            page = document.new_page(width=144, height=144)
            page.insert_text((20, 40), f"Invoice page {index + 1}")
        document.save(path)
    return path


@pytest.mark.parametrize("extension", ["parquet", "csv"])
def test_all_pages_are_cached_and_repeated_runs_do_not_render(
    pdf_path, tmp_path, extension, monkeypatch
):
    path = tmp_path / f"documents.{extension}"
    frame = pd.DataFrame({"document_id": ["invoice"], "pdf_path": [str(pdf_path)]})
    if extension == "csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)
    cache = DocileImageCache(path, tmp_path / "images")
    output = cache.run()
    assert output == (tmp_path / "images").resolve()
    for page_index in range(2):
        with Image.open(
            output / "dpi_150" / "invoice" / f"page_{page_index:04d}.png"
        ) as image:
            assert image.mode == "RGB"
            assert image.size == (300, 300)
    render = Mock(side_effect=AssertionError("Cached pages must not be rendered"))
    monkeypatch.setattr(pymupdf.Page, "get_pixmap", render)
    assert cache.run() == output
    render.assert_not_called()


def test_cli_uses_dataset_and_layoutlm_cache_defaults(
    pdf_path, tmp_path, monkeypatch, capsys
):
    dataset = tmp_path / "docile.parquet"
    pd.DataFrame({"document_id": ["doc"], "pdf_path": [str(pdf_path)]}).to_parquet(
        dataset
    )
    settings = Settings(
        _env_file=None,
        DATA_PATH="unused",
        DEBUG_ARTIFACTS_PATH="unused",
        DOCILE=DocileInvoicesConfig(_env_file=None, OUTPUT_PATH=str(dataset)),
        LAYOUTLM=LayoutLMConfig(
            _env_file=None, IMAGES_PATH=str(tmp_path / "default-cache")
        ),
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    cli.main(["datasets", "cache-images", "docile"])
    assert capsys.readouterr().out.strip() == str(
        (tmp_path / "default-cache").resolve()
    )
    assert (tmp_path / "default-cache" / "dpi_150" / "doc" / "page_0001.png").is_file()
    cli.main(
        [
            "datasets",
            "cache-images",
            "docile",
            "--dataset-path",
            str(dataset),
            "--images-path",
            str(tmp_path / "custom-cache"),
        ]
    )
    assert (tmp_path / "custom-cache" / "dpi_150" / "doc" / "page_0000.png").is_file()


@pytest.mark.parametrize(
    "failure", ["empty", "duplicate", "blank", "null", "unsupported"]
)
def test_invalid_dataset_fails_before_creating_cache(tmp_path, failure):
    dataset = tmp_path / "data.parquet"
    frame = pd.DataFrame({"document_id": ["doc"], "pdf_path": ["missing.pdf"]})
    if failure == "empty":
        frame = frame.iloc[:0]
    elif failure == "duplicate":
        frame = pd.concat([frame, frame])
    elif failure == "blank":
        frame.loc[0, "pdf_path"] = " "
    elif failure == "null":
        frame.loc[0, "pdf_path"] = None
    else:
        dataset = tmp_path / "data.txt"
    if failure != "unsupported":
        frame.to_parquet(dataset, index=False)
    with pytest.raises(ValueError):
        DocileImageCache(dataset, tmp_path / "cache").run()
    assert not (tmp_path / "cache").exists()


def test_missing_pdf_error_identifies_document_and_preserves_cause(tmp_path):
    dataset = tmp_path / "data.csv"
    pd.DataFrame({"document_id": ["bad-doc"], "pdf_path": ["missing.pdf"]}).to_csv(
        dataset, index=False
    )
    with pytest.raises(RuntimeError, match="bad-doc.*missing.pdf") as caught:
        DocileImageCache(dataset, tmp_path / "cache").run()
    assert caught.value.__cause__ is not None


def test_cache_images_help_exposes_all_options(capsys):
    settings = Settings(
        _env_file=None, DATA_PATH="unused", DEBUG_ARTIFACTS_PATH="unused"
    )
    with pytest.raises(SystemExit) as caught:
        cli.build_parser(settings).parse_args(
            ["datasets", "cache-images", "docile", "--help"]
        )
    assert caught.value.code == 0
    help_text = capsys.readouterr().out
    assert "--dataset-path" in help_text
    assert "--images-path" in help_text
