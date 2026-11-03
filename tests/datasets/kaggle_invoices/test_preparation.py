import json
from pathlib import Path
from unittest.mock import Mock
from zipfile import ZipFile

import pandas as pd

from app.datasets.kaggle_invoices.downloader import InvoiceDatasetDownloader
from app.datasets.kaggle_invoices.preparation import KaggleInvoicesPreparation


def test_preparation_exports_reusable_csv_with_absolute_image_paths(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    target = {"invoice": {"seller_name": "Acme"}}
    source = pd.DataFrame(
        [
            {
                "File Name": "image.png",
                "OCRed Text": "Acme\nCorporation",
                "Json Data": json.dumps(target),
            }
        ]
    )
    archive_path = tmp_path / "archive.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("batch/invoices.csv", source.to_csv(index=False))
        archive.writestr("batch/invoices/", "")
    downloader = InvoiceDatasetDownloader("extracted", archive_path)
    preparation = KaggleInvoicesPreparation(downloader, Path("prepared/invoices.json"))
    output_path = preparation.run()
    assert output_path == tmp_path / "prepared/invoices.csv"
    frame = pd.read_csv(output_path, keep_default_na=False)
    assert frame["ocred_text"].tolist() == ["Acme\nCorporation"]
    assert json.loads(frame.iloc[0]["json_data"]) == target
    assert frame.iloc[0]["path"] == str(tmp_path / "extracted/batch/invoices/image.png")
    assert "Unnamed: 0" not in frame.columns
    assert preparation.run() == output_path
    assert len(pd.read_csv(output_path)) == 1


def test_download_cli_dispatches_preparation_and_prints_absolute_path(
    tmp_path, monkeypatch, capsys
):
    from app.config import Settings
    from commands import cli

    output_path = tmp_path / "prepared.csv"
    preparation = Mock()
    preparation.run.return_value = output_path
    factory = Mock(return_value=preparation)
    monkeypatch.setattr(cli, "KaggleInvoicesPreparation", factory)
    settings = Settings(
        _env_file=None,
        DATA_PATH="input",
        DEBUG_ARTIFACTS_PATH="debug",
        DOCILE_TOKEN="unused",
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    cli.main(
        [
            "datasets",
            "download",
            "kaggle-invoices",
            "--destination-path",
            str(tmp_path / "extracted"),
            "--archive-path",
            str(tmp_path / "archive.zip"),
            "--output-path",
            str(output_path),
        ]
    )
    assert capsys.readouterr().out.strip() == str(output_path)
    kwargs = factory.call_args.kwargs
    assert kwargs["output_path"] == output_path
    assert kwargs["downloader"].destination == tmp_path / "extracted"
    assert kwargs["downloader"].archive_path == tmp_path / "archive.zip"
