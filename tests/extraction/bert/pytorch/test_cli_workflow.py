import json
from unittest.mock import Mock
from zipfile import ZipFile

import pandas as pd
import pytest
from transformers import BertConfig, BertModel

from app.config import Settings
from app.datasets.docile.downloader import DocileDatasetDownloader
from app.datasets.kaggle_invoices.downloader import InvoiceDatasetDownloader
from app.extraction.bert.pytorch import model as model_module
from app.extraction.bert.pytorch import pipeline
from commands import cli


@pytest.mark.parametrize(
    "result_filename", ["results/experiment", "results/experiment.json"]
)
def test_prepared_dataset_can_train_offline_with_csv_history(
    tmp_path,
    monkeypatch,
    capsys,
    fast_tokenizer,
    result_filename,
):
    monkeypatch.chdir(tmp_path)
    settings = Settings(
        _env_file=None,
        DATA_PATH="unused",
        DEBUG_ARTIFACTS_PATH="unused",
        DOCILE_TOKEN="unused",
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    archive_path = tmp_path / "invoices.zip"
    frame = pd.DataFrame(
        [
            {
                "File Name": f"invoice-{index}.png",
                "OCRed Text": f"Invoice no: {1000 + index} Seller: Acme Corporation",
                "Json Data": json.dumps(
                    {
                        "invoice": {
                            "invoice_number": str(1000 + index),
                            "seller_name": "Acme Corporation",
                        }
                    }
                ),
            }
            for index in range(10)
        ]
    )
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("batch/invoices.csv", frame.to_csv(index=False))
        archive.writestr("batch/invoices/", "")
    dataset_path = tmp_path / "prepared/invoices.csv"
    cli.main(
        [
            "datasets",
            "download",
            "kaggle-invoices",
            "--archive-path",
            str(archive_path),
            "--destination-path",
            str(tmp_path / "extracted"),
            "--output-path",
            str(dataset_path),
        ]
    )
    assert capsys.readouterr().out.strip() == str(dataset_path)
    assert len(pd.read_csv(dataset_path)) == 10

    download = Mock(side_effect=AssertionError("Training must not download a dataset"))
    monkeypatch.setattr(InvoiceDatasetDownloader, "download_and_extract", download)
    monkeypatch.setattr(
        pipeline.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: fast_tokenizer,
    )
    encoder = BertModel(
        BertConfig(
            vocab_size=len(fast_tokenizer),
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
        )
    )
    monkeypatch.setattr(
        model_module.AutoModel, "from_pretrained", lambda **kwargs: encoder
    )
    monkeypatch.setattr(pipeline.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(pipeline.torch.backends.mps, "is_available", lambda: False)
    cli.main(
        [
            "train",
            "bert-pytorch",
            "--dataset-path",
            str(dataset_path),
            "--result-filename",
            result_filename,
            "--num-epoch",
            "1",
            "--batch-size",
            "2",
            "--max-length-tokens",
            "32",
        ]
    )
    history_path = tmp_path / "results/experiment.csv"
    history = pd.read_csv(history_path)
    assert history["epoch"].tolist() == [1]
    assert history["train_loss"].notna().all()
    assert history["validation_f1"].notna().all()
    assert "Unnamed: 0" not in history.columns
    assert not (tmp_path / "results/experiment.json").exists()
    download.assert_not_called()


def test_docile_cli_download_and_training_use_shared_dataset(
    docile_archive,
    tmp_path,
    monkeypatch,
    capsys,
    fast_tokenizer,
):
    monkeypatch.chdir(tmp_path)
    settings = Settings(
        _env_file=None, DATA_PATH="unused", DEBUG_ARTIFACTS_PATH="unused"
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    output = tmp_path / "docile.parquet"
    cli.main(
        [
            "datasets",
            "download",
            "docile",
            "--token",
            "dummy",
            "--destination-path",
            str(docile_archive.parent),
            "--output-path",
            str(output),
            "--minimum-archive-bytes",
            "0",
        ]
    )
    assert capsys.readouterr().out.strip() == str(output)
    download = Mock(side_effect=AssertionError("Training must not download archives"))
    monkeypatch.setattr(DocileDatasetDownloader, "download", download)
    monkeypatch.setattr(
        pipeline.AutoTokenizer, "from_pretrained", lambda *a, **kw: fast_tokenizer
    )
    encoder = BertModel(
        BertConfig(
            vocab_size=len(fast_tokenizer),
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
        )
    )
    monkeypatch.setattr(model_module.AutoModel, "from_pretrained", lambda **kw: encoder)
    monkeypatch.setattr(pipeline.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(pipeline.torch.backends.mps, "is_available", lambda: False)
    history = tmp_path / "docile-history.csv"
    cli.main(
        [
            "train",
            "bert-pytorch",
            "--dataset-name",
            "docile",
            "--dataset-path",
            str(output),
            "--num-epoch",
            "1",
            "--batch-size",
            "4",
            "--result-filename",
            str(history),
            "--max-length-tokens",
            "32",
        ]
    )
    result = pd.read_csv(history)
    assert result["epoch"].tolist() == [1]
    assert result["train_loss"].notna().all()
    download.assert_not_called()
