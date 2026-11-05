import json
from copy import deepcopy
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
    "result_filename,diagnostic_dir",
    [("results/experiment", "results/diagnostics"), ("results/experiment.json", "")],
)
def test_prepared_dataset_can_train_offline_with_csv_history(
    tmp_path,
    monkeypatch,
    capsys,
    fast_tokenizer,
    result_filename,
    diagnostic_dir,
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
    inspector_factory = Mock(side_effect=AssertionError("Diagnostics are disabled"))
    if not diagnostic_dir:
        monkeypatch.setattr(pipeline, "EntityErrorInspector", inspector_factory)
    split = Mock(wraps=pipeline.BertPytorchPipeline.split_records)
    monkeypatch.setattr(pipeline.BertPytorchPipeline, "split_records", split)
    cli.main(
        [
            "train",
            "bert-pytorch",
            "--dataset-path",
            str(dataset_path),
            "--result-filename",
            result_filename,
            "--num-epoch",
            "2",
            "--diagnostic-dir",
            diagnostic_dir,
            "--batch-size",
            "2",
            "--max-length-tokens",
            "32",
        ]
    )
    history_path = tmp_path / "results/experiment.csv"
    split.assert_called_once()
    assert all("entities" not in row for row in split.call_args.args[0])
    history = pd.read_csv(history_path)
    assert history["epoch"].tolist() == [1, 2]
    assert history["train_loss"].notna().all()
    assert history["validation_f1"].notna().all()
    assert "Unnamed: 0" not in history.columns
    assert not (tmp_path / "results/experiment.json").exists()
    download.assert_not_called()
    if diagnostic_dir:
        for epoch in (1, 2):
            folder = tmp_path / diagnostic_dir / f"epoch_{epoch:03d}"
            assert (
                "SELLER_NAME"
                in pd.read_csv(folder / "per_entity.csv")["entity"].tolist()
            )
            assert pd.read_csv(folder / "per_entity.csv")[
                "epoch"
            ].unique().tolist() == [epoch]
            for filename in (
                "entity_errors.csv",
                "false_positive.csv",
                "false_negative.csv",
            ):
                errors = pd.read_csv(folder / filename)
                assert "epoch" in errors.columns
                assert errors["epoch"].eq(epoch).all()
            assert "context" in pd.read_csv(folder / "false_positive.csv").columns
            assert (folder / "false_negative.csv").is_file()
    else:
        inspector_factory.assert_not_called()
        assert not (tmp_path / "data/interim/bert_diagnostics").exists()


@pytest.mark.parametrize("line_items_only", [False, True, None])
def test_docile_cli_download_and_training_use_shared_dataset(
    docile_archive,
    tmp_path,
    monkeypatch,
    capsys,
    fast_tokenizer,
    line_items_only,
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
    if line_items_only is not False:
        frame = pd.read_parquet(output)
        for index in frame.index:
            data = json.loads(frame.at[index, "json_data"])
            data["items"] = [
                {
                    "line_item_id": 7,
                    "fields": {"description": data["invoice"]["seller_name"]},
                }
            ]
            frame.at[index, "json_data"] = json.dumps(data)
        frame.to_parquet(output, index=False)
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
    monkeypatch.setattr(
        model_module.AutoModel, "from_pretrained", lambda **kw: deepcopy(encoder)
    )
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
            *(
                ["--line-items-only"]
                if line_items_only is True
                else ["--body-only"]
                if line_items_only is False
                else []
            ),
            "--max-length-tokens",
            "32",
        ]
    )
    if line_items_only is None:
        manifests = []
        held_out = []
        for target in ("body", "items"):
            result = pd.read_csv(history.with_name(f"{history.stem}_{target}.csv"))
            assert result["epoch"].tolist() == [1]
            checkpoint = tmp_path / "data/interim/bert_pytorch_checkpoint" / target
            snapshot = json.loads((checkpoint / "validation.json").read_text())
            assert snapshot["decoder"] == "ocr_word"
            assert all(
                path.startswith("items.") == (target == "items")
                for path in snapshot["field_to_entity"]
            )
            manifests.append(
                json.loads((checkpoint / "split_manifest.json").read_text())
            )
            held_out.append(
                {
                    r["document_id"]
                    for r in json.loads((checkpoint / "test.json").read_text())[
                        "records"
                    ]
                }
            )
            cli.main(
                [
                    "debug",
                    "bert-pytorch",
                    "--checkpoint-dir",
                    str(checkpoint),
                    "--output-dir",
                    str(tmp_path / f"debug-{target}"),
                    "--device",
                    "cpu",
                ]
            )
            report = pd.read_csv(tmp_path / f"debug-{target}/decoder_comparison.csv")
            micro = report.loc[
                report.entity.eq("micro avg") & report.decoder.eq("ocr_word"),
                "f1-score",
            ].iloc[0]
            assert micro == pytest.approx(result["validation_f1"].iloc[0])
        assert manifests[0] == manifests[1]
        assert held_out[0] == held_out[1]
        assert not history.exists()
        download.assert_not_called()
        return
    result = pd.read_csv(history)
    assert result["epoch"].tolist() == [1]
    assert result["train_loss"].notna().all()
    assert (
        tmp_path / "data/interim/bert_diagnostics/epoch_001/per_entity.csv"
    ).is_file()
    download.assert_not_called()
    checkpoint = tmp_path / "data/interim/bert_pytorch_checkpoint"
    assert (checkpoint / "model.pt").is_file()
    snapshot = json.loads((checkpoint / "validation.json").read_text())
    assert snapshot["epoch"] == 1
    assert snapshot["decoder"] == "ocr_word"
    assert snapshot["max_length"] == 32
    assert snapshot["records"][0]["text_with_boxes"]
    if line_items_only:
        assert set(snapshot["field_to_entity"]) == {
            "items.description",
            "items.quantity",
            "items.unit_price",
            "items.total_price",
        }
        assert all(
            e["line_item_id"] == 7 for r in snapshot["records"] for e in r["entities"]
        )
    assert (checkpoint / "test.json").is_file()
    saved_test_dir = tmp_path / "test-comparison"
    cli.main(
        [
            "debug",
            "bert-pytorch",
            "--checkpoint-dir",
            str(checkpoint),
            "--output-dir",
            str(saved_test_dir),
            "--device",
            "cpu",
            "--split",
            "test",
        ]
    )
    test_comparison = pd.read_csv(saved_test_dir / "decoder_comparison.csv")
    assert test_comparison["split"].eq("test").all()
    # An older checkpoint can reconstruct test membership from unchanged inputs.
    (checkpoint / "test.json").unlink()
    with pytest.raises(ValueError, match="No saved test split"):
        cli.main(
            [
                "debug",
                "bert-pytorch",
                "--checkpoint-dir",
                str(checkpoint),
                "--output-dir",
                str(tmp_path / "missing-test"),
                "--device",
                "cpu",
                "--split",
                "test",
            ]
        )
    rebuilt_dir = tmp_path / "rebuilt-test"
    cli.main(
        [
            "debug",
            "bert-pytorch",
            "--checkpoint-dir",
            str(checkpoint),
            "--output-dir",
            str(rebuilt_dir),
            "--device",
            "cpu",
            "--split",
            "test",
            "--dataset-path",
            str(output),
            "--random-seed",
            "42",
        ]
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(rebuilt_dir / "decoder_comparison.csv"), test_comparison
    )
    snapshot["records"][0]["text"] += " changed"
    (checkpoint / "validation.json").write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="validation split differs"):
        cli.main(
            [
                "debug",
                "bert-pytorch",
                "--checkpoint-dir",
                str(checkpoint),
                "--output-dir",
                str(tmp_path / "mismatched-test"),
                "--device",
                "cpu",
                "--split",
                "test",
                "--dataset-path",
                str(output),
            ]
        )
    snapshot["records"][0]["text"] = snapshot["records"][0]["text"].removesuffix(
        " changed"
    )
    (checkpoint / "validation.json").write_text(json.dumps(snapshot))
    # Validation debug must use saved inputs, with no access to the original dataset.
    output.unlink()
    debug_dir = tmp_path / "comparison"
    cli.main(
        [
            "debug",
            "bert-pytorch",
            "--checkpoint-dir",
            str(checkpoint),
            "--output-dir",
            str(debug_dir),
            "--device",
            "cpu",
        ]
    )
    comparison = pd.read_csv(debug_dir / "decoder_comparison.csv")
    assert set(comparison["decoder"]) == {"token_argmax", "ocr_word"}
    original = pd.read_csv(
        tmp_path / "data/interim/bert_diagnostics/epoch_001/per_entity.csv"
    )
    baseline = comparison.loc[comparison.decoder.eq("ocr_word")].reset_index(drop=True)
    pd.testing.assert_frame_equal(baseline[original.columns], original)
    for decoder in ("token_argmax", "ocr_word"):
        for filename in (
            "per_entity.csv",
            "entity_errors.csv",
            "false_positive.csv",
            "false_negative.csv",
        ):
            assert (debug_dir / decoder / filename).is_file()
