import json
from copy import deepcopy
from unittest.mock import Mock

import pandas as pd
import pytest
import torch
from transformers import BertConfig, BertModel

from app.config import Settings
from app.datasets.docile.downloader import DocileDatasetDownloader
from app.extraction.bert import model as model_module
from app.pipelines.bert import pipeline
from commands import cli


@pytest.mark.parametrize("line_items_only", [False, True])
def test_docile_cli_download_and_training_use_shared_dataset(
    docile_archive,
    tmp_path,
    monkeypatch,
    capsys,
    fast_tokenizer,
    line_items_only,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline.ClearMLPublisher, "run", Mock(return_value="model-id"))
    settings = Settings(_env_file=None)
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
            "--minimum-archive-bytes",
            "0",
        ]
    )
    assert capsys.readouterr().out.strip() == str(docile_archive.parent)
    assert docile_archive.is_file()
    assert not output.exists()
    assert not (docile_archive.parent / "annotations").exists()
    raw_path = tmp_path / "raw-docile.parquet"
    cli.main(
        [
            "datasets",
            "load",
            "docile",
            "--source-path",
            str(docile_archive.parent),
            "--output-path",
            str(raw_path),
        ]
    )
    assert capsys.readouterr().out.strip() == str(raw_path)
    raw_bytes = raw_path.read_bytes()
    cli.main(
        [
            "preprocess",
            "bert",
            "--source-path",
            str(raw_path),
            "--output-path",
            str(output),
        ]
    )
    assert capsys.readouterr().out.strip().splitlines()[-1] == str(output)
    assert raw_path.read_bytes() == raw_bytes
    if line_items_only:
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
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    history = tmp_path / "docile-history.csv"
    cli.main(
        [
            "train",
            "bert",
            "--dataset-path",
            str(output),
            "--num-epoch",
            "1",
            "--batch-size",
            "4",
            "--result-filename",
            str(history),
            *(["--line-items-only"] if line_items_only else []),
            "--max-length-tokens",
            "32",
        ]
    )
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
    assert snapshot["decoder"] == "token_argmax"
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
            "bert",
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
                "bert",
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
            "bert",
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
                "bert",
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
            "bert",
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
    baseline = comparison.loc[comparison.decoder.eq("token_argmax")].reset_index(
        drop=True
    )
    pd.testing.assert_frame_equal(baseline[original.columns], original)
    for decoder in ("token_argmax", "ocr_word"):
        for filename in (
            "per_entity.csv",
            "entity_errors.csv",
            "false_positive.csv",
            "false_negative.csv",
        ):
            assert (debug_dir / decoder / filename).is_file()
