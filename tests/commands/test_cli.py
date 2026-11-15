from unittest.mock import Mock

import pandas as pd
import pytest

from app.config import (
    BertPytorchConfig,
    BertPytorchDebugConfig,
    DocileInvoicesConfig,
    Settings,
)
from app.datasets.docile.enums import DocileSplit
from commands import cli


@pytest.fixture
def settings():
    return Settings(
        _env_file=None,
        DOCILE_TOKEN="unused",
        RANDOM_SEED=123,
        BERT_PYTORCH=BertPytorchConfig(),
    )


def test_pytorch_cli_maps_all_config_fields_to_pipeline(settings, monkeypatch):
    pipeline = Mock()
    factory = Mock(return_value=pipeline)
    monkeypatch.setattr(cli, "BertTrainingPipeline", factory)
    args = cli.build_parser(settings).parse_args(
        [
            "train",
            "bert-pytorch",
            "--dataset-path",
            "prepared.parquet",
            "--result-filename",
            "experiment",
            "--model-name",
            "offline-model",
            "--batch-size",
            "2",
            "--num-epoch",
            "3",
            "--max-length-tokens",
            "64",
            "--learning-rate",
            "0.001",
            "--weight-decay",
            "0.02",
            "--random-seed",
            "7",
        ]
    )
    args.handler(args, settings)
    params = factory.call_args.kwargs["params"]
    assert params.random_seed == 7
    for name in BertPytorchConfig.model_fields:
        assert getattr(params, name.lower()) == getattr(args, name)


def test_cli_uses_pytorch_configuration_defaults(settings):
    args = cli.build_parser(settings).parse_args(["train", "bert-pytorch"])
    for name in BertPytorchConfig.model_fields:
        assert getattr(args, name) == getattr(settings.BERT_PYTORCH, name)


def test_line_items_flag_uses_environment_and_cli_override(settings, monkeypatch):
    monkeypatch.setenv("BERTPYTORCH_LINE_ITEMS_ONLY", "true")
    settings.BERT_PYTORCH = BertPytorchConfig(_env_file=None)
    parser = cli.build_parser(settings)
    assert parser.parse_args(["train", "bert-pytorch"]).LINE_ITEMS_ONLY is True
    assert (
        parser.parse_args(
            ["train", "bert-pytorch", "--no-line-items-only"]
        ).LINE_ITEMS_ONLY
        is False
    )
    pipeline = Mock()
    factory = Mock(return_value=pipeline)
    monkeypatch.setattr(cli, "BertTrainingPipeline", factory)
    args = parser.parse_args(["train", "bert-pytorch", "--line-items-only"])
    args.handler(args, settings)
    assert factory.call_args.kwargs["params"].line_items_only is True


def test_debug_cli_maps_config_and_environment(settings, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("BERT_PYTORCH_DEBUG_OUTPUT_DIR", str(tmp_path / "debug"))
    settings.BERT_PYTORCH_DEBUG = BertPytorchDebugConfig(_env_file=None)
    pipeline = Mock()
    pipeline.run.return_value = tmp_path / "comparison.csv"
    monkeypatch.setattr(cli, "BertPytorchDebugPipeline", lambda: pipeline)
    args = cli.build_parser(settings).parse_args(
        ["debug", "bert-pytorch", "--checkpoint-dir", "saved-model", "--device", "cpu"]
    )
    args.handler(args, settings)
    params = pipeline.run.call_args.args[0]
    assert params.checkpoint_dir == "saved-model"
    assert params.output_dir == str(tmp_path / "debug")
    assert params.device == "cpu"
    assert capsys.readouterr().out.strip() == str(pipeline.run.return_value)


def test_pytorch_environment_defaults_are_exposed(monkeypatch, settings):
    monkeypatch.setenv("BERTPYTORCH_BATCH_SIZE", "3")
    settings.BERT_PYTORCH = BertPytorchConfig()
    args = cli.build_parser(settings).parse_args(["train", "bert-pytorch"])
    assert args.BATCH_SIZE == 3


def test_pytorch_config_reads_dotenv_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BERTPYTORCH_BATCH_SIZE", raising=False)
    (tmp_path / ".env").write_text("BERTPYTORCH_BATCH_SIZE=7\n", encoding="utf-8")
    assert BertPytorchConfig().BATCH_SIZE == 7


def test_main_dispatches_and_invalid_commands_fail(settings, monkeypatch):
    pipeline = Mock()
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    factory = Mock(return_value=pipeline)
    monkeypatch.setattr(cli, "BertTrainingPipeline", factory)
    cli.main(["train", "bert-pytorch"])
    pipeline.run.assert_called_once()
    with pytest.raises(SystemExit):
        cli.build_parser(settings).parse_args(["train", "bert"])


def test_training_rejects_removed_download_options(settings):
    with pytest.raises(SystemExit):
        cli.build_parser(settings).parse_args(
            [
                "train",
                "bert-pytorch",
                "--destination-path",
                "downloaded",
            ]
        )


def test_dataset_command_uses_docile_config_defaults(settings):
    args = cli.build_parser(settings).parse_args(["datasets", "download", "docile"])
    assert args.DESTINATION_PATH == settings.DOCILE.DESTINATION_PATH
    assert args.OUTPUT_PATH == settings.DOCILE.OUTPUT_PATH
    assert args.SPLIT is settings.DOCILE.SPLIT


def test_docile_split_cli_uses_enum_and_rejects_invalid_values(settings):
    parser = cli.build_parser(settings)
    args = parser.parse_args(["datasets", "download", "docile", "--split", "test"])
    assert args.SPLIT is DocileSplit.TEST
    with pytest.raises(SystemExit):
        parser.parse_args(["datasets", "download", "docile", "--split", "unknown"])


def test_docile_corrections_path_uses_environment_and_cli_override(
    settings, monkeypatch
):
    monkeypatch.setenv("DOCILE_CORRECTIONS_PATH", "reviewed.json")
    settings.DOCILE = DocileInvoicesConfig(_env_file=None)
    parser = cli.build_parser(settings)
    assert (
        parser.parse_args(["datasets", "download", "docile"]).CORRECTIONS_PATH
        == "reviewed.json"
    )
    assert (
        parser.parse_args(
            ["datasets", "download", "docile", "--corrections-path", ""]
        ).CORRECTIONS_PATH
        == ""
    )


def test_missing_parquet_fails_before_tokenizer_initialization(
    settings, monkeypatch, tmp_path
):
    from app.pipelines.bert import pipeline

    tokenizer_factory = Mock()
    monkeypatch.setattr(pipeline.AutoTokenizer, "from_pretrained", tokenizer_factory)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    missing = tmp_path / "missing.parquet"
    with pytest.raises(FileNotFoundError):
        cli.main(["train", "bert-pytorch", "--dataset-path", str(missing)])
    tokenizer_factory.assert_not_called()


def test_invalid_dataset_fails_before_model_loading(settings, monkeypatch, tmp_path):
    from app.pipelines.bert import pipeline

    tokenizer_factory = Mock()
    monkeypatch.setattr(pipeline.AutoTokenizer, "from_pretrained", tokenizer_factory)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    dataset = tmp_path / "dataset.parquet"
    pd.DataFrame({"wrong_column": ["value"]}).to_parquet(dataset, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        cli.main(["train", "bert-pytorch", "--dataset-path", str(dataset)])
    tokenizer_factory.assert_not_called()


def test_docile_token_privacy(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DOCILE_TOKEN", "private-token")
    settings = Settings(_env_file=None)
    cli.build_parser(settings).parse_args(["train", "bert-pytorch"])
    assert settings.DOCILE.TOKEN == "private-token"
    with pytest.raises(SystemExit) as result:
        cli.build_parser(settings).parse_args(
            ["datasets", "download", "docile", "--help"]
        )
    assert result.value.code == 0
    assert "private-token" not in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.build_parser(settings).parse_args(
            ["train", "bert-pytorch", "--dataset-name", "unknown"]
        )
    monkeypatch.delenv("DOCILE_TOKEN")
    assert Settings(_env_file=None).DOCILE.TOKEN == ""
