from unittest.mock import Mock

import pytest

from app.aggregates.bert.params import DatasetName
from app.config import (
    BertConfig,
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
        DATA_PATH="input",
        DEBUG_ARTIFACTS_PATH="debug",
        DOCILE_TOKEN="unused",
        RANDOM_SEED=123,
        BERT=BertConfig(),
        BERT_PYTORCH=BertPytorchConfig(),
    )


def test_pytorch_cli_maps_all_config_fields_to_pipeline(settings, monkeypatch):
    pipeline = Mock()
    monkeypatch.setattr(cli, "BertPytorchPipeline", lambda: pipeline)
    args = cli.build_parser(settings).parse_args(
        [
            "train",
            "bert-pytorch",
            "--dataset-name",
            "docile",
            "--dataset-path",
            "prepared.csv",
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
    params = pipeline.run.call_args.args[0]
    assert params.random_seed == 7
    assert params.dataset_type == DatasetName.DOCILE
    for name in BertPytorchConfig.model_fields:
        assert getattr(params, name.lower()) == getattr(args, name)


def test_transformers_command_keeps_existing_options_and_handler(settings, monkeypatch):
    factory = Mock()
    monkeypatch.setattr(cli, "BertTrainingPipeline", factory)
    args = cli.build_parser(settings).parse_args(
        [
            "train",
            "bert-transformers",
            "--data-path",
            "custom-input",
            "--num-epoch",
            "2",
            "--no-load-best-model-at-end",
        ]
    )
    args.handler(args, settings)
    params = factory.call_args.kwargs["params"]
    assert params.data_path == "custom-input"
    assert params.num_epoch == 2
    assert params.load_best_model_at_end is False
    assert params.random_seed == 123
    factory.return_value.run.assert_called_once_with()


def test_cli_uses_pytorch_configuration_defaults(settings):
    args = cli.build_parser(settings).parse_args(["train", "bert-pytorch"])
    for name in BertPytorchConfig.model_fields:
        assert getattr(args, name) == getattr(settings.BERT_PYTORCH, name)


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
    monkeypatch.setattr(cli, "BertPytorchPipeline", lambda: pipeline)
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


def test_dataset_command_uses_dataset_config_defaults(settings):
    args = cli.build_parser(settings).parse_args(
        ["datasets", "download", "kaggle-invoices"]
    )
    assert args.DESTINATION_PATH == settings.KAGGLE_INVOICES.DESTINATION_PATH
    assert args.ARCHIVE_PATH == settings.KAGGLE_INVOICES.ARCHIVE_PATH
    assert args.OUTPUT_PATH == settings.KAGGLE_INVOICES.OUTPUT_PATH


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


def test_missing_csv_fails_before_tokenizer_initialization(
    settings, monkeypatch, tmp_path
):
    from app.extraction.bert.pytorch import pipeline

    tokenizer_factory = Mock()
    monkeypatch.setattr(pipeline.AutoTokenizer, "from_pretrained", tokenizer_factory)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError, match="Prepared dataset CSV not found"):
        cli.main(["train", "bert-pytorch", "--dataset-path", str(missing)])
    tokenizer_factory.assert_not_called()


@pytest.mark.parametrize(
    "filename,content,message",
    [
        ("dataset.txt", "ocred_text,json_data\n", "must be a CSV"),
        ("dataset.csv", "wrong_column\nvalue\n", "missing columns"),
    ],
)
def test_invalid_dataset_fails_before_model_loading(
    settings, monkeypatch, tmp_path, filename, content, message
):
    from app.extraction.bert.pytorch import pipeline

    tokenizer_factory = Mock()
    monkeypatch.setattr(pipeline.AutoTokenizer, "from_pretrained", tokenizer_factory)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    dataset = tmp_path / filename
    dataset.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        cli.main(["train", "bert-pytorch", "--dataset-path", str(dataset)])
    tokenizer_factory.assert_not_called()


def test_dataset_selection_config_and_token_privacy(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATASET_NAME", "docile")
    monkeypatch.setenv("DOCILE_TOKEN", "private-token")
    settings = Settings(
        _env_file=None, DATA_PATH="unused", DEBUG_ARTIFACTS_PATH="unused"
    )
    args = cli.build_parser(settings).parse_args(["train", "bert-pytorch"])
    assert args.dataset_name == DatasetName.DOCILE
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
    assert (
        Settings(
            _env_file=None, DATA_PATH="unused", DEBUG_ARTIFACTS_PATH="unused"
        ).DOCILE.TOKEN
        == ""
    )
