from unittest.mock import Mock

import pytest

from app.config import BertConfig, BertPytorchConfig, Settings
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
