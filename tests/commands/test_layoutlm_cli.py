from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from app.aggregates.layoutlm.params import LayoutLMDevice, LayoutLMTrainingParams
from app.config import LayoutLMConfig, Settings
from app.pipelines.layoutlm import pipeline as layoutlm_pipeline
from commands import cli


@pytest.fixture
def settings():
    return Settings(
        _env_file=None,
        DATA_PATH="unused",
        DEBUG_ARTIFACTS_PATH="unused",
        LAYOUTLM=LayoutLMConfig(_env_file=None),
    )


def test_layoutlm_command_passes_every_option_to_pipeline(settings, monkeypatch):
    pipeline = Mock()
    factory = Mock(return_value=pipeline)
    monkeypatch.setattr(layoutlm_pipeline, "LayoutLMTrainingPipeline", factory)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    cli.main(
        [
            "train",
            "layoutlm",
            "--dataset-path",
            "prepared.parquet",
            "--model-name",
            "local-layoutlm",
            "--output-dir",
            "reports",
            "--checkpoint-path",
            "weights/best.pt",
            "--images-path",
            "images",
            "--num-epoch",
            "3",
            "--batch-size",
            "2",
            "--max-length-tokens",
            "64",
            "--learning-rate",
            "0.001",
            "--weight-decay",
            "0.02",
            "--warmup-ratio",
            "0.2",
            "--min-word-coverage",
            "0.75",
            "--dataloader-num-workers",
            "1",
            "--random-seed",
            "7",
            "--device",
            "cpu",
        ]
    )
    factory.assert_called_once_with(
        params=LayoutLMTrainingParams(
            dataset_path="prepared.parquet",
            model_name="local-layoutlm",
            output_dir="reports",
            checkpoint_path="weights/best.pt",
            images_path="images",
            num_epoch=3,
            batch_size=2,
            max_length_tokens=64,
            learning_rate=0.001,
            weight_decay=0.02,
            warmup_ratio=0.2,
            min_word_coverage=0.75,
            dataloader_num_workers=1,
            random_seed=7,
            device=LayoutLMDevice.CPU,
        )
    )
    pipeline.run.assert_called_once_with()


def test_config_uses_dotenv_then_environment_then_cli(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LAYOUTLM_BATCH_SIZE", raising=False)
    (tmp_path / ".env").write_text(
        "LAYOUTLM_BATCH_SIZE=7\nLAYOUTLM_WARMUP_RATIO=0.15\n",
        encoding="utf-8",
    )
    assert LayoutLMConfig().BATCH_SIZE == 7
    monkeypatch.setenv("LAYOUTLM_BATCH_SIZE", "4")
    monkeypatch.setenv("LAYOUTLM_DEVICE", "cpu")
    settings = Settings(DATA_PATH="unused", DEBUG_ARTIFACTS_PATH="unused")
    assert settings.LAYOUTLM.BATCH_SIZE == 4
    assert settings.LAYOUTLM.WARMUP_RATIO == 0.15
    assert settings.LAYOUTLM.DEVICE is LayoutLMDevice.CPU
    parser = cli.build_parser(settings)
    assert parser.parse_args(["train", "layoutlm"]).BATCH_SIZE == 4
    assert parser.parse_args(["train", "layoutlm", "--batch-size", "2"]).BATCH_SIZE == 2


def test_layoutlm_help_does_not_run_pipeline(settings, monkeypatch, capsys):
    factory = Mock()
    monkeypatch.setattr(layoutlm_pipeline, "LayoutLMTrainingPipeline", factory)
    with pytest.raises(SystemExit) as result:
        cli.build_parser(settings).parse_args(["train", "layoutlm", "--help"])
    assert result.value.code == 0
    output = capsys.readouterr().out
    assert "--warmup-ratio" in output
    assert "--checkpoint-path" in output
    assert "--device" in output
    factory.assert_not_called()


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--num-epoch", "0"),
        ("--batch-size", "0"),
        ("--max-length-tokens", "513"),
        ("--learning-rate", "0"),
        ("--weight-decay", "-0.1"),
        ("--warmup-ratio", "1"),
        ("--min-word-coverage", "0"),
        ("--dataloader-num-workers", "-1"),
        ("--random-seed", "-1"),
    ],
)
def test_invalid_training_values_fail_before_pipeline(
    settings, monkeypatch, option, value
):
    factory = Mock()
    monkeypatch.setattr(layoutlm_pipeline, "LayoutLMTrainingPipeline", factory)
    args = cli.build_parser(settings).parse_args(["train", "layoutlm", option, value])
    with pytest.raises(ValidationError):
        args.handler(args, settings)
    factory.assert_not_called()


def test_device_choices_are_checked_by_parser(settings):
    with pytest.raises(SystemExit):
        cli.build_parser(settings).parse_args(
            ["train", "layoutlm", "--device", "invalid"]
        )
