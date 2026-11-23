import json
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import torch

from app.aggregates.layoutlm.params import LayoutLMDevice, LayoutLMTrainingParams
from app.config import LayoutLMConfig, Settings
from app.datasets.split import DatasetsSplit
from app.extraction.layoutlm.model import LayoutLMFieldExtractor
from app.extraction.layoutlm.publisher import ClearMLPublisher
from app.pipelines.layoutlm import pipeline as pipeline_module
from app.pipelines.layoutlm.pipeline import LayoutLMTrainingPipeline
from commands import cli


@pytest.fixture
def pipeline_params(tmp_path):
    config = LayoutLMConfig(
        _env_file=None,
        DATASET_PATH=str(tmp_path / "data.parquet"),
        MODEL_NAME="local-tiny",
        OUTPUT_DIR=str(tmp_path / "results"),
        CHECKPOINT_PATH=str(tmp_path / "weights" / "best.pt"),
        IMAGES_PATH=str(tmp_path / "images"),
        NUM_EPOCH=1,
        BATCH_SIZE=4,
        MAX_LENGTH_TOKENS=8,
        DEVICE=LayoutLMDevice.CPU,
        LEARNING_RATE=0.001,
        WARMUP_RATIO=0.25,
        WEIGHT_DECAY=0.03,
        MIN_WORD_COVERAGE=0.75,
        DATALOADER_NUM_WORKERS=0,
    )
    return LayoutLMTrainingParams(
        **{name.lower(): getattr(config, name) for name in LayoutLMConfig.model_fields}
    )


@pytest.mark.parametrize(
    "backend,diagnostics", [("pipefunc", False), ("pipefunc", True), ("clearml", True)]
)
def test_cli_runs_actual_pipefunc_graph_and_writes_reloadable_artifacts(
    pipeline_params,
    document_row,
    layout_tokenizer,
    image_processor,
    layout_backbone,
    tmp_path,
    monkeypatch,
    backend,
    diagnostics,
):
    pipeline_params = replace(
        pipeline_params,
        diagnostic_dir=str(tmp_path / "diagnostics") if diagnostics else "",
    )
    frame = pd.DataFrame(
        [{**vars(document_row), "document_id": f"doc-{index}"} for index in range(10)]
    )
    frame.to_parquet(pipeline_params.dataset_path, index=False)
    monkeypatch.setattr(
        pipeline_module.AutoTokenizer,
        "from_pretrained",
        Mock(return_value=layout_tokenizer),
    )
    monkeypatch.setattr(
        pipeline_module.LayoutLMv3ImageProcessor,
        "from_pretrained",
        Mock(return_value=image_processor),
    )
    models = []
    initial_weights = []

    def create_model(**kwargs):
        model = LayoutLMFieldExtractor(**kwargs)
        models.append(model)
        initial_weights.append(model.classifier.weight.detach().clone())
        return model

    monkeypatch.setattr(pipeline_module, "LayoutLMFieldExtractor", create_model)
    settings = Settings(
        _env_file=None,
        LAYOUTLM=LayoutLMConfig(
            _env_file=None,
            **{
                name: getattr(pipeline_params, name.lower())
                for name in LayoutLMConfig.model_fields
            },
        ),
    )

    def publish():
        checkpoint_path = Path(pipeline_params.checkpoint_path)
        output_dir = Path(pipeline_params.output_dir)
        assert checkpoint_path.is_file()
        assert (output_dir / "test_metrics.json").is_file()
        assert (output_dir / "history.csv").is_file()
        return "clearml-model-id"

    upload = Mock(side_effect=publish)
    monkeypatch.setattr(ClearMLPublisher, "run", upload)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    inspector_factory = Mock(wraps=pipeline_module.EntityErrorInspector)
    monkeypatch.setattr(pipeline_module, "EntityErrorInspector", inspector_factory)
    if backend == "pipefunc":
        cli.main(["train", "layoutlm"])
    else:
        from app.pipelines.clearml import stages

        monkeypatch.setattr(stages.Task, "current_task", lambda: Mock())
        monkeypatch.setattr(ClearMLPublisher, "publish_to", lambda self, task: upload())
        values = asdict(pipeline_params)
        prepared = stages.prepare_model("layoutlm", values)
        trained = stages.train_model("layoutlm", values, prepared)
        evaluated = stages.evaluate_model("layoutlm", values, prepared, trained)
        assert stages.publish_model("layoutlm", values, evaluated) == "clearml-model-id"
    upload.assert_called_once()

    output = tmp_path / "results"
    history = pd.read_csv(output / "history.csv")
    assert history["epoch"].tolist() == [1]
    assert not torch.equal(initial_weights[0], models[0].classifier.weight)
    saved = torch.load(pipeline_params.checkpoint_path, weights_only=True)
    models[0].load_state_dict(saved["model_state_dict"])
    assert saved["max_length"] == 8
    metrics = json.loads((output / "test_metrics.json").read_text())
    assert 0 <= metrics["entity_f1"] <= 1
    assert metrics["entity_report"]["SELLER_NAME"]["support"] == 1
    config = json.loads((output / "config.json").read_text())
    assert config["warmup_ratio"] == 0.25
    assert config["min_word_coverage"] == 0.75
    assert config["device"] == "cpu"
    assert config["diagnostic_dir"] == pipeline_params.diagnostic_dir
    assert inspector_factory.call_count == (1 if diagnostics else 0)
    if diagnostics:
        directory = Path(pipeline_params.diagnostic_dir) / "epoch_001"
        summary = json.loads((directory / "summary.json").read_text())
        for key in history.columns:
            if key.startswith("validation_"):
                assert summary[key.removeprefix("validation_")] == pytest.approx(
                    history[key].iloc[0]
                )
        assert (directory / "word_predictions.csv").is_file()
        assert (directory / "per_entity.csv").is_file()
    else:
        assert not (tmp_path / "diagnostics").exists()
    assert (output / "processor" / "tokenizer.json").is_file()
    assert (output / "processor" / "preprocessor_config.json").is_file()
    manifest = pd.read_csv(output / "split_manifest.csv")
    assert manifest["document_id"].is_unique
    assert manifest.groupby("split").size().to_dict() == {
        "train": 8,
        "val": 1,
        "test": 1,
    }
    assert manifest["accepted"].all()
    for name in ("train", "val", "test"):
        assert pd.read_csv(output / f"{name}_rejected.csv").empty


@pytest.mark.parametrize("failure", ["missing_columns", "duplicate_ids", "too_small"])
def test_bad_dataset_fails_before_model_download(pipeline_params, monkeypatch, failure):
    frame = pd.DataFrame(
        {
            "document_id": [f"doc-{index}" for index in range(6)],
            "json_data": ["{}"] * 6,
            "text_with_boxes": [[]] * 6,
            "pdf_path": ["missing.pdf"] * 6,
        }
    )
    if failure == "missing_columns":
        frame = frame.drop(columns="json_data")
    elif failure == "duplicate_ids":
        frame.loc[1, "document_id"] = frame.loc[0, "document_id"]
    else:
        frame = frame.iloc[:5]
    frame.to_parquet(pipeline_params.dataset_path, index=False)
    factory = Mock()
    monkeypatch.setattr(pipeline_module.AutoTokenizer, "from_pretrained", factory)
    with pytest.raises(ValueError):
        LayoutLMTrainingPipeline(pipeline_params).run()
    factory.assert_not_called()


def test_split_is_repeatable_and_keeps_documents_disjoint():
    frame = pd.DataFrame({"document_id": range(20)})
    first = DatasetsSplit.create(frame, random_seed=7)
    second = DatasetsSplit.create(frame, random_seed=7)
    train, val, test = [
        set(getattr(first, name)["document_id"]) for name in ("train", "val", "test")
    ]
    assert (len(train), len(val), len(test)) == (16, 2, 2)
    assert not (train & val or train & test or val & test)
    assert train | val | test == set(range(20))
    for name in ("train", "val", "test"):
        pd.testing.assert_frame_equal(getattr(first, name), getattr(second, name))
