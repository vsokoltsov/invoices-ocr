from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from app.aggregates.bert.params import BertPytorchParams
from app.aggregates.layoutlm.params import LayoutLMDevice, LayoutLMTrainingParams
from app.aggregates.pipeline import (
    BertRunMode,
    ClearMLPipelineParams,
    PipelineExecution,
)
from app.config import ClearMLPipelineConfig, Settings
from app.pipelines.clearml import pipeline as pipeline_module
from app.pipelines.clearml.pipeline import BertClearMLPipeline, LayoutLMClearMLPipeline
from commands import cli


@pytest.fixture
def bert_params(tmp_path):
    return BertPytorchParams(
        dataset_path=str(tmp_path / "docile.parquet"),
        result_filename=str(tmp_path / "history"),
        model_name="local-tiny",
        max_length_tokens=16,
        random_seed=42,
        batch_size=2,
        num_epoch=1,
        learning_rate=1e-3,
        weight_decay=0.01,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        diagnostic_dir=str(tmp_path / "diagnostics"),
    )


@pytest.mark.parametrize(
    "mode,branches",
    [
        (BertRunMode.MODEL, [False]),
        (BertRunMode.LINE_ITEMS, [True]),
        (BertRunMode.BOTH, [False, True]),
    ],
)
def test_bert_modes_build_real_stage_dependencies_and_isolate_outputs(
    bert_params, monkeypatch, mode, branches
):
    controller = Mock()
    factory = Mock(return_value=controller)
    monkeypatch.setattr(pipeline_module, "PipelineController", factory)
    pipeline = BertClearMLPipeline(bert_params, ClearMLPipelineParams(), mode).build()
    assert [run.params["line_items_only"] for run in pipeline.runs] == branches
    assert pipeline.create() is controller
    steps = [call.kwargs for call in controller.add_function_step.call_args_list]
    assert len(steps) == 4 * len(branches)
    for index, run in enumerate(pipeline.runs):
        prepare, train, evaluate, publish = steps[index * 4 : (index + 1) * 4]
        assert (
            train["function_kwargs"]["prepared"]
            == "${" + prepare["name"] + ".prepared}"
        )
        assert (
            evaluate["function_kwargs"]["trained"] == "${" + train["name"] + ".trained}"
        )
        assert (
            publish["function_kwargs"]["evaluated"]
            == "${" + evaluate["name"] + ".evaluated}"
        )
        assert not any(
            step["cache_executed_step"] for step in (prepare, train, evaluate, publish)
        )
        if index:
            assert prepare["parents"] == [steps[index * 4 - 1]["name"]]
        assert Path(run.params["dataset_path"]).is_absolute()
    if mode == BertRunMode.BOTH:
        left, right = [run.params for run in pipeline.runs]
        for key in ("checkpoint_dir", "result_filename", "diagnostic_dir"):
            assert left[key] != right[key]
        assert left["random_seed"] == right["random_seed"]
    factory.assert_called_once()


@pytest.mark.parametrize("execution", list(PipelineExecution))
@pytest.mark.parametrize("success", [False, True])
def test_execution_uses_selected_backend_and_propagates_stage_failures(
    bert_params, monkeypatch, execution, success
):
    controller = Mock()
    controller.is_successful.return_value = success
    monkeypatch.setattr(
        pipeline_module, "PipelineController", Mock(return_value=controller)
    )
    settings = ClearMLPipelineParams(
        project="test-project", execution=execution, queue="gpu"
    )
    pipeline = BertClearMLPipeline(bert_params, settings)
    if success:
        assert pipeline.run() is controller
    else:
        with pytest.raises(RuntimeError, match="ClearML pipeline failed"):
            pipeline.run()
    controller.set_default_execution_queue.assert_called_once_with("gpu")
    controller.start_locally.assert_called_once_with(
        run_pipeline_steps_locally=execution == PipelineExecution.LOCAL
    )


def test_layoutlm_builds_one_branch_and_preserves_model_settings(tmp_path, monkeypatch):
    params = LayoutLMTrainingParams(
        dataset_path=str(tmp_path / "docile.parquet"),
        model_name="local-tiny",
        output_dir=str(tmp_path / "results"),
        checkpoint_path=str(tmp_path / "best.pt"),
        images_path=str(tmp_path / "images"),
        num_epoch=2,
        batch_size=4,
        max_length_tokens=16,
        learning_rate=1e-3,
        weight_decay=0.01,
        warmup_ratio=0.1,
        min_word_coverage=0.5,
        dataloader_num_workers=0,
        random_seed=17,
        device=LayoutLMDevice.CPU,
    )
    controller = Mock()
    monkeypatch.setattr(
        pipeline_module, "PipelineController", Mock(return_value=controller)
    )
    pipeline = LayoutLMClearMLPipeline(params, ClearMLPipelineParams()).build()
    pipeline.create()
    assert controller.add_function_step.call_count == 4
    (run,) = pipeline.runs
    assert run.model == "layoutlm"
    assert run.params["random_seed"] == 17
    assert run.params["checkpoint_path"] == str(tmp_path / "best.pt")


@pytest.mark.parametrize("model", ["bert", "layoutlm"])
def test_cli_forwards_model_and_pipeline_options(monkeypatch, capsys, model):
    settings = Settings(
        _env_file=None, CLEARML_PIPELINE=ClearMLPipelineConfig(_env_file=None)
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    factory = Mock()
    factory.return_value.run.return_value.id = "pipeline-id"
    name = "BertClearMLPipeline" if model == "bert" else "LayoutLMClearMLPipeline"
    monkeypatch.setattr(pipeline_module, name, factory)
    options = ["--mode", "both"] if model == "bert" else ["--device", "cpu"]
    cli.main(
        [
            "pipeline",
            model,
            "--project",
            "test-project",
            "--execution",
            "agent",
            "--queue",
            "gpu",
            "--num-epoch",
            "2",
            *options,
        ]
    )
    params, execution = factory.call_args.args
    assert params.num_epoch == 2
    assert execution == ClearMLPipelineParams(
        "test-project", PipelineExecution.AGENT, "gpu"
    )
    if model == "bert":
        assert factory.call_args.kwargs["mode"] == BertRunMode.BOTH
    factory.return_value.run.assert_called_once_with()
    assert capsys.readouterr().out.strip() == "pipeline-id"


def test_both_mode_preserves_disabled_diagnostics(bert_params):
    params = replace(bert_params, diagnostic_dir=None)
    pipeline = BertClearMLPipeline(
        params, ClearMLPipelineParams(), BertRunMode.BOTH
    ).build()
    assert all(run.params["diagnostic_dir"] is None for run in pipeline.runs)


def test_prepared_data_roundtrip_restores_nested_lists_and_rejection_details(tmp_path):
    import pandas as pd

    from app.datasets.split import DatasetsSplit
    from app.pipelines.builder import BuildResult
    from app.pipelines.clearml.artifacts import PreparedData

    source = pd.DataFrame(
        {
            "document_id": ["doc"],
            "text_with_boxes": [[{"text": "Acme", "start": 0, "end": 4}]],
        }
    )
    records = pd.DataFrame(
        {
            "document_id": ["doc"],
            "page": [0],
            "words": [["Acme", "Ltd"]],
            "boxes": [[[0, 0, 10, 10], [10, 0, 20, 10]]],
            "labels": [[1, 2]],
        }
    )
    rejected = pd.DataFrame(
        {
            "document_id": ["rejected"],
            "issues": [[{"status": "missing", "candidates": []}]],
        }
    )
    store = PreparedData.for_run(tmp_path / "checkpoint")
    assert store.directory == tmp_path / "checkpoint-prepared"
    directory = store.save(
        DatasetsSplit(source, source, source),
        BuildResult(
            DatasetsSplit(records, records, records),
            DatasetsSplit(rejected, rejected, rejected),
        ),
    )
    restored_source, restored = PreparedData(Path(directory)).load()
    for name in ("train", "val", "test"):
        assert getattr(restored.records, name).to_dict("records") == records.to_dict(
            "records"
        )
        assert getattr(restored_source, name).iloc[0]["document_id"] == "doc"
        assert getattr(restored.rejected, name).to_dict("records") == rejected.to_dict(
            "records"
        )


@pytest.mark.parametrize("execution", list(PipelineExecution))
def test_layoutlm_cli_builds_and_starts_clearml_pipeline(
    tmp_path, monkeypatch, capsys, execution
):
    settings = Settings(
        _env_file=None,
        CLEARML_PIPELINE=ClearMLPipelineConfig(_env_file=None),
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    controller = Mock(id="layoutlm-pipeline-id")
    controller.is_successful.return_value = True
    factory = Mock(return_value=controller)
    monkeypatch.setattr(pipeline_module, "PipelineController", factory)
    cli.main(
        [
            "pipeline",
            "layoutlm",
            "--execution",
            execution.value,
            "--project",
            "ocr-test",
            "--queue",
            "gpu",
            "--dataset-path",
            str(tmp_path / "docile.parquet"),
            "--output-dir",
            str(tmp_path / "results"),
            "--checkpoint-path",
            str(tmp_path / "best.pt"),
            "--images-path",
            str(tmp_path / "images"),
            "--device",
            "cpu",
            "--num-epoch",
            "2",
        ]
    )
    assert factory.call_args.kwargs["name"] == "layoutlm-training"
    assert factory.call_args.kwargs["project"] == "ocr-test"
    steps = [call.kwargs for call in controller.add_function_step.call_args_list]
    assert [step["name"] for step in steps] == [
        "layoutlm-prepare",
        "layoutlm-train",
        "layoutlm-evaluate",
        "layoutlm-publish",
    ]
    for step in steps:
        assert step["function_kwargs"]["model"] == "layoutlm"
        params = step["function_kwargs"]["params"]
        assert params["num_epoch"] == 2
        assert params["device"] == "cpu"
        assert params["checkpoint_path"] == str(tmp_path / "best.pt")
    controller.set_default_execution_queue.assert_called_once_with("gpu")
    controller.start_locally.assert_called_once_with(
        run_pipeline_steps_locally=execution == PipelineExecution.LOCAL
    )
    assert capsys.readouterr().out.strip() == "layoutlm-pipeline-id"
