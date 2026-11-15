import json
from unittest.mock import Mock

import pandas as pd
import pytest

from app.extraction import publisher as publisher_module
from app.extraction.layoutlm.artifacts import LayoutLMArtifacts
from app.extraction.layoutlm.publisher import ClearMLPublisher
from app.extraction.results import EvaluationResult
from app.pipelines.layoutlm.publish import PublishModel


@pytest.fixture
def publication(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"model_name": "local-tiny"}))
    (tmp_path / "test_metrics.json").write_text(
        json.dumps(
            {
                "entity_precision": 0.8,
                "entity_recall": 0.9,
                "entity_f1": 0.85,
            }
        )
    )
    artifacts = LayoutLMArtifacts(tmp_path)
    artifacts.save_history(pd.DataFrame({"epoch": [1]}))
    artifacts.save_manifest(pd.DataFrame({"document_id": ["doc"], "split": ["test"]}))
    artifacts.processor_dir.mkdir()
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    task = Mock()
    task.upload_artifact.return_value = True
    model = Mock(id="model-id")
    monkeypatch.setattr(publisher_module.Task, "init", Mock(return_value=task))
    monkeypatch.setattr(publisher_module, "OutputModel", Mock(return_value=model))
    return checkpoint, task, model


def test_publisher_uploads_checkpoint_and_artifacts_without_deleting_local_model(
    publication, tmp_path
):
    checkpoint, task, model = publication
    assert (
        ClearMLPublisher(LayoutLMArtifacts(tmp_path, checkpoint_path=checkpoint)).run()
        == "model-id"
    )
    model.update_weights.assert_called_once_with(
        weights_filename=str(checkpoint.resolve()),
        auto_delete_file=False,
        async_enable=False,
    )
    assert [call.kwargs["name"] for call in task.upload_artifact.call_args_list] == [
        "test_metrics.json",
        "history.csv",
        "config.json",
        "split_manifest.csv",
        "processor",
    ]
    assert all(
        call.kwargs["wait_on_upload"] for call in task.upload_artifact.call_args_list
    )
    assert task.get_logger.return_value.report_scalar.call_count == 3
    assert checkpoint.is_file()
    task.mark_failed.assert_not_called()
    task.close.assert_called_once()


@pytest.mark.parametrize("failure", ["weights", "artifact"])
def test_failed_upload_marks_task_failed_and_closes_it(publication, tmp_path, failure):
    checkpoint, task, model = publication
    if failure == "weights":
        model.update_weights.side_effect = RuntimeError("Upload failed")
    else:
        task.upload_artifact.return_value = False
    with pytest.raises(RuntimeError):
        ClearMLPublisher(LayoutLMArtifacts(tmp_path, checkpoint_path=checkpoint)).run()
    task.mark_failed.assert_called_once()
    task.close.assert_called_once()


def test_publication_consumes_evaluation_checkpoint_and_keeps_paths_in_constructor(
    tmp_path,
):
    publisher = Mock(
        artifacts=LayoutLMArtifacts(
            tmp_path, checkpoint_path=tmp_path / "weights" / "best.pt"
        )
    )
    publisher.run.return_value = "model-id"
    checkpoint = tmp_path / "weights" / "best.pt"
    result = EvaluationResult(checkpoint, tmp_path / "test_metrics.json")
    assert PublishModel(publisher).publish(result) == "model-id"
    publisher.run.assert_called_once_with()
    publisher.artifacts = LayoutLMArtifacts(
        tmp_path / "different-run", checkpoint_path=checkpoint
    )
    with pytest.raises(ValueError, match="publication directory"):
        PublishModel(publisher).publish(result)
