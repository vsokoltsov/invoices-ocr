from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
import torch
from torch.utils.data import TensorDataset

from app.datasets.split import DatasetsSplit
from app.extraction.layoutlm.artifacts import LayoutLMArtifacts
from app.extraction.metrics import TrainResult
from app.extraction.results import TrainingResult
from app.pipelines.dataloader import DataLoaders
from app.pipelines.evaluate import EvaluateCheckpoint
from app.pipelines.preprocess import PreprocessResult
from app.pipelines.training import TrainingRun


class FrameLoaders(DataLoaders):
    def create_dataset(self, frame, *, training):
        return TensorDataset(torch.tensor(frame["value"].tolist()))


def collate_values(samples):
    return torch.stack([sample[0] for sample in samples]) + 100


def test_loaders_repeat_training_shuffle_keep_evaluation_order_and_use_collator():
    frame = pd.DataFrame({"value": range(20)})
    split = DatasetsSplit(frame, frame.iloc[:3], frame.iloc[3:6])
    step = FrameLoaders(
        device=torch.device("cpu"),
        batch_size=4,
        random_seed=17,
        collate_fn=collate_values,
    )
    result = PreprocessResult(split, split)
    first, second = step.build(result), step.build(result)
    order = torch.cat(list(first["train"]))
    assert torch.equal(order, torch.cat(list(second["train"])))
    assert sorted(order.tolist()) == list(range(100, 120))
    assert order.tolist() != list(range(100, 120))
    assert torch.cat(list(first["val"])).tolist() == [100, 101, 102]
    assert torch.cat(list(first["test"])).tolist() == [103, 104, 105]
    assert not first["train"].pin_memory


@pytest.mark.parametrize("failed", [False, True])
def test_training_run_releases_model_on_success_or_failure(tmp_path, failed):
    trainer, model = Mock(), Mock()
    artifacts = SimpleNamespace(
        checkpoint_location=tmp_path / "best.pt", history_path=tmp_path / "history.csv"
    )
    run = TrainingRun(trainer, model, artifacts)
    if failed:
        trainer.train.side_effect = RuntimeError("training failed")
        with pytest.raises(RuntimeError, match="training failed"):
            run.run()
    else:
        assert run.run() == TrainingResult(
            artifacts.checkpoint_location, artifacts.history_path
        )
    trainer.train.assert_called_once_with()
    model.to.assert_called_once_with("cpu")


@pytest.mark.parametrize("failed", [False, True])
def test_evaluation_releases_model_and_writes_metrics_only_on_success(tmp_path, failed):
    model, evaluator = Mock(), Mock()
    evaluator.evaluate.return_value = TrainResult(0.25)
    if failed:
        evaluator.evaluate.side_effect = RuntimeError("evaluation failed")

    class EvaluationStep(EvaluateCheckpoint):
        def load_evaluator(self):
            return model, evaluator, 3

    artifacts = LayoutLMArtifacts(tmp_path)
    step = EvaluationStep(artifacts=artifacts, device=torch.device("cpu"))
    train_result = TrainingResult(artifacts.checkpoint_path, artifacts.history_path)
    loader = Mock()
    if failed:
        with pytest.raises(RuntimeError, match="evaluation failed"):
            step.eval(train_result, {"test": loader})
        assert not artifacts.test_metrics_path.exists()
    else:
        result = step.eval(train_result, {"test": loader})
        assert result.checkpoint_path == artifacts.checkpoint_path
        assert artifacts.read_json(result.metrics_path) == {"loss": 0.25}
    evaluator.evaluate.assert_called_once_with(loader, epoch=3)
    model.to.assert_called_once_with("cpu")
