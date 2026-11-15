from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clearml import Task
from transformers import set_seed

from app.aggregates.bert.params import BertPytorchParams
from app.aggregates.layoutlm.params import LayoutLMTrainingParams
from app.datasets.split import DatasetsSplit
from app.extraction.results import EvaluationResult, TrainingResult
from app.pipelines.bert.pipeline import BertTrainingPipeline
from app.pipelines.clearml.artifacts import PreparedData
from app.pipelines.components import ModelPipeline
from app.pipelines.layoutlm.pipeline import LayoutLMTrainingPipeline


@dataclass
class ModelStages:
    """Execute one stage using fresh components and filesystem handoffs.

    Attributes:
        components: Lazy model components reconstructed from serializable settings.
    """

    components: ModelPipeline

    @classmethod
    def create(cls, model: str, params: dict[str, Any]) -> "ModelStages":
        """Reconstruct model-specific components in a pipeline worker.

        Args:
            model: Registered component type, bert or layoutlm.
            params: Serialized model parameters from the controller.

        Returns:
            Stage runner with lazy, model-specific components.
        """
        factories = {
            "bert": lambda: BertTrainingPipeline(BertPytorchParams(**params)),
            "layoutlm": lambda: LayoutLMTrainingPipeline(
                LayoutLMTrainingParams(**params)
            ),
        }
        components = factories[model]()
        set_seed(components.random_seed)
        return cls(components)

    def prepare(self) -> str:
        """Split documents, prepare labels, and persist worker inputs.

        Returns:
            Absolute directory containing prepared split records.
        """
        frame = self.components.load_source()
        self.components.prepare_run()
        split = DatasetsSplit.create(frame, self.components.random_seed)
        result = self.components.build_step.build(split)
        return PreparedData.for_run(self.components.artifacts.output_dir).save(
            split, result
        )

    def train(self, prepared: str) -> dict[str, str]:
        """Train a model from preparation outputs in a fresh process.

        Args:
            prepared: Shared directory produced by the preparation task.

        Returns:
            Serializable best-checkpoint and history paths.
        """
        split, result = PreparedData(Path(prepared)).load()
        loaders = self.components.loader_step.build(result)
        trained = self.components.train_prepared(loaders, result, split)
        return {
            "checkpoint_path": str(trained.checkpoint_path),
            "history_path": str(trained.history_path),
        }

    def evaluate(self, prepared: str, trained: dict[str, str]) -> dict[str, str]:
        """Reload best weights and evaluate held-out records in a fresh process.

        Args:
            prepared: Shared preparation directory.
            trained: Paths returned by the training task.

        Returns:
            Serializable checkpoint and test-metric paths.
        """
        _, result = PreparedData(Path(prepared)).load()
        loaders = {
            "test": self.components.loader_step.create_loader(
                result.records.test, training=False
            )
        }
        evaluation = self.components.evaluation_step.eval(
            TrainingResult(**{key: Path(value) for key, value in trained.items()}),
            loaders,
        )
        return {
            "checkpoint_path": str(evaluation.checkpoint_path),
            "metrics_path": str(evaluation.metrics_path),
        }

    def publish(self, evaluated: dict[str, str]) -> str:
        """Publish artifacts into the current ClearML worker task.

        Args:
            evaluated: Checkpoint and metric paths produced by test evaluation.

        Returns:
            Registered model ID.

        Raises:
            RuntimeError: This stage is invoked outside a ClearML task.
        """
        result = EvaluationResult(
            **{key: Path(value) for key, value in evaluated.items()}
        )
        self.components.artifacts.validate_evaluation(
            result.checkpoint_path, result.metrics_path
        )
        task = Task.current_task()
        if task is None:
            raise RuntimeError("ClearML publication requires an active pipeline task")
        return self.components.publisher_step.publisher.publish_to(task)


# ClearML extracts these functions into standalone task scripts. Keep imports inside
# each entry point so those scripts do not depend on controller globals.
def prepare_model(model: str, params: dict) -> str:
    """Prepare data in a ClearML task.

    Args:
        model: Model component name.
        params: Serialized model parameters.

    Returns:
        Shared preparation directory.
    """
    from app.pipelines.clearml.stages import ModelStages

    return ModelStages.create(model, params).prepare()


def train_model(model: str, params: dict, prepared: str) -> dict:
    """Train using preparation outputs in a ClearML task.

    Args:
        model: Model component name.
        params: Serialized model parameters.
        prepared: Directory returned by preparation.

    Returns:
        Best-checkpoint and history paths.
    """
    from app.pipelines.clearml.stages import ModelStages

    return ModelStages.create(model, params).train(prepared)


def evaluate_model(model: str, params: dict, prepared: str, trained: dict) -> dict:
    """Evaluate the saved model in a ClearML task.

    Args:
        model: Model component name.
        params: Serialized model parameters.
        prepared: Directory returned by preparation.
        trained: Training output paths resolved by ClearML.

    Returns:
        Checkpoint and test-metric paths.
    """
    from app.pipelines.clearml.stages import ModelStages

    return ModelStages.create(model, params).evaluate(prepared, trained)


def publish_model(model: str, params: dict, evaluated: dict) -> str:
    """Publish evaluated model artifacts in the current ClearML task.

    Args:
        model: Model component name.
        params: Serialized model parameters.
        evaluated: Test-evaluation output paths.

    Returns:
        Registered model ID.
    """
    from app.pipelines.clearml.stages import ModelStages

    return ModelStages.create(model, params).publish(evaluated)
