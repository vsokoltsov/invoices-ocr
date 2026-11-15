from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from app.extraction.artifacts import RunArtifacts
from app.extraction.protocols import Evaluator
from app.extraction.results import EvaluationResult, TrainingResult


@dataclass(kw_only=True)
class EvaluateCheckpoint(ABC):
    """Evaluate saved weights, release the model, and persist test metrics.

    Attributes:
        artifacts: Storage used to validate checkpoint references and save metrics.
        device: Execution device used by the model-specific evaluator.
    """

    artifacts: RunArtifacts
    device: torch.device

    @abstractmethod
    def load_evaluator(self) -> tuple[nn.Module, Evaluator[Any], int]:
        """Restore the saved model and configure its scoring strategy.

        Returns:
            Model, evaluator returning a dataclass result, and saved epoch number.
        """
        ...

    def eval(
        self, train_result: TrainingResult, dataloaders: dict[str, DataLoader]
    ) -> EvaluationResult:
        """Evaluate held-out test data using the saved best checkpoint.

        Args:
            train_result: Checkpoint reference produced by training.
            dataloaders: Split loaders; only the test loader is evaluated.

        Returns:
            Checkpoint and saved test-metric paths for publication.

        Raises:
            ValueError: The checkpoint reference does not belong to this run.
            FileNotFoundError: A required checkpoint file is absent.

        Moves the loaded model to the CPU on evaluation success or failure.
        Metrics are written only after evaluation succeeds.
        """
        self.artifacts.validate_checkpoint_reference(train_result.checkpoint_path)
        model, evaluator, epoch = self.load_evaluator()
        try:
            metrics = evaluator.evaluate(dataloaders["test"], epoch=epoch)
        finally:
            model.to("cpu")
        metrics_path = self.artifacts.save_metrics(asdict(metrics))
        return EvaluationResult(train_result.checkpoint_path, metrics_path)
