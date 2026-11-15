from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from app.extraction.enums import IGNORE_INDEX
from app.extraction.layoutlm.artifacts import LayoutLMArtifacts
from app.extraction.layoutlm.checkpoint import LayoutLMCheckpointLoader
from app.extraction.layoutlm.evaluation import LayoutLMEvaluation
from app.extraction.layoutlm.metrics import EpochMetrics
from app.extraction.results import EvaluationResult, TrainingResult


@dataclass
class EvaluateTraining:
    """Reload a saved model and evaluate the held-out test split.

    Attributes:
        artifacts: Run storage used to load the checkpoint and save metrics.
        device: Device used for test inference.
    """

    artifacts: LayoutLMArtifacts
    device: torch.device

    def eval(
        self, train_result: TrainingResult, dataloaders: dict[str, DataLoader]
    ) -> EvaluationResult:
        """Evaluate the saved best checkpoint and persist test metrics.

        Loads a new model instance and moves it back to the CPU after inference.
        Metrics are written only after evaluation succeeds.

        Args:
            train_result: Training outputs identifying this run checkpoint.
            dataloaders: Split loaders; the test loader is used for inference.

        Returns:
            Checkpoint and saved test-metric paths for publication.

        Raises:
            ValueError: The checkpoint reference does not belong to this run.
            FileNotFoundError: Required checkpoint files are missing.
        """
        self.artifacts.validate_checkpoint_reference(train_result.checkpoint_path)
        model, schema, checkpoint = LayoutLMCheckpointLoader(
            self.artifacts, self.device
        ).load()
        evaluator = LayoutLMEvaluation(
            model=model,
            metrics=EpochMetrics(schema.labels, checkpoint["fields"]),
            schema=schema,
            device=self.device,
            criterion=nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX),
        )
        try:
            metrics = evaluator.evaluate(dataloaders["test"], epoch=checkpoint["epoch"])
        finally:
            model.to("cpu")
        metrics_path = self.artifacts.save_metrics(asdict(metrics))
        return EvaluationResult(train_result.checkpoint_path, metrics_path)
