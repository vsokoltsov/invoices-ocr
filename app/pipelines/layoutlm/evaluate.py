from dataclasses import dataclass

import torch
from torch import nn

from app.extraction.enums import IGNORE_INDEX
from app.extraction.layoutlm.artifacts import LayoutLMArtifacts
from app.extraction.layoutlm.checkpoint import LayoutLMCheckpointLoader
from app.extraction.layoutlm.evaluation import LayoutLMEvaluation
from app.extraction.metrics import WindowMetrics
from app.pipelines.evaluate import EvaluateCheckpoint


@dataclass
class EvaluateTraining(EvaluateCheckpoint):
    """Reload a saved model and evaluate the held-out test split.

    Attributes:
        artifacts: Run storage used to load the checkpoint and save metrics.
        device: Device used for test inference.
    """

    artifacts: LayoutLMArtifacts
    device: torch.device

    def load_evaluator(self):
        """Restore the checkpoint and configure model-specific test scoring.

        Returns:
            Restored model, evaluator, and checkpoint epoch.
        """
        model, schema, checkpoint = LayoutLMCheckpointLoader(
            self.artifacts, self.device
        ).load()
        evaluator = LayoutLMEvaluation(
            model=model,
            metrics=WindowMetrics(schema.id2label),
            schema=schema,
            device=self.device,
            criterion=nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX),
        )
        return model, evaluator, checkpoint["epoch"]
