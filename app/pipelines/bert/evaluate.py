from dataclasses import dataclass

import torch
from torch import nn

from app.extraction.bert.artifacts import BertArtifacts
from app.extraction.bert.checkpoint import BertCheckpointLoader
from app.extraction.bert.decoding import OCRWordDecoder
from app.extraction.bert.evaluation import BertEvaluation
from app.extraction.bert.metrics import EvalMetrics
from app.extraction.enums import IGNORE_INDEX
from app.pipelines.evaluate import EvaluateCheckpoint


@dataclass
class EvaluateTraining(EvaluateCheckpoint):
    """Reload a saved model and evaluate the held-out test split.

    Attributes:
        artifacts: Run storage used to load the checkpoint and save metrics.
        device: Device used for test inference.
    """

    artifacts: BertArtifacts
    device: torch.device

    def load_evaluator(self):
        """Restore the checkpoint and configure model-specific test scoring.

        Returns:
            Restored model, evaluator, and checkpoint epoch.
        """
        model, _, schema, checkpoint = BertCheckpointLoader(
            self.artifacts, self.device
        ).load()
        word_decoder = (
            OCRWordDecoder(schema.label2id)
            if checkpoint["decoder"] == "ocr_word"
            else None
        )
        evaluator = BertEvaluation(
            model=model,
            metrics=EvalMetrics(schema.id2label, IGNORE_INDEX),
            device=self.device,
            criterion=nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX),
            word_decoder=word_decoder,
        )
        return model, evaluator, checkpoint["epoch"]
