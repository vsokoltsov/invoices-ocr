from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from app.extraction.bert.artifacts import BertArtifacts
from app.extraction.bert.checkpoint import BertPytorchCheckpoint
from app.extraction.bert.decoding import OCRWordDecoder
from app.extraction.bert.evaluation import BertEvaluation
from app.extraction.bert.metrics import EvalMetrics
from app.extraction.enums import IGNORE_INDEX
from app.pipelines.bert.training import TrainingResult


@dataclass(frozen=True)
class EvaluationResult:
    checkpoint_path: Path
    metrics_path: Path


@dataclass
class EvaluateTraining:
    artifacts: BertArtifacts
    device: torch.device

    def eval(
        self, train_result: TrainingResult, dataloaders: dict[str, DataLoader]
    ) -> EvaluationResult:
        """Load the best saved checkpoint and evaluate it on held-out test documents."""
        self.artifacts.validate_checkpoint_reference(train_result.checkpoint_path)
        model, _, schema, checkpoint = BertPytorchCheckpoint(self.artifacts).load(
            self.device
        )
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
        try:
            metrics = evaluator.evaluate(dataloaders["test"], epoch=checkpoint["epoch"])
        finally:
            model.to("cpu")
        metrics_path = self.artifacts.save_metrics(asdict(metrics))
        return EvaluationResult(train_result.checkpoint_path, metrics_path)
