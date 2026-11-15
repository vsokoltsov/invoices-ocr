import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from app.extraction.bert.pytorch.checkpoint import BertPytorchCheckpoint
from app.extraction.bert.pytorch.decoding import OCRWordDecoder
from app.extraction.bert.pytorch.evaluation import BertEvaluation
from app.extraction.bert.pytorch.metrics import EvalMetrics
from app.extraction.enums import IGNORE_INDEX
from app.pipelines.bert.training import TrainingResult


@dataclass(frozen=True)
class EvaluationResult:
    checkpoint_path: Path
    metrics_path: Path


@dataclass
class EvaluateTraining:
    output_dir: Path
    device: torch.device

    def eval(
        self, train_result: TrainingResult, dataloaders: dict[str, DataLoader]
    ) -> EvaluationResult:
        """Load the best saved checkpoint and evaluate it on held-out test documents."""
        model, _, schema, checkpoint = BertPytorchCheckpoint(
            train_result.checkpoint_path
        ).load(self.device)
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
        self.output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = self.output_dir / "test_metrics.json"
        metrics_path.write_text(
            json.dumps(asdict(metrics), indent=2, allow_nan=False), encoding="utf-8"
        )
        return EvaluationResult(train_result.checkpoint_path, metrics_path)
