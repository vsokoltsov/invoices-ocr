import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from app.extraction.enums import IGNORE_INDEX
from app.extraction.layoutlm.evaluation import LayoutLMEvaluation
from app.extraction.layoutlm.metrics import EpochMetrics
from app.extraction.layoutlm.model import LayoutLMFieldExtractor
from app.extraction.layoutlm.schema import InvoiceLabelSchema
from app.pipelines.layoutlm.training import TrainingResult


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
        """Load the best saved checkpoint and evaluate it on held-out test pages."""
        checkpoint = torch.load(
            train_result.checkpoint_path, map_location="cpu", weights_only=True
        )
        schema = InvoiceLabelSchema(checkpoint["field_to_entity"])
        if tuple(checkpoint["labels"]) != schema.labels:
            raise ValueError("Checkpoint labels do not match its field schema")
        model = LayoutLMFieldExtractor(
            model_name=checkpoint["model_name"],
            num_labels=schema.num_labels,
            id2label=schema.id2label,
            label2id=schema.label2id,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
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
        self.output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = self.output_dir / "test_metrics.json"
        metrics_path.write_text(
            json.dumps(asdict(metrics), indent=2, allow_nan=False), encoding="utf-8"
        )
        return EvaluationResult(train_result.checkpoint_path, metrics_path)
