import json
from dataclasses import dataclass

from app.extraction.bert.metrics import EvalResult
from app.extraction.bert.publisher import ClearMLPublisher
from app.pipelines.bert.evaluate import EvaluationResult


@dataclass
class PublishModel:
    publisher: ClearMLPublisher

    def publish(self, evaluation_result: EvaluationResult) -> str:
        metrics = EvalResult(
            **json.loads(evaluation_result.metrics_path.read_text(encoding="utf-8"))
        )

        return self.publisher.run(
            checkpoint_path=evaluation_result.checkpoint_path,
            test_metrics=metrics,
        )
