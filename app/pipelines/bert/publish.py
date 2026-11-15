from dataclasses import dataclass

from app.extraction.bert.publisher import ClearMLPublisher
from app.pipelines.bert.evaluate import EvaluationResult


@dataclass
class PublishModel:
    publisher: ClearMLPublisher

    def publish(self, evaluation_result: EvaluationResult) -> str:
        """Publish only after evaluation has produced this run's metrics."""
        self.publisher.artifacts.validate_evaluation(
            evaluation_result.checkpoint_path, evaluation_result.metrics_path
        )
        return self.publisher.run()
