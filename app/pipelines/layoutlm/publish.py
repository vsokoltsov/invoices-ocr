from dataclasses import dataclass
from pathlib import Path

from app.extraction.layoutlm.publisher import ClearMLPublisher
from app.pipelines.layoutlm.evaluate import EvaluationResult


@dataclass
class PublishModel:
    publisher: ClearMLPublisher
    output_dir: Path

    def publish(self, evaluation_result: EvaluationResult) -> str:
        """Publish only after the evaluation artifact has been produced."""
        if evaluation_result.metrics_path != self.output_dir / "test_metrics.json":
            raise ValueError(
                "Evaluation metrics must belong to the publication directory"
            )
        return self.publisher.run(
            checkpoint_path=evaluation_result.checkpoint_path,
            output_dir=self.output_dir,
        )
