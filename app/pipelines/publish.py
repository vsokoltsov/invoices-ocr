from dataclasses import dataclass

from app.extraction.publisher import ClearMLPublisher
from app.extraction.results import EvaluationResult


@dataclass
class PublishModel:
    """Publish a completed evaluation through the configured model publisher.

    Attributes:
        publisher: Model-specific ClearML publisher with run artifacts.
    """

    publisher: ClearMLPublisher

    def publish(self, evaluation_result: EvaluationResult) -> str:
        """Validate evaluation output references and publish this run.

        Args:
            evaluation_result: Checkpoint and test metrics produced by evaluation.

        Returns:
            Registered ClearML model ID.

        Raises:
            ValueError: Checkpoint or metric paths do not belong to the configured run.
        """
        self.publisher.artifacts.validate_evaluation(
            evaluation_result.checkpoint_path, evaluation_result.metrics_path
        )
        return self.publisher.run()
