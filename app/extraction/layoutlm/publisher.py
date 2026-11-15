from dataclasses import dataclass

from clearml import OutputModel

from app.extraction.layoutlm.artifacts import LayoutLMArtifacts
from app.extraction.publisher import ClearMLPublisher as BaseClearMLPublisher


@dataclass
class ClearMLPublisher(BaseClearMLPublisher):
    """Adapt shared ClearML publication to LayoutLM artifacts.

    Attributes:
        artifacts: LayoutLM checkpoint and run output storage.
        task_name: Default task name and registered model name prefix.
    """

    artifacts: LayoutLMArtifacts
    task_name: str = "layoutlmv3"

    def upload_weights(self, model: OutputModel) -> None:
        """Upload the LayoutLM best.pt checkpoint.

        Waits for upload completion and keeps local files.

        Args:
            model: ClearML output model receiving the weights.
        """
        model.update_weights(
            weights_filename=str(self.artifacts.checkpoint_path),
            auto_delete_file=False,
            async_enable=False,
        )

    def test_scalars(self, metrics: dict) -> dict[str, float]:
        """Select the saved test metrics for ClearML scalar reporting.

        Args:
            metrics: Saved test metric dictionary.

        Returns:
            Entity precision, recall, and F1 as Python floats.
        """
        return {
            name: float(metrics[name])
            for name in ("entity_precision", "entity_recall", "entity_f1")
        }
