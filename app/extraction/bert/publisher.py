from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from clearml import OutputModel

from app.extraction.bert.artifacts import BertArtifacts
from app.extraction.bert.metrics import EvalResult
from app.extraction.publisher import ClearMLPublisher as BaseClearMLPublisher


@dataclass
class ClearMLPublisher(BaseClearMLPublisher):
    """Adapt shared ClearML publication to BERT artifacts.

    Attributes:
        artifacts: BERT checkpoint and run output storage.
        task_name: Default task name and registered model name prefix.
    """

    artifacts: BertArtifacts
    task_name: str = "bert-pytorch"

    def upload_weights(self, model: OutputModel) -> None:
        """Upload the complete BERT checkpoint directory as a ZIP package.

        Waits for upload completion and keeps local files.

        Args:
            model: ClearML output model receiving the weights.
        """
        model.update_weights_package(
            weights_path=str(self.artifacts.checkpoint_dir),
            target_filename=f"{self.task_name}-best.zip",
            auto_delete_file=False,
            async_enable=False,
        )

    def artifact_payloads(
        self, files: dict[str, Path], config: dict, metrics: dict
    ) -> dict[str, Any]:
        """Attach history files, structured test metrics, and configuration.

        Args:
            files: Named files validated by the artifact store.
            config: Saved training configuration.
            metrics: Saved BERT evaluation result.

        Returns:
            Named file and dictionary payloads uploaded by the shared publisher.
        """
        return {
            **files,
            "test_metrics": asdict(EvalResult.from_dict(metrics)),
            "config": config,
        }

    def test_scalars(self, metrics: dict) -> dict[str, float]:
        """Select the saved test metrics for ClearML scalar reporting.

        Args:
            metrics: Saved test metric dictionary.

        Returns:
            BERT token and entity scalars.
        """
        return EvalResult.from_dict(metrics).scalars
