from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clearml import OutputModel, Task

from app.extraction.artifacts import RunArtifacts


@dataclass
class ClearMLPublisher(ABC):
    """Publish a completed run with model-specific weight and metric adapters.

    Attributes:
        artifacts: Validated run storage and output paths.
        project_name: ClearML project receiving the task.
        task_name: Task name and prefix used for the registered model.
    """

    artifacts: RunArtifacts
    project_name: str = "ocr"
    task_name: str = "extraction"

    @abstractmethod
    def upload_weights(self, model: OutputModel) -> None:
        """Upload model-specific weights synchronously.

        Args:
            model: ClearML output model to receive the checkpoint.
        """
        ...

    @abstractmethod
    def test_scalars(self, metrics: dict) -> dict[str, float]:
        """Select test metrics suitable for scalar reporting.

        Args:
            metrics: Saved test metrics decoded from JSON.

        Returns:
            Series names mapped to numeric values.
        """
        ...

    def artifact_payloads(
        self, files: dict[str, Path], config: dict, metrics: dict
    ) -> dict[str, Any]:
        """Select the artifacts attached to the ClearML task.

        Args:
            files: Named artifact paths validated by the storage component.
            config: Saved training configuration; available to subclass implementations.
            metrics: Saved test metrics; available to subclass implementations.

        Returns:
            Artifact names mapped to files or serializable objects.
        """
        return dict(files)

    def run(self) -> str:
        """Create a task and publish configuration, weights, artifacts, and test scalars.

        Validates local outputs before creating a remote task. Uploads are synchronous.
        Publication errors mark the created task as failed and propagate to the caller;
        the task is closed on both success and failure. SDK errors also propagate.

        Returns:
            Registered ClearML model ID.

        Raises:
            FileNotFoundError: A required local artifact is missing.
            RuntimeError: ClearML reports a failed artifact upload.
        """
        # Fail on incomplete local outputs before creating a remote task.
        self.artifacts.publication_files()
        task = Task.init(
            project_name=self.project_name,
            task_name=self.task_name,
            reuse_last_task_id=False,
            output_uri=True,
            auto_connect_frameworks=False,
            auto_connect_arg_parser=False,
        )
        try:
            return self.publish_to(task)
        except Exception as error:
            # Preserve the upload failure in both ClearML and the pipeline caller.
            task.mark_failed(status_reason=str(error))
            raise
        finally:
            task.close()

    def publish_to(self, task: Task) -> str:
        """Publish into an existing task without changing its lifecycle.

        Args:
            task: Active ClearML task owned by a pipeline worker or caller.

        Returns:
            Registered ClearML model ID.

        Raises:
            RuntimeError: An artifact upload fails.
            FileNotFoundError: A required local output is absent.

        The caller owns task completion and failure reporting. This allows the
        pipeline worker to upload its return value before closing its task.
        """
        files = self.artifacts.publication_files()
        config = self.artifacts.load_config()
        metrics = self.artifacts.load_metrics()
        task.connect(config, name="Training")
        model = OutputModel(
            task=task,
            name=f"{self.task_name}-best",
            framework="PyTorch",
            config_dict=config,
        )
        self.upload_weights(model)
        for name, artifact in self.artifact_payloads(files, config, metrics).items():
            if not task.upload_artifact(
                name=name, artifact_object=artifact, wait_on_upload=True
            ):
                raise RuntimeError(f"ClearML upload failed: {name}")
        for name, value in self.test_scalars(metrics).items():
            task.get_logger().report_scalar(
                title="Test", series=name, value=value, iteration=0
            )
        return model.id
