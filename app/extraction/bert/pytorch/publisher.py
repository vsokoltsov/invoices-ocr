from dataclasses import asdict, dataclass
from pathlib import Path

from clearml import OutputModel, Task

from app.extraction.bert.pytorch.metrics import EvalResult


@dataclass
class ClearMLPublisher:
    config: dict
    history_path: Path
    project_name: str = "ocr"
    task_name: str = "bert-pytorch"

    def run(
        self,
        checkpoint_path: Path,
        test_metrics: EvalResult,
    ) -> str:
        checkpoint_dir = checkpoint_path.expanduser().resolve()
        history_path = self.history_path.expanduser().resolve()

        for path in (
            checkpoint_dir / "model.pt",
            checkpoint_dir / "encoder",
            checkpoint_dir / "tokenizer",
            checkpoint_dir / "validation.json",
            history_path,
        ):
            if not path.exists():
                raise FileNotFoundError(f"Missing artifact: {path}")

        task = Task.init(
            project_name=self.project_name,
            task_name=self.task_name,
            reuse_last_task_id=False,
            output_uri=True,
            auto_connect_frameworks=False,
            auto_connect_arg_parser=False,
        )

        try:
            task.connect(self.config, name="Training")

            model = OutputModel(
                task=task,
                name=f"{self.task_name}-best",
                framework="PyTorch",
                config_dict=self.config,
            )

            # Upload all checkpoint files together for offline loading.
            model.update_weights_package(
                weights_path=str(checkpoint_dir),
                target_filename=f"{self.task_name}-best.zip",
                auto_delete_file=False,
                async_enable=False,
            )

            artifacts = {
                "history.csv": history_path,
                "test_metrics": asdict(test_metrics),
                "config": self.config,
            }

            manifest = checkpoint_dir / "split_manifest.json"
            if manifest.exists():
                artifacts["split_manifest.json"] = manifest

            for name, artifact in artifacts.items():
                uploaded = task.upload_artifact(
                    name=name,
                    artifact_object=artifact,
                    wait_on_upload=True,
                )

                if not uploaded:
                    raise RuntimeError(f"ClearML upload failed: {name}")

            for name in (
                "loss",
                "precision",
                "recall",
                "f1",
                "token_accuracy",
                "token_macro_f1",
            ):
                task.get_logger().report_scalar(
                    title="Test",
                    series=name,
                    value=float(getattr(test_metrics, name)),
                    iteration=0,
                )

            return model.id

        except Exception as error:
            task.mark_failed(status_reason=str(error))
            raise
        finally:
            task.close()
