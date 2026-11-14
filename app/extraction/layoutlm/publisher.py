import json
from dataclasses import dataclass
from pathlib import Path

from clearml import OutputModel, Task


@dataclass
class ClearMLPublisher:
    project_name: str = "ocr"
    task_name: str = "layoutlmv3"

    def run(self, checkpoint_path: Path, output_dir: Path) -> str:
        checkpoint_path = checkpoint_path.resolve()
        output_dir = output_dir.resolve()

        config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
        metrics = json.loads(
            (output_dir / "test_metrics.json").read_text(encoding="utf-8")
        )

        task = Task.init(
            project_name=self.project_name,
            task_name=self.task_name,
            reuse_last_task_id=False,
            output_uri=True,
            auto_connect_frameworks=False,
            auto_connect_arg_parser=False,
        )
        try:
            task.connect(config, name="Training")

            model = OutputModel(
                task=task,
                name="layoutlmv3-best",
                framework="PyTorch",
                config_dict=config,
            )
            model.update_weights(
                weights_filename=str(checkpoint_path),
                auto_delete_file=False,
                async_enable=False,
            )

            for name in (
                "test_metrics.json",
                "history.csv",
                "config.json",
                "split_manifest.csv",
                "processor",
            ):
                uploaded = task.upload_artifact(
                    name=name,
                    artifact_object=output_dir / name,
                    wait_on_upload=True,
                )
                if not uploaded:
                    raise RuntimeError(f"ClearML upload failed: {name}")

            for name in ("entity_precision", "entity_recall", "entity_f1"):
                task.get_logger().report_scalar(
                    title="Test",
                    series=name,
                    value=float(metrics[name]),
                    iteration=0,
                )

            return model.id
        except Exception as error:
            task.mark_failed(status_reason=str(error))
            raise
        finally:
            task.close()
