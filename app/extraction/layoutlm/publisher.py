from dataclasses import dataclass

from clearml import OutputModel, Task

from app.extraction.layoutlm.artifacts import LayoutLMArtifacts


@dataclass
class ClearMLPublisher:
    artifacts: LayoutLMArtifacts
    project_name: str = "ocr"
    task_name: str = "layoutlmv3"

    def run(self) -> str:
        files = self.artifacts.publication_files()
        config = self.artifacts.load_config()
        metrics = self.artifacts.load_metrics()

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
                weights_filename=str(self.artifacts.checkpoint_path),
                auto_delete_file=False,
                async_enable=False,
            )

            for name, path in files.items():
                uploaded = task.upload_artifact(
                    name=name,
                    artifact_object=path,
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
