from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from clearml import PipelineController

from app.aggregates.bert.params import BertPytorchParams
from app.aggregates.layoutlm.params import LayoutLMTrainingParams
from app.aggregates.pipeline import (
    BertRunMode,
    ClearMLPipelineParams,
    PipelineExecution,
)
from app.pipelines.clearml.stages import (
    evaluate_model,
    prepare_model,
    publish_model,
    train_model,
)


@dataclass(frozen=True)
class ModelRun:
    """Define one model branch in a ClearML pipeline.

    Attributes:
        name: Unique branch name used to prefix stage task names.
        model: Component type used by the workers.
        params: Model settings containing absolute input and output paths.
    """

    name: str
    model: str
    params: dict[str, Any]


@dataclass
class ClearMLTrainingPipeline:
    """Build real ClearML tasks with explicit filesystem and artifact dependencies.

    Attributes:
        name: Pipeline name shown in ClearML.
        runs: Model branches executed sequentially to avoid GPU contention.
        settings: Project, execution mode, and agent queue settings.
    """

    name: str
    runs: tuple[ModelRun, ...]
    settings: ClearMLPipelineParams

    def create(self) -> PipelineController:
        """Register preparation, training, evaluation, and publication tasks.

        Returns:
            Configured controller; no model training has started yet.
        """
        controller = PipelineController(
            name=self.name,
            project=self.settings.project,
            version="1.0",
            abort_on_failure=True,
            output_uri=True,
            skip_global_imports=True,
            packages=False,
            enable_local_imports=True,
        )
        controller.set_default_execution_queue(self.settings.queue)
        previous = None
        for run in self.runs:
            prepare, train, evaluate, publish = (
                f"{run.name}-{stage}"
                for stage in ("prepare", "train", "evaluate", "publish")
            )
            inputs = {"model": run.model, "params": run.params}
            prepared = "${" + prepare + ".prepared}"
            trained = "${" + train + ".trained}"
            evaluated = "${" + evaluate + ".evaluated}"
            controller.add_function_step(
                name=prepare,
                function=prepare_model,
                function_kwargs=inputs,
                function_return=["prepared"],
                parents=[previous] if previous else None,
                cache_executed_step=False,
            )
            controller.add_function_step(
                name=train,
                function=train_model,
                function_kwargs={**inputs, "prepared": prepared},
                function_return=["trained"],
                cache_executed_step=False,
            )
            controller.add_function_step(
                name=evaluate,
                function=evaluate_model,
                function_kwargs={**inputs, "prepared": prepared, "trained": trained},
                function_return=["evaluated"],
                cache_executed_step=False,
            )
            controller.add_function_step(
                name=publish,
                function=publish_model,
                function_kwargs={**inputs, "evaluated": evaluated},
                function_return=["model_id"],
                cache_executed_step=False,
            )
            previous = publish
        return controller

    def run(self) -> PipelineController:
        """Run the controller locally with subprocess or agent-backed stages.

        Returns:
            Completed controller with task and branch results available in ClearML.

        Raises:
            RuntimeError: A pipeline stage failed or execution was aborted.

        Agent execution requires the same dataset, PDFs, cache, and run paths on
        every worker. The project and its dependencies must be installed there.
        """
        controller = self.create()
        controller.start_locally(
            run_pipeline_steps_locally=self.settings.execution
            == PipelineExecution.LOCAL
        )
        if not controller.is_successful():
            raise RuntimeError(f"ClearML pipeline failed: {self.name}")
        return controller


@dataclass
class BertClearMLPipeline:
    """Select invoice, line-item, or both BERT model branches.

    Attributes:
        params: BERT hyperparameters and output paths.
        settings: ClearML execution and project configuration.
        mode: model, line-items, or both. Both uses independent output directories.
    """

    params: BertPytorchParams
    settings: ClearMLPipelineParams
    mode: BertRunMode = BertRunMode.MODEL

    def build(self) -> ClearMLTrainingPipeline:
        """Create isolated model branches with a shared split seed.

        Returns:
            Controller configuration for the requested BERT models.
        """
        selections = {
            BertRunMode.MODEL: (("bert-model", False),),
            BertRunMode.LINE_ITEMS: (("bert-items", True),),
            BertRunMode.BOTH: (("bert-model", False), ("bert-items", True)),
        }
        runs = []
        for name, items in selections[self.mode]:
            params = replace(self.params, line_items_only=items)
            if self.mode == BertRunMode.BOTH:
                branch = "items" if items else "model"
                history = Path(params.result_filename).expanduser()
                params = replace(
                    params,
                    checkpoint_dir=str(Path(params.checkpoint_dir) / branch),
                    result_filename=str(history.parent / branch / history.name),
                    diagnostic_dir=str(Path(params.diagnostic_dir) / branch)
                    if params.diagnostic_dir
                    else None,
                )
            values = asdict(params)
            for key in (
                "dataset_path",
                "checkpoint_dir",
                "result_filename",
                "diagnostic_dir",
            ):
                if values[key]:
                    values[key] = str(Path(values[key]).expanduser().resolve())
            runs.append(ModelRun(name, "bert", values))
        return ClearMLTrainingPipeline("bert-training", tuple(runs), self.settings)

    def run(self) -> PipelineController:
        """Execute selected BERT branches and return their ClearML controller."""
        return self.build().run()


@dataclass
class LayoutLMClearMLPipeline:
    """Configure a LayoutLM branch for ClearML execution.

    Attributes:
        params: LayoutLM training, preprocessing, and output settings.
        settings: ClearML project and execution settings.
    """

    params: LayoutLMTrainingParams
    settings: ClearMLPipelineParams

    def build(self) -> ClearMLTrainingPipeline:
        """Create a LayoutLM branch with absolute shared filesystem paths.

        Returns:
            Controller configuration for LayoutLM preparation through publication.
        """
        values = asdict(self.params)
        # Keep worker arguments JSON-native; no controller enum classes are needed.
        values["device"] = str(self.params.device)
        for key in ("dataset_path", "output_dir", "checkpoint_path", "images_path"):
            values[key] = str(Path(values[key]).expanduser().resolve())
        return ClearMLTrainingPipeline(
            "layoutlm-training",
            (ModelRun("layoutlm", "layoutlm", values),),
            self.settings,
        )

    def run(self) -> PipelineController:
        """Execute LayoutLM stages and return the completed ClearML controller."""
        return self.build().run()
