from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
from pipefunc import PipeFunc, Pipeline
from torch.utils.data import DataLoader

from app.extraction.results import EvaluationResult, TrainingResult
from app.pipelines.builder import BuildResult
from app.pipelines.split import DatasetsSplit


@dataclass
class TrainingPipeline:
    """Compose model-specific steps into the shared training dependency graph.

    Attributes:
        random_seed: Seed used for document splitting.
        build: Step preparing records from source document partitions.
        dataloaders: Step constructing split loaders from prepared records.
        train: Model-specific training step; its signature defines dependencies.
        evaluate: Step evaluating the saved best checkpoint.
        publish: Step publishing evaluated artifacts and returning a model ID.
    """

    random_seed: int
    build: Callable[[DatasetsSplit], BuildResult]
    dataloaders: Callable[[BuildResult], dict[str, DataLoader]]
    train: Callable[..., TrainingResult]
    evaluate: Callable[[TrainingResult, dict[str, DataLoader]], EvaluationResult]
    publish: Callable[[EvaluationResult], str]

    def run(self, df: pd.DataFrame) -> str:
        """Execute preparation, training, evaluation, and publication in order.

        Args:
            df: Validated source documents from prepared DocILE Parquet data.

        Returns:
            Registered model ID from the publication step.

        Step failures propagate with PipeFunc's error snapshot. Data produced by
        earlier steps remains an explicit dependency rather than constructor state.
        """
        pipeline = Pipeline(
            [
                PipeFunc(DatasetsSplit.create, output_name="split"),
                PipeFunc(self.build, output_name="build_result"),
                PipeFunc(self.dataloaders, output_name="dataloaders"),
                PipeFunc(self.train, output_name="train_result"),
                PipeFunc(self.evaluate, output_name="evaluation_result"),
                PipeFunc(self.publish, output_name="model_id"),
            ]
        )
        return pipeline.run(
            "model_id", kwargs={"df": df, "random_seed": self.random_seed}
        )
