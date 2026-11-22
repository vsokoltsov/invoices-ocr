from abc import ABC, abstractmethod

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import set_seed

from app.datasets.split import DatasetsSplit
from app.extraction.artifacts import RunArtifacts
from app.extraction.results import TrainingResult
from app.pipelines.dataloader import DataLoaders
from app.pipelines.evaluate import EvaluateCheckpoint
from app.pipelines.pipeline import TrainingPipeline
from app.pipelines.preprocess import PreprocessPartitions, PreprocessResult
from app.pipelines.publish import PublishModel


class ModelPipeline(ABC):
    """Share local orchestration while exposing lazy components to ClearML tasks.

    Attributes:
        artifacts: Model-specific output storage.
        device: Resolved execution device.
        preprocess_step: Model-specific record preparation.
        loader_step: Model-specific dataset and loader construction.
        evaluation_step: Saved-checkpoint test evaluation.
        publisher_step: Model-specific ClearML publication.
    """

    artifacts: RunArtifacts
    device: torch.device
    preprocess_step: PreprocessPartitions
    loader_step: DataLoaders
    evaluation_step: EvaluateCheckpoint
    publisher_step: PublishModel

    @property
    @abstractmethod
    def random_seed(self) -> int:
        """Return the seed used for document splitting and training."""
        ...

    @abstractmethod
    def load_source(self) -> pd.DataFrame:
        """Return validated source documents without initializing model weights."""
        ...

    @abstractmethod
    def prepare_run(self) -> None:
        """Create run storage and save fixed configuration and preprocessing files."""
        ...

    @abstractmethod
    def train_prepared(
        self,
        dataloaders: dict[str, DataLoader],
        preprocess_result: PreprocessResult,
        split: DatasetsSplit,
    ) -> TrainingResult:
        """Train using prepared records, loaders, and original document partitions.

        Args:
            dataloaders: Training and evaluation loaders.
            preprocess_result: Prepared model-specific records.
            split: Original source document partitions.

        Returns:
            Saved best-checkpoint and history locations.
        """
        ...

    def run(self) -> str:
        """Run the shared PipeFunc graph on the current machine.

        Returns:
            Published ClearML model ID.

        Validation happens before model initialization. ClearML orchestration uses
        the same components in separate processes and exchanges prepared files.
        """
        set_seed(self.random_seed)
        frame = self.load_source()
        # Validate explicit accelerator requests before preparation has side effects.
        _ = self.device
        self.prepare_run()
        return TrainingPipeline(
            random_seed=self.random_seed,
            preprocess=self.preprocess_step.run,
            dataloaders=self.loader_step.build,
            train=self.train_prepared,
            evaluate=self.evaluation_step.eval,
            publish=self.publisher_step.publish,
        ).run(frame)
