from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from app.pipelines.preprocess import PreprocessResult


@dataclass(kw_only=True)
class DataLoaders(ABC):
    """Construct split loaders with model-specific dataset creation.

    Attributes:
        device: Execution device; CUDA enables pinned memory.
        batch_size: Number of samples per batch.
        max_length_tokens: Token window length passed to datasets.
        num_workers: Number of worker processes used by each loader.
        collate_fn: Optional collator for variable-length or metadata-rich samples.
        random_seed: Optional independent seed for training shuffle order.
    """

    device: torch.device
    batch_size: int
    max_length_tokens: int = 512
    num_workers: int = 0
    collate_fn: Callable | None = None
    random_seed: int | None = None

    @abstractmethod
    def create_dataset(self, frame: pd.DataFrame, *, training: bool) -> Dataset:
        """Create a dataset for one partition.

        Args:
            frame: Prepared records for this partition.
            training: Whether evaluation-only metadata can be omitted.

        Returns:
            Dataset accepted by the configured collator.
        """
        ...

    def loader_options(self, *, training: bool) -> dict[str, Any]:
        """Build shared loader settings and an optional shuffle generator.

        Args:
            training: Whether to enable shuffling and the configured seed.

        Returns:
            Keyword arguments for torch DataLoader.
        """
        options = {
            "batch_size": self.batch_size,
            "shuffle": training,
            "num_workers": self.num_workers,
            "pin_memory": self.device.type == "cuda",
            "collate_fn": self.collate_fn,
        }
        if training and self.random_seed is not None:
            options["generator"] = torch.Generator().manual_seed(self.random_seed)
        return options

    def build(self, preprocess_result: PreprocessResult) -> dict[str, DataLoader]:
        """Create shuffled training and ordered evaluation loaders.

        Args:
            preprocess_result: Prepared records produced by the preprocessing step.

        Returns:
            Loaders keyed by train, val, and test.
        """
        return {
            name: self.create_loader(
                getattr(preprocess_result.records, name), training=name == "train"
            )
            for name in ("train", "val", "test")
        }

    def create_loader(self, frame: pd.DataFrame, *, training: bool) -> DataLoader:
        """Build one partition without loading unrelated training records.

        Args:
            frame: Prepared records for the selected partition.
            training: Whether to enable training shuffle and omit scoring metadata.

        Returns:
            Configured loader for this partition.
        """
        return DataLoader(
            self.create_dataset(frame, training=training),
            **self.loader_options(training=training),
        )
