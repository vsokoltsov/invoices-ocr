from abc import ABC, abstractmethod
from typing import ClassVar, NamedTuple

import pandas as pd

from app.datasets.split import DatasetsSplit
from app.extraction.artifacts import RunArtifacts


class BuildResult(NamedTuple):
    """Carry prepared and rejected records through the pipeline.

    Attributes:
        records: Prepared training, validation, and test records.
        rejected: Rejection reports for the same source partitions.
    """

    records: DatasetsSplit
    rejected: DatasetsSplit


class BuildPartitions(ABC):
    """Prepare fixed document partitions with model-specific record builders.

    Attributes:
        artifacts: Storage for rejection reports and split membership.
        record_name: Record unit used in empty-partition error messages.
    """

    artifacts: RunArtifacts
    record_name: ClassVar[str]

    @abstractmethod
    def prepare_partition(
        self, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare one source partition.

        Args:
            frame: Source documents belonging to one split.

        Returns:
            Accepted records and rejection report as DataFrames.
        """
        ...

    @abstractmethod
    def save_manifest(self, split: DatasetsSplit, rejected: DatasetsSplit) -> None:
        """Persist model-specific document membership and rejection metadata.

        Args:
            split: Original document partitions.
            rejected: Rejection reports produced for each partition.
        """
        ...

    def build(self, split: DatasetsSplit) -> BuildResult:
        """Prepare every split without changing document membership.

        Args:
            split: Source document partitions produced by the split step.

        Returns:
            Prepared records and rejection reports for all three splits.

        Raises:
            ValueError: A partition contains no accepted records.
        """
        records, rejected = {}, {}
        for name in ("train", "val", "test"):
            accepted, failures = self.prepare_partition(getattr(split, name))
            records[name], rejected[name] = accepted, failures
            self.artifacts.save_rejections(name, failures)
            if accepted.empty:
                raise ValueError(f"No accepted {self.record_name} in {name}")
        result = BuildResult(DatasetsSplit(**records), DatasetsSplit(**rejected))
        self.save_manifest(split, result.rejected)
        return result
