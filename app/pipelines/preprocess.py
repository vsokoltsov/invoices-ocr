import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from inspect import isroutine
from typing import Any, ClassVar, NamedTuple

import numpy as np
import pandas as pd
from pipefunc import PipeFunc, Pipeline

from app.datasets.split import DatasetsSplit
from app.extraction.artifacts import RunArtifacts


@dataclass
class DataFramePipeline:
    """Run ordered DataFrame steps with explicit PipeFunc dependencies.

    Attributes:
        steps: Callables that each accept and return a DataFrame named frame.
    """

    steps: Sequence[Callable[[pd.DataFrame], pd.DataFrame]]

    def create(self) -> Pipeline:
        """Create a PipeFunc graph without processing data.

        Returns:
            A graph with one named output for each step.

        Raises:
            ValueError: No preprocessing steps are configured.
        """
        if not self.steps:
            raise ValueError("A preprocessing pipeline needs at least one step")
        functions = []
        previous = "frame"
        for index, step in enumerate(self.steps):
            output = f"frame_{index}"
            function = step if isroutine(step) else step.__call__
            functions.append(
                PipeFunc(function, output_name=output, renames={"frame": previous})
            )
            previous = output
        return Pipeline(functions)

    def run(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Process a frame through every configured step.

        Args:
            frame: Source documents or intermediate records.

        Returns:
            The DataFrame produced by the last step.
        """
        return self.create().run(
            f"frame_{len(self.steps) - 1}", kwargs={"frame": frame}
        )


class DecodeDocuments:
    """Restore Python containers from JSON strings and Parquet arrays."""

    @classmethod
    def python_value(cls, value: Any) -> Any:
        """Convert nested NumPy values without changing their contents.

        Args:
            value: A cell or nested value read from a prepared dataset.

        Returns:
            Equivalent Python lists, dictionaries, scalars, or unchanged values.
        """
        if isinstance(value, np.ndarray):
            return cls.python_value(value.tolist())
        if isinstance(value, list):
            return [cls.python_value(item) for item in value]
        if isinstance(value, dict):
            return {key: cls.python_value(item) for key, item in value.items()}
        if isinstance(value, np.generic):
            return value.item()
        return value

    def __call__(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Decode annotations and OCR words into new containers.

        Args:
            frame: Documents with JSON strings or nested Python/NumPy values.

        Returns:
            A copy with decoded annotation and OCR columns.
        """
        result = frame.copy()
        for column in ("json_data", "text_with_boxes"):
            if column in result:
                result[column] = result[column].map(
                    lambda value: self.python_value(
                        json.loads(value) if isinstance(value, str) else value
                    )
                )
        return result


class DocumentStep(ABC):
    """Add one intermediate column and preserve per-document validation errors.

    Attributes:
        output_column: Name of the column produced by this step.
    """

    output_column: ClassVar[str]

    @abstractmethod
    def process(self, row: pd.Series) -> Any:
        """Process one document.

        Args:
            row: Source document and outputs from preceding steps.

        Returns:
            The value to store in output_column.

        Raises:
            ValueError: The document must be rejected.
        """
        ...

    def __call__(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Process documents that preceding steps have not rejected.

        Args:
            frame: Documents with an optional reason column.

        Returns:
            A copy with the step output and a reason for each rejected document.
        """
        result = frame.copy()
        outputs, reasons = [], []
        for _, row in frame.iterrows():
            reason = row.get("reason", "")
            output = None
            if not reason:
                try:
                    output = self.process(row)
                except ValueError as error:
                    reason = str(error)
            outputs.append(output)
            reasons.append(reason)
        result[self.output_column] = outputs
        result["reason"] = reasons
        return result


class PreprocessResult(NamedTuple):
    """Carry prepared and rejected records through the pipeline.

    Attributes:
        records: Prepared training, validation, and test records.
        rejected: Rejection reports for the same source partitions.
    """

    records: DatasetsSplit
    rejected: DatasetsSplit


class PreprocessPartitions(ABC):
    """Prepare fixed document partitions with model-specific preprocessing.

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

    def run(self, split: DatasetsSplit) -> PreprocessResult:
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
        result = PreprocessResult(DatasetsSplit(**records), DatasetsSplit(**rejected))
        self.save_manifest(split, result.rejected)
        return result
