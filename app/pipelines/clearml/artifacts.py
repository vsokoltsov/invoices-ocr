from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from app.datasets.split import DatasetsSplit
from app.pipelines.preprocess import PreprocessResult


@dataclass(frozen=True)
class PreparedData:
    """Own files passed between preparation, training, and evaluation processes.

    Attributes:
        directory: Shared filesystem directory visible to every pipeline worker.
    """

    directory: Path

    @classmethod
    def for_run(cls, output_dir: Path) -> "PreparedData":
        """Keep prepared records beside the run's publishable outputs.

        Args:
            output_dir: Model-specific run output directory.

        Returns:
            Storage in a sibling directory ending in -prepared. BERT checkpoint
            packages therefore exclude intermediate dataset copies.
        """
        return cls(output_dir.with_name(output_dir.name + "-prepared"))

    @staticmethod
    def _python_value(value):
        """Restore nested Python containers after a Parquet round trip.

        Args:
            value: Cell or nested value returned by the Parquet reader.

        Returns:
            Equivalent native lists, dictionaries, scalars, or unchanged values.
        """
        if isinstance(value, np.ndarray):
            return PreparedData._python_value(value.tolist())
        if isinstance(value, list):
            return [PreparedData._python_value(item) for item in value]
        if isinstance(value, dict):
            return {
                key: PreparedData._python_value(item) for key, item in value.items()
            }
        if isinstance(value, np.generic):
            return value.item()
        return value

    def save(self, split: DatasetsSplit, result: PreprocessResult) -> str:
        """Persist preparation outputs without serializing models or dataloaders.

        Args:
            split: Source partitions; only IDs and OCR words are retained.
            result: Prepared records and structured rejection reports.

        Returns:
            Absolute prepared-data directory.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        for name in ("train", "val", "test"):
            getattr(result.records, name).to_parquet(
                self.directory / f"{name}.parquet", index=False
            )
            source = getattr(split, name)
            columns = [
                column
                for column in ("document_id", "text_with_boxes")
                if column in source
            ]
            source[columns].to_parquet(
                self.directory / f"{name}_source.parquet", index=False
            )
            getattr(result.rejected, name).to_json(
                self.directory / f"{name}_rejected.json",
                orient="table",
                index=False,
            )
        return str(self.directory.resolve())

    def load(self) -> tuple[DatasetsSplit, PreprocessResult]:
        """Restore prepared records for a fresh training or evaluation process.

        Returns:
            Source document partitions and the full preparation result.

        Raises:
            FileNotFoundError: Preparation did not produce a required file.
        """
        source, records, rejected = {}, {}, {}
        for name in ("train", "val", "test"):
            # Dataset constructors expect Python lists, not Arrow-derived ndarrays.
            records[name] = pd.read_parquet(self.directory / f"{name}.parquet").map(
                self._python_value
            )
            source[name] = pd.read_parquet(self.directory / f"{name}_source.parquet")
            text = (self.directory / f"{name}_rejected.json").read_text(
                encoding="utf-8"
            )
            rejected[name] = pd.read_json(StringIO(text), orient="table")
        return DatasetsSplit(**source), PreprocessResult(
            DatasetsSplit(**records), DatasetsSplit(**rejected)
        )
