"""Run shared invoice preprocessing before model-specific label construction."""

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

import pandas as pd
from pipefunc import PipeFunc, Pipeline

from app.datasets.source import DatasetSource
from app.extraction.order import OCRReadingOrder
from app.extraction.xy_cut import XYCut


class InvoiceSource(Protocol):
    """Define source access and normalization for invoice preprocessing."""

    dataset_path: str | Path

    def load(self) -> pd.DataFrame:
        """Return document records read from a raw Parquet snapshot."""
        ...

    def normalization_steps(self) -> Sequence[Callable[[pd.DataFrame], pd.DataFrame]]:
        """Return source-specific steps that produce normalized invoice words."""
        ...


class OrderOCRWords:
    """Apply visual-row reading order and rebuild text and character offsets."""

    def __call__(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Order each document by visual row and horizontal position.

        Args:
            frame: Documents containing flattened OCRWord records.

        Returns:
            A copy with ocr_text and ordered OCR word dictionaries.
        """
        result = frame.copy()
        documents = [
            OCRReadingOrder().build(words) for words in frame["text_with_boxes"]
        ]
        result["ocr_text"] = [text for text, _ in documents]
        result["text_with_boxes"] = [
            [asdict(word) for word in words] for _, words in documents
        ]
        return result


@dataclass
class PreprocessProgress:
    """Report a DataFrame step without changing its output or error handling.

    Attributes:
        name: Step name shown in the console.
        step: Transformation that accepts and returns a DataFrame.
    """

    name: str
    step: Callable[[pd.DataFrame], pd.DataFrame]

    def __call__(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Print step progress and elapsed time, then return its output.

        Args:
            frame: Input documents for this step.

        Returns:
            The transformation's output DataFrame.
        """
        print(f"[preprocess] Running {self.name}...", flush=True)
        started = perf_counter()
        try:
            result = self.step(frame)
        except Exception:
            print(
                f"[preprocess] {self.name} failed after {perf_counter() - started:.1f}s",
                flush=True,
            )
            raise
        print(
            f"[preprocess] {self.name}: {len(result):,} documents "
            f"in {perf_counter() - started:.1f}s",
            flush=True,
        )
        return result


@dataclass
class InvoicePreprocess:
    """Run source adaptation, reading order, and invoice export through PipeFunc.

    Attributes:
        source: Adapter that reads and normalizes the source dataset.
        output_path: Destination Parquet or CSV file.
    """

    source: InvoiceSource
    output_path: Path

    def load(self) -> pd.DataFrame:
        """Read the raw snapshot and report the source document count.

        Returns:
            Original document records returned by the source adapter.
        """
        print("[preprocess] Loading raw Parquet...", flush=True)
        started = perf_counter()
        frame = self.source.load()
        print(
            f"[preprocess] Loaded {len(frame):,} documents "
            f"in {perf_counter() - started:.1f}s",
            flush=True,
        )
        return frame

    def save(self, frame: pd.DataFrame) -> Path:
        """Write model-independent invoice documents.

        Args:
            frame: Normalized documents with OCR words and rebuilt text offsets.

        Returns:
            Absolute output file path.
        """
        print("[preprocess] Saving processed export...", flush=True)
        started = perf_counter()
        result = frame.copy()
        output = Path(self.output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        for column in ("json_data", "annotation_corrections"):
            if column in result:
                result[column] = result[column].map(
                    lambda value: json.dumps(value, ensure_ascii=False)
                )
        if output.suffix.lower() == ".parquet":
            result.to_parquet(output, index=False)
        else:
            result["text_with_boxes"] = result["text_with_boxes"].map(
                lambda value: json.dumps(value, ensure_ascii=False)
            )
            result.to_csv(output, index=False)
        word_counts = frame["text_with_boxes"].map(len)
        empty_count = int(word_counts.eq(0).sum())
        empty_percentage = 100 * empty_count / len(frame) if len(frame) else 0.0
        print(
            f"[preprocess] Exported {len(frame):,} documents; "
            f"{int(word_counts.sum()):,} OCR words; "
            f"{empty_count:,} documents without OCR words ({empty_percentage:.1f}%) "
            f"in {perf_counter() - started:.1f}s",
            flush=True,
        )
        print(
            "[preprocess] Model rejection checks are not run here; "
            "counts and reasons are reported during train/pipeline preparation.",
            flush=True,
        )
        return output

    def pipeline(self) -> Pipeline:
        """Create ordered dependencies without downloading or loading a model.

        Returns:
            A PipeFunc graph from raw Parquet to a processed invoice export.
        """
        steps = (*self.source.normalization_steps(), OrderOCRWords(), XYCut())
        functions = [
            PipeFunc(self.load, output_name="raw_documents"),
        ]
        previous = "raw_documents"
        for index, step in enumerate(steps):
            output = f"documents_{index}"
            progress = PreprocessProgress(type(step).__name__, step)
            functions.append(
                PipeFunc(
                    progress.__call__, output_name=output, renames={"frame": previous}
                )
            )
            previous = output
        functions.append(
            PipeFunc(self.save, output_name="output_path", renames={"frame": previous})
        )
        return Pipeline(functions)

    def run(self) -> Path:
        """Execute the shared invoice preprocessing pipeline.

        Returns:
            Absolute path of the exported dataset.

        Raises:
            ValueError: The output extension is unsupported or input would be overwritten.
        """
        if Path(self.output_path).suffix.lower() not in {".csv", ".parquet"}:
            raise ValueError("Invoice output must have a .csv or .parquet extension")
        if (
            not DatasetSource(self.source.dataset_path).is_clearml
            and Path(self.source.dataset_path).expanduser().resolve()
            == Path(self.output_path).expanduser().resolve()
        ):
            raise ValueError(
                "Raw input and preprocessed output must have different paths"
            )
        started = perf_counter()
        output = self.pipeline().run("output_path", kwargs={})
        print(f"[preprocess] Finished in {perf_counter() - started:.1f}s", flush=True)
        return output
