"""Run shared invoice preprocessing before model-specific label construction."""

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd
from pipefunc import PipeFunc, Pipeline

from app.extraction.order import OCRReadingOrder
from app.extraction.xy_cut import XYCut


class InvoiceSource(Protocol):
    """Define source access and normalization for invoice preprocessing."""

    dataset_path: Path

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
class InvoicePreprocess:
    """Run source adaptation, reading order, and invoice export through PipeFunc.

    Attributes:
        source: Adapter that reads and normalizes the source dataset.
        output_path: Destination Parquet or CSV file.
    """

    source: InvoiceSource
    output_path: Path

    def save(self, frame: pd.DataFrame) -> Path:
        """Write model-independent invoice documents.

        Args:
            frame: Normalized documents with OCR words and rebuilt text offsets.

        Returns:
            Absolute output file path.
        """
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
        return output

    def pipeline(self) -> Pipeline:
        """Create ordered dependencies without downloading or loading a model.

        Returns:
            A PipeFunc graph from raw Parquet to a processed invoice export.
        """
        steps = (*self.source.normalization_steps(), OrderOCRWords(), XYCut())
        functions = [
            PipeFunc(self.source.load, output_name="raw_documents"),
        ]
        previous = "raw_documents"
        for index, step in enumerate(steps):
            output = f"documents_{index}"
            functions.append(
                PipeFunc(step.__call__, output_name=output, renames={"frame": previous})
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
            Path(self.source.dataset_path).expanduser().resolve()
            == Path(self.output_path).expanduser().resolve()
        ):
            raise ValueError(
                "Raw input and preprocessed output must have different paths"
            )
        return self.pipeline().run("output_path", kwargs={})
