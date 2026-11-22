import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from tqdm.std import tqdm

from app.datasets.docile.enums import DocileSplit
from app.datasets.docile.models import DocileRecord


@dataclass(init=False)
class DocileDatasetLoader:
    """Read original DocILE documents without changing OCR or annotations.

    Attributes:
        root: Directory containing extracted DocILE files.
    """

    root: Path

    def __init__(self, root: str | Path):
        """Set the extracted source directory.

        Args:
            root: Directory containing original DocILE JSON files and PDFs.
        """
        self.root = Path(root).expanduser().resolve()

    def document_ids(
        self, split: DocileSplit | str = DocileSplit.TRAINVAL
    ) -> list[str]:
        """Read and validate an official split manifest.

        Args:
            split: Official split name to read.

        Returns:
            Unique document IDs in their source order.

        Raises:
            ValueError: The manifest does not contain unique string IDs.
        """
        path = Path(self.root) / f"{DocileSplit(split)}.json"
        with path.open(encoding="utf-8") as file:
            document_ids = json.load(file)
        if not isinstance(document_ids, list) or not all(
            isinstance(value, str) for value in document_ids
        ):
            raise ValueError(f"{path} must contain a list of document IDs.")
        if len(document_ids) != len(set(document_ids)):
            raise ValueError(f"{path} contains duplicate document IDs.")
        return document_ids

    def load(self, document_id: str) -> DocileRecord:
        """Read one document's original JSON files and PDF location.

        Args:
            document_id: File stem from a split manifest.

        Returns:
            Unmodified OCR and annotation objects with their PDF path.

        Raises:
            ValueError: The ID is not a valid file stem.
        """
        if (
            not document_id
            or document_id in {".", ".."}
            or "/" in document_id
            or "\\" in document_id
        ):
            raise ValueError("document_id must be a file stem.")
        root = Path(self.root)
        with (root / "annotations" / f"{document_id}.json").open(
            encoding="utf-8"
        ) as file:
            annotations = json.load(file)
        with (root / "ocr" / f"{document_id}.json").open(encoding="utf-8") as file:
            ocr = json.load(file)
        return DocileRecord(
            document_id, ocr, annotations, str(root / "pdfs" / f"{document_id}.pdf")
        )

    def iter_split(
        self, split: DocileSplit | str = DocileSplit.TRAIN
    ) -> Iterator[DocileRecord]:
        """Read documents in manifest order.

        Args:
            split: Official split to read.

        Yields:
            Original DocILE document records.
        """
        for document_id in self.document_ids(split):
            yield self.load(document_id)

    def to_dataframe(
        self,
        split: DocileSplit | str = DocileSplit.TRAINVAL,
        *,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Read a split into a DataFrame without preprocessing it.

        Args:
            split: Official split to read.
            show_progress: Whether to show document loading progress.

        Returns:
            Rows containing document_id, ocr_data, json_data, and pdf_path.

        Raises:
            RuntimeError: A source document cannot be read.
        """
        rows = []
        for document_id in tqdm(
            self.document_ids(split),
            desc=f"Loading {split}",
            unit="document",
            disable=not show_progress,
        ):
            try:
                rows.append(asdict(self.load(document_id)))
            except Exception as error:
                raise RuntimeError(
                    f"Failed to load document {document_id}: {error}"
                ) from error
        return pd.DataFrame(rows)
