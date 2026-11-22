import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from tqdm.std import tqdm

from app.datasets.docile.enums import DocileSplit
from app.datasets.docile.models import DocileRecord


@dataclass(init=False)
class DocileDatasetLoader:
    """Read original DocILE documents and save raw Parquet snapshots.

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

    def extract(self, split: DocileSplit | str = DocileSplit.TRAINVAL) -> Path:
        """Extract the requested archive when its source files are absent.

        Args:
            split: Official split whose source files are required.

        Returns:
            Absolute directory containing extracted source files.

        Raises:
            FileNotFoundError: The required archive was not downloaded.
            ValueError: An archive contains a path outside the source directory.
        """
        split = DocileSplit(split)
        root = self.root
        if (
            (root / f"{split}.json").is_file()
            and (root / "annotations").is_dir()
            and (root / "ocr").is_dir()
        ):
            return root
        name = "test" if split == DocileSplit.TEST else "annotated-trainval"
        path = root / f"{name}.zip"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Run datasets download docile first."
            )
        with ZipFile(path) as archive:
            for member in archive.infolist():
                if not (root / member.filename).resolve().is_relative_to(root):
                    raise ValueError(
                        f"Archive entry is outside the dataset directory: {member.filename}"
                    )
            archive.extractall(root)
        return root

    def add_split_membership(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Attach official manifest membership without changing document data.

        Args:
            frame: Original source records selected for storage.

        Returns:
            A copy with available official split metadata.

        Raises:
            ValueError: Official split assignments overlap or are incomplete.
        """
        membership = {}
        for split in (DocileSplit.TRAIN, DocileSplit.VAL, DocileSplit.TEST):
            if not (self.root / f"{split}.json").is_file():
                continue
            for document_id in self.document_ids(split):
                if document_id in membership:
                    raise ValueError(
                        f"Document appears in both {membership[document_id]} and {split}: {document_id}"
                    )
                membership[document_id] = split
        result = frame.copy()
        if membership:
            result["split"] = result["document_id"].map(membership)
            if result["split"].isna().any():
                raise ValueError("Some documents have no official split assignment")
        return result

    def save(
        self, output_path: str | Path, split: DocileSplit | str = DocileSplit.TRAINVAL
    ) -> Path:
        """Extract local archives, read original records, and save raw Parquet.

        OCR and annotations are stored as JSON strings. This preserves nested
        source data without selecting geometry, reordering words, or changing labels.
        PDFs remain separate files referenced by pdf_path.

        Args:
            output_path: Destination raw Parquet file.
            split: Official source split to load.

        Returns:
            Absolute path of the raw snapshot.

        Raises:
            ValueError: The output extension or selected records are invalid.
        """
        output = Path(output_path).expanduser().resolve()
        if output.suffix.lower() != ".parquet":
            raise ValueError("Raw DocILE output must have a .parquet extension")
        self.extract(split)
        frame = self.add_split_membership(self.to_dataframe(split))
        if frame.empty:
            raise ValueError("Selected split contains no documents")
        for column in ("ocr_data", "json_data"):
            frame[column] = frame[column].map(
                lambda value: json.dumps(value, ensure_ascii=False)
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output, index=False)
        return output
