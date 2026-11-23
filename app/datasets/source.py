"""Resolve local paths and versioned ClearML dataset inputs."""

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pandas as pd
from clearml import Dataset


@dataclass(frozen=True)
class ResolvedDataset:
    """Hold a local input path and its optional ClearML dataset root.

    Attributes:
        path: Local file or image-directory path.
        root: ClearML cache root; None for ordinary local paths.
    """

    path: Path
    root: Path | None = None

    def read_parquet(self, **kwargs) -> pd.DataFrame:
        """Read a Parquet input and resolve remote PDF references in memory.

        Args:
            **kwargs: Options passed to pandas.read_parquet.

        Returns:
            Records with local PDF paths when loaded from ClearML. Cached files
            and local source files remain unchanged.

        Raises:
            ValueError: Stored PDF references do not match the dataset layout.
        """
        frame = pd.read_parquet(self.path, **kwargs)
        if self.root is not None and "pdf_path" in frame:
            required = {"document_id", "pdf_path"}
            if not required <= set(frame.columns):
                raise ValueError("ClearML PDF references require document_id")
            references = [f"pdfs/{value}.pdf" for value in frame["document_id"]]
            if frame["pdf_path"].tolist() != references:
                raise ValueError("Stored PDF paths must use pdfs/<document_id>.pdf")
            if any(
                "/" in str(value) or "\\" in str(value) or str(value) in {".", ".."}
                for value in frame["document_id"]
            ):
                raise ValueError("Invalid document ID in ClearML dataset")
            paths = [self.root / reference for reference in references]
            if missing := next((path for path in paths if not path.is_file()), None):
                raise FileNotFoundError(f"Missing dataset PDF: {missing}")
            frame["pdf_path"] = [str(path) for path in paths]
        return frame


@dataclass(frozen=True)
class DatasetSource:
    """Resolve an existing local input or a clearml:// dataset reference.

    Attributes:
        location: Local path or clearml://<dataset-id>/<member> reference.
    """

    location: str | Path

    @property
    def is_clearml(self) -> bool:
        """Return whether the input requests ClearML dataset resolution."""
        return str(self.location).startswith("clearml:")

    def resolve(self, default_member: str) -> ResolvedDataset:
        """Resolve an input on the machine that consumes it.

        Args:
            default_member: raw.parquet, preprocessed.parquet, or images when the
                reference omits its member name.

        Returns:
            Local input and its optional read-only ClearML cache root.

        Raises:
            ValueError: The reference is invalid or the dataset is not final.
            FileNotFoundError: The requested member is absent.
        """
        if not self.is_clearml:
            return ResolvedDataset(Path(self.location).expanduser().resolve())
        reference = urlsplit(str(self.location))
        member = unquote(reference.path).removeprefix("/") or default_member
        if (
            not str(self.location).startswith("clearml://")
            or not reference.netloc
            or reference.username is not None
            or ":" in reference.netloc
            or reference.query
            or reference.fragment
            or member not in {"raw.parquet", "preprocessed.parquet", "images"}
            or (default_member == "images") != (member == "images")
        ):
            raise ValueError(
                "Use clearml://<dataset-id>/raw.parquet, preprocessed.parquet, or images"
            )
        dataset = Dataset.get(dataset_id=reference.netloc)
        if not dataset.is_final():
            raise ValueError("The ClearML dataset must be finalized before loading")
        files = dataset.list_files()
        if member == "images":
            selected = sorted(
                path
                for path in files
                if path.startswith("images/dpi_150/") and path.endswith(".png")
            )
        else:
            selected = (
                [member]
                + sorted(
                    path
                    for path in files
                    if path.startswith("pdfs/") and path.endswith(".pdf")
                )
                if member in files
                else []
            )
        if not selected:
            raise FileNotFoundError(
                f"Dataset {reference.netloc} does not contain {member}"
            )
        root = Path(dataset.get_local_copy(files=selected)).resolve()
        path = root / member
        if not path.exists():
            raise FileNotFoundError(f"Missing cached dataset input: {path}")
        return ResolvedDataset(path, root)
