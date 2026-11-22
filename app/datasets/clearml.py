"""Store existing invoice Parquet files and page caches in ClearML."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pymupdf
from clearml import Dataset
from PIL import Image


@dataclass
class DocileParquet:
    """Validate stored files without running invoice preprocessing.

    Attributes:
        path: Existing local Parquet file.
        kind: Raw or preprocessed invoice schema.
    """

    path: Path
    kind: str = "raw"

    def read(self) -> pd.DataFrame:
        """Read records and check the columns needed for dataset storage.

        Returns:
            Original records with unchanged OCR and annotation values.

        Raises:
            ValueError: The file extension, schema, or document IDs are invalid.
        """
        if self.path.suffix.lower() != ".parquet":
            raise ValueError("Dataset storage requires a Parquet file")
        frame = pd.read_parquet(self.path)
        required = {"document_id", "json_data", "pdf_path"}
        required |= (
            {"ocr_data"} if self.kind == "raw" else {"ocr_text", "text_with_boxes"}
        )
        if missing := required - set(frame.columns):
            raise ValueError(f"Missing {self.kind} columns: {sorted(missing)}")
        if frame.empty or frame[["document_id", "pdf_path"]].isna().any().any():
            raise ValueError("Dataset records and PDF references must not be empty")
        if frame.document_id.duplicated().any():
            raise ValueError("Document IDs must be unique")
        for value in frame.document_id:
            if (
                not isinstance(value, str)
                or not value.strip()
                or value in {".", ".."}
                or "/" in value
                or "\\" in value
            ):
                raise ValueError(f"Invalid document ID: {value!r}")
        if any(
            not isinstance(value, str) or not value.strip() for value in frame.pdf_path
        ):
            raise ValueError("PDF references must be nonempty strings")
        return frame

    def pdf_path(self, value: str) -> Path:
        """Resolve a stored PDF reference against this Parquet file.

        Args:
            value: Local absolute path or dataset-relative PDF path.

        Returns:
            Absolute local file path.
        """
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.path.parent / path


@dataclass
class RawDocileDatasetUpload:
    """Upload original records and PDFs in one portable dataset.

    Attributes:
        dataset_path: Raw Parquet written by the DocILE loader.
        project: ClearML project for the dataset.
        name: Dataset name shown in ClearML.
        output_uri: Optional upload storage URI.
    """

    dataset_path: Path
    project: str
    name: str
    output_uri: str = ""

    def run(self) -> str:
        """Validate files, upload a portable snapshot, and finalize it.

        Returns:
            Finalized ClearML dataset ID.

        Raises:
            ValueError: A document ID cannot be used as a file stem.
            FileNotFoundError: A referenced PDF is absent.
        """
        path = self.dataset_path.expanduser().resolve()
        snapshot = DocileParquet(path)
        documents = snapshot.read()
        for document_id, pdf_path in documents[["document_id", "pdf_path"]].itertuples(
            index=False, name=None
        ):
            if not snapshot.pdf_path(pdf_path).is_file():
                raise FileNotFoundError(f"Missing PDF for {document_id}: {pdf_path}")

        # Stage references without copying large PDFs or changing the source file.
        with TemporaryDirectory(prefix="docile-upload-") as directory:
            root = Path(directory)
            (root / "pdfs").mkdir()
            portable = documents.copy()
            portable["pdf_path"] = [
                f"pdfs/{document_id}.pdf" for document_id in documents.document_id
            ]
            portable.to_parquet(root / "raw.parquet", index=False)
            for document_id, pdf_path in documents[
                ["document_id", "pdf_path"]
            ].itertuples(index=False, name=None):
                (root / "pdfs" / f"{document_id}.pdf").symlink_to(
                    snapshot.pdf_path(pdf_path)
                )
            dataset = Dataset.create(
                dataset_project=self.project, dataset_name=self.name
            )
            dataset.add_files(path=str(root))
            dataset.set_metadata(
                {"format": "docile-raw-v1", "documents": len(documents)}
            )
            dataset.upload(output_url=self.output_uri or None)
            dataset.finalize()
            return dataset.id


@dataclass
class PreprocessedDocileDatasetUpload:
    """Upload an existing processed Parquet as a child of a raw dataset.

    Attributes:
        dataset_path: Parquet already produced by local preprocessing.
        parent_id: Finalized dataset containing raw.parquet and PDFs.
        project: ClearML dataset project.
        name: Child dataset name.
        output_uri: Optional upload storage URI.
    """

    dataset_path: Path
    parent_id: str
    project: str
    name: str
    output_uri: str = ""

    def run(self) -> str:
        """Publish processed records without calling preprocessing.

        Returns:
            Finalized child dataset ID.

        Raises:
            ValueError: The parent is not final or lacks selected document PDFs.
        """
        frame = DocileParquet(
            self.dataset_path.expanduser().resolve(), "preprocessed"
        ).read()
        parent = Dataset.get(dataset_id=self.parent_id)
        if not parent.is_final():
            raise ValueError("The parent dataset must be finalized")
        files = set(parent.list_files())
        if "raw.parquet" not in files:
            raise ValueError("The parent must contain raw.parquet")
        references = [f"pdfs/{value}.pdf" for value in frame.document_id]
        if missing := set(references) - files:
            raise ValueError(f"The parent is missing document PDFs: {sorted(missing)}")
        with TemporaryDirectory(prefix="docile-preprocessed-upload-") as directory:
            root = Path(directory)
            frame["pdf_path"] = references
            frame.to_parquet(root / "preprocessed.parquet", index=False)
            dataset = Dataset.create(
                dataset_project=self.project,
                dataset_name=self.name,
                parent_datasets=[self.parent_id],
            )
            dataset.add_files(path=str(root))
            dataset.set_metadata(
                {"format": "docile-preprocessed-v1", "parent_id": self.parent_id}
            )
            dataset.upload(output_url=self.output_uri or None)
            dataset.finalize()
            return dataset.id


@dataclass
class DocileImagesDatasetUpload:
    """Upload a page cache as a child of raw or preprocessed data.

    Attributes:
        parent_id: Finalized dataset containing raw.parquet and PDFs.
        images_path: Cache directory containing dpi_150 document folders.
        project: ClearML project for the child dataset.
        name: Child dataset name shown in ClearML.
        output_uri: Optional upload storage URI.
    """

    parent_id: str
    images_path: Path
    project: str
    name: str
    output_uri: str = ""

    def run(self) -> str:
        """Check page coverage, upload only images, and finalize the child.

        Returns:
            Child dataset ID. The child inherits the parent's Parquet and PDFs.

        Raises:
            ValueError: The parent layout or cached page set is invalid.
        """
        parent = Dataset.get(dataset_id=self.parent_id)
        if not parent.is_final():
            raise ValueError("The parent dataset must be finalized")
        root = Path(parent.get_local_copy())
        snapshot = DocileParquet(root / "raw.parquet")
        documents = snapshot.read()
        portable = documents[["document_id", "pdf_path"]]
        if any(
            pdf_path != f"pdfs/{document_id}.pdf"
            for document_id, pdf_path in portable.itertuples(index=False, name=None)
        ):
            raise ValueError(
                "The parent must use portable pdfs/<document_id>.pdf paths"
            )
        cache = self.images_path.expanduser().resolve() / "dpi_150"
        expected = set()
        for document_id, pdf_path in documents[["document_id", "pdf_path"]].itertuples(
            index=False, name=None
        ):
            with pymupdf.open(snapshot.pdf_path(pdf_path)) as document:
                for page_index in range(len(document)):
                    expected.add(cache / document_id / f"page_{page_index:04d}.png")
        actual = set(cache.rglob("*.png"))
        if not expected or actual != expected:
            raise ValueError(
                f"Image cache does not match the parent PDFs: "
                f"{len(expected - actual)} missing, {len(actual - expected)} extra pages"
            )
        for path in sorted(expected):
            with Image.open(path) as image:
                image.verify()
        dataset = Dataset.create(
            dataset_project=self.project,
            dataset_name=self.name,
            parent_datasets=[self.parent_id],
        )
        dataset.add_files(
            path=str(cache), wildcard="*.png", dataset_path="images/dpi_150"
        )
        dataset.set_metadata(
            {"format": "docile-images-v1", "dpi": 150, "parent_id": self.parent_id}
        )
        dataset.upload(output_url=self.output_uri or None)
        dataset.finalize()
        return dataset.id


@dataclass
class ClearMLDatasetDownload:
    """Download a finalized dataset and its inherited files.

    Attributes:
        dataset_id: Exact ClearML dataset version to retrieve.
        destination_path: New or empty directory for a writable local copy.
    """

    dataset_id: str
    destination_path: Path

    def run(self) -> Path:
        """Retrieve the dataset without overwriting existing local files.

        Returns:
            Absolute directory with Parquet files, PDFs, and optional images.

        Raises:
            ValueError: The destination contains files or the dataset is not final.
        """
        destination = self.destination_path.expanduser().resolve()
        if destination.exists() and (
            not destination.is_dir() or any(destination.iterdir())
        ):
            raise ValueError(
                "The download destination must be a new or empty directory"
            )
        dataset = Dataset.get(dataset_id=self.dataset_id)
        if not dataset.is_final():
            raise ValueError("The dataset must be finalized before download")
        dataset.get_mutable_local_copy(target_folder=str(destination), overwrite=False)
        # Rebase only the downloaded copy. Stored dataset references stay portable.
        for filename, kind in (
            ("raw.parquet", "raw"),
            ("preprocessed.parquet", "preprocessed"),
        ):
            path = destination / filename
            if not path.is_file():
                continue
            frame = DocileParquet(path, kind).read()
            references = [f"pdfs/{value}.pdf" for value in frame.document_id]
            if frame.pdf_path.tolist() != references:
                raise ValueError("Stored PDF paths must use pdfs/<document_id>.pdf")
            frame["pdf_path"] = [str(destination / value) for value in references]
            frame.to_parquet(path, index=False)
        return destination
