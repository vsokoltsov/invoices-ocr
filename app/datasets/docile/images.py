from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pymupdf
from tqdm.auto import tqdm

from app.extraction.layoutlm.converter import AttachmentToImageConverter


@dataclass
class DocileImageCache:
    """Cache all PDF pages listed in a prepared DocILE CSV or Parquet file."""

    dataset_path: Path
    images_path: Path

    def run(self) -> Path:
        dataset_path = self.dataset_path.expanduser().resolve()
        images_path = self.images_path.expanduser().resolve()
        columns = ["document_id", "pdf_path"]
        if dataset_path.suffix.lower() == ".parquet":
            documents = pd.read_parquet(dataset_path, columns=columns)
        elif dataset_path.suffix.lower() == ".csv":
            documents = pd.read_csv(
                dataset_path,
                usecols=pd.Index(columns),
                dtype={column: str for column in columns},
                keep_default_na=False,
            )
        else:
            raise ValueError("Prepared DocILE dataset must be a .parquet or .csv file")
        if documents.empty:
            raise ValueError("Prepared DocILE dataset has no documents")
        if documents.isna().to_numpy().any() or any(
            documents[column].astype(str).str.strip().eq("").any() for column in columns
        ):
            raise ValueError("Document IDs and PDF paths must not be empty")
        if documents["document_id"].duplicated().any():
            raise ValueError("Document IDs must be unique")

        converter = AttachmentToImageConverter(images_path)
        for row in tqdm(
            documents.itertuples(index=False),
            total=len(documents),
            desc="Cache DocILE images",
            unit="document",
        ):
            try:
                with pymupdf.open(row.pdf_path) as document:
                    page_count = len(document)
                for page_index in range(page_count):
                    with converter.load_page_image(
                        {
                            "document_id": row.document_id,
                            "pdf_path": row.pdf_path,
                            "page": page_index,
                        }
                    ):
                        pass
            except Exception as error:
                raise RuntimeError(
                    f"Failed to cache document {row.document_id!r} from {row.pdf_path}"
                ) from error
        return images_path
