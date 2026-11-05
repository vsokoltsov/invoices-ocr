import json
from dataclasses import dataclass
from pathlib import Path

from app.datasets.docile.downloader import DocileDatasetDownloader
from app.datasets.docile.enums import DocileSplit
from app.datasets.docile.loader import DocileDatasetLoader


@dataclass
class DocileInvoicesPreparation:
    downloader: DocileDatasetDownloader
    output_path: Path
    split: DocileSplit = DocileSplit.TRAINVAL
    use_snapped_boxes: bool = True
    corrections_path: str | Path | None = None

    def run(self) -> Path:
        """Save nested invoice data and available official split assignments.

        Reuse extracted data when the selected split and data directories exist.
        Store annotations as JSON to preserve empty sections and variable metadata.
        Parquet keeps OCR word structures; CSV stores these as JSON too.
        Optional corrections are applied before conversion and saved as audit JSON.
        Return the absolute path to the prepared file.
        """
        output_path = self.output_path.expanduser().resolve()
        if output_path.suffix.lower() not in {".csv", ".parquet"}:
            raise ValueError("DocILE output must have a .csv or .parquet extension")
        selected_split = DocileSplit(self.split)

        root = self.downloader.target_dir.resolve()
        loader = DocileDatasetLoader(
            root,
            use_snapped_boxes=self.use_snapped_boxes,
            corrections_path=self.corrections_path,
        )
        extracted = (
            (root / f"{selected_split}.json").is_file()
            and (root / "annotations").is_dir()
            and (root / "ocr").is_dir()
        )
        if not extracted:
            if not self.downloader.unzip:
                raise ValueError("Dataset preparation requires unzip=True")
            root = self.downloader.download().resolve()

        loader.root = root
        frame = loader.to_dataframe(selected_split)
        if frame.empty:
            raise ValueError("Selected split contains no documents")
        membership = {}
        for split in (DocileSplit.TRAIN, DocileSplit.VAL, DocileSplit.TEST):
            if (root / f"{split}.json").is_file():
                for document_id in loader.document_ids(split):
                    if document_id in membership:
                        raise ValueError(
                            f"Document appears in both {membership[document_id]} and {split}: {document_id}"
                        )
                    membership[document_id] = split
        if membership:
            frame["split"] = frame["document_id"].map(membership)
            if frame["split"].isna().any():
                raise ValueError("Some documents have no official split assignment")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        for column in ("json_data", "annotation_corrections"):
            if column in frame:
                frame[column] = frame[column].map(
                    lambda value: json.dumps(value, ensure_ascii=False)
                )
        if output_path.suffix.lower() == ".parquet":
            frame.to_parquet(output_path, index=False)
        else:
            frame["text_with_boxes"] = frame["text_with_boxes"].map(
                lambda value: json.dumps(value, ensure_ascii=False)
            )
            frame.to_csv(output_path, index=False)
        return output_path
