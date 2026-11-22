import json
from dataclasses import asdict, dataclass
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from pipefunc import PipeFunc, Pipeline

from app.datasets.docile.converter import DocileAnnotationConverter
from app.datasets.docile.corrections import DocileAnnotationCorrections
from app.datasets.docile.enums import DocileSplit
from app.datasets.docile.loader import DocileDatasetLoader
from app.datasets.docile.models import OCRWord
from app.datasets.docile.order import OCRReadingOrder


@dataclass
class ApplyCorrections:
    """Apply reviewed annotation corrections before field conversion.

    Attributes:
        corrections: Validated correction manifest; None keeps the source labels.
    """

    corrections: DocileAnnotationCorrections | None

    def __call__(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return corrected annotations with per-document audit records.

        Args:
            frame: Original DocILE document rows.

        Returns:
            A copy with optional corrected annotations and audit records.

        Raises:
            RuntimeError: A correction does not match the selected source document.
        """
        result = frame.copy()
        if self.corrections is None:
            return result
        values, audits = [], []
        for _, row in frame.iterrows():
            try:
                values.append(self.corrections.apply(row.document_id, row.json_data))
            except ValueError as error:
                raise RuntimeError(
                    f"Failed to preprocess document {row.document_id}: {error}"
                ) from error
            audits.append(
                [
                    rule.model_dump(mode="json")
                    for rule in self.corrections.for_document(row.document_id)
                ]
            )
        result["json_data"] = values
        result["annotation_corrections"] = audits
        return result


@dataclass
class ExtractOCRWords:
    """Convert original OCR blocks into word records without ordering them.

    Attributes:
        use_snapped_boxes: Select snapped word geometry when available.
    """

    use_snapped_boxes: bool = True

    def __call__(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Flatten OCR hierarchy and preserve source word metadata.

        Args:
            frame: Original rows containing ocr_data.

        Returns:
            A copy with text_with_boxes. Character offsets are assigned later.

        Raises:
            ValueError: OCR geometry does not contain two box corners.
        """
        result = frame.copy()
        documents = []
        for _, row in frame.iterrows():
            words = []
            for page_position, page in enumerate(row.ocr_data["pages"]):
                page_index = int(page.get("page_idx", page_position))
                for block_index, block in enumerate(page["blocks"]):
                    for line_index, line in enumerate(block["lines"]):
                        for word_index, word in enumerate(line["words"]):
                            if not word["value"].strip():
                                continue
                            geometry = (
                                word.get("snapped_geometry")
                                if self.use_snapped_boxes
                                else None
                            )
                            geometry = geometry or word["geometry"]
                            if len(geometry) != 2:
                                raise ValueError(
                                    f"Invalid OCR geometry in {row.document_id}: {geometry!r}"
                                )
                            (left, top), (right, bottom) = geometry
                            words.append(
                                OCRWord(
                                    text=word["value"],
                                    bbox=(
                                        float(left),
                                        float(top),
                                        float(right),
                                        float(bottom),
                                    ),
                                    page=page_index,
                                    start=0,
                                    end=0,
                                    confidence=word.get("confidence"),
                                    block_index=block_index,
                                    line_index=line_index,
                                    word_index=word_index,
                                )
                            )
            documents.append(words)
        result["text_with_boxes"] = documents
        return result


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


class ConvertAnnotations:
    """Map DocILE field types and line items to the invoice schema."""

    def __call__(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Convert annotations without changing their boxes or source values.

        Args:
            frame: Documents with original or corrected DocILE annotations.

        Returns:
            A copy containing invoice-schema annotations.
        """
        result = frame.copy()
        result["json_data"] = result["json_data"].map(
            DocileAnnotationConverter().convert
        )
        return result.drop(columns=["ocr_data"])


@dataclass
class AddSplitMembership:
    """Attach available official split assignments without selecting training splits.

    Attributes:
        root: Directory containing official split manifests.
    """

    root: Path

    def __call__(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Attach split metadata and validate manifest consistency.

        Args:
            frame: Documents selected for export.

        Returns:
            A copy with a split column when official manifests are available.

        Raises:
            ValueError: Split assignments are duplicate or incomplete.
        """
        membership = {}
        loader = DocileDatasetLoader(self.root)
        for split in (DocileSplit.TRAIN, DocileSplit.VAL, DocileSplit.TEST):
            if not (self.root / f"{split}.json").is_file():
                continue
            for document_id in loader.document_ids(split):
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


@dataclass
class DocilePreprocess:
    """Extract local archives and export model-independent invoice documents.

    Attributes:
        root: Directory containing downloaded archives or extracted files.
        output_path: Destination Parquet or CSV file.
        split: Official source split to preprocess.
        use_snapped_boxes: Whether to select snapped OCR geometry.
        corrections_path: Optional reviewed annotation correction manifest.
    """

    root: Path
    output_path: Path
    split: DocileSplit = DocileSplit.TRAINVAL
    use_snapped_boxes: bool = True
    corrections_path: str | Path | None = None

    def extract(self) -> Path:
        """Extract the requested archive when its source files are absent.

        Returns:
            Absolute directory containing extracted source files.

        Raises:
            FileNotFoundError: The required archive was not downloaded.
            ValueError: An archive contains a path outside the source directory.
        """
        root = Path(self.root).expanduser().resolve()
        if (
            (root / f"{self.split}.json").is_file()
            and (root / "annotations").is_dir()
            and (root / "ocr").is_dir()
        ):
            return root
        name = "test" if self.split == DocileSplit.TEST else "annotated-trainval"
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

    def load(self, root: Path) -> pd.DataFrame:
        """Read original records from the selected split.

        Args:
            root: Directory produced by archive extraction.

        Returns:
            Original documents without preprocessing transformations.
        """
        frame = DocileDatasetLoader(root).to_dataframe(self.split)
        if frame.empty:
            raise ValueError("Selected split contains no documents")
        return frame

    def save(self, frame: pd.DataFrame) -> Path:
        """Write the transformed document frame.

        Args:
            frame: Model-independent preprocessed documents.

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
        """Create explicit download-independent preprocessing dependencies.

        Returns:
            A PipeFunc graph from local source archives to a prepared file.
        """
        corrections = (
            DocileAnnotationCorrections.from_file(self.corrections_path)
            if self.corrections_path
            else None
        )
        steps = [
            ApplyCorrections(corrections),
            ExtractOCRWords(self.use_snapped_boxes),
            OrderOCRWords(),
            ConvertAnnotations(),
            AddSplitMembership(Path(self.root).expanduser().resolve()),
        ]
        functions = [
            PipeFunc(self.extract, output_name="root"),
            PipeFunc(self.load, output_name="raw_documents"),
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
        """Execute preprocessing without downloading or initializing models.

        Returns:
            Absolute path of the exported dataset.

        Raises:
            ValueError: The output extension is unsupported.
        """
        if Path(self.output_path).suffix.lower() not in {".csv", ".parquet"}:
            raise ValueError("DocILE output must have a .csv or .parquet extension")
        return self.pipeline().run("output_path", kwargs={})
