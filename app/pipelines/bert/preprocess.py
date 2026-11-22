from dataclasses import dataclass

import pandas as pd

from app.datasets.split import DatasetsSplit
from app.extraction.bert.artifacts import BertArtifacts
from app.extraction.bert.preprocess import InvoicePreprocessor
from app.pipelines.preprocess import PreprocessPartitions


@dataclass
class BertPreprocess(PreprocessPartitions):
    """Prepare BERT spans and preserve OCR words and JSON split membership.

    Attributes:
        preprocessor: Span preprocessor for invoice fields or line-item fields.
        artifacts: Storage for rejection reports and document membership.
    """

    record_name = "records"
    preprocessor: InvoicePreprocessor
    artifacts: BertArtifacts

    def prepare_partition(
        self, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare span records and retain OCR words for word-level decoding.

        Args:
            frame: Source documents in one fixed partition.

        Returns:
            Prepared span records and a structured rejection report.
        """
        accepted, failures = self.preprocessor.run(frame)
        rejected = pd.DataFrame(
            failures, columns=pd.Index(["row_index", "document_id", "issues"])
        )
        if "text_with_boxes" in frame:
            words = dict(
                zip(frame["document_id"], frame["text_with_boxes"], strict=True)
            )
            for record in accepted:
                record["text_with_boxes"] = words[record["document_id"]]
        return pd.DataFrame(accepted), rejected

    def save_manifest(self, split: DatasetsSplit, rejected: DatasetsSplit) -> None:
        """Write source document membership using BERT's JSON manifest format.

        Args:
            split: Original document partitions, including rejected documents.
            rejected: Rejection reports; unused by this manifest format.
        """
        manifest = {
            str(document_id): "validation" if name == "val" else name
            for name in ("train", "val", "test")
            for document_id in getattr(split, name)["document_id"]
        }
        self.artifacts.save_manifest(manifest)
