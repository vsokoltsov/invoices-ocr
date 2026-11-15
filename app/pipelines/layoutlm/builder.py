from dataclasses import dataclass

import pandas as pd

from app.extraction.layoutlm.artifacts import LayoutLMArtifacts
from app.extraction.layoutlm.builder import InvoiceBuilder
from app.pipelines.builder import BuildPartitions
from app.pipelines.split import DatasetsSplit


@dataclass
class BuildInvoices(BuildPartitions):
    """Prepare LayoutLM pages and preserve CSV document acceptance metadata.

    Attributes:
        builder: Page builder that assigns labels from OCR and annotation boxes.
        artifacts: Storage for rejection reports and document acceptance metadata.
    """

    record_name = "pages"
    builder: InvoiceBuilder
    artifacts: LayoutLMArtifacts

    def prepare_partition(
        self, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare labeled page records and document rejection reports.

        Args:
            frame: Source documents in one fixed partition.

        Returns:
            Prepared page records and the builder's rejection DataFrame.
        """
        pages, rejected = self.builder.build(frame)
        return pd.DataFrame(pages), rejected

    def save_manifest(self, split: DatasetsSplit, rejected: DatasetsSplit) -> None:
        """Write source membership and acceptance using the LayoutLM CSV format.

        Args:
            split: Original document partitions.
            rejected: Reports identifying rejected documents in each partition.
        """
        manifest = []
        for name in ("train", "val", "test"):
            rejected_ids = set(getattr(rejected, name)["document_id"])
            manifest.extend(
                {
                    "document_id": document_id,
                    "split": name,
                    "accepted": document_id not in rejected_ids,
                }
                for document_id in getattr(split, name)["document_id"]
            )
        self.artifacts.save_manifest(pd.DataFrame(manifest))
