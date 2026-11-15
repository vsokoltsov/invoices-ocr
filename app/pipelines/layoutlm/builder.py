from dataclasses import dataclass
from typing import NamedTuple

import pandas as pd

from app.extraction.layoutlm.artifacts import LayoutLMArtifacts
from app.extraction.layoutlm.builder import InvoiceBuilder
from app.pipelines.layoutlm.split import DatasetsSplit


class BuildResult(NamedTuple):
    records: DatasetsSplit
    rejected: DatasetsSplit


@dataclass
class BuildInvoices:
    builder: InvoiceBuilder
    artifacts: LayoutLMArtifacts

    def build(self, split: DatasetsSplit) -> BuildResult:
        records = {}
        rejected = {}
        manifest = []
        for name in ("train", "val", "test"):
            frame = getattr(split, name)
            pages, rejected_frame = self.builder.build(frame)
            records[name] = pd.DataFrame(pages)
            rejected[name] = rejected_frame
            rejected_ids = set(rejected_frame["document_id"])
            manifest.extend(
                {
                    "document_id": document_id,
                    "split": name,
                    "accepted": document_id not in rejected_ids,
                }
                for document_id in frame["document_id"]
            )
            self.artifacts.save_rejections(name, rejected_frame)
            if not pages:
                raise ValueError(f"No accepted pages in {name}")
        self.artifacts.save_manifest(pd.DataFrame(manifest))
        return BuildResult(
            records=DatasetsSplit(**records),
            rejected=DatasetsSplit(**rejected),
        )
