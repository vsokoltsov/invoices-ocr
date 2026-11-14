from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from app.extraction.layoutlm.builder import InvoiceBuilder
from app.pipelines.layoutlm.split import DatasetsSplit


class BuildResult(NamedTuple):
    records: DatasetsSplit
    rejected: DatasetsSplit


@dataclass
class BuildInvoices:
    builder: InvoiceBuilder
    output_dir: Path | None = None

    def build(self, split: DatasetsSplit) -> BuildResult:
        records = {}
        rejected = {}
        manifest = []
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
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
            if self.output_dir is not None:
                rejected_frame.to_csv(
                    self.output_dir / f"{name}_rejected.csv", index=False
                )
            if not pages:
                raise ValueError(f"No accepted pages in {name}")
        if self.output_dir is not None:
            pd.DataFrame(manifest).to_csv(
                self.output_dir / "split_manifest.csv", index=False
            )
        return BuildResult(
            records=DatasetsSplit(**records),
            rejected=DatasetsSplit(**rejected),
        )
