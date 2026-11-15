from dataclasses import dataclass
from typing import NamedTuple

import pandas as pd

from app.extraction.bert.artifacts import BertArtifacts
from app.extraction.bert.builder import InvoiceRecordBuilder
from app.pipelines.bert.split import DatasetsSplit


class BuildResult(NamedTuple):
    records: DatasetsSplit
    rejected: DatasetsSplit


@dataclass
class InvoicesBuilder:
    builder: InvoiceRecordBuilder
    artifacts: BertArtifacts

    def build(self, split: DatasetsSplit) -> BuildResult:
        """Prepare fixed partitions and keep OCR words for DocILE decoding."""
        records, rejected, manifest = {}, {}, {}
        for name in ("train", "val", "test"):
            frame = getattr(split, name)
            accepted, failures = self.builder.build(frame)
            rejected[name] = pd.DataFrame(
                failures,
                columns=pd.Index(["row_index", "document_id", "issues"]),
            )
            self.artifacts.save_rejections(name, rejected[name])
            if not accepted:
                raise ValueError(f"No accepted records in {name}")
            if "text_with_boxes" in frame:
                words = dict(
                    zip(frame["document_id"], frame["text_with_boxes"], strict=True)
                )
                for record in accepted:
                    record["text_with_boxes"] = words[record["document_id"]]
            records[name] = pd.DataFrame(accepted)
            for document_id in frame["document_id"]:
                manifest[str(document_id)] = "validation" if name == "val" else name
        self.artifacts.save_manifest(manifest)
        return BuildResult(DatasetsSplit(**records), DatasetsSplit(**rejected))
