import json
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from app.extraction.bert.pytorch.docile.builder import DocileRecordBuilder
from app.pipelines.bert.split import DatasetsSplit


class BuildResult(NamedTuple):
    records: DatasetsSplit
    rejected: DatasetsSplit


@dataclass
class InvoicesBuilder:
    builder: DocileRecordBuilder
    output_dir: Path

    def build(self, split: DatasetsSplit) -> BuildResult:
        """Prepare fixed partitions and keep OCR words for DocILE decoding."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        records, rejected, manifest = {}, {}, {}
        for name in ("train", "val", "test"):
            frame = getattr(split, name)
            accepted, failures = self.builder.build(frame)
            rejected[name] = pd.DataFrame(
                failures,
                columns=pd.Index(["row_index", "document_id", "issues"]),
            )
            rejected[name].to_csv(self.output_dir / f"{name}_rejected.csv", index=False)
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
        (self.output_dir / "split_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return BuildResult(DatasetsSplit(**records), DatasetsSplit(**rejected))
