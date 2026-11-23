"""Store model preparation results with portable document records."""

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.extraction.bert.preprocess import InvoicePreprocessor as BertPreprocessor
from app.extraction.layoutlm.preprocess import (
    InvoicePreprocessor as LayoutLMPreprocessor,
)
from app.pipelines.preprocess import DecodeDocuments, PreprocessStatistics


@dataclass
class ModelPreparation(PreprocessStatistics):
    """Run model rules once and retain their results in a document export.

    Attributes:
        model: Model family, either bert or layoutlm.
        preprocessor: Model rules used to create spans or labeled pages.
        rejection_path: CSV destination required when exporting documents.
    """

    model: str
    preprocessor: BertPreprocessor | LayoutLMPreprocessor
    rejection_path: Path | None = None
    record_name = "records"

    @property
    def configuration(self) -> dict:
        """Return the settings that must match when training reuses records."""
        if isinstance(self.preprocessor, BertPreprocessor):
            schema = self.preprocessor.label_schema
            return {
                "version": 1,
                "model": self.model,
                "fields": schema.field_to_entity,
                "labels": schema.label2id,
            }
        return {
            "version": 1,
            "model": self.model,
            "fields": self.preprocessor.field_to_entity,
            "labels": self.preprocessor.label2id,
            "min_coverage": self.preprocessor.min_coverage,
        }

    def rejection_frame(self, failures) -> pd.DataFrame:
        """Keep report columns present when no documents are rejected.

        Args:
            failures: Model rejection records, as a list or DataFrame.

        Returns:
            A report with the model's fixed rejection columns.
        """
        columns = (
            ["row_index", "document_id", "issues"]
            if self.model == "bert"
            else ["document_id", "reason"]
        )
        return pd.DataFrame(failures, columns=pd.Index(columns))

    def __call__(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Prepare documents, report rejections, and return accepted documents.

        Args:
            frame: Common invoice records after reading-order steps.

        Returns:
            Accepted documents with JSON model records in prepared_records.

        Raises:
            ValueError: No report path is set or no documents pass model rules.
        """
        if self.rejection_path is None:
            raise ValueError("Model export requires a rejection report path")
        records, failures = self.preprocessor.run(frame)
        rejected = self.rejection_frame(failures)
        report = rejected.copy()
        if "issues" in report:
            report["issues"] = report["issues"].map(json.dumps)
        self.rejection_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(self.rejection_path, index=False)
        grouped = defaultdict(list)
        for record in records:
            # Resolve PDF paths from the document on each machine that trains.
            values = {key: value for key, value in record.items() if key != "pdf_path"}
            grouped[record["document_id"]].append(DecodeDocuments.python_value(values))
        self.print_statistics(
            "total",
            {
                "documents": len(frame),
                "accepted": len(grouped),
                "rejected": rejected["document_id"].nunique(),
                "records": len(records),
            },
        )
        self.print_rejection_reasons(rejected)
        print(f"[preprocess] Rejection report: {self.rejection_path}", flush=True)
        if not records:
            raise ValueError(f"No accepted documents for {self.model}")
        result = frame.loc[frame["document_id"].isin(grouped)].copy()
        result["prepared_records"] = [
            json.dumps(
                {"configuration": self.configuration, "records": grouped[doc_id]},
                ensure_ascii=False,
            )
            for doc_id in result["document_id"]
        ]
        return result

    def load(self, frame: pd.DataFrame) -> list[dict] | None:
        """Read saved records and check that their model settings match.

        Args:
            frame: A document partition read from a processed export.

        Returns:
            Prepared records, or None for an older common-only export.

        Raises:
            ValueError: Saved model settings or record membership differ.
        """
        if "prepared_records" not in frame:
            return None
        records = []
        for _, row in frame.iterrows():
            payload = json.loads(row.prepared_records)
            if payload["configuration"] != self.configuration:
                raise ValueError(
                    "Prepared model settings differ; run preprocess again with the training settings"
                )
            if not payload["records"]:
                raise ValueError("Prepared document has no model records")
            for stored in payload["records"]:
                record = dict(stored)
                if record["document_id"] != row.document_id:
                    raise ValueError("Prepared record belongs to another document")
                if self.model == "layoutlm":
                    record["pdf_path"] = row.pdf_path
                records.append(record)
        return records
