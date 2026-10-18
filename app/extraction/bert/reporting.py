from dataclasses import dataclass

import pandas as pd

from app.extraction.bert.alignment import DocumentAligner
from app.extraction.bert.models import (
    DetailReportRow,
    DocumentReportRow,
    EntityReport,
    FinancialReport,
    InvoiceTarget,
)
from app.extraction.bert.utils import group_spans_by_entity


@dataclass(slots=True)
class AlignmentReport:
    documents: pd.DataFrame
    details: pd.DataFrame
    entities: pd.DataFrame
    financial: pd.DataFrame

    @classmethod
    def build(
        cls,
        frame: pd.DataFrame,
        aligner: DocumentAligner,
    ) -> "AlignmentReport":
        document_rows = []
        detail_rows = []

        for row_index, row in frame.iterrows():
            text = str(row["ocred_text"])
            target = cls._get_target(row)
            file_name = row.get(
                "file_name",
                str(row_index),
            )

            expected_fields = tuple(
                target.extract_fields()
            )

            spans = tuple(
                aligner.find_spans(
                    text=text,
                    target=target,
                )
            )

            spans_by_entity = (
                group_spans_by_entity(spans)
            )

            document_rows.append(
                DocumentReportRow.create(
                    row_index=row_index,
                    file_name=file_name,
                    expected_fields=expected_fields,
                    spans=spans,
                )
            )

            detail_rows.extend(
                DetailReportRow.create(
                    row_index=row_index,
                    file_name=file_name,
                    text=text,
                    entity=entity,
                    value=value,
                    selected_spans=(
                        spans_by_entity.get(entity, ())
                    ),
                )
                for entity, value in expected_fields
            )

        documents = pd.DataFrame(
            document_rows
        )

        details = pd.DataFrame(
            detail_rows
        )
    
        entities: pd.DataFrame = EntityReport(df=details).build()

        financial = FinancialReport(df=details).build()

        return cls(
            documents=documents,
            details=details,
            entities=entities,
            financial=financial,
        )

    @staticmethod
    def _get_target(
        row: pd.Series,
    ) -> InvoiceTarget:
        target = row.get("target")

        if isinstance(target, InvoiceTarget):
            return target

        if isinstance(target, dict):
            return InvoiceTarget.model_validate(
                target
            )

        json_data = row["json_data"]

        if isinstance(json_data, str):
            return (
                InvoiceTarget
                .model_validate_json(json_data)
            )

        return InvoiceTarget.model_validate(
            json_data
        )

    def missing_counts(self) -> pd.DataFrame:
        missing = (
            self.details.loc[
                ~self.details["aligned"],
                "entity",
            ]
            .value_counts()
            .rename("missing_count")
        )

        return missing.to_frame()

    def low_coverage(
        self,
        limit: int = 10,
    ) -> pd.DataFrame:
        return self.documents.nsmallest(
            limit,
            "alignment_coverage",
        )