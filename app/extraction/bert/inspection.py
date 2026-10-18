from dataclasses import dataclass
import pandas as pd
from app.extraction.bert.reporting import AlignmentReport
from app.extraction.bert.utils import money_digits
from app.extraction.bert.enums import EntityType

@dataclass(frozen=True, slots=True)
class AlignmentInspector:
    report: AlignmentReport

    def document(
        self,
        row_index: int,
    ) -> pd.DataFrame:
        return (
            self.report.details.loc[
                self.report.details[
                    "row_index"
                ] == row_index
            ]
            .sort_values("entity")
            .reset_index(drop=True)
        )

    def missing(
        self,
    ) -> pd.DataFrame:
        return (
            self.report.details.loc[
                ~self.report.details["aligned"]
            ]
            .sort_values(
                [
                    "entity",
                    "file_name",
                ]
            )
            .reset_index(drop=True)
        )

    def ambiguous(
        self,
    ) -> pd.DataFrame:
        return (
            self.report.details.loc[
                self.report.details["ambiguous"]
            ]
            .sort_values(
                [
                    "entity",
                    "file_name",
                ]
            )
            .reset_index(drop=True)
        )

    def partial_matches(
        self,
    ) -> pd.DataFrame:
        return (
            self.report.details.loc[
                self.report.details[
                    "partially_raw_matched"
                ]
            ]
            .sort_values(
                [
                    "entity",
                    "file_name",
                ]
            )
            .reset_index(drop=True)
        )

    def worst_documents(
        self,
        limit: int = 10,
    ) -> pd.DataFrame:
        return self.report.low_coverage(
            limit
        )

    def financial_mismatches(
        self,
    ) -> pd.DataFrame:
        financial = self.report.details[
            self.report.details["entity"].isin(
                [
                    EntityType.TAX.value,
                    EntityType.TOTAL.value,
                ]
            )
        ].copy()

        financial["value_matches_json"] = (
            financial.apply(
                lambda row: (
                    row["observed"] is not None
                    and money_digits(
                        row["observed"]
                    )
                    == money_digits(
                        row["expected"]
                    )
                ),
                axis=1,
            )
        )

        return (
            financial.loc[
                ~financial[
                    "value_matches_json"
                ]
            ]
            .reset_index(drop=True)
        )