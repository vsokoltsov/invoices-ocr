from dataclasses import dataclass
from collections.abc import Hashable
import pandas as pd
from pydantic import BaseModel
from app.extraction.bert.fields.field import Field
from app.extraction.bert.fields.account import AccountNumberField
from app.extraction.bert.fields.identifier import IdentifierField
from app.extraction.bert.fields.address import AddressField
from app.extraction.bert.fields.money import MoneyField
from app.extraction.bert.enums import EntityType
from app.extraction.bert.spans import EntitySpan
from app.extraction.bert.utils import money_digits

class InvoiceData(BaseModel):
    invoice_number: IdentifierField | None = None
    invoice_date: Field | None = None
    invoice_due_date: Field | None = None
    seller_name: Field | None = None
    seller_address: AddressField | None = None
    client_name: Field | None = None
    client_address: AddressField | None = None


class SubtotalData(BaseModel):
    tax: MoneyField | None = None
    total: MoneyField | None = None
    discount: MoneyField | None = None


class PaymentInstructions(BaseModel):
    due_date: Field | None = None
    account_number: AccountNumberField | None = None
    bank_name: Field | None = None
    payment_method: Field | None = None


class InvoiceTarget(BaseModel):
    invoice: InvoiceData
    subtotal: SubtotalData
    payment_instructions: PaymentInstructions
    items: list[dict]

    def extract_fields(self) -> list[tuple[EntityType, Field]]:
        fields = [
            ( EntityType.INVOICE_NUMBER, self.invoice.invoice_number ),
            ( EntityType.INVOICE_DATE, self.invoice.invoice_date ),
            ( EntityType.INVOICE_DUE_DATE, self.invoice.invoice_due_date ),
            ( EntityType.SELLER_NAME, self.invoice.seller_name ),
            ( EntityType.SELLER_ADDRESS, self.invoice.seller_address ),
            ( EntityType.CLIENT_NAME, self.invoice.client_name ),
            ( EntityType.CLIENT_ADDRESS, self.invoice.client_address ),
            ( EntityType.TAX, self.subtotal.tax ),
            ( EntityType.DISCOUNT, self.subtotal.discount ),
            ( EntityType.TOTAL, self.subtotal.total ),
            ( EntityType.PAYMENT_DUE_DATE, self.payment_instructions.due_date ),
            ( EntityType.BANK_NAME, self.payment_instructions.bank_name ),
            ( EntityType.ACCOUNT_NUMBER, self.payment_instructions.account_number ),
            ( EntityType.PAYMENT_METHOD, self.payment_instructions.payment_method )
        ]
        return [
            (entity, value)
            for entity, value in fields
            if (
                value is not None
                and value.normalized()
            )
        ]

@dataclass
class DocumentReportRow:
    row_index: Hashable
    file_name: str
    expected_entities: int
    matched_entities: int
    alignment_coverage: float
    missing_entities: tuple[str, ...]

    @classmethod
    def create(
        cls,
        *,
        row_index: Hashable,
        file_name: str,
        expected_fields: tuple[
            tuple[EntityType, Field],
            ...
        ],
        spans: tuple[EntitySpan, ...],
    ) -> "DocumentReportRow":
        expected = {
            entity
            for entity, _ in expected_fields
        }

        matched = {
            span.entity
            for span in spans
        }

        matched_expected = expected & matched

        return cls(
            row_index=row_index,
            file_name=file_name,
            expected_entities=len(expected),
            matched_entities=len(
                matched_expected
            ),
            alignment_coverage=(
                len(matched_expected) / len(expected)
                if expected
                else 1.0
            ),
            missing_entities=tuple(
                entity.value
                for entity in sorted(
                    expected - matched,
                    key=lambda item: item.value,
                )
            ),
        )


@dataclass
class DetailReportRow:
    row_index: Hashable
    file_name: str
    entity: str
    expected: str
    observed: str | None
    aligned: bool
    span_count: int
    parts: int
    matched_parts: int
    fully_raw_matched: bool
    partially_raw_matched: bool
    ambiguous: bool
    candidate_counts: tuple[int, ...]
    selected_spans: tuple[EntitySpan, ...]

    @classmethod
    def create(
        cls,
        *,
        row_index: Hashable,
        file_name: str,
        text: str,
        entity: EntityType,
        value: Field,
        selected_spans: tuple[
            EntitySpan,
            ...
        ],
    ) -> "DetailReportRow":
        value_parts = value.alignment_parts()

        part_candidates = [
            part.find_matches(text)
            for part in value_parts
        ]

        matched_parts = sum(
            bool(candidates)
            for candidates in part_candidates
        )

        candidate_counts = tuple(
            len(candidates)
            for candidates in part_candidates
        )

        return cls(
            row_index=row_index,
            file_name=file_name,
            entity=entity.value,
            expected=str(value),
            observed=(
                "\n".join(
                    span.value
                    for span in selected_spans
                )
                if selected_spans
                else None
            ),
            aligned=bool(selected_spans),
            span_count=len(selected_spans),
            parts=len(value_parts),
            matched_parts=matched_parts,
            fully_raw_matched=(
                matched_parts == len(value_parts)
            ),
            partially_raw_matched=(
                0 < matched_parts < len(value_parts)
            ),
            ambiguous=any(
                count > 1
                for count in candidate_counts
            ),
            candidate_counts=candidate_counts,
            selected_spans=selected_spans,
        )

@dataclass
class EntityReport:
    df: pd.DataFrame

    def build(self) -> pd.DataFrame:
        if self.df.empty:
            return pd.DataFrame()

        result = (
            self.df
            .groupby("entity", as_index=False)
            .agg(
                expected_documents=(
                    "entity",
                    "size",
                ),
                aligned_documents=(
                    "aligned",
                    "sum",
                ),
                fully_raw_matched=(
                    "fully_raw_matched",
                    "sum",
                ),
                partially_raw_matched=(
                    "partially_raw_matched",
                    "sum",
                ),
                ambiguous_documents=(
                    "ambiguous",
                    "sum",
                ),
            )
        )

        result["alignment_coverage"] = (
            result["aligned_documents"]
            / result["expected_documents"]
        )

        result["raw_match_rate"] = (
            result["fully_raw_matched"]
            / result["expected_documents"]
        )

        result["partial_rate"] = (
            result["partially_raw_matched"]
            / result["expected_documents"]
        )

        result["ambiguous_rate"] = (
            result["ambiguous_documents"]
            / result["expected_documents"]
        )

        return result.sort_values(
            "alignment_coverage"
        )

@dataclass
class FinancialReport:
    df: pd.DataFrame

    def build(self) -> pd.DataFrame:
        if self.df.empty:
            return pd.DataFrame()

        financial_details = self.df[
            self.df["entity"].isin(
                [
                    EntityType.TAX.value,
                    EntityType.TOTAL.value,
                ]
            )
        ].copy()

        if financial_details.empty:
            return pd.DataFrame()

        financial_details["value_matches_json"] = (
            financial_details.apply(
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

        result = (
            financial_details
            .groupby("entity", as_index=False)
            .agg(
                documents=("entity", "size"),
                spans_found=("aligned", "sum"),
                values_matching_json=(
                    "value_matches_json",
                    "sum",
                ),
            )
        )

        result["span_coverage"] = (
            result["spans_found"]
            / result["documents"]
        )

        result["ocr_value_accuracy"] = (
            result["values_matching_json"]
            / result["documents"]
        )

        return result