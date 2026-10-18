import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
)

from app.extraction.bert.enums import EntityType, MatchType
from app.extraction.bert.fields.account import AccountNumberField
from app.extraction.bert.fields.identifier import IdentifierField
from app.extraction.bert.models import (
    DocumentOutput,
    InvoiceData,
    PaymentInstructions,
    PredictedEntity,
    SubtotalData,
)

ACCOUNT_NUMBER_PATH = (
    "payment_instructions.account_number"
)

ACCEPTED_ACCOUNT_MATCHES = {
    MatchType.EXACT,
    MatchType.NORMALIZED,
    MatchType.OCR_EQUIVALENT,
}


DocumentData = (
    BaseModel
    | Mapping[str, Any]
)


def normalize_money(
    value: object,
) -> str:
    normalized = unicodedata.normalize(
        "NFKC",
        str(value),
    )

    normalized = re.sub(
        r"[^\d,.\-]",
        "",
        normalized,
    )

    if "," in normalized and "." in normalized:
        # The final separator is treated as the
        # decimal separator.
        if (
            normalized.rfind(",")
            > normalized.rfind(".")
        ):
            normalized = (
                normalized
                .replace(".", "")
                .replace(",", ".")
            )
        else:
            normalized = (
                normalized.replace(",", "")
            )

    elif "," in normalized:
        normalized = normalized.replace(
            ",",
            ".",
        )

    try:
        return str(
            Decimal(normalized).normalize()
        )
    except InvalidOperation:
        return normalized

MONEY_FIELDS = {
    "tax",
    "total",
    "subtotal",
    "discount",
}

IDENTIFIER_FIELDS = {
    "invoice_number",
    "account_number",
}

def normalize_value(
    field: str,
    value: object,
) -> str | None:
    if value is None:
        return None

    normalized = unicodedata.normalize(
        "NFKC",
        str(value),
    ).strip()

    leaf_name = field.split(".")[-1]

    if leaf_name in MONEY_FIELDS:
        return normalize_money(
            normalized
        )

    if leaf_name in IDENTIFIER_FIELDS:
        return re.sub(
            r"[^A-Za-z0-9]",
            "",
            normalized,
        ).upper()

    # Treat newlines and repeated spaces equally.
    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    normalized = normalized.casefold()

    # Normalize common OCR punctuation differences.
    normalized = normalized.replace(
        ":",
        ".",
    )

    normalized = re.sub(
        r"\s*([,.;])\s*",
        r"\1",
        normalized,
    )

    return normalized

def document_to_dict(
    document: BaseModel | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(document, BaseModel):
        return document.model_dump(
            exclude_none=True
        )

    return dict(document)


def flatten_json(
    document: BaseModel | Mapping[str, Any],
    prefix: str = "",
) -> dict[str, Any]:
    source = document_to_dict(
        document
    )

    result: dict[str, Any] = {}

    for key, value in source.items():
        path = (
            f"{prefix}.{key}"
            if prefix
            else key
        )

        if isinstance(value, BaseModel):
            result.update(
                flatten_json(
                    value,
                    path,
                )
            )

        elif isinstance(value, Mapping):
            result.update(
                flatten_json(
                    value,
                    path,
                )
            )

        elif not isinstance(value, list):
            result[path] = value

    return result

def optional_text(
    value: object,
) -> str | None:
    if value is None:
        return None

    result = str(value).strip()

    return result or None


def evaluate_json_records(
    records: Iterable[
        Mapping[str, Any]
    ],
) -> pd.DataFrame:
    records = list(records)

    flattened_records: list[
        tuple[
            dict[str, Any],
            dict[str, Any],
        ]
    ] = []

    field_names: set[str] = set()

    for record in records:
        actual = flatten_json(
            record["actual"]
        )

        predicted = flatten_json(
            record["predicted"]
        )

        flattened_records.append(
            (actual, predicted)
        )

        field_names.update(actual)
        field_names.update(predicted)

    rows = []

    for field in sorted(field_names):
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        expected_count = 0

        for actual, predicted in flattened_records:
            actual_value = actual.get(
                field
            )

            predicted_value = predicted.get(
                field
            )

            actual_present = (
                actual_value not in (None, "")
            )

            predicted_present = (
                predicted_value
                not in (None, "")
            )

            # The dataset has incomplete account-number
            # annotations. Missing means unknown, not absent.
            if (
                field == ACCOUNT_NUMBER_PATH
                and not actual_present
            ):
                continue

            if actual_present:
                expected_count += 1

            correct = (
                actual_present
                and predicted_present
                and normalize_value(
                    field,
                    actual_value,
                )
                == normalize_value(
                    field,
                    predicted_value,
                )
            )

            if correct:
                true_positives += 1
                continue

            if predicted_present:
                false_positives += 1

            if actual_present:
                false_negatives += 1

        precision_denominator = (
            true_positives
            + false_positives
        )

        recall_denominator = (
            true_positives
            + false_negatives
        )

        precision = (
            true_positives
            / precision_denominator
            if precision_denominator
            else 0.0
        )

        recall = (
            true_positives
            / recall_denominator
            if recall_denominator
            else 0.0
        )

        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
            if precision + recall
            else 0.0
        )

        exact_match_rate = (
            true_positives
            / expected_count
            if expected_count
            else np.nan
        )

        rows.append({
            "field": field,
            "support": expected_count,
            "exact_matches": (
                true_positives
            ),
            "exact_match_rate": (
                exact_match_rate
            ),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    return pd.DataFrame(rows)

def document_is_exact(
    record: Mapping[str, Any],
    *,
    penalize_extra_fields: bool = False,
) -> bool:
    actual = flatten_json(
        record["actual"]
    )

    predicted = flatten_json(
        record["predicted"]
    )

    # Every expected field must be present and correct.
    for field, expected_value in actual.items():
        if optional_text(expected_value) is None:
            continue
        predicted_value = predicted.get(
            field
        )

        if (
            normalize_value(
                field,
                expected_value,
            )
            != normalize_value(
                field,
                predicted_value,
            )
        ):
            return False

    if penalize_extra_fields:
        for field, predicted_value in predicted.items():
            predicted_present = (
                optional_text(predicted_value)
                is not None
            )

            actual_present = (
                 optional_text(
                    actual.get(field)
                )
                is not None
            )

            if (
                predicted_present
                and not actual_present
            ):
                return False

    return True

def extract_anchored_account_number(
    text: str,
) -> str | None:
    match = (
        AccountNumberField
        .alias_pattern()
        .search(text)
    )

    if match is None:
        return None

    return optional_text(
        match.group("value")
    )


def account_values_equal(
    left: str | None,
    right: str | None,
) -> bool:
    if left is None or right is None:
        return False

    return (
        IdentifierField.compact_value(left)
        == IdentifierField.compact_value(right)
    )


def build_account_evaluation(
    records: Iterable[
        Mapping[str, Any]
    ],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    rows = []

    for record in records:
        actual = flatten_json(
            record["actual"]
        )

        predicted = flatten_json(
            record["predicted"]
        )

        expected_json = optional_text(
            actual.get(
                ACCOUNT_NUMBER_PATH
            )
        )

        predicted_value = optional_text(
            predicted.get(
                ACCOUNT_NUMBER_PATH
            )
        )

        observed_ocr = (
            extract_anchored_account_number(
                str(record["ocred_text"])
            )
        )

        match_type = None

        if (
            expected_json is not None
            and predicted_value is not None
        ):
            match_type = (
                AccountNumberField(
                    expected_json
                )
                .classify_match(
                    predicted_value
                )
            )

        rows.append({
            "row_index": (
                record["row_index"]
            ),
            "file_name": (
                record.get("file_name")
            ),
            "annotated": (
                expected_json is not None
            ),
            "expected_json": expected_json,
            "observed_ocr": observed_ocr,
            "predicted": predicted_value,
            "predicted_present": (
                predicted_value is not None
            ),
            "matches_ocr": (
                account_values_equal(
                    predicted_value,
                    observed_ocr,
                )
            ),
            "matches_json_exact": (
                account_values_equal(
                    predicted_value,
                    expected_json,
                )
                if expected_json is not None
                else None
            ),
            "json_match_type": (
                match_type.value
                if match_type is not None
                else None
            ),
            "json_match_accepted": (
                match_type
                in ACCEPTED_ACCOUNT_MATCHES
                if expected_json is not None
                else None
            ),
        })

    details = pd.DataFrame(rows)

    annotated = details[
        details["annotated"]
    ]

    with_ocr_anchor = details[
        details["observed_ocr"].notna()
    ]

    def metric_row(
        name: str,
        correct: int,
        support: int,
    ) -> dict[str, object]:
        return {
            "metric": name,
            "correct": correct,
            "support": support,
            "accuracy": (
                correct / support
                if support
                else np.nan
            ),
        }

    metrics = pd.DataFrame([
        metric_row(
            name="OCR account detected",
            correct=int(
                details[
                    "observed_ocr"
                ].notna().sum()
            ),
            support=len(details),
        ),
        metric_row(
            name="Prediction matches OCR",
            correct=int(
                with_ocr_anchor[
                    "matches_ocr"
                ].sum()
            ),
            support=len(
                with_ocr_anchor
            ),
        ),
        metric_row(
            name="Exact JSON match",
            correct=int(
                annotated[
                    "matches_json_exact"
                ].sum()
            ),
            support=len(annotated),
        ),
        metric_row(
            name=(
                "Exact or OCR-equivalent "
                "JSON match"
            ),
            correct=int(
                annotated[
                    "json_match_accepted"
                ].sum()
            ),
            support=len(annotated),
        ),
    ])

    return details, metrics

@dataclass(frozen=True, slots=True)
class DocumentOutputFormatter:
    def format(
        self,
        entities: list[PredictedEntity],
    ) -> DocumentOutput:
        grouped = self._group_entities(
            entities
        )

        return DocumentOutput(
            invoice=InvoiceData(
                invoice_number=self._first(
                    grouped,
                    EntityType.INVOICE_NUMBER,
                ),
                invoice_date=self._first(
                    grouped,
                    EntityType.INVOICE_DATE,
                ),
                invoice_due_date=self._first(
                    grouped,
                    EntityType.INVOICE_DUE_DATE,
                ),
                seller_name=self._first(
                    grouped,
                    EntityType.SELLER_NAME,
                ),
                seller_address=self._multiline(
                    grouped,
                    EntityType.SELLER_ADDRESS,
                ),
                client_name=self._first(
                    grouped,
                    EntityType.CLIENT_NAME,
                ),
                client_address=self._multiline(
                    grouped,
                    EntityType.CLIENT_ADDRESS,
                ),
            ),
            subtotal=SubtotalData(
                tax=self._last(
                    grouped,
                    EntityType.TAX,
                ),
                total=self._last(
                    grouped,
                    EntityType.TOTAL,
                ),
                discount=self._last(
                    grouped,
                    EntityType.DISCOUNT,
                ),
            ),
            payment_instructions=(
                PaymentInstructions(
                    due_date=self._first(
                        grouped,
                        EntityType.PAYMENT_DUE_DATE,
                    ),
                    account_number=self._first(
                        grouped,
                        EntityType.ACCOUNT_NUMBER,
                    ),
                    bank_name=self._first(
                        grouped,
                        EntityType.BANK_NAME,
                    ),
                    payment_method=self._first(
                        grouped,
                        EntityType.PAYMENT_METHOD,
                    ),
                )
            ),
            items=None,
        )

    def format_json(
        self,
        entities: list[PredictedEntity],
        *,
        indent: int | None = 2,
    ) -> str:
        output = self.format(entities)

        return output.model_dump_json(
            exclude_none=True,
            indent=indent,
        )

    @staticmethod
    def _group_entities(
        entities: list[PredictedEntity],
    ) -> dict[
        EntityType,
        list[PredictedEntity],
    ]:
        grouped: dict[
            EntityType,
            list[PredictedEntity],
        ] = defaultdict(list)

        for entity in sorted(
            entities,
            key=lambda item: item.start,
        ):
            grouped[entity.entity].append(
                entity
            )

        return dict(grouped)

    @staticmethod
    def _first(
        grouped: dict[
            EntityType,
            list[PredictedEntity],
        ],
        entity_type: EntityType,
    ) -> str | None:
        candidates = grouped.get(
            entity_type,
            [],
        )

        if not candidates:
            return None

        return candidates[0].text.strip()

    @staticmethod
    def _last(
        grouped: dict[
            EntityType,
            list[PredictedEntity],
        ],
        entity_type: EntityType,
    ) -> str | None:
        candidates = grouped.get(
            entity_type,
            [],
        )

        if not candidates:
            return None

        return candidates[-1].text.strip()

    @staticmethod
    def _multiline(
        grouped: dict[
            EntityType,
            list[PredictedEntity],
        ],
        entity_type: EntityType,
    ) -> str | None:
        candidates = grouped.get(
            entity_type,
            [],
        )

        values = [
            candidate.text.strip()
            for candidate in candidates
            if candidate.text.strip()
        ]

        if not values:
            return None

        return "\n".join(values)