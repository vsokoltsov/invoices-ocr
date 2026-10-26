import re
import unicodedata
from collections.abc import Iterable, Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

ACCOUNT_NUMBER_PATH = "payment_instructions.account_number"

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


def normalize_money(value: object) -> str:
    """Normalize OCR money formatting into a comparable decimal representation."""

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
        if normalized.rfind(",") > normalized.rfind("."):
            normalized = normalized.replace(".", "").replace(",", ".")
        else:
            normalized = normalized.replace(",", "")
    elif "," in normalized:
        normalized = normalized.replace(",", ".")

    try:
        return str(Decimal(normalized).normalize())
    except InvalidOperation:
        return normalized


def normalize_value(field: str, value: object) -> str | None:
    """Normalize a value according to the semantics of its output field."""

    if value is None:
        return None

    normalized = unicodedata.normalize(
        "NFKC",
        str(value),
    ).strip()
    leaf_name = field.split(".")[-1]

    if leaf_name in MONEY_FIELDS:
        return normalize_money(normalized)

    if leaf_name in IDENTIFIER_FIELDS:
        return re.sub(r"[^A-Za-z0-9]", "", normalized).upper()

    normalized = re.sub(r"\s+", " ", normalized).casefold()
    normalized = normalized.replace(":", ".")
    return re.sub(r"\s*([,.;])\s*", r"\1", normalized)


def document_to_dict(document: BaseModel | Mapping[str, Any]) -> dict[str, Any]:
    """Convert supported Pydantic and mapping documents into dictionaries."""

    if isinstance(document, BaseModel):
        return document.model_dump(exclude_none=True)

    return dict(document)


def flatten_json(
    document: BaseModel | Mapping[str, Any],
    prefix: str = "",
) -> dict[str, Any]:
    """Flatten nested document mappings while intentionally excluding lists."""

    source = document_to_dict(document)
    result: dict[str, Any] = {}

    for key, value in source.items():
        path = f"{prefix}.{key}" if prefix else key

        if isinstance(value, BaseModel | Mapping):
            result.update(flatten_json(value, path))
        elif not isinstance(value, list):
            result[path] = value

    return result


def optional_text(value: object) -> str | None:
    """Return stripped text when a value is meaningfully present."""

    if value is None:
        return None

    result = str(value).strip()
    return result or None


def evaluate_json_records(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Calculate field-level precision, recall, F1, and exact-match rate."""

    flattened_records: list[tuple[dict[str, Any], dict[str, Any]]] = []
    field_names: set[str] = set()

    for record in records:
        actual = flatten_json(record["actual"])
        predicted = flatten_json(record["predicted"])
        flattened_records.append((actual, predicted))
        field_names.update(actual)
        field_names.update(predicted)

    rows = []

    for field in sorted(field_names):
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        expected_count = 0

        for actual, predicted in flattened_records:
            actual_value = actual.get(field)
            predicted_value = predicted.get(field)
            actual_present = actual_value not in (None, "")
            predicted_present = predicted_value not in (None, "")

            if field == ACCOUNT_NUMBER_PATH and not actual_present:
                continue

            if actual_present:
                expected_count += 1

            correct = (
                actual_present
                and predicted_present
                and normalize_value(field, actual_value)
                == normalize_value(field, predicted_value)
            )

            if correct:
                true_positives += 1
                continue

            if predicted_present:
                false_positives += 1
            if actual_present:
                false_negatives += 1

        precision_denominator = true_positives + false_positives
        recall_denominator = true_positives + false_negatives
        precision = (
            true_positives / precision_denominator
            if precision_denominator
            else 0.0
        )
        recall = (
            true_positives / recall_denominator if recall_denominator else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        rows.append(
            {
                "field": field,
                "support": expected_count,
                "exact_matches": true_positives,
                "exact_match_rate": (
                    true_positives / expected_count
                    if expected_count
                    else np.nan
                ),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return pd.DataFrame(rows)


def document_is_exact(
    record: Mapping[str, Any],
    *,
    penalize_extra_fields: bool = False,
) -> bool:
    """Determine whether a predicted document contains every expected field value."""

    actual = flatten_json(record["actual"])
    predicted = flatten_json(record["predicted"])

    for field, expected_value in actual.items():
        if optional_text(expected_value) is None:
            continue

        if normalize_value(field, expected_value) != normalize_value(
            field,
            predicted.get(field),
        ):
            return False

    if not penalize_extra_fields:
        return True

    return not any(
        optional_text(predicted_value) is not None
        and optional_text(actual.get(field)) is None
        for field, predicted_value in predicted.items()
    )
