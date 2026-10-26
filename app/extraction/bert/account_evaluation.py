from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from app.extraction.bert.enums import MatchType
from app.extraction.bert.fields.account import AccountNumberField
from app.extraction.bert.fields.identifier import IdentifierField
from app.extraction.bert.json_evaluation import (
    ACCOUNT_NUMBER_PATH,
    flatten_json,
    optional_text,
)

ACCEPTED_ACCOUNT_MATCHES = {
    MatchType.EXACT,
    MatchType.NORMALIZED,
    MatchType.OCR_EQUIVALENT,
}


def extract_anchored_account_number(text: str) -> str | None:
    """Extract the identifier following an accepted account-number OCR alias."""

    match = AccountNumberField.alias_pattern().search(text)
    if match is None:
        return None

    return optional_text(match.group("value"))


def account_values_equal(left: str | None, right: str | None) -> bool:
    """Compare account identifiers while ignoring formatting and letter casing."""

    if left is None or right is None:
        return False

    return IdentifierField.compact_value(left) == IdentifierField.compact_value(right)


def build_account_evaluation(
    records: Iterable[Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-document account diagnostics and aggregate account metrics."""

    rows = []

    for record in records:
        actual = flatten_json(record["actual"])
        predicted = flatten_json(record["predicted"])
        expected_json = optional_text(actual.get(ACCOUNT_NUMBER_PATH))
        predicted_value = optional_text(predicted.get(ACCOUNT_NUMBER_PATH))
        observed_ocr = extract_anchored_account_number(str(record["ocred_text"]))
        match_type = None

        if expected_json is not None and predicted_value is not None:
            match_type = AccountNumberField(expected_json).classify_match(
                predicted_value
            )

        rows.append(
            {
                "row_index": record["row_index"],
                "file_name": record.get("file_name"),
                "annotated": expected_json is not None,
                "expected_json": expected_json,
                "observed_ocr": observed_ocr,
                "predicted": predicted_value,
                "predicted_present": predicted_value is not None,
                "matches_ocr": account_values_equal(predicted_value, observed_ocr),
                "matches_json_exact": (
                    account_values_equal(predicted_value, expected_json)
                    if expected_json is not None
                    else None
                ),
                "json_match_type": match_type.value if match_type is not None else None,
                "json_match_accepted": (
                    match_type in ACCEPTED_ACCOUNT_MATCHES
                    if expected_json is not None
                    else None
                ),
            }
        )

    details = pd.DataFrame(rows)
    annotated = details[details["annotated"]]
    with_ocr_anchor = details[details["observed_ocr"].notna()]

    def metric_row(name: str, correct: int, support: int) -> dict[str, object]:
        return {
            "metric": name,
            "correct": correct,
            "support": support,
            "accuracy": correct / support if support else np.nan,
        }

    metrics = pd.DataFrame(
        [
            metric_row(
                name="OCR account detected",
                correct=int(details["observed_ocr"].notna().sum()),
                support=len(details),
            ),
            metric_row(
                name="Prediction matches OCR",
                correct=int(with_ocr_anchor["matches_ocr"].sum()),
                support=len(with_ocr_anchor),
            ),
            metric_row(
                name="Exact JSON match",
                correct=int(annotated["matches_json_exact"].sum()),
                support=len(annotated),
            ),
            metric_row(
                name="Exact or OCR-equivalent JSON match",
                correct=int(annotated["json_match_accepted"].sum()),
                support=len(annotated),
            ),
        ]
    )

    return details, metrics
