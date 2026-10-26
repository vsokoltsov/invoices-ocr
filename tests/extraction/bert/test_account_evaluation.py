from typing import Any

import pandas as pd
import pytest

from app.extraction.bert.account_evaluation import (
    account_values_equal,
    build_account_evaluation,
    extract_anchored_account_number,
)


def test_extract_anchored_account_number_returns_value_after_known_alias() -> None:
    """Extract only an account number introduced by an accepted OCR alias."""

    assert (
        extract_anchored_account_number(
            "Payment details. IBAN: GB12ABCD12345678901234"
        )
        == "GB12ABCD12345678901234"
    )
    assert extract_anchored_account_number("Reference GB12ABCD12345678901234") is None


def test_account_values_equal_compares_compacted_identifiers() -> None:
    """Ignore casing and separators but reject absent account values."""

    assert account_values_equal("GB12-AB34", "gb12ab34")
    assert not account_values_equal("GB12AB34", "GB12AB35")
    assert not account_values_equal("GB12AB34", None)


@pytest.fixture
def account_evaluation_records() -> list[dict[str, Any]]:
    """Provide exact, OCR-equivalent, and unannotated account prediction records."""

    exact = "GB12ABCD12345678901234"
    expected_equivalent = "GB55XDOT36152114286113"
    observed_equivalent = "GBSSXDOT36152114286113"
    unannotated = "GB99ABCD12345678901234"

    return [
        {
            "row_index": 1,
            "file_name": "exact.jpg",
            "ocred_text": f"IBAN: {exact}",
            "actual": {
                "payment_instructions": {"account_number": exact}
            },
            "predicted": {
                "payment_instructions": {"account_number": exact}
            },
        },
        {
            "row_index": 2,
            "file_name": "equivalent.jpg",
            "ocred_text": f"IBAN: {observed_equivalent}",
            "actual": {
                "payment_instructions": {
                    "account_number": expected_equivalent
                }
            },
            "predicted": {
                "payment_instructions": {
                    "account_number": observed_equivalent
                }
            },
        },
        {
            "row_index": 3,
            "file_name": "unknown.jpg",
            "ocred_text": f"IBAN: {unannotated}",
            "actual": {"payment_instructions": {}},
            "predicted": {
                "payment_instructions": {"account_number": unannotated}
            },
        },
    ]


def test_build_account_evaluation_separates_json_and_ocr_quality(
    account_evaluation_records: list[dict[str, Any]],
) -> None:
    """Report exact JSON, OCR-equivalent JSON, and OCR-anchor accuracy separately."""

    details, metrics = build_account_evaluation(account_evaluation_records)
    details = details.set_index("row_index")
    metrics = metrics.set_index("metric")

    assert details.loc[1, "json_match_type"] == "exact"
    assert details.loc[2, "json_match_type"] == "ocr_equivalent"
    assert bool(details.loc[2, "json_match_accepted"])
    assert pd.isna(details.loc[3, "json_match_accepted"])
    assert bool(details["matches_ocr"].all())

    assert metrics.loc["OCR account detected", "accuracy"] == pytest.approx(1.0)
    assert metrics.loc["Prediction matches OCR", "accuracy"] == pytest.approx(1.0)
    assert metrics.loc["Exact JSON match", "accuracy"] == pytest.approx(0.5)
    assert (
        metrics.loc[
            "Exact or OCR-equivalent JSON match",
            "accuracy",
        ]
        == pytest.approx(1.0)
    )
