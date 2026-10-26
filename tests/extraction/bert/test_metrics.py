from collections.abc import Mapping

import numpy as np
import pytest
from transformers import EvalPrediction

from app.extraction.bert.metrics import TokenClassificationMetrics


def _logits_for(
    predicted_ids: np.ndarray,
    number_of_labels: int,
) -> np.ndarray:
    logits = np.full(
        (*predicted_ids.shape, number_of_labels),
        fill_value=-10.0,
        dtype=np.float32,
    )

    for sequence_index, sequence in enumerate(predicted_ids):
        for token_index, label_id in enumerate(sequence):
            logits[sequence_index, token_index, label_id] = 10.0

    return logits


@pytest.fixture
def id2label() -> dict[int, str]:
    """Provide the smallest complete BIO label mapping for metric tests."""

    return {
        0: "O",
        1: "B-SELLER_NAME",
        2: "I-SELLER_NAME",
    }


@pytest.fixture
def token_metrics(
    id2label: Mapping[int, str],
) -> TokenClassificationMetrics:
    """Provide metrics configured with the BIO labels used by test predictions."""

    return TokenClassificationMetrics(id2label=id2label)


@pytest.fixture
def label_ids() -> np.ndarray:
    """Provide two sequences with masked special tokens and one missing entity."""

    return np.array(
        [
            [-100, 1, 2, 0, -100],
            [-100, 1, 2, 0, -100],
        ],
        dtype=np.int64,
    )


@pytest.fixture
def prediction_logits(
    label_ids: np.ndarray,
    id2label: Mapping[int, str],
) -> np.ndarray:
    """Provide logits with one exact entity and one entity omitted entirely."""

    predicted_ids = np.array(
        [
            [0, 1, 2, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    return _logits_for(
        predicted_ids,
        number_of_labels=len(id2label),
    )


def test_number_of_labels_comes_from_configured_mapping(
    token_metrics: TokenClassificationMetrics,
) -> None:
    """Expose the class count TorchMetrics needs from the injected label mapping."""

    assert token_metrics.number_of_labels == 3


def test_decode_uses_argmax_and_ignores_masked_tokens(
    token_metrics: TokenClassificationMetrics,
    prediction_logits: np.ndarray,
    label_ids: np.ndarray,
) -> None:
    """Return parallel BIO label sequences without special or padding tokens."""

    predicted, actual = token_metrics.decode(
        predictions=prediction_logits,
        label_ids=label_ids,
    )

    assert predicted == [
        ["B-SELLER_NAME", "I-SELLER_NAME", "O"],
        ["O", "O", "O"],
    ]
    assert actual == [
        ["B-SELLER_NAME", "I-SELLER_NAME", "O"],
        ["B-SELLER_NAME", "I-SELLER_NAME", "O"],
    ]


def test_call_reports_entity_and_token_metrics(
    token_metrics: TokenClassificationMetrics,
    prediction_logits: np.ndarray,
    label_ids: np.ndarray,
) -> None:
    """Report entity exactness with seqeval and token quality with TorchMetrics."""

    result = token_metrics(
        EvalPrediction(
            predictions=prediction_logits,
            label_ids=label_ids,
        )
    )

    assert set(result) == {
        "precision",
        "recall",
        "f1",
        "token_accuracy",
        "token_macro_f1",
    }
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["token_accuracy"] == pytest.approx(4 / 6)
    assert result["token_macro_f1"] == pytest.approx(2 / 3)


def test_call_accepts_tuple_wrapped_logits_and_labels(
    token_metrics: TokenClassificationMetrics,
    prediction_logits: np.ndarray,
    label_ids: np.ndarray,
) -> None:
    """Use first tuple entries when Hugging Face supplies tuple-wrapped outputs."""

    result = token_metrics(
        EvalPrediction(
            predictions=(prediction_logits,),
            label_ids=(label_ids,),
        )
    )

    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(0.5)
    assert result["token_accuracy"] == pytest.approx(4 / 6)
