from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest

from app.extraction.bert.enums import EntityType
from app.extraction.bert.evaluation import ModelEvaluator
from app.extraction.bert.formatter import DocumentOutputFormatter
from app.extraction.bert.inference import DocumentModelPredictor
from app.extraction.bert.metrics import TokenClassificationMetrics
from app.extraction.bert.models import PredictedEntity
from app.extraction.bert.operations import DocumentModelTrainer


@dataclass
class FakeTrainer:
    """Record the requested split and return deterministic trainer output."""

    requested_splits: list[str]

    def predict_dataset(self, split: str) -> SimpleNamespace:
        self.requested_splits.append(split)
        return SimpleNamespace(
            predictions=(np.zeros((1, 1, 2)),),
            label_ids=np.zeros((1, 1), dtype=np.int64),
            metrics={"test_loss": 0.25},
        )


@dataclass
class FakeMetrics:
    """Record decoded model output while returning one valid BIO sequence."""

    decoded_inputs: list[tuple[np.ndarray, np.ndarray]]

    def decode(
        self,
        *,
        predictions: np.ndarray,
        label_ids: np.ndarray,
    ) -> tuple[list[list[str]], list[list[str]]]:
        self.decoded_inputs.append((predictions, label_ids))
        return (
            [["B-INVOICE_NUMBER"]],
            [["B-INVOICE_NUMBER"]],
        )


class FakePredictor:
    """Return deterministic entities for two documents in the evaluation frame."""

    def __init__(self) -> None:
        self.predicted_texts: list[str] = []

    def predict(self, text: str) -> list[PredictedEntity]:
        self.predicted_texts.append(text)
        if "1001" in text:
            return [
                PredictedEntity(
                    entity=EntityType.INVOICE_NUMBER,
                    start=0,
                    end=4,
                    text="1001",
                    confidence=0.99,
                ),
                PredictedEntity(
                    entity=EntityType.TAX,
                    start=5,
                    end=10,
                    text="10,00",
                    confidence=0.98,
                ),
                PredictedEntity(
                    entity=EntityType.TOTAL,
                    start=11,
                    end=17,
                    text="110,00",
                    confidence=0.97,
                ),
                PredictedEntity(
                    entity=EntityType.ACCOUNT_NUMBER,
                    start=18,
                    end=40,
                    text="GB12ABCD12345678901234",
                    confidence=0.96,
                ),
            ]

        return [
            PredictedEntity(
                entity=EntityType.INVOICE_NUMBER,
                start=0,
                end=4,
                text="9999",
                confidence=0.91,
            ),
            PredictedEntity(
                entity=EntityType.TAX,
                start=5,
                end=10,
                text="20,00",
                confidence=0.90,
            ),
            PredictedEntity(
                entity=EntityType.TOTAL,
                start=11,
                end=17,
                text="220,00",
                confidence=0.89,
            ),
            PredictedEntity(
                entity=EntityType.ACCOUNT_NUMBER,
                start=18,
                end=40,
                text="GB99ABCD12345678901234",
                confidence=0.88,
            ),
        ]


@pytest.fixture
def evaluation_frame() -> pd.DataFrame:
    """Provide one fully correct and one partially incorrect test document."""

    return pd.DataFrame(
        [
            {
                "file_name": "correct.jpg",
                "ocred_text": (
                    "Invoice no: 1001 Tax: 10,00 Total: 110,00 "
                    "IBAN: GB12ABCD12345678901234"
                ),
                "json_data": (
                    '{"invoice":{"invoice_number":"1001"},'
                    '"subtotal":{"tax":"10.00","total":"110.00"},'
                    '"payment_instructions":{"account_number":'
                    '"GB12ABCD12345678901234"},"items":[]}'
                ),
            },
            {
                "file_name": "incorrect.jpg",
                "ocred_text": (
                    "Invoice no: 2002 Tax: 20,00 Total: 220,00 "
                    "IBAN: GB99ABCD12345678901234"
                ),
                "json_data": (
                    '{"invoice":{"invoice_number":"2002"},'
                    '"subtotal":{"tax":"20.00","total":"220.00"},'
                    '"payment_instructions":{},"items":[]}'
                ),
            },
        ],
        index=pd.Index([101, 202]),
    )


@pytest.fixture
def evaluator_dependencies() -> tuple[
    FakeTrainer,
    FakeMetrics,
    FakePredictor,
    DocumentOutputFormatter,
]:
    """Provide recording collaborators for testing evaluator orchestration."""

    return (
        FakeTrainer(requested_splits=[]),
        FakeMetrics(decoded_inputs=[]),
        FakePredictor(),
        DocumentOutputFormatter(),
    )


@pytest.fixture
def model_evaluator(
    evaluator_dependencies: tuple[
        FakeTrainer,
        FakeMetrics,
        FakePredictor,
        DocumentOutputFormatter,
    ],
) -> ModelEvaluator:
    """Build ModelEvaluator with deterministic collaborators."""

    trainer, metrics, predictor, formatter = evaluator_dependencies
    return ModelEvaluator(
        trainer=cast(DocumentModelTrainer, trainer),
        metrics=cast(TokenClassificationMetrics, metrics),
        predictor=cast(DocumentModelPredictor, predictor),
        output_formatter=formatter,
    )


def test_build_field_records_parses_targets_and_formats_predictions(
    model_evaluator: ModelEvaluator,
    evaluator_dependencies: tuple[
        FakeTrainer,
        FakeMetrics,
        FakePredictor,
        DocumentOutputFormatter,
    ],
    evaluation_frame: pd.DataFrame,
) -> None:
    """Build one comparable actual/predicted JSON record for every test document."""

    records = model_evaluator._build_field_records(evaluation_frame)
    predictor = evaluator_dependencies[2]

    assert [record["row_index"] for record in records] == [101, 202]
    assert [record["file_name"] for record in records] == [
        "correct.jpg",
        "incorrect.jpg",
    ]
    assert records[0]["actual"]["invoice"]["invoice_number"] == "1001"
    assert records[0]["predicted"]["invoice"]["invoice_number"] == "1001"
    assert records[1]["predicted"]["invoice"]["invoice_number"] == "9999"
    assert predictor.predicted_texts == evaluation_frame["ocred_text"].tolist()


def test_evaluate_orchestrates_all_reports(
    model_evaluator: ModelEvaluator,
    evaluator_dependencies: tuple[
        FakeTrainer,
        FakeMetrics,
        FakePredictor,
        DocumentOutputFormatter,
    ],
    evaluation_frame: pd.DataFrame,
) -> None:
    """Combine trainer, model, field, document, and account-level evaluation outputs."""

    report = model_evaluator.evaluate(evaluation_frame)
    trainer, metrics, predictor, _ = evaluator_dependencies

    assert trainer.requested_splits == ["test"]
    assert len(metrics.decoded_inputs) == 1
    assert metrics.decoded_inputs[0][0].shape == (1, 1, 2)
    assert metrics.decoded_inputs[0][1].shape == (1, 1)
    assert predictor.predicted_texts == evaluation_frame["ocred_text"].tolist()

    assert report.summary == {"test_loss": 0.25}
    assert report.document_accuracy == pytest.approx(0.5)
    assert len(report.records) == 2

    field_metrics = report.field_metrics.set_index("field")
    assert field_metrics.loc[
        "invoice.invoice_number",
        "exact_match_rate",
    ] == pytest.approx(0.5)
    assert field_metrics.loc[
        "subtotal.tax",
        "exact_match_rate",
    ] == pytest.approx(1.0)
    assert field_metrics.loc[
        "subtotal.total",
        "exact_match_rate",
    ] == pytest.approx(1.0)

    account_metrics = report.account_metrics.set_index("metric")
    assert account_metrics.loc[
        "OCR account detected",
        "accuracy",
    ] == pytest.approx(1.0)
    assert account_metrics.loc[
        "Exact JSON match",
        "accuracy",
    ] == pytest.approx(1.0)

    assert not report.entity_metrics.empty
    assert set(report.account_records["row_index"]) == {101, 202}
