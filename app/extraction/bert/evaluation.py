from typing import Any
from dataclasses import dataclass
from seqeval.metrics import classification_report
import pandas as pd
import numpy as np
from app.extraction.bert.operations import DocumentModelTrainer
from app.extraction.bert.metrics import TokenClassificationMetrics
from app.extraction.bert.inference import DocumentModelPredictor
from app.extraction.bert.formatter import (
    DocumentOutputFormatter,
    evaluate_json_records,
    build_account_evaluation,
    document_is_exact
)
from app.extraction.bert.models import (
    ModelEvaluationReport,
    InvoiceTarget
)

@dataclass(slots=True)
class ModelEvaluator:
    trainer: DocumentModelTrainer
    metrics: TokenClassificationMetrics
    predictor: DocumentModelPredictor
    output_formatter: DocumentOutputFormatter

    def evaluate(
        self,
        test_frame: pd.DataFrame,
    ) -> ModelEvaluationReport:
        prediction_output = (
            self.trainer.predict_dataset(
                "test"
            )
        )

        predictions = (
            prediction_output.predictions
        )

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        predicted_labels, actual_labels = (
            self.metrics.decode(
                predictions=predictions,
                label_ids=(
                    prediction_output.label_ids
                ),
            )
        )

        entity_metrics = pd.DataFrame(
            classification_report(
                actual_labels,
                predicted_labels,
                output_dict=True,
                zero_division=0,
            )
        ).transpose()

        evaluation_records = (
            self._build_field_records(
                test_frame
            )
        )

        field_metrics = (
            evaluate_json_records(
                evaluation_records
            )
        )

        (
            account_records,
            account_metrics,
        ) = build_account_evaluation(
            evaluation_records
        )

        document_accuracy = float(
            np.mean([
                document_is_exact(record)
                for record
                in evaluation_records
            ])
        )

        return ModelEvaluationReport(
            summary=dict(
                prediction_output.metrics
            ),
            entity_metrics=entity_metrics,
            field_metrics=field_metrics,
            document_accuracy=(
                document_accuracy
            ),
            records=pd.DataFrame(
                evaluation_records
            ),
            account_metrics=(
                account_metrics
            ),
            account_records=(
                account_records
            ),
        )

    def _build_field_records(
        self,
        frame: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        records = []

        for row_index, row in frame.iterrows():
            text = str(
                row["ocred_text"]
            )

            target = (
                InvoiceTarget
                .model_validate_json(
                    row["json_data"]
                )
            )

            predicted_entities = (
                self.predictor.predict(
                    text
                )
            )

            predicted_output = (
                self.output_formatter.format(
                    predicted_entities
                )
            )

            records.append({
                "row_index": row_index,
                "file_name": row.get(
                    "file_name"
                ),
                "ocred_text": text,
                "actual": target.model_dump(
                    exclude_none=True
                ),
                "predicted": (
                    predicted_output.model_dump(
                        exclude_none=True
                    )
                ),
            })

        return records