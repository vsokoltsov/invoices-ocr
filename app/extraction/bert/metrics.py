from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from seqeval.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_f1_score,
)
from transformers import EvalPrediction

# seqeval annotates this parameter as str, but only the
# integer 0 returns a score of 0 without a warning.
_ZERO_DIVISION = cast(str, 0)

@dataclass(frozen=True, slots=True)
class TokenClassificationMetrics:
    id2label: Mapping[int, str]

    @property
    def number_of_labels(self) -> int:
        return len(self.id2label)

    def decode(
        self,
        predictions: np.ndarray,
        label_ids: np.ndarray,
    ) -> tuple[
        list[list[str]],
        list[list[str]],
    ]:
        predicted_ids = np.argmax(
            predictions,
            axis=-1,
        )

        predicted_labels: list[
            list[str]
        ] = []

        actual_labels: list[
            list[str]
        ] = []

        for predicted_sequence, actual_sequence in zip(
            predicted_ids,
            label_ids,
        ):
            sequence_predictions = []
            sequence_actual = []

            for predicted_id, actual_id in zip(
                predicted_sequence,
                actual_sequence,
            ):
                if actual_id == -100:
                    continue

                sequence_predictions.append(
                    self.id2label[
                        int(predicted_id)
                    ]
                )

                sequence_actual.append(
                    self.id2label[
                        int(actual_id)
                    ]
                )

            predicted_labels.append(
                sequence_predictions
            )

            actual_labels.append(
                sequence_actual
            )

        return (
            predicted_labels,
            actual_labels,
        )

    def __call__(
        self,
        evaluation: EvalPrediction,
    ) -> dict[str, float]:
        predictions = evaluation.predictions

        # Some models return a tuple where logits are
        # the first item.
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        label_ids = evaluation.label_ids

        # Some models return labels as a one-element tuple.
        if isinstance(label_ids, tuple):
            label_ids = label_ids[0]

        predicted_ids = np.argmax(
            predictions,
            axis=-1,
        )

        valid_mask = label_ids != -100

        token_predictions = torch.as_tensor(
            predicted_ids[valid_mask],
            dtype=torch.long,
        )

        token_targets = torch.as_tensor(
            label_ids[valid_mask],
            dtype=torch.long,
        )

        predicted_labels, actual_labels = (
            self.decode(
                predictions=predictions,
                label_ids=label_ids,
            )
        )

        return {
            # Entity-level BIO metrics.
            "precision": precision_score(
                actual_labels,
                predicted_labels,
                zero_division=_ZERO_DIVISION,
            ),
            "recall": recall_score(
                actual_labels,
                predicted_labels,
                zero_division=_ZERO_DIVISION,
            ),
            "f1": f1_score(
                actual_labels,
                predicted_labels,
                zero_division=_ZERO_DIVISION,
            ),

            # Token-level TorchMetrics.
            "token_accuracy": float(
                multiclass_accuracy(
                    token_predictions,
                    token_targets,
                    num_classes=(
                        self.number_of_labels
                    ),
                    average="micro",
                ).item()
            ),
            "token_macro_f1": float(
                multiclass_f1_score(
                    token_predictions,
                    token_targets,
                    num_classes=(
                        self.number_of_labels
                    ),
                    average="macro",
                ).item()
            ),
        }