from dataclasses import dataclass

import torch
from seqeval.metrics import classification_report
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from app.extraction.enums import IGNORE_INDEX


@dataclass(frozen=True, slots=True)
class EvalResult:
    loss: float
    precision: float
    recall: float
    f1: float
    token_accuracy: float
    token_macro_f1: float
    per_entity: dict[str, dict[str, float | int]]


class EvalMetrics:
    """Accumulate token metrics and BIO entities for each validation window.

    Entity scoring uses seqeval's default BIO interpretation. Optional decoded
    predictions affect scores, while loss remains based on the original logits.
    """

    def __init__(self, id2label: dict[int, str], ignore_index: int = IGNORE_INDEX):
        self.id2label = dict(id2label)
        self.ignore_index = ignore_index
        self.accuracy: MulticlassAccuracy = MulticlassAccuracy(
            num_classes=len(id2label), average="micro"
        )
        self.macro_f1: MulticlassF1Score = MulticlassF1Score(
            num_classes=len(id2label), average="macro"
        )
        self.reset()

    def reset(self) -> None:
        self.accuracy.reset()
        self.macro_f1.reset()
        self.loss_sum = 0.0
        self.token_count = 0
        self.actual_sequences: list[list[str]] = []
        self.predicted_sequences: list[list[str]] = []

    @torch.inference_mode()
    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss: torch.Tensor,
        *,
        predictions: torch.Tensor | None = None,
    ) -> None:
        valid = targets != self.ignore_index
        count = int(valid.sum().item())
        if count == 0:
            return

        if predictions is None:
            predictions = logits.argmax(dim=-1)

        targets = targets.detach().cpu()
        predictions = predictions.detach().cpu()
        valid = valid.cpu()
        # TorchMetrics wraps these methods at runtime; ty misreads their binding.
        self.accuracy.update(predictions[valid], targets[valid])  # ty: ignore[invalid-argument-type]
        self.macro_f1.update(predictions[valid], targets[valid])  # ty: ignore[invalid-argument-type]
        self.loss_sum += loss.detach().item() * count
        self.token_count += count

        for truth, predicted, mask in zip(targets, predictions, valid):
            if not mask.any():
                continue
            self.actual_sequences.append(
                [self.id2label[index] for index in truth[mask].tolist()]
            )
            self.predicted_sequences.append(
                [self.id2label[index] for index in predicted[mask].tolist()]
            )

    def compute(self) -> EvalResult:
        if self.token_count == 0:
            raise ValueError("Evaluation loader contains no supervised tokens.")

        report = classification_report(
            self.actual_sequences,
            self.predicted_sequences,
            output_dict=True,
            zero_division=0,
        )
        per_entity = {
            entity: {
                key: int(value) if key == "support" else float(value)
                for key, value in metrics.items()
            }
            for entity, metrics in report.items()
        }
        micro = per_entity["micro avg"]
        return EvalResult(
            loss=self.loss_sum / self.token_count,
            precision=float(micro["precision"]),
            recall=float(micro["recall"]),
            f1=float(micro["f1-score"]),
            token_accuracy=self.accuracy.compute().item(),  # ty: ignore[missing-argument]
            token_macro_f1=self.macro_f1.compute().item(),  # ty: ignore[missing-argument]
            per_entity=per_entity,
        )
