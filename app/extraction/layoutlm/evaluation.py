from dataclasses import dataclass
from typing import cast

import torch
from seqeval.scheme import IOB2
from torch.utils.data import DataLoader

from app.extraction.batches import TensorBatchAdapter
from app.extraction.evaluation import TokenClassificationEvaluation
from app.extraction.layoutlm.dataset import InvoiceDataset
from app.extraction.layoutlm.metrics import EpochMetrics, ValidationMetricsResult
from app.extraction.metrics import BioEntityScorer
from app.extraction.schema import InvoiceLabelSchema


@dataclass
class LayoutLMEvaluation(TokenClassificationEvaluation[ValidationMetricsResult]):
    """Score LayoutLM words and strict BIO entities after page reconstruction.

    The caller supplies the model, execution device, and criterion through the
    shared evaluation constructor.

    Attributes:
        metrics: Word-level confusion matrix and loss accumulator.
        schema: BIO mappings used to reconstruct page label sequences.
    """

    metrics: EpochMetrics
    schema: InvoiceLabelSchema

    batch_adapter = TensorBatchAdapter(
        ("input_ids", "attention_mask", "bbox", "pixel_values")
    )

    @property
    def ignore_index(self) -> int:
        """Return the target ID excluded from scoring.

        Returns:
            Ignore index configured on the metric accumulator.
        """
        return self.metrics.ignore_index

    def reset(self) -> None:
        """Clear metrics and initialize prediction collection for a fresh pass."""
        self.metrics.reset()
        self.word_predictions: dict[tuple[int, int], int] = {}

    def update(
        self,
        batch: dict,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss: torch.Tensor,
    ) -> None:
        """Collect first-subtoken predictions keyed by page and OCR word.

        Args:
            batch: Original batch with word/page metadata or decoding metadata.
            logits: Raw token scores.
            labels: Target label IDs with ignored positions masked.
            loss: Scalar mean loss over supervised tokens.

        Raises:
            ValueError: A page word appears more than once in the evaluated windows.
        """
        self.metrics.update(logits, labels, loss)
        predicted = logits.argmax(dim=-1).cpu()
        for batch_index, page_index in enumerate(batch["page_index"].tolist()):
            for token_index, word_index in enumerate(
                batch["word_ids"][batch_index].tolist()
            ):
                if word_index < 0:
                    continue

                key = (page_index, word_index)

                if key in self.word_predictions:
                    raise ValueError("A word was evaluated more than once")

                self.word_predictions[key] = int(
                    predicted[batch_index, token_index].item()
                )

    def compute(self, loader: DataLoader) -> ValidationMetricsResult:
        # A token-window boundary must not split an entity during page scoring.
        """Reconstruct complete pages and score strict IOB2 entities.

        Args:
            loader: Loader whose dataset supplies complete page records.

        Returns:
            Word metrics plus strict page-level entity scores.

        Raises:
            KeyError: The loader omitted an expected page word.
        """
        true_sequences = []
        pred_sequences = []

        dataset = cast(InvoiceDataset, loader.dataset)
        for page_index, page in enumerate(dataset.pages):
            true_sequences.append(
                [self.schema.id2label[label] for label in page["labels"]]
            )

            pred_sequences.append(
                [
                    self.schema.id2label[
                        self.word_predictions[(page_index, word_index)]
                    ]
                    for word_index in range(len(page["words"]))
                ]
            )

        report = BioEntityScorer(mode="strict", scheme=IOB2).score(
            true_sequences, pred_sequences
        )
        return ValidationMetricsResult(
            word_metrics=self.metrics.compute(),
            entity_precision=report.precision,
            entity_recall=report.recall,
            entity_f1=report.f1,
            entity_report=report.per_entity,
        )
