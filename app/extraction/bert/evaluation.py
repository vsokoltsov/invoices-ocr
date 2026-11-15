from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from app.extraction.batches import ArgmaxDecoder, TensorBatchAdapter
from app.extraction.bert.decoding import OCRWordBatchDecoder, OCRWordDecoder
from app.extraction.bert.metrics import EvalMetrics, EvalResult
from app.extraction.evaluation import TokenClassificationEvaluation
from app.extraction.protocols import PredictionDecoder


@dataclass
class BertEvaluation(TokenClassificationEvaluation[EvalResult]):
    """Score BERT token windows using a composed prediction decoder.

    The caller supplies the model, execution device, and criterion through the
    shared evaluation constructor.

    Attributes:
        metrics: Token and default BIO metric accumulator.
        word_decoder: Optional OCR-word pooling strategy; None selects token argmax.
    """

    metrics: EvalMetrics
    word_decoder: OCRWordDecoder | None = None

    batch_adapter = TensorBatchAdapter(("input_ids", "attention_mask"))

    @property
    def ignore_index(self) -> int:
        """Return the target ID excluded from scoring.

        Returns:
            Ignore index configured on the metric accumulator.
        """
        return self.metrics.ignore_index

    def reset(self) -> None:
        """Clear metrics and select the configured prediction decoder."""
        self.metrics.reset()
        self.decoder: PredictionDecoder = (
            OCRWordBatchDecoder(self.word_decoder)
            if self.word_decoder is not None
            else ArgmaxDecoder()
        )

    def update(
        self,
        batch: dict,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss: torch.Tensor,
    ) -> None:
        """Accumulate metrics using token or OCR-word predictions.

        Args:
            batch: Original batch with word/page metadata or decoding metadata.
            logits: Raw token scores.
            labels: Target label IDs with ignored positions masked.
            loss: Scalar mean loss over supervised tokens.

        Raises:
            ValueError: OCR-word decoding is enabled but its metadata is absent.
        """
        self.metrics.update(
            logits, labels, loss, predictions=self.decoder.decode(logits, batch)
        )

    def compute(self, loader: DataLoader) -> EvalResult:
        """Return accumulated token and window-level entity scores.

        Args:
            loader: Evaluated loader; unused for window-level scoring.

        Returns:
            Token loss, accuracy, macro F1, and default BIO entity scores.

        Raises:
            ValueError: No supervised tokens have been accumulated.
        """
        return self.metrics.compute()
