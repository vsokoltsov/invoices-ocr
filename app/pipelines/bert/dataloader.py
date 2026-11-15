from dataclasses import dataclass

import pandas as pd
from transformers.tokenization_utils_tokenizers import TokenizersBackend

from app.extraction.bert.collation import InvoiceDataCollator
from app.extraction.bert.datasets import InvoiceDataset
from app.extraction.schema import InvoiceLabelSchema
from app.pipelines.dataloader import DataLoaders


@dataclass(kw_only=True)
class InvoiceDataLoader(DataLoaders):
    """Create BERT span datasets with optional OCR-word decoding metadata.

    Loader size, workers, device, and token-window settings are inherited from
    DataLoaders. Evaluation order stays fixed while training samples are shuffled.

    Attributes:
        schema: Shared field mapping and BIO label IDs.
        tokenizer: Fast tokenizer used for span alignment.
        collate_fn: Collator that pads token windows and retains decoder metadata.
        random_seed: Independent seed used for training shuffle order.
        include_decoder_metadata: Whether evaluation windows retain OCR offsets.
    """

    schema: InvoiceLabelSchema
    tokenizer: TokenizersBackend
    collate_fn: InvoiceDataCollator
    random_seed: int = 42
    include_decoder_metadata: bool = False

    def create_dataset(self, frame: pd.DataFrame, *, training: bool) -> InvoiceDataset:
        """Create a token-window dataset for one prepared partition.

        Args:
            frame: Prepared BERT span records with OCR words for evaluation.
            training: Excludes evaluation-only metadata when True.

        Returns:
            Token windows compatible with the configured BERT collator.
        """
        return InvoiceDataset(
            records=frame.to_dict(orient="records"),
            tokenizer=self.tokenizer,
            label2id=self.schema.label2id,
            field_to_entity=self.schema.field_to_entity,
            max_length=self.max_length_tokens,
            include_decoder_metadata=not training and self.include_decoder_metadata,
        )
