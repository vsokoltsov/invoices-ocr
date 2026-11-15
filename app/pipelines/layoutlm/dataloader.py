from dataclasses import dataclass

import pandas as pd
from transformers.image_processing_utils import BaseImageProcessor
from transformers.tokenization_utils_tokenizers import TokenizersBackend

from app.extraction.layoutlm.converter import AttachmentToImageConverter
from app.extraction.layoutlm.dataset import InvoiceDataset
from app.pipelines.dataloader import DataLoaders


@dataclass
class InvoiceDataLoaders(DataLoaders):
    """Create LayoutLM datasets using configured image and token preprocessing.

    Loader size, workers, device, and token-window settings are inherited from
    DataLoaders. Fixed-length samples use the default PyTorch collator.

    Attributes:
        tokenizer: Fast tokenizer used to build complete-word windows.
        image_processor: Image preprocessing with built-in OCR disabled.
        converter: Loader for cached or rendered document page images.
    """

    tokenizer: TokenizersBackend
    image_processor: BaseImageProcessor
    converter: AttachmentToImageConverter

    def create_dataset(self, frame: pd.DataFrame, *, training: bool) -> InvoiceDataset:
        """Create image, token, and layout samples for one partition.

        Args:
            frame: Prepared page records with words, boxes, and labels.
            training: Identifies the training split; page representation is unchanged.

        Returns:
            Complete-word windows with page images and evaluation coordinates.
        """
        return InvoiceDataset(
            pages=frame.to_dict(orient="records"),
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            converter=self.converter,
            max_length=self.max_length_tokens,
        )
