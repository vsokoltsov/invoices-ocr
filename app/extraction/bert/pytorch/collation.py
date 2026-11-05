from dataclasses import dataclass, field

from transformers import DataCollatorForTokenClassification, PreTrainedTokenizerBase


@dataclass
class InvoiceDataCollator:
    """Pad model inputs while keeping OCR decoding metadata outside tensors."""

    tokenizer: PreTrainedTokenizerBase
    collator: DataCollatorForTokenClassification = field(init=False)

    def __post_init__(self):
        self.collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer, label_pad_token_id=-100
        )

    def __call__(self, examples):
        features = [dict(example) for example in examples]
        metadata = [feature.pop("decoder_metadata", None) for feature in features]
        batch = self.collator(features)
        if any(item is not None for item in metadata):
            if any(item is None for item in metadata):
                raise ValueError("A decoding batch requires metadata for every example")
            batch["decoder_metadata"] = metadata
        return batch
