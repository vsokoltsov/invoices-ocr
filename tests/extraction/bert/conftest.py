import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from app.extraction.bert.normalization import Normalizer


@pytest.fixture
def normalizer():
    return Normalizer()


@pytest.fixture
def fast_tokenizer() -> PreTrainedTokenizerFast:
    """Provide an offline fast tokenizer with real offsets and overflow windows."""

    tokens = [
        "[UNK]",
        "[PAD]",
        "[CLS]",
        "[SEP]",
        "Invoice",
        "no",
        ":",
        "40108666",
        "Seller",
        "Acme",
        "Corporation",
        "prefix",
        "middle",
        "suffix",
        "IBAN",
        "GB12ABCD12345678901234",
    ]
    vocabulary = {token: token_id for token_id, token in enumerate(tokens)}
    backend = Tokenizer(
        WordLevel(
            vocabulary,
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[
            ("[CLS]", vocabulary["[CLS]"]),
            ("[SEP]", vocabulary["[SEP]"]),
        ],
    )

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )
    tokenizer.model_input_names = [
        "input_ids",
        "token_type_ids",
        "attention_mask",
    ]
    return tokenizer
