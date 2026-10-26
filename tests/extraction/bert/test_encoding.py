from collections import Counter
from collections.abc import Mapping

import pytest
from transformers import PreTrainedTokenizerFast

from app.extraction.bert.alignment import DocumentAligner
from app.extraction.bert.encoding import TokenizedBatch, TokenLabelBatchEncoder
from app.extraction.bert.models import InvoiceTarget


@pytest.fixture
def invoice_text() -> str:
    """Provide OCR text containing one single-token and one multi-token entity."""

    return "Invoice no: 40108666 Seller: Acme Corporation"


@pytest.fixture
def invoice_target(target_factory) -> InvoiceTarget:
    """Provide expected values that can be aligned in the encoder OCR fixture."""

    return target_factory(
        invoice={
            "invoice_number": "40108666",
            "seller_name": "Acme Corporation",
        }
    )


def _label_names(
    batch: TokenizedBatch,
    label2id: Mapping[str, int],
) -> list[str]:
    id2label = {
        label_id: label
        for label, label_id in label2id.items()
    }
    return [
        id2label[label_id]
        for chunk in batch["labels"]
        for label_id in chunk
        if label_id != -100
    ]


def test_encoder_converts_aligned_spans_to_bio_labels(
    batch_encoder: TokenLabelBatchEncoder,
    invoice_text: str,
    invoice_target: InvoiceTarget,
    label2id: Mapping[str, int],
) -> None:
    """Label the first entity token with B and its continuation with I."""

    result = batch_encoder(
        {
            "ocred_text": [invoice_text],
            "json_data": [invoice_target],
        }
    )

    counts = Counter(_label_names(result, label2id))

    assert counts["B-INVOICE_NUMBER"] == 1
    assert counts["I-INVOICE_NUMBER"] == 0
    assert counts["B-SELLER_NAME"] == 1
    assert counts["I-SELLER_NAME"] == 1
    assert counts["O"] > 0


def test_encoder_keeps_model_inputs_and_labels_positionally_aligned(
    batch_encoder: TokenLabelBatchEncoder,
    invoice_text: str,
    invoice_target: InvoiceTarget,
) -> None:
    """Return one label for every token in every generated model input chunk."""

    result = batch_encoder(
        {
            "ocred_text": [invoice_text],
            "json_data": [invoice_target],
        }
    )

    chunk_count = len(result["input_ids"])

    assert chunk_count == 1
    assert len(result["attention_mask"]) == chunk_count
    assert len(result["token_type_ids"]) == chunk_count
    assert len(result["labels"]) == chunk_count
    assert all(
        len(input_ids)
        == len(attention_mask)
        == len(token_type_ids)
        == len(labels)
        for input_ids, attention_mask, token_type_ids, labels in zip(
            result["input_ids"],
            result["attention_mask"],
            result["token_type_ids"],
            result["labels"],
            strict=True,
        )
    )


def test_encoder_ignores_special_tokens(
    batch_encoder: TokenLabelBatchEncoder,
    invoice_text: str,
    invoice_target: InvoiceTarget,
) -> None:
    """Assign -100 to special tokens so they do not contribute to model loss."""

    result = batch_encoder(
        {
            "ocred_text": [invoice_text],
            "json_data": [invoice_target],
        }
    )

    assert result["labels"][0][0] == -100
    assert result["labels"][0][-1] == -100


@pytest.mark.parametrize("target_format", ["model", "dict", "json"])
def test_encoder_accepts_all_supported_target_formats(
    batch_encoder: TokenLabelBatchEncoder,
    invoice_text: str,
    invoice_target: InvoiceTarget,
    label2id: Mapping[str, int],
    target_format: str,
) -> None:
    """Parse targets supplied as Pydantic models, dictionaries, or JSON strings."""

    match target_format:
        case "model":
            target: object = invoice_target
        case "dict":
            target = invoice_target.model_dump()
        case "json":
            target = invoice_target.model_dump_json()
        case _:
            raise AssertionError(target_format)

    result = batch_encoder(
        {
            "ocred_text": [invoice_text],
            "json_data": [target],
        }
    )

    counts = Counter(_label_names(result, label2id))
    assert counts["B-INVOICE_NUMBER"] == 1
    assert counts["B-SELLER_NAME"] == 1


def test_encoder_assigns_overlapping_entity_to_only_one_chunk(
    fast_tokenizer: PreTrainedTokenizerFast,
    document_aligner: DocumentAligner,
    label2id: Mapping[str, int],
    target_factory,
) -> None:
    """Avoid duplicate BIO supervision when stride makes an entity span repeat."""

    encoder = TokenLabelBatchEncoder(
        tokenizer=fast_tokenizer,
        aligner=document_aligner,
        label2id=label2id,
        max_length=8,
        window_stride=3,
    )
    text = "prefix prefix prefix prefix Acme Corporation suffix suffix"
    target = target_factory(
        invoice={"seller_name": "Acme Corporation"},
    )

    result = encoder(
        {
            "ocred_text": [text],
            "json_data": [target],
        }
    )
    counts = Counter(_label_names(result, label2id))

    assert len(result["input_ids"]) > 1
    assert counts["B-SELLER_NAME"] == 1
    assert counts["I-SELLER_NAME"] == 1


def test_encoder_labels_entity_near_end_of_long_document(
    fast_tokenizer: PreTrainedTokenizerFast,
    document_aligner: DocumentAligner,
    label2id: Mapping[str, int],
    target_factory,
) -> None:
    """Preserve an entity beyond the first model window instead of truncating it."""

    encoder = TokenLabelBatchEncoder(
        tokenizer=fast_tokenizer,
        aligner=document_aligner,
        label2id=label2id,
        max_length=8,
        window_stride=2,
    )
    text = " ".join([*["prefix"] * 24, "40108666"])
    target = target_factory(
        invoice={"invoice_number": "40108666"},
    )

    result = encoder(
        {
            "ocred_text": [text],
            "json_data": [target],
        }
    )
    counts = Counter(_label_names(result, label2id))

    assert len(result["input_ids"]) > 1
    assert counts["B-INVOICE_NUMBER"] == 1


def test_encoder_ignores_unannotated_account_number_tokens(
    batch_encoder: TokenLabelBatchEncoder,
    fast_tokenizer: PreTrainedTokenizerFast,
    target_factory,
) -> None:
    """Exclude an OCR IBAN from loss when the JSON target leaves it unannotated."""

    account_number = "GB12ABCD12345678901234"
    result = batch_encoder(
        {
            "ocred_text": [f"IBAN: {account_number}"],
            "json_data": [target_factory()],
        }
    )
    account_token_id = fast_tokenizer.convert_tokens_to_ids(account_number)
    account_positions = [
        (chunk_index, token_index)
        for chunk_index, input_ids in enumerate(result["input_ids"])
        for token_index, token_id in enumerate(input_ids)
        if token_id == account_token_id
    ]

    assert len(account_positions) == 1
    chunk_index, token_index = account_positions[0]
    assert result["labels"][chunk_index][token_index] == -100


def test_encoder_labels_annotated_account_number(
    batch_encoder: TokenLabelBatchEncoder,
    target_factory,
    label2id: Mapping[str, int],
) -> None:
    """Use normal ACCOUNT_NUMBER supervision when JSON contains the OCR IBAN."""

    account_number = "GB12ABCD12345678901234"
    target = target_factory(
        payment_instructions={
            "account_number": account_number,
        }
    )

    result = batch_encoder(
        {
            "ocred_text": [f"IBAN: {account_number}"],
            "json_data": [target],
        }
    )
    counts = Counter(_label_names(result, label2id))

    assert counts["B-ACCOUNT_NUMBER"] == 1


@pytest.mark.parametrize(
    ("max_length", "window_stride", "message"),
    [
        (0, 0, "max_length must be positive"),
        (8, -1, "window_stride must be between"),
        (8, 8, "window_stride must be between"),
    ],
)
def test_encoder_rejects_invalid_window_configuration(
    fast_tokenizer: PreTrainedTokenizerFast,
    document_aligner: DocumentAligner,
    label2id: Mapping[str, int],
    max_length: int,
    window_stride: int,
    message: str,
) -> None:
    """Reject chunk settings that cannot produce valid tokenizer windows."""

    with pytest.raises(ValueError, match=message):
        TokenLabelBatchEncoder(
            tokenizer=fast_tokenizer,
            aligner=document_aligner,
            label2id=label2id,
            max_length=max_length,
            window_stride=window_stride,
        )


def test_encoder_requires_outside_label(
    fast_tokenizer: PreTrainedTokenizerFast,
    document_aligner: DocumentAligner,
) -> None:
    """Fail early when the mandatory O label is absent from the mapping."""

    with pytest.raises(ValueError, match="must contain the O label"):
        TokenLabelBatchEncoder(
            tokenizer=fast_tokenizer,
            aligner=document_aligner,
            label2id={"B-INVOICE_NUMBER": 0},
        )


def test_encoder_reports_missing_entity_label(
    fast_tokenizer: PreTrainedTokenizerFast,
    document_aligner: DocumentAligner,
    target_factory,
) -> None:
    """Identify an entity label omitted from label2id instead of mislabelling it."""

    encoder = TokenLabelBatchEncoder(
        tokenizer=fast_tokenizer,
        aligner=document_aligner,
        label2id={"O": 0},
    )
    target = target_factory(
        invoice={"invoice_number": "40108666"},
    )

    with pytest.raises(KeyError, match="B-INVOICE_NUMBER"):
        encoder(
            {
                "ocred_text": ["Invoice no: 40108666"],
                "json_data": [target],
            }
        )
