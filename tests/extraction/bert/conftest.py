from collections.abc import Callable
from typing import Mapping, Protocol

import pandas as pd
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from app.extraction.bert.alignment import DocumentAligner, FinancialSummaryAligner
from app.extraction.bert.candidates import CandidatesSelector
from app.extraction.bert.encoding import TokenLabelBatchEncoder
from app.extraction.bert.enums import EntityType
from app.extraction.bert.models import InvoiceTarget, PredictedEntity
from app.extraction.bert.reporting import AlignmentReport
from app.extraction.bert.spans import CandidateSpan


class _SpanFor(Protocol):
    def __call__(
        self,
        text: str,
        value: str,
        occurrence: int = 1,
    ) -> CandidateSpan: ...


class _TargetFactory(Protocol):
    def __call__(
        self,
        *,
        invoice: dict[str, object] | None = None,
        subtotal: dict[str, object] | None = None,
        payment_instructions: dict[str, object] | None = None,
    ) -> InvoiceTarget: ...


class _PredictedEntityFactory(Protocol):
    def __call__(
        self,
        *,
        entity: EntityType,
        start: int,
        text: str,
        confidence: float = 0.9,
    ) -> PredictedEntity: ...


@pytest.fixture
def candidate_factory() -> Callable[[int, int], CandidateSpan]:
    def create(start: int, end: int) -> CandidateSpan:
        return CandidateSpan(start=start, end=end)

    return create


@pytest.fixture
def span_for() -> _SpanFor:
    def create(text: str, value: str, occurrence: int = 1) -> CandidateSpan:
        start = -1

        for _ in range(occurrence):
            start = text.index(value, start + 1)

        return CandidateSpan(start=start, end=start + len(value))

    return create


@pytest.fixture
def document_aligner() -> DocumentAligner:
    return DocumentAligner(
        financial_summary_aligner=FinancialSummaryAligner(),
        selector=CandidatesSelector(),
    )


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
    vocabulary = {
        token: token_id
        for token_id, token in enumerate(tokens)
    }
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


@pytest.fixture
def label2id() -> dict[str, int]:
    """Provide BIO label IDs for every entity supported by the application."""

    labels = ["O"]
    for entity in EntityType:
        labels.extend(
            [
                f"B-{entity.value}",
                f"I-{entity.value}",
            ]
        )
    return {
        label: label_id
        for label_id, label in enumerate(labels)
    }


@pytest.fixture
def batch_encoder(
    fast_tokenizer: PreTrainedTokenizerFast,
    document_aligner: DocumentAligner,
    label2id: Mapping[str, int],
) -> TokenLabelBatchEncoder:
    """Build a token-label encoder with deterministic offline collaborators."""

    return TokenLabelBatchEncoder(
        tokenizer=fast_tokenizer,
        aligner=document_aligner,
        label2id=label2id,
        max_length=32,
        window_stride=4,
    )


@pytest.fixture
def target_factory() -> _TargetFactory:
    def create(
        *,
        invoice: dict[str, object] | None = None,
        subtotal: dict[str, object] | None = None,
        payment_instructions: dict[str, object] | None = None,
    ) -> InvoiceTarget:
        return InvoiceTarget.model_validate(
            {
                "invoice": invoice or {},
                "subtotal": subtotal or {},
                "payment_instructions": payment_instructions or {},
                "items": [],
            }
        )

    return create


@pytest.fixture
def predicted_entity_factory() -> _PredictedEntityFactory:
    """Create valid predicted entities without repeating span bookkeeping."""

    def create(
        *,
        entity: EntityType,
        start: int,
        text: str,
        confidence: float = 0.9,
    ) -> PredictedEntity:
        return PredictedEntity(
            entity=entity,
            start=start,
            end=start + len(text),
            text=text,
            confidence=confidence,
        )

    return create


@pytest.fixture
def alignment_frame(target_factory: _TargetFactory) -> pd.DataFrame:
    complete_target = target_factory(
        invoice={
            "invoice_number": "1001",
            "seller_name": "Acme",
        },
        subtotal={
            "tax": "10.00",
            "total": "100.00",
        },
    )
    partial_target = target_factory(
        invoice={
            "invoice_number": "2002",
            "seller_name": "Missing Corp",
        },
        subtotal={
            "tax": "5.00",
            "total": "55.00",
        },
    )

    return pd.DataFrame(
        [
            {
                "file_name": "complete.jpg",
                "ocred_text": (
                    "Invoice no: 1001 Seller: Acme "
                    "Total $ 90.00 $ 10.00 $ 100.00"
                ),
                "target": complete_target,
            },
            {
                "file_name": "partial.jpg",
                "ocred_text": (
                    "Invoice no: 2002 "
                    "Total $ 45.00 $ 5.00 $ 50.00"
                ),
                "target": partial_target,
            },
        ],
        index=pd.Index([10, 20]),
    )


@pytest.fixture
def alignment_report(
    alignment_frame: pd.DataFrame,
    document_aligner: DocumentAligner,
) -> AlignmentReport:
    return AlignmentReport.build(
        frame=alignment_frame,
        aligner=document_aligner,
    )
