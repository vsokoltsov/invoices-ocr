from collections.abc import Iterator, Mapping
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from app.extraction.bert.enums import EntityType
from app.extraction.bert.inference import DocumentModelPredictor


class DeterministicTokenClassifier:
    """A minimal classifier that emits fixed labels for configured token IDs."""

    def __init__(
        self,
        *,
        number_of_labels: int,
        labels_by_token_id: Mapping[int, int],
    ) -> None:
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.number_of_labels = number_of_labels
        self.labels_by_token_id = labels_by_token_id
        self.forward_calls = 0

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """Provide the parameter Predictor uses to determine its inference device."""

        yield self.anchor

    def eval(self) -> None:
        """Match the model evaluation-mode interface used by the predictor."""

    def __call__(
        self,
        input_ids: torch.Tensor,
        **_: torch.Tensor,
    ) -> SimpleNamespace:
        self.forward_calls += 1
        batch_size, sequence_length = input_ids.shape
        logits = torch.full(
            (batch_size, sequence_length, self.number_of_labels),
            fill_value=-10.0,
            device=input_ids.device,
        )
        logits[..., 0] = 10.0

        for token_id, label_id in self.labels_by_token_id.items():
            if label_id == 0:
                continue

            token_mask = input_ids.eq(token_id)
            logits[..., 0] = logits[..., 0].masked_fill(token_mask, -10.0)
            logits[..., label_id] = logits[..., label_id].masked_fill(
                token_mask,
                10.0,
            )

        return SimpleNamespace(logits=logits)


@pytest.fixture
def predictor_id2label() -> dict[int, str]:
    """Provide labels predicted by the deterministic token-classification model."""

    return {
        0: "O",
        1: "B-INVOICE_NUMBER",
        2: "B-SELLER_NAME",
        3: "I-SELLER_NAME",
    }


@pytest.fixture
def deterministic_model(
    fast_tokenizer,
    predictor_id2label: Mapping[int, str],
) -> DeterministicTokenClassifier:
    """Provide a CPU model that labels known invoice and seller tokens."""

    return DeterministicTokenClassifier(
        number_of_labels=len(predictor_id2label),
        labels_by_token_id={
            fast_tokenizer.convert_tokens_to_ids("40108666"): 1,
            fast_tokenizer.convert_tokens_to_ids("Acme"): 2,
            fast_tokenizer.convert_tokens_to_ids("Corporation"): 3,
        },
    )


@pytest.fixture
def predictor(
    deterministic_model: DeterministicTokenClassifier,
    fast_tokenizer,
    predictor_id2label: Mapping[int, str],
) -> DocumentModelPredictor:
    """Build a predictor using the real fast-tokenizer fixture and fixed model."""

    return DocumentModelPredictor(
        model=cast(PreTrainedModel, deterministic_model),
        tokenizer=fast_tokenizer,
        id2label=predictor_id2label,
        max_length=32,
        window_stride=4,
    )


@pytest.fixture
def inference_text() -> str:
    """Provide OCR text containing the token IDs configured by the test model."""

    return "Invoice no: 40108666 Seller: Acme Corporation"


def test_predict_returns_entities_with_original_text_offsets(
    predictor: DocumentModelPredictor,
    deterministic_model: DeterministicTokenClassifier,
    inference_text: str,
) -> None:
    """Reconstruct nested BIO predictions as entities from the original OCR string."""

    entities = predictor.predict(inference_text)

    assert deterministic_model.forward_calls == 1
    assert [(entity.entity, entity.text) for entity in entities] == [
        (EntityType.INVOICE_NUMBER, "40108666"),
        (EntityType.SELLER_NAME, "Acme Corporation"),
    ]
    assert [(entity.start, entity.end) for entity in entities] == [
        (inference_text.index("40108666"), inference_text.index("40108666") + 8),
        (
            inference_text.index("Acme"),
            inference_text.index("Corporation") + len("Corporation"),
        ),
    ]
    assert all(entity.confidence > 0.99 for entity in entities)


def test_predict_returns_no_entities_for_blank_text(
    predictor: DocumentModelPredictor,
    deterministic_model: DeterministicTokenClassifier,
) -> None:
    """Avoid tokenization and model inference when OCR text has no content."""

    assert predictor.predict(" \n\t ") == []
    assert deterministic_model.forward_calls == 0


def test_predict_merges_repeated_window_tokens_into_one_entity(
    deterministic_model: DeterministicTokenClassifier,
    fast_tokenizer,
    predictor_id2label: Mapping[int, str],
) -> None:
    """Emit one entity when the same entity tokens occur in overlapping windows."""

    predictor = DocumentModelPredictor(
        model=cast(PreTrainedModel, deterministic_model),
        tokenizer=fast_tokenizer,
        id2label=predictor_id2label,
        max_length=8,
        window_stride=3,
    )
    text = "prefix prefix prefix prefix Acme Corporation suffix suffix"

    entities = predictor.predict(text)

    assert [(entity.entity, entity.text) for entity in entities] == [
        (EntityType.SELLER_NAME, "Acme Corporation"),
    ]
    assert deterministic_model.forward_calls == 1


def test_merge_chunk_predictions_averages_logits_by_character_offset(
    predictor: DocumentModelPredictor,
) -> None:
    """Average repeated token logits and return one sorted prediction per offset."""

    logits = torch.tensor(
        [
            [[0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]],
            [[0.0, 4.0, 0.0, 0.0], [0.0, 0.0, 4.0, 0.0]],
        ]
    )

    predictions = predictor._merge_chunk_predictions(
        logits=logits,
        offset_chunks=[
            [[0, 4], [5, 9]],
            [[0, 4], [5, 9]],
        ],
    )

    assert [(start, end, label) for start, end, label, _ in predictions] == [
        (0, 4, "B-INVOICE_NUMBER"),
        (5, 9, "B-SELLER_NAME"),
    ]
    assert all(confidence > 0.85 for *_, confidence in predictions)


def test_tokens_to_entities_recovers_from_initial_i_label(
    predictor: DocumentModelPredictor,
) -> None:
    """Treat I-ENTITY without a preceding B-ENTITY as a new entity boundary."""

    text = "x 1001 2002 Acme Corporation"
    entities = predictor._tokens_to_entities(
        text=text,
        token_predictions=[
            (0, 1, "O", 1.0),
            (2, 6, "I-INVOICE_NUMBER", 0.8),
            (7, 11, "I-INVOICE_NUMBER", 0.6),
            (12, 16, "B-SELLER_NAME", 0.9),
            (17, 28, "I-SELLER_NAME", 0.7),
        ],
    )

    assert [(entity.entity, entity.text) for entity in entities] == [
        (EntityType.INVOICE_NUMBER, "1001 2002"),
        (EntityType.SELLER_NAME, "Acme Corporation"),
    ]
    assert [entity.confidence for entity in entities] == [
        pytest.approx(0.7),
        pytest.approx(0.8),
    ]


@pytest.mark.parametrize(
    ("label", "message"),
    [
        ("SELLER_NAME", "Invalid BIO label"),
        ("X-SELLER_NAME", "Invalid BIO prefix"),
    ],
)
def test_tokens_to_entities_rejects_invalid_bio_labels(
    predictor: DocumentModelPredictor,
    label: str,
    message: str,
) -> None:
    """Reject malformed labels instead of silently producing an invalid output span."""

    with pytest.raises(ValueError, match=message):
        predictor._tokens_to_entities(
            text="Acme",
            token_predictions=[(0, 4, label, 0.9)],
        )


@pytest.mark.parametrize(
    ("max_length", "window_stride", "message"),
    [
        (0, 0, "max_length must be positive"),
        (8, -1, "window_stride must be between"),
        (8, 8, "window_stride must be between"),
    ],
)
def test_predictor_rejects_invalid_window_configuration(
    deterministic_model: DeterministicTokenClassifier,
    fast_tokenizer,
    predictor_id2label: Mapping[int, str],
    max_length: int,
    window_stride: int,
    message: str,
) -> None:
    """Reject tokenizer window settings that cannot produce valid overflow chunks."""

    with pytest.raises(ValueError, match=message):
        DocumentModelPredictor(
            model=cast(PreTrainedModel, deterministic_model),
            tokenizer=fast_tokenizer,
            id2label=predictor_id2label,
            max_length=max_length,
            window_stride=window_stride,
        )


def test_predictor_requires_fast_tokenizer(
    deterministic_model: DeterministicTokenClassifier,
    predictor_id2label: Mapping[int, str],
) -> None:
    """Fail at construction when tokenizer offsets are unavailable for reconstruction."""

    tokenizer = cast(
        PreTrainedTokenizerBase,
        SimpleNamespace(is_fast=False),
    )

    with pytest.raises(ValueError, match="requires a fast tokenizer"):
        DocumentModelPredictor(
            model=cast(PreTrainedModel, deterministic_model),
            tokenizer=tokenizer,
            id2label=predictor_id2label,
        )
