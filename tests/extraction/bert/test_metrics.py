from dataclasses import asdict

import pytest
import torch

from app.extraction.bert.metrics import EvalMetrics


def logits_for(predictions):
    return torch.nn.functional.one_hot(torch.tensor(predictions), 3).float() * 5


def test_accumulates_masked_tokens_weighted_loss_and_entity_boundaries():
    metrics = EvalMetrics({0: "O", 1: "B-NAME", 2: "I-NAME"})
    metrics.update(
        logits_for([[1, 2, 1]]),
        torch.tensor([[1, 2, -100]]),
        torch.tensor(0.2),
    )
    metrics.update(
        logits_for([[0, 0, 0]]),
        torch.tensor([[0, 1, 0]]),
        torch.tensor(0.6),
    )
    # Fully masked batches must not add NaN loss or change counts.
    metrics.update(
        logits_for([[2, 2, 2]]),
        torch.full((1, 3), -100),
        torch.tensor(float("nan")),
    )
    result = metrics.compute()
    assert result.loss == pytest.approx(0.44)
    assert result.word_metrics.bio_word_accuracy == pytest.approx(0.8)
    assert result.word_metrics.field_word_macro_f1 == pytest.approx(0.8)
    assert result.precision == 1
    assert result.recall == 0.5
    assert result.f1 == pytest.approx(2 / 3)
    assert result.per_entity["NAME"]["support"] == 2
    assert isinstance(result.per_entity["NAME"]["support"], int)


def test_reset_clears_loss_token_metrics_and_entity_sequences():
    metrics = EvalMetrics({0: "O", 1: "B-NAME", 2: "I-NAME"})
    targets = torch.tensor([[1, 2]])
    metrics.update(logits_for([[0, 0]]), targets, torch.tensor(2.0))
    metrics.reset()
    with pytest.raises(ValueError, match="no supervised tokens"):
        metrics.compute()
    metrics.update(logits_for([[1, 2]]), targets, torch.tensor(0.1))
    result = metrics.compute()
    assert result.loss == pytest.approx(0.1)
    assert result.f1 == result.word_metrics.bio_word_accuracy == 1
    assert result.per_entity["NAME"]["support"] == 1
    assert asdict(metrics.compute()) == asdict(result)


def test_decoded_predictions_override_argmax_and_custom_mask_is_respected():
    metrics = EvalMetrics({0: "O", 1: "B-NAME", 2: "I-NAME"}, ignore_index=-1)
    metrics.update(
        logits_for([[0, 0, 0]]),
        torch.tensor([[1, 2, -1]]),
        torch.tensor(0.7),
        predictions=torch.tensor([[1, 2, 2]]),
    )
    result = metrics.compute()
    assert result.f1 == result.word_metrics.bio_word_accuracy == 1
    assert result.loss == pytest.approx(0.7)
