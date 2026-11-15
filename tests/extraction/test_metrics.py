import pytest
import torch
from seqeval.scheme import IOB2

from app.extraction.metrics import BioEntityScorer, LossMetrics


def test_loss_metrics_weight_supervised_tokens_and_ignore_masked_batches():
    metrics = LossMetrics()
    logits = torch.zeros(1, 3, 3)
    metrics.update(logits, torch.tensor([[1, -100, -100]]), torch.tensor(0.25))
    metrics.update(logits, torch.tensor([[1, 2, 0]]), torch.tensor(0.75))
    metrics.update(logits, torch.full((1, 3), -100), torch.tensor(float("nan")))
    assert metrics.compute().loss == pytest.approx(0.625)
    metrics.reset()
    with pytest.raises(ValueError, match="no supervised tokens"):
        metrics.compute()


def test_bio_scorers_preserve_default_and_strict_boundary_interpretations():
    actual = [["B-NAME", "I-NAME"]]
    predicted = [["I-NAME", "I-NAME"]]
    default = BioEntityScorer().score(actual, predicted)
    strict = BioEntityScorer(mode="strict", scheme=IOB2).score(actual, predicted)
    assert default.f1 == 1
    assert strict.f1 == 0
    assert default.per_entity["NAME"]["support"] == 1
    assert isinstance(default.per_entity["NAME"]["support"], int)
