import math

import pytest
import torch
from torch.nn import functional as F

from app.extraction.layoutlm.metrics import EpochMetrics


def test_metrics_ignore_masked_tokens_and_weight_loss_by_word_count():
    fields = ("NUMBER", "DATE", "DUE_DATE", "SELLER", "CLIENT")
    labels = ["O"] + [f"{prefix}-{field}" for field in fields for prefix in ("B", "I")]
    metrics = EpochMetrics(labels, fields)
    for truth, guesses in (
        ([1, 2, 0, -100], [2, 2, 1, 4]),
        ([3, 5, -100, -100], [3, 0, 7, 8]),
    ):
        targets = torch.tensor([truth])
        logits = F.one_hot(torch.tensor([guesses]), num_classes=len(labels)).float() * 4
        loss = F.cross_entropy(logits.reshape(-1, len(labels)), targets.reshape(-1))
        metrics.update(logits, targets, loss)
    result = metrics.compute()
    assert metrics.word_count == 5
    assert result.loss == pytest.approx(math.log(math.exp(4) + 10) - 4 * (2 / 5))
    assert result.bio_word_accuracy == pytest.approx(2 / 5)
    assert result.field_word_micro_precision == pytest.approx(3 / 4)
    assert result.field_word_micro_recall == pytest.approx(3 / 4)
    assert result.field_word_micro_f1 == pytest.approx(0.75)
    assert result.field_word_macro_f1 == pytest.approx(0.6)
    assert result.per_field["NUMBER"].precision == pytest.approx(2 / 3)
    assert result.per_field["NUMBER"].recall == 1
    assert result.per_field["NUMBER"].support == 2
    assert result.per_field["DUE_DATE"].f1 == 0
    assert result.per_field["DUE_DATE"].support == 1
    metrics.reset()
    with pytest.raises(ValueError, match="No supervised words"):
        metrics.compute()


@pytest.mark.parametrize("labels", [("O", "B-X"), ("O", "B-X", "I-X", "B-Y")])
def test_field_count_must_match_bio_pairs(labels):
    with pytest.raises(ValueError, match="B/I pair"):
        EpochMetrics(labels, ["X"])
