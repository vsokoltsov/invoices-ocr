import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from app.extraction.bert.pytorch.decoding import OCRWordDecoder
from app.extraction.bert.pytorch.evaluation import BertEvaluation
from app.extraction.bert.pytorch.metrics import EvalMetrics


class FixedTagger(nn.Module):
    def forward(self, input_ids, attention_mask):
        assert not self.training
        assert not torch.is_grad_enabled()
        return nn.functional.one_hot(input_ids, num_classes=3).float() * 5


class BatchDataset(Dataset):
    def __init__(self, batches):
        self.batches = batches

    def __getitem__(self, index):
        return self.batches[index]

    def __len__(self):
        return len(self.batches)


def make_loader(batches):
    return DataLoader(BatchDataset(batches), batch_size=None)


def make_evaluator():
    return BertEvaluation(
        model=FixedTagger(),
        metrics=EvalMetrics({0: "O", 1: "B-NAME", 2: "I-NAME"}),
        device=torch.device("cpu"),
        criterion=nn.CrossEntropyLoss(ignore_index=-100),
    )


def make_batch(labels):
    return {
        "input_ids": torch.tensor([[1, 2, 0]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "labels": torch.tensor([labels]),
    }


def test_evaluates_without_training_state_and_resets_between_loaders():
    evaluator = make_evaluator()
    loader = make_loader([make_batch([1, 2, -100])])
    result = evaluator.evaluate(loader, epoch=1)
    assert result.f1 == result.precision == result.recall == 1
    assert result.token_accuracy == 1
    assert result.per_entity["NAME"]["support"] == 1

    # An earlier evaluation must not contribute to a new loader's scores.
    loader = make_loader([make_batch([0, 0, 0])])
    result = evaluator.evaluate(loader, epoch=2)
    assert result.f1 == 0
    assert result.token_accuracy == pytest.approx(1 / 3)
    assert result.per_entity["NAME"]["support"] == 0


@pytest.mark.parametrize("batches", [[], [make_batch([-100, -100, -100])]])
def test_empty_supervision_raises(batches):
    with pytest.raises(ValueError, match="no supervised tokens"):
        make_evaluator().evaluate(make_loader(batches), epoch=1)


def test_nonfinite_loss_raises():
    evaluator = make_evaluator()
    evaluator.criterion = nn.CrossEntropyLoss(
        weight=torch.full((3,), float("nan")), ignore_index=-100
    )
    loader = make_loader([make_batch([1, 2, -100])])
    with pytest.raises(FloatingPointError, match="Evaluation loss is not finite"):
        evaluator.evaluate(loader, epoch=1)


def test_word_decoder_uses_metadata_without_moving_or_modifying_it():
    evaluator = make_evaluator()
    evaluator.word_decoder = OCRWordDecoder(
        {label: index for index, label in evaluator.metrics.id2label.items()}
    )
    batch = make_batch([1, 2, -100])
    metadata = [{"offsets": [(0, 2), (2, 4)], "words": [{"start": 0, "end": 4}]}]
    batch["decoder_metadata"] = metadata
    result = evaluator.evaluate(make_loader([batch]), epoch=1)
    assert result.f1 == 1
    assert metadata[0]["offsets"] == [(0, 2), (2, 4)]
