import math
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from app.extraction.bert.pytorch.collation import InvoiceDataCollator
from app.extraction.bert.pytorch.decoding import OCRWordDecoder
from app.extraction.bert.pytorch.training import (
    EvalResult,
    OCRInvoiceTraining,
    TrainResult,
)


class TinyTagger(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(3, 3)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.eye(3) * 5)

    def forward(self, input_ids, attention_mask):  # ty: ignore[invalid-mutable-override]
        return self.embedding(input_ids)


class BatchDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, index):
        return self.rows[index]

    def __len__(self):
        return len(self.rows)


def make_trainer(labels=(1, 2, -100), epochs=1):
    model = TinyTagger()
    loader = DataLoader(
        BatchDataset(
            [
                {
                    "input_ids": torch.tensor([1, 2, 0]),
                    "attention_mask": torch.ones(3, dtype=torch.long),
                    "labels": torch.tensor(labels),
                }
            ]
        ),
        batch_size=1,
    )
    return OCRInvoiceTraining(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        criterion=nn.CrossEntropyLoss(ignore_index=-100),
        train_loader=loader,
        val_loader=loader,
        device=torch.device("cpu"),
        epochs=epochs,
        id2label={0: "O", 1: "B-NAME", 2: "I-NAME"},
    )


def test_training_updates_weights_and_evaluation_reports_bio_metrics():
    trainer = make_trainer()
    before = trainer.model.embedding.weight.detach().clone()
    training = trainer.train_one_epoch(1)
    assert math.isfinite(training.loss)
    assert not torch.equal(before, trainer.model.embedding.weight)
    result = trainer.eval_one_epoch(1)
    assert result.f1 == result.precision == result.recall == 1.0
    assert result.token_accuracy == 1.0
    assert result.per_entity["NAME"]["support"] == 1
    assert not trainer.model.training
    assert math.isfinite(result.loss)


@pytest.mark.parametrize("method", ["train_one_epoch", "eval_one_epoch"])
def test_empty_supervision_raises_without_changing_weights(method):
    trainer = make_trainer(labels=(-100, -100, -100))
    before = trainer.model.embedding.weight.detach().clone()
    with pytest.raises(ValueError, match="no supervised tokens"):
        getattr(trainer, method)(1)
    assert torch.equal(before, trainer.model.embedding.weight)


def test_all_masked_batch_is_skipped_when_other_batches_have_labels():
    trainer = make_trainer()
    supervised = trainer.train_loader.dataset[0]
    masked = {**supervised, "labels": torch.full((3,), -100)}
    loader = DataLoader(BatchDataset([masked, supervised]), batch_size=1)
    trainer.train_loader = trainer.val_loader = loader
    assert math.isfinite(trainer.train_one_epoch(1).loss)
    assert trainer.eval_one_epoch(1).f1 == 1.0


def test_train_restores_best_checkpoint_and_returns_scalar_history(
    monkeypatch, tmp_path
):
    trainer = make_trainer(epochs=2)
    inspected_weights = []
    records = [{"text": "Acme", "entities": []}]

    def inspect(validation_records, output_dir, *, epoch):
        assert validation_records is records
        assert epoch == len(inspected_weights) + 1
        assert output_dir == tmp_path / f"epoch_{epoch:03d}"
        inspected_weights.append(trainer.model.embedding.weight[0, 0].item())

    inspector = Mock()
    inspector.run.side_effect = inspect
    trainer.inspector = inspector
    trainer.validation_records = records
    trainer.diagnostic_dir = tmp_path

    def train_epoch(epoch):
        with torch.no_grad():
            trainer.model.embedding.weight.fill_(epoch)
        return TrainResult(loss=1.0 / epoch)

    def evaluate(epoch):
        return EvalResult(0.1, 0.9, 0.9, 0.9 if epoch == 1 else 0.5, 1.0, 1.0, {})

    monkeypatch.setattr(trainer, "train_one_epoch", train_epoch)
    monkeypatch.setattr(trainer, "eval_one_epoch", evaluate)
    history = trainer.train()
    assert history["validation_f1"].tolist() == [0.9, 0.5]
    assert "validation_per_entity" not in history
    assert torch.all(trainer.model.embedding.weight == 1)
    assert inspected_weights == [1, 2]


def test_training_rejects_nonpositive_epoch_count():
    with pytest.raises(ValueError, match="epochs must be positive"):
        make_trainer(epochs=0)


def test_word_decoding_changes_metrics_without_changing_loss_or_padding(fast_tokenizer):
    trainer = make_trainer()
    examples = [
        {
            "input_ids": [1, 0, 2],
            "attention_mask": [1, 1, 1],
            "labels": [1, 2, 2],
            "decoder_metadata": {
                "offsets": [(0, 2), (2, 3), (3, 6)],
                "words": [{"start": 0, "end": 6}],
            },
        },
        {
            "input_ids": [1],
            "attention_mask": [1],
            "labels": [1],
            "decoder_metadata": {
                "offsets": [(0, 2)],
                "words": [{"start": 0, "end": 2}],
            },
        },
    ]
    trainer.val_loader = DataLoader(
        BatchDataset(examples),
        batch_size=2,
        collate_fn=InvoiceDataCollator(fast_tokenizer),
    )
    baseline = trainer.eval_one_epoch(1)
    trainer.word_decoder = OCRWordDecoder({"O": 0, "B-NAME": 1, "I-NAME": 2})
    decoded = trainer.eval_one_epoch(1)
    assert baseline.f1 < decoded.f1 == 1
    assert baseline.loss == pytest.approx(decoded.loss)
    assert decoded.per_entity["NAME"]["support"] == 2
    assert examples[0]["decoder_metadata"]["offsets"] == [(0, 2), (2, 3), (3, 6)]


def test_word_evaluation_requires_offsets_and_words():
    trainer = make_trainer()
    trainer.word_decoder = OCRWordDecoder({"O": 0, "B-NAME": 1, "I-NAME": 2})
    with pytest.raises(ValueError, match="requires decoding metadata"):
        trainer.eval_one_epoch(1)
