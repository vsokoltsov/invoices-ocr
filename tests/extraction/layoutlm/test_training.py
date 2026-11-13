from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from app.extraction.layoutlm.metrics import EpochMetrics
from app.extraction.layoutlm.schema import InvoiceLabelSchema
from app.extraction.layoutlm.training import OCRInvoiceTraining


class TinyPageDataset(Dataset):
    def __init__(self):
        self.pages = [{"words": ["Acme", "Ltd", "Other"], "labels": [1, 2, 0]}]
        self.samples = [
            {
                "input_ids": torch.tensor(tokens),
                "attention_mask": torch.ones(3, dtype=torch.long),
                "bbox": torch.zeros(3, 4, dtype=torch.long),
                "pixel_values": torch.full((3, 2, 2), 0.25),
                "labels": torch.tensor(labels),
                "word_ids": torch.tensor(words),
                "page_index": torch.tensor(0),
            }
            for tokens, labels, words in (
                ([1, 0, 0], [1, -100, -100], [0, -1, -1]),
                ([2, 0, 0], [2, 0, -100], [1, 2, -1]),
            )
        ]

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


class TinyLayoutTagger(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(3, 3)
        self.backbone = SimpleNamespace(
            config=SimpleNamespace(_name_or_path="tiny-layout-model")
        )
        with torch.no_grad():
            self.embedding.weight.copy_(torch.eye(3) * 5)

    def forward(self, input_ids, attention_mask, bbox, pixel_values):
        assert pixel_values.dtype == torch.float32
        assert bbox.dtype == torch.long
        return self.embedding(input_ids)


def make_trainer(tmp_path, epochs=1):
    schema = InvoiceLabelSchema({"invoice.seller_name": "SELLER_NAME"})
    model = TinyLayoutTagger()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return OCRInvoiceTraining(
        model=model,
        metrics=EpochMetrics(schema.labels, list(schema.field_to_entity)),
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(ignore_index=-100),
        train_loader=DataLoader(TinyPageDataset(), batch_size=1),
        val_loader=DataLoader(TinyPageDataset(), batch_size=1),
        device=torch.device("cpu"),
        scheduler=LambdaLR(optimizer, lr_lambda=lambda _: 1.0),
        schema=schema,
        epochs=epochs,
        output_dir=tmp_path / "reports",
        checkpoint_path=tmp_path / "weights" / "best.pt",
    )


def test_training_and_evaluation_reset_metrics_and_restore_page_entities(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.metrics.confusion.fill_(100)
    trainer.metrics.loss_sum = 1000
    trainer.metrics.word_count = 1000
    before = trainer.model.embedding.weight.detach().clone()
    previous_step = trainer.scheduler.last_epoch

    training = trainer.train_one_epoch(1)
    assert training.field_word_micro_f1 == 1.0
    assert not torch.equal(before, trainer.model.embedding.weight)
    assert trainer.scheduler.last_epoch == previous_step + len(trainer.train_loader)

    result = trainer.eval_one_epoch(1)
    # The B/I entity spans two windows and must be counted only once.
    assert result.entity_f1 == result.entity_precision == result.entity_recall == 1.0
    assert result.entity_report["SELLER_NAME"]["support"] == 1
    assert result.word_metrics.per_field["invoice.seller_name"].support == 2
    assert trainer.metrics.word_count == 3
    assert not trainer.model.training
    assert trainer.eval_one_epoch(2) == result


def test_validation_uses_strict_bio_boundaries(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.val_loader.dataset.samples[0]["input_ids"][0] = 2
    result = trainer.eval_one_epoch(1)
    assert result.word_metrics.field_word_micro_f1 == 1.0
    assert result.entity_f1 == 0.0


def test_train_saves_scalar_history_and_restores_best_checkpoint(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path, epochs=2)
    validation = trainer.eval_one_epoch(1)
    training = validation.word_metrics

    def train_epoch(epoch):
        with torch.no_grad():
            trainer.model.embedding.weight.fill_(epoch)
        return training

    monkeypatch.setattr(trainer, "train_one_epoch", train_epoch)
    monkeypatch.setattr(
        trainer,
        "eval_one_epoch",
        lambda epoch: replace(validation, entity_f1=1.0 / epoch),
    )
    history = trainer.train()
    saved_history = pd.read_csv(trainer.output_dir / "history.csv")
    pd.testing.assert_frame_equal(history, saved_history)
    assert history["validation_entity_f1"].tolist() == [1.0, 0.5]
    assert all(dtype.kind in "if" for dtype in history.dtypes)
    checkpoint = torch.load(trainer.checkpoint_path, weights_only=True)
    assert checkpoint["epoch"] == 1
    assert checkpoint["val_entity_f1"] == 1.0
    assert checkpoint["labels"] == list(trainer.schema.labels)
    assert checkpoint["model_name"] == "tiny-layout-model"
    assert checkpoint["max_length"] == 3
    assert checkpoint["field_to_entity"] == trainer.schema.field_to_entity
    for key, value in trainer.model.state_dict().items():
        assert torch.all(value == 1)
        assert torch.equal(value, checkpoint["model_state_dict"][key])


@pytest.mark.parametrize("method", ["train_one_epoch", "eval_one_epoch"])
def test_nonfinite_loss_fails_before_saving(tmp_path, method):
    trainer = make_trainer(tmp_path)
    with torch.no_grad():
        trainer.model.embedding.weight.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="loss is not finite"):
        getattr(trainer, method)(1)
    assert not trainer.checkpoint_path.exists()


def test_duplicate_word_evaluation_is_rejected(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.val_loader.dataset.samples[1]["word_ids"][0] = 0
    with pytest.raises(ValueError, match="word was evaluated more than once"):
        trainer.eval_one_epoch(1)


def test_epoch_count_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="epochs must be positive"):
        make_trainer(tmp_path, epochs=0)
