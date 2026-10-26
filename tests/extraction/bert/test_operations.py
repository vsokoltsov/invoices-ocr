from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from datasets import Dataset, DatasetDict
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainingArguments

import app.extraction.bert.operations as operations
from app.extraction.bert.metrics import TokenClassificationMetrics
from app.extraction.bert.operations import DocumentModelTrainer


class RecordingTrainer:
    """Record Trainer calls without constructing or training a neural network."""

    instance: "RecordingTrainer | None" = None

    def __init__(self, **kwargs: Any) -> None:
        type(self).instance = self
        self.model = kwargs["model"]
        self.args = kwargs["args"]
        self.train_dataset = kwargs["train_dataset"]
        self.eval_dataset = kwargs["eval_dataset"]
        self.data_collator = kwargs["data_collator"]
        self.processing_class = kwargs["processing_class"]
        self.compute_metrics = kwargs["compute_metrics"]
        self.evaluate_calls: list[tuple[Any, str]] = []
        self.predict_calls: list[Any] = []
        self.saved_directories: list[str] = []

    def train(self) -> dict[str, bool]:
        return {"trained": True}

    def evaluate(
        self,
        *,
        eval_dataset: Any,
        metric_key_prefix: str,
    ) -> dict[str, Any]:
        self.evaluate_calls.append((eval_dataset, metric_key_prefix))
        return {f"{metric_key_prefix}_loss": 0.1}

    def predict(self, dataset: Any, *, metric_key_prefix: str) -> dict[str, Any]:
        self.predict_calls.append((dataset, metric_key_prefix))
        return {"split": metric_key_prefix}

    def save_model(self, output_directory: str) -> None:
        self.saved_directories.append(output_directory)


class MinimalModel:
    """Provide the model interface needed by the recording Trainer and tests."""

    def parameters(self):
        yield torch.nn.Parameter(torch.zeros(1))


@pytest.fixture
def trainer_datasets() -> DatasetDict:
    """Provide all required tokenized dataset splits."""

    def split(value: int) -> Dataset:
        return Dataset.from_dict(
            {
                "input_ids": [[value, value + 1]],
                "attention_mask": [[1, 1]],
                "token_type_ids": [[0, 0]],
                "labels": [[0, 0]],
            }
        )

    return DatasetDict(
        {
            "train": split(1),
            "validation": split(2),
            "test": split(3),
        }
    )


@pytest.fixture
def training_arguments(tmp_path: Path) -> TrainingArguments:
    """Provide minimal Hugging Face training configuration for initialization."""

    return TrainingArguments(
        output_dir=str(tmp_path / "model"),
        report_to=[],
    )


@pytest.fixture
def document_model_trainer(
    monkeypatch: pytest.MonkeyPatch,
    trainer_datasets: DatasetDict,
    training_arguments: TrainingArguments,
    fast_tokenizer: PreTrainedTokenizerBase,
    label2id: Mapping[str, int],
) -> DocumentModelTrainer:
    """Build the project trainer around a recording Hugging Face Trainer."""

    RecordingTrainer.instance = None
    monkeypatch.setattr(operations, "Trainer", RecordingTrainer)

    return DocumentModelTrainer(
        model=cast(PreTrainedModel, MinimalModel()),
        tokenizer=fast_tokenizer,
        datasets=trainer_datasets,
        training_arguments=training_arguments,
        metrics=TokenClassificationMetrics(id2label={
            label_id: label
            for label, label_id in label2id.items()
        }),
    )


def test_initialization_wires_model_data_collator_and_metrics(
    document_model_trainer: DocumentModelTrainer,
    trainer_datasets: DatasetDict,
    fast_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Pass model, splits, tokenizer, collator, and metrics to Hugging Face Trainer."""

    recording_trainer = cast(RecordingTrainer, RecordingTrainer.instance)

    assert document_model_trainer.trainer is recording_trainer
    assert recording_trainer.train_dataset is trainer_datasets["train"]
    assert recording_trainer.eval_dataset is trainer_datasets["validation"]
    assert recording_trainer.processing_class is fast_tokenizer
    assert recording_trainer.data_collator is not None
    assert recording_trainer.compute_metrics is document_model_trainer.metrics


def test_train_and_evaluation_methods_delegate_to_hugging_face_trainer(
    document_model_trainer: DocumentModelTrainer,
    trainer_datasets: DatasetDict,
) -> None:
    """Delegate training and validation/test evaluation with the expected prefixes."""

    recording_trainer = cast(RecordingTrainer, RecordingTrainer.instance)

    assert document_model_trainer.train() == {"trained": True}
    assert document_model_trainer.evaluate_validation() == {
        "validation_loss": 0.1
    }
    assert document_model_trainer.evaluate_test() == {"test_loss": 0.1}
    assert recording_trainer.evaluate_calls == [
        (trainer_datasets["validation"], "validation"),
        (trainer_datasets["test"], "test"),
    ]


def test_predict_dataset_selects_requested_split(
    document_model_trainer: DocumentModelTrainer,
    trainer_datasets: DatasetDict,
) -> None:
    """Predict from the requested dataset split and pass its name as metric prefix."""

    recording_trainer = cast(RecordingTrainer, RecordingTrainer.instance)

    result = document_model_trainer.predict_dataset("validation")

    assert result == {"split": "validation"}
    assert recording_trainer.predict_calls == [
        (trainer_datasets["validation"], "validation")
    ]


def test_predict_dataset_rejects_unknown_split(
    document_model_trainer: DocumentModelTrainer,
) -> None:
    """Raise a clear error rather than silently predicting from the wrong split."""

    with pytest.raises(KeyError, match="Unknown dataset split: production"):
        document_model_trainer.predict_dataset("production")


def test_save_delegates_model_export_to_trainer_and_tokenizer(
    document_model_trainer: DocumentModelTrainer,
    fast_tokenizer: PreTrainedTokenizerBase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Save both model artifacts and tokenizer configuration to one directory."""

    recording_trainer = cast(RecordingTrainer, RecordingTrainer.instance)
    saved_tokenizer_directories: list[str] = []
    monkeypatch.setattr(
        fast_tokenizer,
        "save_pretrained",
        lambda directory: saved_tokenizer_directories.append(directory),
    )
    output_directory = str(tmp_path / "saved-model")

    document_model_trainer.save(output_directory)

    assert recording_trainer.saved_directories == [output_directory]
    assert saved_tokenizer_directories == [output_directory]


def test_initialization_rejects_missing_required_dataset_split(
    monkeypatch: pytest.MonkeyPatch,
    trainer_datasets: DatasetDict,
    training_arguments: TrainingArguments,
    fast_tokenizer: PreTrainedTokenizerBase,
    label2id: Mapping[str, int],
) -> None:
    """Fail before Trainer construction when train, validation, or test is absent."""

    monkeypatch.setattr(operations, "Trainer", RecordingTrainer)
    incomplete_datasets = DatasetDict(
        {
            "train": trainer_datasets["train"],
            "test": trainer_datasets["test"],
        }
    )

    with pytest.raises(
        ValueError,
        match=r"Missing dataset splits: \['validation'\]",
    ):
        DocumentModelTrainer(
            model=cast(PreTrainedModel, MinimalModel()),
            tokenizer=fast_tokenizer,
            datasets=incomplete_datasets,
            training_arguments=training_arguments,
            metrics=TokenClassificationMetrics(id2label={
                label_id: label
                for label, label_id in label2id.items()
            }),
        )
