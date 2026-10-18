from dataclasses import dataclass, field
from typing import cast

from datasets import DatasetDict
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    DataCollatorForTokenClassification,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from app.extraction.bert.metrics import TokenClassificationMetrics


@dataclass(slots=True)
class DocumentModelTrainer:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    datasets: DatasetDict
    training_arguments: TrainingArguments
    metrics: TokenClassificationMetrics

    trainer: Trainer = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        required_splits = {
            "train",
            "validation",
            "test",
        }

        missing_splits = (
            required_splits
            - set(self.datasets)
        )

        if missing_splits:
            raise ValueError(
                "Missing dataset splits: "
                f"{sorted(missing_splits)}"
            )

        data_collator = (
            DataCollatorForTokenClassification(
                tokenizer=self.tokenizer
            )
        )

        self.trainer = Trainer(
            model=self.model,
            args=self.training_arguments,
            train_dataset=(
                self.datasets["train"]
            ),
            eval_dataset=(
                self.datasets["validation"]
            ),
            data_collator=data_collator,
            processing_class=self.tokenizer,
            compute_metrics=self.metrics,
        )

    def train(self):
        return self.trainer.train()

    def _split_dataset(
        self,
        split: str,
    ) -> TorchDataset:
        # evaluate() and predict() accept a Hugging Face
        # Dataset, but only annotate torch's Dataset.
        return cast(
            TorchDataset,
            self.datasets[split],
        )

    def evaluate_validation(
        self,
    ) -> dict[str, float]:
        return self.trainer.evaluate(
            eval_dataset=self._split_dataset(
                "validation"
            ),
            metric_key_prefix="validation",
        )

    def evaluate_test(
        self,
    ) -> dict[str, float]:
        return self.trainer.evaluate(
            eval_dataset=self._split_dataset(
                "test"
            ),
            metric_key_prefix="test",
        )

    def predict_dataset(
        self,
        split: str = "test",
    ):
        if split not in self.datasets:
            raise KeyError(
                f"Unknown dataset split: {split}"
            )

        return self.trainer.predict(
            self._split_dataset(split),
            metric_key_prefix=split,
        )

    def save(
        self,
        output_directory: str,
    ) -> None:
        self.trainer.save_model(
            output_directory
        )

        self.tokenizer.save_pretrained(
            output_directory
        )
