from dataclasses import dataclass
from enum import StrEnum


class DatasetName(StrEnum):
    KAGGLE_INVOICES = "kaggle_invoices"
    DOCILE = "docile"


@dataclass(frozen=True, slots=True)
class BertPytorchParams:
    dataset_path: str
    result_filename: str
    model_name: str
    max_length_tokens: int
    random_seed: int
    batch_size: int
    num_epoch: int
    learning_rate: float
    weight_decay: float
    dataset_type: DatasetName = DatasetName.KAGGLE_INVOICES
    diagnostic_dir: str | None = "./data/interim/bert_diagnostics"
    checkpoint_dir: str = "./data/interim/bert_pytorch_checkpoint"
    line_items_only: bool = False
    body_only: bool = False


@dataclass(frozen=True, slots=True)
class BertPytorchDebugParams:
    checkpoint_dir: str
    output_dir: str
    device: str = "auto"
    split: str = "validation"
    dataset_path: str = ""
    random_seed: int = 42


@dataclass(frozen=True, slots=True)
class BertTrainingParams:
    """Resolved parameters for one BERT training run."""

    data_path: str
    debug_artifacts_path: str
    random_seed: int
    model_name: str
    max_length_tokens: int
    model_window_stride: int
    batch_size: int
    num_epoch: int
    learning_rate: float
    weight_decay: float
    gradient_accumulation_steps: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    eval_strategy: str
    save_strategy: str
    logging_strategy: str
    logging_steps: int
    load_best_model_at_end: bool
    metric_for_best_model: str
    greater_is_better: bool
    dataloader_num_workers: int
