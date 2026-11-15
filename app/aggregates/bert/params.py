from dataclasses import dataclass


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
    diagnostic_dir: str | None = "./data/interim/bert_diagnostics"
    checkpoint_dir: str = "./data/interim/bert_pytorch_checkpoint"
    line_items_only: bool = False


@dataclass(frozen=True, slots=True)
class BertPytorchDebugParams:
    checkpoint_dir: str
    output_dir: str
    device: str = "auto"
    split: str = "validation"
    dataset_path: str = ""
    random_seed: int = 42
