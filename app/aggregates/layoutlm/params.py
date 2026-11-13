from dataclasses import dataclass
from enum import StrEnum


class LayoutLMDevice(StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


@dataclass(frozen=True, slots=True)
class LayoutLMTrainingParams:
    """Resolved inputs for constructing datasets, model, optimizer, and trainer."""

    dataset_path: str
    model_name: str
    output_dir: str
    checkpoint_path: str
    images_path: str
    num_epoch: int
    batch_size: int
    max_length_tokens: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    min_word_coverage: float
    dataloader_num_workers: int
    random_seed: int
    device: LayoutLMDevice
