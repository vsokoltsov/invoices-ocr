from dataclasses import dataclass

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
