from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BertConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BERT_",
        extra="ignore",
    )

    MODEL_NAME: str = Field(
        default="google-bert/bert-base-uncased",
        description="Hugging Face checkpoint used to initialize the token-classification model.",
    )
    MAX_LENGTH_TOKENS: int = Field(
        default=512,
        description="Maximum tokenized input length for each document window.",
    )
    MODEL_WINDOW_STRIDE: int = Field(
        default=64,
        description="Number of overlapping tokens between windows for long documents.",
    )
    BATCH_SIZE: int = Field(
        default=64,
        description="Number of documents processed in one dataset-encoding batch.",
    )
    NUM_EPOCH: int = Field(
        default=5,
        description="Number of complete passes over the training dataset.",
    )
    LEARNING_RATE: float = Field(
        default=2e-5,
        description="Optimizer learning rate used during fine-tuning.",
    )
    WEIGHT_DECAY: float = Field(
        default=0.01,
        description="Optimizer weight-decay coefficient for regularization.",
    )
    GRADIENT_ACCUMULATION_STEPS: int = Field(
        default=1,
        description="Number of batches accumulated before an optimizer update.",
    )
    PER_DEVICE_TRAIN_BATCH_SIZE: int = Field(
        default=16,
        description="Training examples per device in each optimization batch.",
    )
    PER_DEVICE_EVAL_BATCH_SIZE: int = Field(
        default=16,
        description="Evaluation examples per device in each evaluation batch.",
    )
    EVAL_STRATEGY: str = Field(
        default="epoch",
        description="When to evaluate during training, for example 'epoch' or 'steps'.",
    )
    SAVE_STRATEGY: str = Field(
        default="epoch",
        description="When to save checkpoints, for example 'epoch' or 'steps'.",
    )
    LOGGING_STRATEGY: str = Field(
        default="steps",
        description="When to log training metrics, for example 'steps' or 'epoch'.",
    )
    LOGGING_STEPS: int = Field(
        default=20,
        description="Number of training steps between logs when logging by steps.",
    )
    LOAD_BEST_MODEL_AT_END: bool = Field(
        default=True,
        description="Load the checkpoint with the best evaluation metric after training.",
    )
    METRIC_FOR_BEST_MODEL: str = Field(
        default="f1",
        description="Evaluation metric used to select the best checkpoint.",
    )
    GREATER_IS_BETTER: bool = Field(
        default=True,
        description="Whether a larger best-model metric value indicates improvement.",
    )
    DATALOADER_NUM_WORKERS: int = Field(
        default=0,
        description="Worker processes used by the training data loader.",
    )

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    DATA_PATH: str
    DEBUG_ARTIFACTS_PATH: str
    BERT: BertConfig = Field(default_factory=BertConfig)
    RANDOM_SEED: int = 42
    DOCILE_TOKEN: str


@lru_cache
def get_settings() -> Settings:
    return Settings()
