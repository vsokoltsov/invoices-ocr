from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.aggregates.bert.params import DatasetName
from app.datasets.docile.enums import DocileSplit


class DocileInvoicesConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCILE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    TOKEN: str = Field(
        default="",
        repr=False,
        description="Token for retrieving docile archive",
    )
    DESTINATION_PATH: str = Field(
        default="./data/raw/docile",
        description="Directory for the extracted dataset archive.",
    )
    OUTPUT_PATH: str = Field(
        default="./data/raw/docile.parquet",
        description="Prepared dataset output, with .parquet or .csv extension.",
    )
    SPLIT: DocileSplit = Field(
        default=DocileSplit.TRAINVAL,
        description="Document split to prepare: train, val, trainval, or test.",
    )
    USE_SNAPPED_BOXES: bool = Field(
        default=True,
        description="Use OCR boxes snapped to the text when available.",
    )
    CORRECTIONS_PATH: str = Field(
        default="",
        description="Optional JSON annotation-correction file; empty preserves original labels.",
    )
    MINIMUM_ARCHIVE_BYTES: int = Field(
        default=100_000_000,
        ge=0,
        description="Minimum accepted archive size in bytes.",
    )


class KaggleInvoicesConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KAGGLE_INVOICES_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    DESTINATION_PATH: str = Field(
        default="./data/raw/kaggle-invoices",
        description="Directory for the extracted dataset archive.",
    )
    ARCHIVE_PATH: str = Field(
        default="./data/raw/invoices-ocr.zip",
        description="Dataset archive path; an existing archive is reused.",
    )
    OUTPUT_PATH: str = Field(
        default="./data/raw/kaggle-invoices.csv",
        description="Path for the prepared dataset CSV.",
    )


class BertPytorchConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BERTPYTORCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    DATASET_PATH: str = Field(
        default="./data/raw/kaggle-invoices.csv",
        description="Existing prepared CSV or DocILE Parquet; fail if it is missing.",
    )
    RESULT_FILENAME: str = Field(
        default="bert_training_pipeline",
        description="Training-history filename or path; the extension is always .csv.",
    )
    DIAGNOSTIC_DIR: str = Field(
        default="./data/interim/bert_diagnostics",
        description="Directory for per-epoch validation diagnostics; an empty value disables inspection.",
    )
    CHECKPOINT_DIR: str = Field(
        default="./data/interim/bert_pytorch_checkpoint",
        description="Directory for best weights, tokenizer, schema, and validation inputs.",
    )
    LINE_ITEMS_ONLY: bool = Field(
        default=False,
        description="Train only DocILE item descriptions, quantities, gross unit prices, and gross amounts.",
    )
    BODY_ONLY: bool = Field(
        default=False,
        description="Train only invoice fields, without the line-item model.",
    )
    MODEL_NAME: str = Field(
        default="google-bert/bert-base-uncased",
        description="Hugging Face checkpoint used to initialize the token-classification model.",
    )
    BATCH_SIZE: int = Field(
        default=16,
        description="Number of documents processed in one dataset-encoding batch.",
    )
    NUM_EPOCH: int = Field(
        default=5,
        description="Number of complete passes over the training dataset.",
    )
    MAX_LENGTH_TOKENS: int = Field(
        default=512,
        description="Maximum tokenized input length for each document window.",
    )
    LEARNING_RATE: float = Field(
        default=2e-5,
        description="Optimizer learning rate used during fine-tuning.",
    )
    WEIGHT_DECAY: float = Field(
        default=0.01,
        description="Optimizer weight-decay coefficient for regularization.",
    )


class BertPytorchDebugConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BERT_PYTORCH_DEBUG_", env_file=".env", extra="ignore"
    )

    CHECKPOINT_DIR: str = Field(
        default="./data/interim/bert_pytorch_checkpoint",
        description="Saved PyTorch training checkpoint and validation inputs.",
    )
    OUTPUT_DIR: str = Field(
        default="./data/interim/bert_decoder_debug",
        description="Output directory for decoder metrics and entity errors.",
    )
    DEVICE: str = Field(
        default="auto", description="Inference device: auto, cpu, cuda, or mps."
    )
    SPLIT: str = Field(
        default="validation", description="Evaluation split: validation or test."
    )
    DATASET_PATH: str = Field(
        default="",
        description="Original prepared DocILE dataset for checkpoints without saved test records.",
    )
    RANDOM_SEED: int = Field(
        default=42,
        description="Original training split seed for rebuilding test records.",
    )


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
    BERT_PYTORCH: BertPytorchConfig = Field(default_factory=BertPytorchConfig)
    BERT_PYTORCH_DEBUG: BertPytorchDebugConfig = Field(
        default_factory=BertPytorchDebugConfig
    )
    KAGGLE_INVOICES: KaggleInvoicesConfig = Field(default_factory=KaggleInvoicesConfig)
    DOCILE: DocileInvoicesConfig = Field(default_factory=DocileInvoicesConfig)
    RANDOM_SEED: int = 42
    DATASET_NAME: DatasetName = DatasetName.KAGGLE_INVOICES
    DOCILE_TOKEN: str = Field(default="", repr=False)


@lru_cache
def get_settings() -> Settings:
    return Settings()
