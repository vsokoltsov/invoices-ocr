from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.aggregates.layoutlm.params import LayoutLMDevice
from app.aggregates.pipeline import BertRunMode, PipelineExecution
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
        description="Directory for downloaded archives and extracted source files.",
    )
    OUTPUT_PATH: str = Field(
        default="./data/raw/docile.parquet",
        description="Raw DocILE Parquet output written by datasets load docile.",
    )
    SOURCE_PATH: str = Field(
        default="",
        description="Raw input Parquet path or clearml://<dataset-id>/raw.parquet; empty uses DOCILE_OUTPUT_PATH.",
    )
    SPLIT: DocileSplit = Field(
        default=DocileSplit.TRAINVAL,
        description="Source split to download or load: train, val, trainval, or test.",
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

    @property
    def raw_source_path(self) -> str:
        """Return the explicit raw input or the local loader output path."""
        return self.SOURCE_PATH or self.OUTPUT_PATH


class BertPytorchConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BERTPYTORCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    DATASET_PATH: str = Field(
        default="./data/interim/docile.parquet",
        description="Prepared local Parquet or clearml://<dataset-id>/preprocessed.parquet.",
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
    MODEL_NAME: str = Field(
        default="google-bert/bert-base-uncased",
        description="Hugging Face checkpoint used to initialize the token-classification model.",
    )
    BATCH_SIZE: int = Field(
        default=16,
        description="Token windows per training or evaluation batch.",
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


class LayoutLMConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LAYOUTLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    DATASET_PATH: str = Field(
        default="./data/interim/docile.parquet",
        description="Prepared local Parquet or clearml://<dataset-id>/preprocessed.parquet.",
    )
    MODEL_NAME: str = Field(
        default="microsoft/layoutlmv3-base",
        description="Hugging Face checkpoint used to initialize LayoutLMv3.",
    )
    OUTPUT_DIR: str = Field(
        default="./data/interim/layoutlm_experiment",
        description="Directory for history.csv and training reports.",
    )
    CHECKPOINT_PATH: str = Field(
        default="./data/interim/layoutlm_experiment/best.pt",
        description="File path for the checkpoint with the best validation entity F1.",
    )
    IMAGES_PATH: str = Field(
        default="./data/interim/docile_images",
        description="Local page cache or clearml://<dataset-id>/images.",
    )
    NUM_EPOCH: int = Field(default=20, gt=0, description="Number of training epochs.")
    BATCH_SIZE: int = Field(
        default=16, gt=0, description="Token windows per training or evaluation batch."
    )
    MAX_LENGTH_TOKENS: int = Field(
        default=512,
        ge=4,
        le=512,
        description="Maximum tokenized window length, including special tokens.",
    )
    LEARNING_RATE: float = Field(
        default=3e-5, gt=0, description="Peak AdamW learning rate."
    )
    WEIGHT_DECAY: float = Field(
        default=0.01, ge=0, description="AdamW weight decay for decay parameter groups."
    )
    WARMUP_RATIO: float = Field(
        default=0.1,
        ge=0,
        lt=1,
        description="Fraction of optimizer steps used for linear warmup before linear decay.",
    )
    MIN_WORD_COVERAGE: float = Field(
        default=0.5,
        gt=0,
        le=1,
        description="Minimum fraction of an OCR word box covered by an annotation.",
    )
    DATALOADER_NUM_WORKERS: int = Field(
        default=0, ge=0, description="Data loader worker processes."
    )
    RANDOM_SEED: int = Field(
        default=42,
        ge=0,
        le=2**32 - 1,
        description="Seed for data splitting and training.",
    )
    DEVICE: LayoutLMDevice = Field(
        default=LayoutLMDevice.AUTO,
        description="Training device: auto, cpu, cuda, or mps.",
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


class ClearMLPipelineConfig(BaseSettings):
    """Configure pipeline orchestration separately from model hyperparameters."""

    model_config = SettingsConfigDict(
        env_prefix="CLEARML_PIPELINE_", env_file=".env", extra="ignore"
    )

    PROJECT: str = Field(
        default="ocr", description="ClearML project for pipeline runs."
    )
    EXECUTION: PipelineExecution = Field(
        default=PipelineExecution.LOCAL,
        description="Run stages as local subprocesses or through a ClearML agent queue.",
    )
    QUEUE: str = Field(default="default", description="ClearML agent execution queue.")
    BERT_MODE: BertRunMode = Field(
        default=BertRunMode.MODEL,
        description="BERT models to train: model, line-items, or both.",
    )


class ClearMLDatasetConfig(BaseSettings):
    """Configure storage for versioned invoice datasets."""

    model_config = SettingsConfigDict(
        env_prefix="CLEARML_DATASET_", env_file=".env", extra="ignore"
    )
    PROJECT: str = Field(default="ocr", description="ClearML project for datasets.")
    OUTPUT_URI: str = Field(
        default="",
        description="Upload storage URI; empty uses ClearML default storage.",
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    BERT_PYTORCH: BertPytorchConfig = Field(default_factory=BertPytorchConfig)
    LAYOUTLM: LayoutLMConfig = Field(default_factory=LayoutLMConfig)
    BERT_PYTORCH_DEBUG: BertPytorchDebugConfig = Field(
        default_factory=BertPytorchDebugConfig
    )
    DOCILE: DocileInvoicesConfig = Field(default_factory=DocileInvoicesConfig)
    CLEARML_PIPELINE: ClearMLPipelineConfig = Field(
        default_factory=ClearMLPipelineConfig
    )
    CLEARML_DATASET: ClearMLDatasetConfig = Field(default_factory=ClearMLDatasetConfig)
    RANDOM_SEED: int = 42
    DOCILE_TOKEN: str = Field(default="", repr=False)


@lru_cache
def get_settings() -> Settings:
    return Settings()
