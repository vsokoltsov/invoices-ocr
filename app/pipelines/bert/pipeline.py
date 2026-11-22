from dataclasses import asdict, dataclass
from functools import cached_property
from typing import cast

from transformers import AutoTokenizer
from transformers.tokenization_utils_tokenizers import TokenizersBackend

from app.aggregates.bert.params import BertPytorchParams
from app.extraction.bert.artifacts import BertArtifacts
from app.extraction.bert.collation import InvoiceDataCollator
from app.extraction.bert.inspector import EntityErrorInspector
from app.extraction.bert.item_preprocess import ItemPreprocessor
from app.extraction.bert.model import BertFieldExtractor
from app.extraction.bert.preprocess import InvoicePreprocessor
from app.extraction.bert.publisher import ClearMLPublisher
from app.pipelines.bert.dataloader import InvoiceDataLoader
from app.pipelines.bert.evaluate import EvaluateTraining
from app.pipelines.bert.preprocess import BertPreprocess
from app.pipelines.bert.training import BertTraining
from app.pipelines.components import ModelPipeline
from app.pipelines.device import DeviceSelector
from app.pipelines.loading import DocileDataset
from app.pipelines.publish import PublishModel


@dataclass
class BertTrainingPipeline(ModelPipeline):
    """Configure reusable BERT components for local and ClearML execution.

    Attributes:
        params: Fixed dataset, model, optimization, and output settings.
    """

    params: BertPytorchParams

    @property
    def random_seed(self):
        """Return the document split and training seed."""
        return self.params.random_seed

    def load_source(self):
        """Load and validate source documents without transforming their contents.

        Returns:
            Prepared DocILE source DataFrame.
        """
        return DocileDataset(
            self.params.dataset_path,
            frozenset({"document_id", "ocr_text", "text_with_boxes", "json_data"}),
        ).load()

    @cached_property
    def device(self):
        """Resolve the preferred available training accelerator."""
        return DeviceSelector().resolve()

    @cached_property
    def artifacts(self):
        """Return the configured checkpoint, history, and diagnostic storage."""
        return BertArtifacts.from_params(self.params)

    @cached_property
    def record_preprocessor(self):
        """Select invoice or line-item record preparation for this model run."""
        return (
            ItemPreprocessor() if self.params.line_items_only else InvoicePreprocessor()
        )

    @cached_property
    def schema(self):
        """Return the schema associated with the selected record preprocessor."""
        return self.record_preprocessor.label_schema

    @cached_property
    def tokenizer(self):
        """Load a fast tokenizer for span alignment.

        Raises:
            ValueError: The checkpoint does not provide a fast tokenizer.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.params.model_name, use_fast=True)
        if tokenizer is None or not tokenizer.is_fast:
            raise ValueError("Span alignment requires a fast tokenizer")
        return cast(TokenizersBackend, tokenizer)

    @cached_property
    def model(self):
        """Initialize training weights only when a training task needs them."""
        return BertFieldExtractor(self.params.model_name, self.schema.num_labels).to(
            self.device
        )

    @cached_property
    def word_decoder(self):
        """Use first-subtoken argmax predictions for shared word scoring."""
        return None

    def prepare_run(self):
        """Create output directories and save fixed model configuration."""
        self.artifacts.prepare()
        self.artifacts.save_config(
            {**asdict(self.params), "field_to_entity": self.schema.field_to_entity}
        )

    @cached_property
    def preprocess_step(self):
        """Configure split record preparation without initializing model weights."""
        return BertPreprocess(self.record_preprocessor, self.artifacts)

    @cached_property
    def loader_step(self):
        """Configure token-window loaders and OCR-word decoding metadata."""
        return InvoiceDataLoader(
            schema=self.schema,
            tokenizer=self.tokenizer,
            collate_fn=InvoiceDataCollator(self.tokenizer),
            device=self.device,
            batch_size=self.params.batch_size,
            max_length_tokens=self.params.max_length_tokens,
            random_seed=self.params.random_seed,
            include_decoder_metadata=True,
        )

    @cached_property
    def training_step(self):
        """Configure training and optional validation diagnostics."""
        inspector = (
            EntityErrorInspector(
                model=self.model,
                tokenizer=self.tokenizer,
                label_schema=self.schema,
                device=self.device,
                max_length=self.params.max_length_tokens,
                word_decoder=self.word_decoder,
            )
            if self.params.diagnostic_dir
            else None
        )
        return BertTraining(
            model=self.model,
            schema=self.schema,
            tokenizer=self.tokenizer,
            device=self.device,
            epochs=self.params.num_epoch,
            learning_rate=self.params.learning_rate,
            weight_decay=self.params.weight_decay,
            artifacts=self.artifacts,
            max_length_tokens=self.params.max_length_tokens,
            inspector=inspector,
            word_decoder=self.word_decoder,
        )

    @cached_property
    def evaluation_step(self):
        """Configure saved-checkpoint evaluation without a live training model."""
        return EvaluateTraining(self.artifacts, self.device)

    @cached_property
    def publisher_step(self):
        """Configure model-specific ClearML artifact publication."""
        return PublishModel(
            ClearMLPublisher(
                artifacts=self.artifacts,
                task_name="bert-items" if self.params.line_items_only else "bert-body",
            )
        )

    def train_prepared(self, dataloaders, preprocess_result, split):
        """Train from persisted preparation outputs.

        Args:
            dataloaders: Constructed split loaders.
            preprocess_result: Prepared span records and rejection reports.
            split: Original document partitions used for OCR offsets.

        Returns:
            Saved best-checkpoint and history locations.
        """
        return self.training_step.train(dataloaders, preprocess_result, split)
