from dataclasses import asdict, dataclass
from functools import cached_property
from typing import cast

from transformers import AutoTokenizer, LayoutLMv3ImageProcessor
from transformers.tokenization_utils_tokenizers import TokenizersBackend

from app.aggregates.layoutlm.params import LayoutLMTrainingParams
from app.extraction.layoutlm.artifacts import LayoutLMArtifacts
from app.extraction.layoutlm.builder import InvoiceBuilder
from app.extraction.layoutlm.converter import AttachmentToImageConverter
from app.extraction.layoutlm.model import LayoutLMFieldExtractor
from app.extraction.layoutlm.publisher import ClearMLPublisher
from app.extraction.schema import InvoiceLabelSchema
from app.pipelines.components import ModelPipeline
from app.pipelines.device import DeviceSelector
from app.pipelines.layoutlm.builder import BuildInvoices
from app.pipelines.layoutlm.dataloader import InvoiceDataLoaders
from app.pipelines.layoutlm.evaluate import EvaluateTraining
from app.pipelines.layoutlm.training import LayoutLMTraining
from app.pipelines.loading import DocileDataset
from app.pipelines.publish import PublishModel


@dataclass
class LayoutLMTrainingPipeline(ModelPipeline):
    """Configure reusable LayoutLM components for local and ClearML execution.

    Attributes:
        params: Fixed dataset, model, image-cache, and optimization settings.
    """

    params: LayoutLMTrainingParams

    @property
    def random_seed(self):
        """Return the document split and training seed."""
        return self.params.random_seed

    def load_source(self):
        """Load validated DocILE documents with OCR boxes and PDF paths.

        Returns:
            Prepared source DataFrame.
        """
        return DocileDataset(
            self.params.dataset_path,
            frozenset(
                {
                    "document_id",
                    "json_data",
                    "text_with_boxes",
                    "pdf_path",
                }
            ),
        ).load()

    @cached_property
    def device(self):
        """Resolve and validate the configured execution device."""
        return DeviceSelector(self.params.device).resolve()

    @cached_property
    def artifacts(self):
        """Return storage for the model, reports, processor, and image cache."""
        return LayoutLMArtifacts.from_params(self.params)

    @cached_property
    def schema(self):
        """Return the configured invoice BIO label mapping."""
        return InvoiceLabelSchema()

    @cached_property
    def tokenizer(self):
        """Load a fast tokenizer with OCR-word alignment support.

        Raises:
            ValueError: The checkpoint does not provide a fast tokenizer.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.params.model_name, use_fast=True)
        if tokenizer is None or not tokenizer.is_fast:
            raise ValueError(
                "LayoutLM training requires a fast tokenizer with word IDs"
            )
        return cast(TokenizersBackend, tokenizer)

    @cached_property
    def image_processor(self):
        """Load image preprocessing with built-in OCR disabled."""
        return LayoutLMv3ImageProcessor.from_pretrained(
            self.params.model_name, apply_ocr=False
        )

    def prepare_run(self):
        """Create output directories and save model and processor configuration."""
        self.artifacts.prepare()
        self.artifacts.save_processor(self.tokenizer, self.image_processor)
        self.artifacts.save_config(
            {**asdict(self.params), "field_to_entity": self.schema.field_to_entity}
        )

    @cached_property
    def model(self):
        """Initialize training weights only when the training task needs them."""
        return LayoutLMFieldExtractor(
            model_name=self.params.model_name,
            num_labels=self.schema.num_labels,
            id2label=self.schema.id2label,
            label2id=self.schema.label2id,
        ).to(self.device)

    @cached_property
    def build_step(self):
        """Configure page preparation without initializing training weights."""
        return BuildInvoices(
            InvoiceBuilder(
                fields=self.schema.field_to_entity,
                label2id=self.schema.label2id,
                min_coverage=self.params.min_word_coverage,
            ),
            self.artifacts,
        )

    @cached_property
    def loader_step(self):
        """Configure image, text, and coordinate dataset construction."""
        return InvoiceDataLoaders(
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            converter=AttachmentToImageConverter(self.artifacts),
            device=self.device,
            batch_size=self.params.batch_size,
            max_length_tokens=self.params.max_length_tokens,
            num_workers=self.params.dataloader_num_workers,
        )

    @cached_property
    def training_step(self):
        """Configure LayoutLM optimizer, schedule, metrics, and checkpoints."""
        return LayoutLMTraining(
            model=self.model,
            schema=self.schema,
            device=self.device,
            epochs=self.params.num_epoch,
            learning_rate=self.params.learning_rate,
            weight_decay=self.params.weight_decay,
            warmup_ratio=self.params.warmup_ratio,
            artifacts=self.artifacts,
        )

    @cached_property
    def evaluation_step(self):
        """Configure saved-checkpoint test evaluation."""
        return EvaluateTraining(self.artifacts, self.device)

    @cached_property
    def publisher_step(self):
        """Configure LayoutLM artifact publication."""
        return PublishModel(ClearMLPublisher(self.artifacts))

    def train_prepared(self, dataloaders, build_result, split):
        """Train LayoutLM from persisted prepared page records.

        Args:
            dataloaders: Constructed split loaders.
            build_result: Preparation outputs; already represented in the loaders.
            split: Source partitions; page training uses the prepared records.

        Returns:
            Saved best-checkpoint and history locations.
        """
        return self.training_step.train(dataloaders)
