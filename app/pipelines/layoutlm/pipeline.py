from dataclasses import asdict, dataclass
from typing import cast

from transformers import AutoTokenizer, LayoutLMv3ImageProcessor, set_seed
from transformers.tokenization_utils_tokenizers import TokenizersBackend

from app.aggregates.layoutlm.params import LayoutLMTrainingParams
from app.extraction.layoutlm.artifacts import LayoutLMArtifacts
from app.extraction.layoutlm.builder import InvoiceBuilder
from app.extraction.layoutlm.converter import AttachmentToImageConverter
from app.extraction.layoutlm.model import LayoutLMFieldExtractor
from app.extraction.layoutlm.publisher import ClearMLPublisher
from app.extraction.schema import InvoiceLabelSchema
from app.pipelines.device import DeviceSelector
from app.pipelines.layoutlm.builder import BuildInvoices
from app.pipelines.layoutlm.dataloader import InvoiceDataLoaders
from app.pipelines.layoutlm.evaluate import EvaluateTraining
from app.pipelines.layoutlm.training import LayoutLMTraining
from app.pipelines.loading import DocileDataset
from app.pipelines.pipeline import TrainingPipeline
from app.pipelines.publish import PublishModel


@dataclass
class LayoutLMTrainingPipeline:
    """Configure LayoutLM-specific components for the shared training graph.

    Attributes:
        params: Fixed dataset, model, image-cache, and optimization settings.
    """

    params: LayoutLMTrainingParams

    def run(self) -> str:
        """Configure LayoutLM steps and execute the shared training graph.

        Returns:
            Registered ClearML model ID.

        Raises:
            ValueError: Source documents, the requested device, or tokenization
                settings are invalid.

        Saves processor settings and run configuration before executing the graph.
        Model, image-cache, and optimization settings come from constructor params.
        """
        set_seed(self.params.random_seed)
        device = DeviceSelector(self.params.device).resolve()
        df = DocileDataset(
            self.params.dataset_path,
            frozenset({"document_id", "json_data", "text_with_boxes", "pdf_path"}),
        ).load()
        artifacts = LayoutLMArtifacts.from_params(self.params)
        artifacts.prepare()
        schema = InvoiceLabelSchema()
        tokenizer = AutoTokenizer.from_pretrained(
            self.params.model_name,
            use_fast=True,
        )
        image_processor = LayoutLMv3ImageProcessor.from_pretrained(
            self.params.model_name,
            apply_ocr=False,
        )
        if tokenizer is None or not tokenizer.is_fast:
            raise ValueError(
                "LayoutLM training requires a fast tokenizer with word IDs"
            )
        artifacts.save_processor(tokenizer, image_processor)
        artifacts.save_config(
            {**asdict(self.params), "field_to_entity": schema.field_to_entity}
        )
        converter = AttachmentToImageConverter(artifacts)
        model = LayoutLMFieldExtractor(
            model_name=self.params.model_name,
            num_labels=len(schema.labels),
            id2label=schema.id2label,
            label2id=schema.label2id,
        ).to(device)
        invoice_builder = InvoiceBuilder(
            fields=schema.field_to_entity,
            label2id=schema.label2id,
            min_coverage=self.params.min_word_coverage,
        )
        build_step = BuildInvoices(builder=invoice_builder, artifacts=artifacts)
        dataloaders_step = InvoiceDataLoaders(
            tokenizer=cast(TokenizersBackend, tokenizer),
            image_processor=image_processor,
            converter=converter,
            device=device,
            batch_size=self.params.batch_size,
            max_length_tokens=self.params.max_length_tokens,
            num_workers=self.params.dataloader_num_workers,
        )
        training_step = LayoutLMTraining(
            model=model,
            schema=schema,
            device=device,
            epochs=self.params.num_epoch,
            learning_rate=self.params.learning_rate,
            weight_decay=self.params.weight_decay,
            warmup_ratio=self.params.warmup_ratio,
            artifacts=artifacts,
        )
        eval_training_step = EvaluateTraining(
            artifacts=artifacts,
            device=device,
        )
        publisher_step = PublishModel(
            publisher=ClearMLPublisher(artifacts),
        )
        return TrainingPipeline(
            random_seed=self.params.random_seed,
            build=build_step.build,
            dataloaders=dataloaders_step.build,
            train=training_step.train,
            evaluate=eval_training_step.eval,
            publish=publisher_step.publish,
        ).run(df)
