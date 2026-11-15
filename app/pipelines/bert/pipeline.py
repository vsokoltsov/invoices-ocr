import json
from dataclasses import asdict, dataclass
from typing import cast

from transformers import AutoTokenizer, set_seed
from transformers.tokenization_utils_tokenizers import TokenizersBackend

from app.aggregates.bert.params import BertPytorchParams
from app.extraction.bert.artifacts import BertArtifacts
from app.extraction.bert.builder import InvoiceRecordBuilder
from app.extraction.bert.collation import InvoiceDataCollator
from app.extraction.bert.decoding import OCRWordDecoder
from app.extraction.bert.inspector import EntityErrorInspector
from app.extraction.bert.item_builder import ItemRecordBuilder
from app.extraction.bert.model import BertFieldExtractor
from app.extraction.bert.publisher import ClearMLPublisher
from app.pipelines.bert.builder import InvoicesBuilder
from app.pipelines.bert.dataloader import InvoiceDataLoader
from app.pipelines.bert.evaluate import EvaluateTraining
from app.pipelines.bert.training import BertTraining
from app.pipelines.device import DeviceSelector
from app.pipelines.loading import DocileDataset
from app.pipelines.pipeline import TrainingPipeline
from app.pipelines.publish import PublishModel


@dataclass
class BertTrainingPipeline:
    """Configure BERT-specific components for the shared training graph.

    Attributes:
        params: Fixed dataset, model, optimization, and output settings.
    """

    params: BertPytorchParams

    def run(self) -> str:
        """Configure BERT steps and execute the shared training graph.

        Returns:
            Registered ClearML model ID.

        Raises:
            ValueError: Source documents are invalid or tokenization is not fast.

        Validates prepared Parquet inputs before loading model weights. Model,
        storage, and optimization settings come from the constructor parameters.
        """
        params = self.params

        set_seed(params.random_seed)
        frame = DocileDataset(
            params.dataset_path,
            frozenset({"document_id", "ocr_text", "text_with_boxes", "json_data"}),
        ).load()
        frame["json_data"] = frame["json_data"].map(
            lambda value: json.loads(value) if isinstance(value, str) else value
        )
        device = DeviceSelector().resolve()
        artifacts = BertArtifacts.from_params(params)
        artifacts.prepare()
        builder = (
            ItemRecordBuilder() if params.line_items_only else InvoiceRecordBuilder()
        )
        schema = builder.label_schema
        config = {**asdict(params), "field_to_entity": schema.field_to_entity}
        artifacts.save_config(config)
        tokenizer = AutoTokenizer.from_pretrained(params.model_name, use_fast=True)
        if tokenizer is None or not tokenizer.is_fast:
            raise ValueError("Span alignment requires a fast tokenizer")
        model = BertFieldExtractor(params.model_name, schema.num_labels).to(device)
        word_decoder = OCRWordDecoder(schema.label2id)
        inspector = (
            EntityErrorInspector(
                model=model,
                tokenizer=tokenizer,
                label_schema=schema,
                device=device,
                max_length=params.max_length_tokens,
                word_decoder=word_decoder,
            )
            if params.diagnostic_dir
            else None
        )
        build_step = InvoicesBuilder(builder, artifacts)
        loader_step = InvoiceDataLoader(
            schema=schema,
            tokenizer=cast(TokenizersBackend, tokenizer),
            collate_fn=InvoiceDataCollator(tokenizer),
            device=device,
            batch_size=params.batch_size,
            max_length_tokens=params.max_length_tokens,
            random_seed=params.random_seed,
            include_decoder_metadata=True,
        )
        training_step = BertTraining(
            model=model,
            schema=schema,
            tokenizer=cast(TokenizersBackend, tokenizer),
            device=device,
            epochs=params.num_epoch,
            learning_rate=params.learning_rate,
            weight_decay=params.weight_decay,
            artifacts=artifacts,
            max_length_tokens=params.max_length_tokens,
            inspector=inspector,
            word_decoder=word_decoder,
        )
        evaluation_step = EvaluateTraining(artifacts, device)
        publisher_step = PublishModel(
            ClearMLPublisher(
                artifacts=artifacts,
                task_name="bert-items" if params.line_items_only else "bert-body",
            )
        )
        return TrainingPipeline(
            random_seed=params.random_seed,
            build=build_step.build,
            dataloaders=loader_step.build,
            train=training_step.train,
            evaluate=evaluation_step.eval,
            publish=publisher_step.publish,
        ).run(frame)
