import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import pandas as pd
import torch
from pipefunc import PipeFunc, Pipeline
from transformers import AutoTokenizer, set_seed
from transformers.tokenization_utils_tokenizers import TokenizersBackend

from app.aggregates.bert.params import BertPytorchParams
from app.extraction.bert.pytorch.collation import InvoiceDataCollator
from app.extraction.bert.pytorch.decoding import OCRWordDecoder
from app.extraction.bert.pytorch.docile.builder import DocileRecordBuilder
from app.extraction.bert.pytorch.docile.item_builder import DocileItemRecordBuilder
from app.extraction.bert.pytorch.inspector import EntityErrorInspector
from app.extraction.bert.pytorch.model import BertFieldExtractor
from app.extraction.bert.pytorch.publisher import ClearMLPublisher
from app.pipelines.bert.builder import InvoicesBuilder
from app.pipelines.bert.dataloader import InvoiceDataLoader
from app.pipelines.bert.evaluate import EvaluateTraining
from app.pipelines.bert.publish import PublishModel
from app.pipelines.bert.split import DatasetsSplit
from app.pipelines.bert.training import BertTraining


@dataclass
class BertTrainingPipeline:
    params: BertPytorchParams

    def run(self) -> str:
        """Train one DocILE model, evaluate its saved checkpoint, and publish it."""
        params = self.params

        set_seed(params.random_seed)
        frame = pd.read_parquet(params.dataset_path)
        required = {"document_id", "ocr_text", "text_with_boxes", "json_data"}
        if missing := required - set(frame.columns):
            raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
        if frame["document_id"].isna().any() or frame["document_id"].duplicated().any():
            raise ValueError("Document IDs must be present and unique before splitting")
        if len(frame) < 6:
            raise ValueError(
                "At least six documents are required for an 80/10/10 split"
            )
        frame["json_data"] = frame["json_data"].map(
            lambda value: json.loads(value) if isinstance(value, str) else value
        )
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        output_dir = Path(params.checkpoint_dir).expanduser().resolve()
        history_path = (
            Path(params.result_filename).expanduser().with_suffix(".csv").resolve()
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        builder = (
            DocileItemRecordBuilder()
            if params.line_items_only
            else DocileRecordBuilder()
        )
        schema = builder.label_schema
        config = {**asdict(params), "field_to_entity": schema.field_to_entity}
        (output_dir / "config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
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
        build_step = InvoicesBuilder(builder, output_dir)
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
            checkpoint_path=output_dir,
            history_path=history_path,
            max_length_tokens=params.max_length_tokens,
            inspector=inspector,
            diagnostic_dir=Path(params.diagnostic_dir)
            if params.diagnostic_dir
            else None,
            word_decoder=word_decoder,
        )
        evaluation_step = EvaluateTraining(output_dir, device)
        publisher_step = PublishModel(
            ClearMLPublisher(
                config=config,
                history_path=history_path,
                task_name="bert-items" if params.line_items_only else "bert-body",
            )
        )
        pipeline = Pipeline(
            [
                PipeFunc(DatasetsSplit.create, output_name="split"),
                PipeFunc(build_step.build, output_name="build_result"),
                PipeFunc(loader_step.build, output_name="dataloaders"),
                PipeFunc(training_step.train, output_name="train_result"),
                PipeFunc(evaluation_step.eval, output_name="evaluation_result"),
                PipeFunc(publisher_step.publish, output_name="model_id"),
            ]
        )
        return pipeline.run(
            "model_id", kwargs={"df": frame, "random_seed": params.random_seed}
        )
