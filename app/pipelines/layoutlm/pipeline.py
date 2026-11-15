import random
from dataclasses import asdict, dataclass
from typing import cast

import numpy as np
import pandas as pd
import torch
from pipefunc import PipeFunc, Pipeline
from transformers import AutoTokenizer, LayoutLMv3ImageProcessor
from transformers.tokenization_utils_tokenizers import TokenizersBackend

from app.aggregates.layoutlm.params import LayoutLMDevice, LayoutLMTrainingParams
from app.extraction.layoutlm.artifacts import LayoutLMArtifacts
from app.extraction.layoutlm.builder import InvoiceBuilder
from app.extraction.layoutlm.converter import AttachmentToImageConverter
from app.extraction.layoutlm.model import LayoutLMFieldExtractor
from app.extraction.layoutlm.publisher import ClearMLPublisher
from app.extraction.schema import InvoiceLabelSchema
from app.pipelines.layoutlm.builder import BuildInvoices
from app.pipelines.layoutlm.dataloader import InvoiceDataLoaders
from app.pipelines.layoutlm.evaluate import EvaluateTraining
from app.pipelines.layoutlm.publish import PublishModel
from app.pipelines.layoutlm.split import DatasetsSplit
from app.pipelines.layoutlm.training import LayoutLMTraining


@dataclass
class LayoutLMTrainingPipeline:
    params: LayoutLMTrainingParams

    def _device(self) -> torch.device:
        if self.params.device == LayoutLMDevice.AUTO:
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        if self.params.device == LayoutLMDevice.CUDA and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        if (
            self.params.device == LayoutLMDevice.MPS
            and not torch.backends.mps.is_available()
        ):
            raise ValueError("MPS was requested but is not available")
        return torch.device(str(self.params.device))

    def run(self) -> str:
        """Train from prepared DocILE data and save run, split, and evaluation artifacts."""
        random.seed(self.params.random_seed)
        np.random.seed(self.params.random_seed)
        torch.manual_seed(self.params.random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.params.random_seed)

        device = self._device()
        df = pd.read_parquet(self.params.dataset_path)
        required = {"document_id", "json_data", "text_with_boxes", "pdf_path"}
        if missing := required - set(df.columns):
            raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
        if df["document_id"].isna().any() or df["document_id"].duplicated().any():
            raise ValueError("Document IDs must be present and unique before splitting")
        if len(df) < 6:
            raise ValueError(
                "At least six documents are required for an 80/10/10 split"
            )
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
        pipeline = Pipeline(
            [
                PipeFunc(DatasetsSplit.create, output_name="split"),
                PipeFunc(build_step.build, output_name="build_result"),
                PipeFunc(dataloaders_step.build, output_name="dataloaders"),
                PipeFunc(training_step.train, output_name="train_result"),
                PipeFunc(eval_training_step.eval, output_name="evaluation_result"),
                PipeFunc(publisher_step.publish, output_name="model_id"),
            ]
        )
        return pipeline.run(
            "model_id", kwargs={"df": df, "random_seed": self.params.random_seed}
        )
