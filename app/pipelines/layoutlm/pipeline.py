import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from pipefunc import PipeFunc, Pipeline
from transformers import AutoTokenizer, LayoutLMv3ImageProcessor
from transformers.tokenization_utils_tokenizers import TokenizersBackend

from app.aggregates.layoutlm.params import LayoutLMDevice, LayoutLMTrainingParams
from app.extraction.layoutlm.builder import InvoiceBuilder
from app.extraction.layoutlm.converter import AttachmentToImageConverter
from app.extraction.layoutlm.metrics import EpochMetrics
from app.extraction.layoutlm.model import LayoutLMFieldExtractor
from app.extraction.layoutlm.schema import InvoiceLabelSchema
from app.pipelines.layoutlm.builder import BuildInvoices
from app.pipelines.layoutlm.dataloader import InvoiceDataLoaders
from app.pipelines.layoutlm.split import DatasetsSplit
from app.pipelines.layoutlm.training import LayoutLMTraining
from app.pipelines.layoutlm.evaluate import EvaluateTraining
from app.pipelines.layoutlm.publish import PublishModel
from app.extraction.layoutlm.publisher import ClearMLPublisher


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

    def run(self) -> pd.DataFrame:
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
        output_dir = Path(self.params.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        schema = InvoiceLabelSchema()
        fields = list(schema.field_to_entity.keys())
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
        tokenizer.save_pretrained(output_dir / "processor")
        image_processor.save_pretrained(output_dir / "processor")
        (output_dir / "config.json").write_text(
            json.dumps(
                {**asdict(self.params), "field_to_entity": schema.field_to_entity},
                indent=2,
            ),
            encoding="utf-8",
        )
        converter = AttachmentToImageConverter(
            images_path=Path(self.params.images_path)
        )
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
        metrics = EpochMetrics(
            labels=schema.labels,
            fields=fields,
        )
        publisher = ClearMLPublisher()
        build_step = BuildInvoices(builder=invoice_builder, output_dir=output_dir)
        dataloaders_step = InvoiceDataLoaders(
            tokenizer=cast(TokenizersBackend, tokenizer),
            image_processor=image_processor,
            converter=converter,
            device=device,
            params=self.params,
        )
        training_step = LayoutLMTraining(
            model=model,
            metrics=metrics,
            schema=schema,
            device=device,
            params=self.params,
        )
        eval_training_step = EvaluateTraining(
            output_dir=Path(self.params.output_dir)
        )
        publisher_step = PublishModel(
            publisher=publisher,
            checkpoint_path=self.params.checkpoint_path
        )
        pipeline = Pipeline(
            [
                PipeFunc(DatasetsSplit.create, output_name="split"),
                PipeFunc(build_step.build, output_name="build_result"),
                PipeFunc(dataloaders_step.build, output_name="dataloaders"),
                PipeFunc(training_step.train, output_name="train_result"),
                PipeFunc(eval_training_step.eval, output_name="output_dir"),
                PipeFunc(publisher_step.publish, output_name="model_id"),
            ]
        )
        return pipeline.run(
            "model_id", kwargs={"df": df, "random_seed": self.params.random_seed}
        )
