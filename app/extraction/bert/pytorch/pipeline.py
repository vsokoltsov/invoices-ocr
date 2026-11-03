import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForTokenClassification, set_seed

from app.aggregates.bert.params import BertPytorchParams
from app.extraction.bert.pytorch.builder import InvoiceRecordBuilder
from app.extraction.bert.pytorch.datasets import InvoiceDataset
from app.extraction.bert.pytorch.model import BertFieldExtractor
from app.extraction.bert.pytorch.schema import InvoiceLabelSchema
from app.extraction.bert.pytorch.spans import (
    Normalizer,
    NumericField,
    OCRSpanMatcher,
    OCRSummarySpan,
)
from app.extraction.bert.pytorch.training import OCRInvoiceTraining

logger = structlog.get_logger()


@dataclass
class BertPytorchPipeline:
    def run(self, params: BertPytorchParams):
        """Train on accepted invoices and evaluate the best weights on held-out data."""
        dataset_path = Path(params.dataset_path).expanduser().resolve()
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Prepared dataset CSV not found: {dataset_path}")
        if dataset_path.suffix.lower() != ".csv":
            raise ValueError(f"Prepared dataset must be a CSV file: {dataset_path}")

        df = pd.read_csv(
            dataset_path,
            keep_default_na=False,
            dtype={"ocred_text": str, "json_data": str},
        )
        missing_columns = {"ocred_text", "json_data"} - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"Dataset CSV is missing columns: {sorted(missing_columns)}"
            )

        random.seed(params.random_seed)
        np.random.seed(params.random_seed)
        torch.manual_seed(params.random_seed)
        set_seed(params.random_seed)

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            params.model_name,
            use_fast=True,
        )
        if tokenizer is None:
            raise RuntimeError(f"Could not load tokenizer for {params.model_name}")
        if not tokenizer.is_fast:
            raise ValueError("Span alignment requires a fast tokenizer")
        collator = DataCollatorForTokenClassification(
            tokenizer=tokenizer,
            label_pad_token_id=-100,
        )

        normalizer = Normalizer()
        numeric_field_handler = NumericField(normalizer=normalizer)
        summary_span = OCRSummarySpan(normalizer=normalizer)
        span_matcher = OCRSpanMatcher(
            summary_span=summary_span,
            normalizer=normalizer,
            numeric=numeric_field_handler,
        )
        builder = InvoiceRecordBuilder(
            span_matcher=span_matcher,
            numeric=numeric_field_handler,
            normalizer=normalizer,
        )
        label_schema = InvoiceLabelSchema()
        logger.info("Prepare training spans", dataset_path=str(dataset_path))
        records, rejected = builder.build(df)
        logger.info(
            "bert_pytorch_records", accepted=len(records), rejected=len(rejected)
        )
        if len(records) < 10:
            raise ValueError("At least 10 accepted invoices are required for the split")
        logger.info("3. Init model")
        model = BertFieldExtractor(
            model_name=params.model_name, num_labels=label_schema.num_labels
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params.learning_rate,
            weight_decay=params.weight_decay,
        )
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        logger.info("4. Split data and init datasets")
        train_records, remaining_records = train_test_split(
            records,
            test_size=0.2,
            shuffle=True,
            random_state=params.random_seed,
        )

        validation_records, test_records = train_test_split(
            remaining_records,
            test_size=0.5,
            shuffle=True,
            random_state=params.random_seed,
        )
        train_dataset = InvoiceDataset(
            train_records,
            tokenizer,
            label_schema.label2id,
            label_schema.field_to_entity,
            max_length=params.max_length_tokens,
        )
        validation_dataset = InvoiceDataset(
            validation_records,
            tokenizer,
            label_schema.label2id,
            label_schema.field_to_entity,
            max_length=params.max_length_tokens,
        )
        test_dataset = InvoiceDataset(
            test_records,
            tokenizer,
            label_schema.label2id,
            label_schema.field_to_entity,
            max_length=params.max_length_tokens,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=params.batch_size,
            shuffle=True,
            collate_fn=collator,
            generator=torch.Generator().manual_seed(params.random_seed),
        )

        validation_loader = DataLoader(
            validation_dataset,
            batch_size=params.batch_size,
            collate_fn=collator,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=params.batch_size,
            collate_fn=collator,
        )
        logger.info("5. Init training process")
        trainer = OCRInvoiceTraining(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=validation_loader,
            device=device,
            epochs=params.num_epoch,
            id2label=label_schema.id2label,
        )
        logger.info("6. Start training")
        history_df = trainer.train()
        result_path = (
            Path(params.result_filename).expanduser().with_suffix(".csv").resolve()
        )
        result_path.parent.mkdir(parents=True, exist_ok=True)
        history_df.to_csv(result_path, index=False)
        logger.info("Saved training history", path=str(result_path))
        trainer.val_loader = test_loader
        logger.info("7. Verify results on test loader")
        test_metrics = trainer.eval_one_epoch(epoch=params.num_epoch)
        logger.info(
            "bert_pytorch_test",
            precision=test_metrics.precision,
            recall=test_metrics.recall,
            f1=test_metrics.f1,
        )
        return history_df, test_metrics
