import json
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

from app.aggregates.bert.params import BertPytorchParams, DatasetName
from app.extraction.bert.pytorch.builder import InvoiceRecordBuilder
from app.extraction.bert.pytorch.datasets import InvoiceDataset
from app.extraction.bert.pytorch.docile.builder import DocileRecordBuilder
from app.extraction.bert.pytorch.inspector import EntityErrorInspector
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
    @staticmethod
    def load_frame(params: BertPytorchParams) -> pd.DataFrame:
        """Read prepared data without downloading archives or changing OCR offsets."""
        path = Path(params.dataset_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Prepared dataset CSV not found: {path}")
        dataset_type = DatasetName(params.dataset_type)
        if dataset_type == DatasetName.DOCILE and path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(path)
        elif path.suffix.lower() == ".csv":
            frame = pd.read_csv(
                path,
                keep_default_na=False,
                dtype={
                    "ocred_text": str,
                    "ocr_text": str,
                    "document_id": str,
                    "json_data": str,
                },
            )
        else:
            formats = "CSV or Parquet" if dataset_type == DatasetName.DOCILE else "CSV"
            raise ValueError(f"Prepared dataset must be a {formats} file: {path}")
        required = (
            {"document_id", "ocr_text", "text_with_boxes", "json_data"}
            if dataset_type == DatasetName.DOCILE
            else {"ocred_text", "json_data"}
        )
        if missing := required - set(frame.columns):
            raise ValueError(f"Dataset CSV is missing columns: {sorted(missing)}")
        if dataset_type == DatasetName.DOCILE:
            if frame["document_id"].duplicated().any():
                raise ValueError("DocILE document IDs must be unique")
            for column in ("text_with_boxes", "json_data"):
                frame[column] = frame[column].map(
                    lambda value: json.loads(value) if isinstance(value, str) else value
                )
        return frame

    @staticmethod
    def prepare_records(df: pd.DataFrame, dataset_type: DatasetName):
        """Use geometry for DocILE and text alignment for Kaggle invoices."""
        label_schema = InvoiceLabelSchema()
        if dataset_type == DatasetName.DOCILE:
            label_schema = InvoiceLabelSchema(
                field_to_entity={
                    **label_schema.field_to_entity,
                    "invoice.due_date": "DUE_DATE",
                    "payment_instructions.payment_terms": "PAYMENT_TERMS",
                }
            )
            records, rejected = DocileRecordBuilder(label_schema).build(df)
        else:
            normalizer = Normalizer()
            numeric = NumericField(normalizer=normalizer)
            matcher = OCRSpanMatcher(
                summary_span=OCRSummarySpan(normalizer=normalizer),
                normalizer=normalizer,
                numeric=numeric,
            )
            records, rejected = InvoiceRecordBuilder(
                span_matcher=matcher,
                numeric=numeric,
                normalizer=normalizer,
            ).build(df)
        return records, rejected, label_schema

    @staticmethod
    def split_records(records: list[dict], df: pd.DataFrame, params: BertPytorchParams):
        """Keep official DocILE validation documents out of training and tuning.

        Use part of the official training set for tuning. Prepared data without
        official split metadata uses a seeded 80/10/10 document split.
        """
        if params.dataset_type == DatasetName.DOCILE and "split" in df:
            membership = dict(zip(df["document_id"], df["split"], strict=True))
            if set(membership.values()) - {"train", "val"}:
                raise ValueError("DocILE split values must be 'train' or 'val'")
            train = [r for r in records if membership[r["document_id"]] == "train"]
            test = [r for r in records if membership[r["document_id"]] == "val"]
            if len(train) < 2 or not test:
                raise ValueError(
                    "Accepted DocILE records must include train and val documents"
                )
            train, validation = train_test_split(
                train,
                test_size=0.1,
                random_state=params.random_seed,
            )
            return train, validation, test
        train, remaining = train_test_split(
            records,
            test_size=0.2,
            random_state=params.random_seed,
        )
        validation, test = train_test_split(
            remaining,
            test_size=0.5,
            random_state=params.random_seed,
        )
        return train, validation, test

    def run(self, params: BertPytorchParams):
        """Train on accepted invoices and evaluate the best weights on held-out data."""
        df = self.load_frame(params)

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

        records, rejected, label_schema = self.prepare_records(df, params.dataset_type)
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
        train_records, validation_records, test_records = self.split_records(
            records, df, params
        )
        logger.info(
            "bert_pytorch_split",
            source="official_docile"
            if params.dataset_type == DatasetName.DOCILE and "split" in df
            else "random",
            training=len(train_records),
            validation=len(validation_records),
            final_evaluation=len(test_records),
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
        inspector = (
            EntityErrorInspector(
                model=model,
                tokenizer=tokenizer,
                label_schema=label_schema,
                device=device,
                max_length=params.max_length_tokens,
            )
            if params.diagnostic_dir
            else None
        )
        trainer = OCRInvoiceTraining(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=validation_loader,
            device=device,
            epochs=params.num_epoch,
            id2label=label_schema.id2label,
            inspector=inspector,
            diagnostic_dir=params.diagnostic_dir,
            validation_records=validation_records if inspector is not None else None,
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
