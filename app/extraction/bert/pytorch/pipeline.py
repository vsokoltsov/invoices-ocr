import json
import random
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, set_seed

from app.aggregates.bert.params import BertPytorchParams, DatasetName
from app.extraction.bert.pytorch.builder import InvoiceRecordBuilder
from app.extraction.bert.pytorch.checkpoint import BertPytorchCheckpoint
from app.extraction.bert.pytorch.collation import InvoiceDataCollator
from app.extraction.bert.pytorch.datasets import InvoiceDataset
from app.extraction.bert.pytorch.decoding import OCRWordDecoder
from app.extraction.bert.pytorch.docile.builder import DocileRecordBuilder
from app.extraction.bert.pytorch.docile.item_builder import DocileItemRecordBuilder
from app.extraction.bert.pytorch.enums import InvoiceBIOLabel, InvoiceField
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
    def prepare_records(
        df: pd.DataFrame, dataset_type: DatasetName, *, line_items_only: bool = False
    ):
        """Use geometry for DocILE and text alignment for Kaggle invoices."""
        if line_items_only:
            if dataset_type != DatasetName.DOCILE:
                raise ValueError("--line-items-only requires --dataset-name docile")
            builder = DocileItemRecordBuilder()
            records, rejected = builder.build(df)
            return records, rejected, builder.label_schema
        label_schema = InvoiceLabelSchema()
        if dataset_type == DatasetName.DOCILE:
            label_schema = InvoiceLabelSchema(
                field_to_entity={
                    **label_schema.field_to_entity,
                    "invoice.due_date": "DUE_DATE",
                    "payment_instructions.payment_terms": "PAYMENT_TERMS",
                    InvoiceField.AMOUNT_DUE.value: InvoiceBIOLabel.AMOUNT_DUE.value,
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
        """Train separate body and item models, or one explicitly selected model.

        DocILE document membership is fixed before task-specific record rejection.
        Kaggle supports only the body model because it has no item-box alignment.
        """
        if params.body_only and params.line_items_only:
            raise ValueError(
                "--body-only and --line-items-only cannot be used together"
            )
        if params.line_items_only and params.dataset_type != DatasetName.DOCILE:
            raise ValueError("--line-items-only requires --dataset-name docile")
        df = self.load_frame(params)
        rows = [{"row_position": position} for position in range(len(df))]
        if params.dataset_type == DatasetName.DOCILE:
            for row, document_id in zip(rows, df["document_id"], strict=True):
                row["document_id"] = document_id
        train_rows, validation_rows, test_rows = self.split_records(rows, df, params)
        train_df = df.iloc[[row["row_position"] for row in train_rows]].copy()
        validation_df = df.iloc[[row["row_position"] for row in validation_rows]].copy()
        test_df = df.iloc[[row["row_position"] for row in test_rows]].copy()
        targets = (
            ("items",)
            if params.line_items_only
            else ("body",)
            if params.body_only or params.dataset_type != DatasetName.DOCILE
            else ("body", "items")
        )
        results = {}
        for target in targets:
            model_params = replace(
                params, line_items_only=target == "items", body_only=target == "body"
            )
            if len(targets) == 2:
                result = Path(params.result_filename).with_suffix(".csv")
                model_params = replace(
                    model_params,
                    result_filename=str(
                        result.with_name(f"{result.stem}_{target}.csv")
                    ),
                    checkpoint_dir=str(Path(params.checkpoint_dir) / target),
                    diagnostic_dir=str(Path(params.diagnostic_dir) / target)
                    if params.diagnostic_dir
                    else params.diagnostic_dir,
                )
            results[target] = self._run_model(
                model_params, train_df, validation_df, test_df
            )
        return results if len(targets) == 2 else results[targets[0]]

    def _run_model(self, params, train_df, validation_df, test_df):
        """Prepare fixed partitions and select weights using evaluation decoding."""

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

        collator = InvoiceDataCollator(tokenizer=tokenizer)

        prepared = [
            self.prepare_records(
                partition, params.dataset_type, line_items_only=params.line_items_only
            )
            for partition in (train_df, validation_df, test_df)
        ]
        train_records, validation_records, test_records = [
            result[0] for result in prepared
        ]
        records = train_records + validation_records + test_records
        rejected = [record for result in prepared for record in result[1]]
        label_schema = prepared[0][2]
        if not all((train_records, validation_records, test_records)):
            raise ValueError(
                "Each fixed document split must contain accepted records for this model"
            )
        df = pd.concat((train_df, validation_df, test_df))
        word_decoder = None
        if params.dataset_type == DatasetName.DOCILE:
            word_decoder = OCRWordDecoder(label_schema.label2id)
            words = dict(zip(df["document_id"], df["text_with_boxes"], strict=True))
            for record in records:
                record["text_with_boxes"] = words[record["document_id"]]
        logger.info(
            "bert_pytorch_records",
            accepted=len(records),
            rejected=len(rejected),
            decoder="ocr_word" if word_decoder is not None else "token_argmax",
            training_target="line_items"
            if params.line_items_only
            else "invoice_fields",
            amount_due_suppressed_words=sum(
                record.get("amount_due_suppressed_words", 0) for record in records
            ),
        )
        if len(records) < 10:
            raise ValueError("At least 10 accepted invoices are required for training")
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

        logger.info("4. Init datasets from prepared partitions")
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
            include_decoder_metadata=word_decoder is not None,
        )
        test_dataset = InvoiceDataset(
            test_records,
            tokenizer,
            label_schema.label2id,
            label_schema.field_to_entity,
            max_length=params.max_length_tokens,
            include_decoder_metadata=word_decoder is not None,
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
                word_decoder=word_decoder,
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
            word_decoder=word_decoder,
        )
        logger.info("6. Start training")
        history_df = trainer.train()
        checkpoint_dir = Path(params.checkpoint_dir).expanduser().resolve()
        best_epoch = history_df.loc[history_df["validation_f1"].idxmax(), "epoch"]
        BertPytorchCheckpoint(checkpoint_dir).save(
            model,
            tokenizer,
            label_schema,
            validation_records,
            df,
            epoch=best_epoch,
            max_length=params.max_length_tokens,
            test_records=test_records,
            decoder="ocr_word" if word_decoder is not None else "token_argmax",
        )
        if params.dataset_type == DatasetName.DOCILE:
            manifest = {
                document_id: split
                for split, partition in zip(
                    ("train", "validation", "test"),
                    (train_df, validation_df, test_df),
                    strict=True,
                )
                for document_id in partition["document_id"]
            }
            (checkpoint_dir / "split_manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
        logger.info(
            "Saved best checkpoint and validation inputs", path=str(checkpoint_dir)
        )
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
