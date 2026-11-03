import numpy as np
import random
from dataclasses import dataclass
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    set_seed
)
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from app.datasets.kaggle_invoices.downloader import InvoiceDatasetDownloader
from app.datasets.kaggle_invoices.loader import InvoiceDatasetLoader
from app.aggregates.bert.params import BertPytorchParams
from app.extraction.bert.pytorch.spans import OCRSpanMatcher, OCRSummarySpan, NumericField, Normalizer
from app.extraction.bert.pytorch.builder import InvoiceRecordBuilder
from app.extraction.bert.pytorch.schema import InvoiceLabelSchema
from app.extraction.bert.pytorch.model import BertFieldExtractor
from app.extraction.bert.pytorch.datasets import InvoiceDataset
from app.extraction.bert.pytorch.training import OCRInvoiceTraining


@dataclass
class BertPytorchPipeline:
    
    def run(self, params: BertPytorchParams):
        random.seed(params.random_seed)
        np.random.seed(params.random_seed)
        torch.manual_seed(params.random_seed)
        set_seed(params.random_seed)

        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            params.model_name,
            use_fast=True,
        )
        if tokenizer is None:
            raise RuntimeError(
                f"Could not load tokenizer for {params.model_name}"
            )
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
            numeric=numeric_field_handler
        )
        builder = InvoiceRecordBuilder(
            span_matcher=span_matcher,
            numeric=numeric_field_handler,
            normalizer=normalizer
        )
        label_schema = InvoiceLabelSchema()
        downloader = InvoiceDatasetDownloader(
            destination=params.destination_path, 
            archive_path=params.archive_part,
        )
        invoices_path = downloader.download_and_extract()
        df = InvoiceDatasetLoader(invoices_path).load()
        records, rejected = builder.build(df)
        model = BertFieldExtractor(
            model_name=params.model_name,
            num_labels=label_schema.labels
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=2e-5,
            weight_decay=0.01,
        )
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        shuffled = records.copy()

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
            train_records, tokenizer, label_schema.label2id, params.max_length_tokens
        )
        validation_dataset = InvoiceDataset(
            validation_records, tokenizer, label_schema.label2id, params.max_length_tokens
        )
        test_dataset = InvoiceDataset(
            test_records, tokenizer, label_schema.label2id, params.max_length_tokens
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
        trainer = OCRInvoiceTraining(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=validation_loader,
            device=device,
            epochs=params.num_epoch,
            id2label=label_schema.id2label
        )
        history_df = trainer.train()
        history_df.to_csv('bert_training_colab.csv')




