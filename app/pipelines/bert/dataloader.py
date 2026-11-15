from dataclasses import dataclass

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers.tokenization_utils_tokenizers import TokenizersBackend

from app.extraction.bert.collation import InvoiceDataCollator
from app.extraction.bert.datasets import InvoiceDataset
from app.extraction.schema import InvoiceLabelSchema
from app.pipelines.bert.builder import BuildResult


@dataclass
class InvoiceDataLoader:
    schema: InvoiceLabelSchema
    tokenizer: TokenizersBackend
    collate_fn: InvoiceDataCollator
    device: torch.device
    batch_size: int
    max_length_tokens: int = 512
    num_workers: int = 0
    random_seed: int = 42
    include_decoder_metadata: bool = False

    def _create_datasets(self, records: list[pd.DataFrame]) -> list[InvoiceDataset]:
        datasets = []
        train_record, *other_records = records
        datasets.append(
            InvoiceDataset(
                records=train_record.to_dict(orient="records"),
                tokenizer=self.tokenizer,
                label2id=self.schema.label2id,
                field_to_entity=self.schema.field_to_entity,
                max_length=self.max_length_tokens,
            )
        )
        for record in other_records:
            ds = InvoiceDataset(
                records=record.to_dict(orient="records"),
                tokenizer=self.tokenizer,
                label2id=self.schema.label2id,
                field_to_entity=self.schema.field_to_entity,
                max_length=self.max_length_tokens,
                include_decoder_metadata=self.include_decoder_metadata,
            )
            datasets.append(ds)
        return datasets

    def _create_dataloaders(self, datasets: list[InvoiceDataset]) -> list[DataLoader]:
        loaders = []
        train_dataset, *other_datasets = datasets
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=self.collate_fn,
            generator=torch.Generator().manual_seed(self.random_seed),
        )
        for ld in other_datasets:
            loaders.append(
                DataLoader(
                    ld,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=self.device.type == "cuda",
                    collate_fn=self.collate_fn,
                )
            )
        return [train_loader, *loaders]

    def build(self, build_result: BuildResult) -> dict[str, DataLoader]:
        train_df = build_result.records.train
        val_df = build_result.records.val
        test_df = build_result.records.test
        datasets = self._create_datasets(records=[train_df, val_df, test_df])
        dataloaders = self._create_dataloaders(datasets=datasets)
        return {"train": dataloaders[0], "val": dataloaders[1], "test": dataloaders[2]}
