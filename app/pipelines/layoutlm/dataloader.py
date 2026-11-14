from dataclasses import dataclass

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers.image_processing_utils import BaseImageProcessor
from transformers.tokenization_utils_tokenizers import TokenizersBackend

from app.extraction.layoutlm.converter import AttachmentToImageConverter
from app.extraction.layoutlm.dataset import InvoiceDataset
from app.pipelines.layoutlm.builder import BuildResult


@dataclass
class InvoiceDataLoaders:
    tokenizer: TokenizersBackend
    image_processor: BaseImageProcessor
    converter: AttachmentToImageConverter
    device: torch.device
    batch_size: int
    max_length_tokens: int = 512
    num_workers: int = 0

    def _create_datasets(self, pages: list[pd.DataFrame]) -> list[InvoiceDataset]:
        datasets = []
        for page in pages:
            ds = InvoiceDataset(
                pages=page.to_dict(orient="records"),
                tokenizer=self.tokenizer,
                image_processor=self.image_processor,
                converter=self.converter,
                max_length=self.max_length_tokens,
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
        )
        for ld in other_datasets:
            loaders.append(
                DataLoader(
                    ld,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=self.device.type == "cuda",
                )
            )
        return [train_loader, *loaders]

    def build(self, build_result: BuildResult) -> dict[str, DataLoader]:
        train_df = build_result.records.train
        val_df = build_result.records.val
        test_df = build_result.records.test
        datasets = self._create_datasets(pages=[train_df, val_df, test_df])
        dataloaders = self._create_dataloaders(datasets=datasets)
        return {"train": dataloaders[0], "val": dataloaders[1], "test": dataloaders[2]}
