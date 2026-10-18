from dataclasses import dataclass

import pandas as pd
from datasets import Dataset, DatasetDict

from app.extraction.bert.encoding import TokenLabelBatchEncoder


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    raw: DatasetDict
    tokenized: DatasetDict


@dataclass(slots=True)
class HuggingFaceDatasetBuilder:
    batch_encoder: TokenLabelBatchEncoder
    batch_size: int = 16
    text_column: str = "ocred_text"
    target_column: str = "json_data"

    def _from_dataframe(
        self,
        frame: pd.DataFrame,
    ) -> Dataset:
        return Dataset.from_pandas(
            frame[
                [
                    self.text_column,
                    self.target_column,
                ]
            ],
            preserve_index=False,
        )

    def build(
        self,
        *,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        test: pd.DataFrame,
    ) -> DatasetBundle:
        raw = DatasetDict({
            "train": self._from_dataframe(
                train
            ),
            "validation": self._from_dataframe(
                validation
            ),
            "test": self._from_dataframe(
                test
            ),
        })

        tokenized = raw.map(
            self.batch_encoder,
            batched=True,
            batch_size=self.batch_size,
            remove_columns=[
                self.text_column,
                self.target_column,
            ],
            load_from_cache_file=False,
        )

        return DatasetBundle(
            raw=raw,
            tokenized=tokenized,
        )