from dataclasses import dataclass
from typing import NamedTuple, Protocol
import pandas as pd

from app.pipelines.bert.split import DatasetsSplit

class Builder(Protocol):
    def build(self, df: pd.DataFrame, /) -> tuple[list[dict], list[dict]]: ...

class BuildResult(NamedTuple):
    records: DatasetsSplit
    rejected: DatasetsSplit

@dataclass
class InvoicesBuilder:
    builder: Builder

    def build(self, split: DatasetsSplit) -> BuildResult:
        records_dfs, rejected_dfs = [], []
        for df in [split.train, split.val, split.test]:
            records, rejected = self.builder.build(df)
            records = pd.DataFrame(records)
            rejected = pd.DataFrame(rejected)
            records_dfs.append(records)
            rejected_dfs.append(rejected)
        
        record_train, record_val, record_test = records_dfs
        rejected_train, rejected_val, rejected_test = rejected_dfs
        return BuildResult(
            records=DatasetsSplit(
                train=record_train,
                val=record_val,
                test=record_test
            ),
            rejected=DatasetsSplit(
                train=rejected_train,
                val=rejected_val,
                test=rejected_test
            )
        )

    def build_results(self, builder: Builder, split: DatasetsSplit) -> BuildResult:
        records_dfs, rejected_dfs = [], []
        for df in [split.train, split.val, split.test]:
            records, rejected = builder.build(df)
            records = pd.DataFrame(records)
            rejected = pd.DataFrame(rejected)
            records_dfs.append(records)
            rejected_dfs.append(rejected)
        
        record_train, record_val, record_test = records_dfs
        rejected_train, rejected_val, rejected_test = rejected_dfs
        return BuildResult(
            records=DatasetsSplit(
                train=record_train,
                val=record_val,
                test=record_test
            ),
            rejected=DatasetsSplit(
                train=rejected_train,
                val=rejected_val,
                test=rejected_test
            )
        )