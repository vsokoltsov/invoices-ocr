from dataclasses import dataclass
from unittest.mock import Mock

import pandas as pd
import pytest

from app.pipelines.preprocess import DataFramePipeline, DocumentStep


@dataclass
class AddOffset:
    offset: int = 3

    def __call__(self, frame):
        return frame.assign(value=frame.value + self.offset)


def double_values(frame):
    return frame.assign(value=frame.value * 2)


def test_pipefunc_runs_function_and_class_steps_in_order_without_config_leaks():
    source = pd.DataFrame({"value": [2, 4]})
    pipeline = DataFramePipeline((AddOffset(), double_values))
    assert pipeline.run(source).value.tolist() == [10, 14]
    assert source.value.tolist() == [2, 4]
    assert DataFramePipeline((AddOffset(10), double_values)).run(
        source
    ).value.tolist() == [24, 28]


def test_step_failure_stops_downstream_work():
    def fail(frame):
        raise RuntimeError("preprocessing failed")

    downstream = Mock(side_effect=double_values)

    def next_step(frame):
        return downstream(frame)

    with pytest.raises(RuntimeError, match="preprocessing failed"):
        DataFramePipeline((fail, next_step)).run(pd.DataFrame({"value": [1]}))
    downstream.assert_not_called()
    with pytest.raises(ValueError, match="at least one step"):
        DataFramePipeline(()).run(pd.DataFrame())


class ValidatePositive(DocumentStep):
    output_column = "validated"

    def process(self, row):
        if row.value < 0:
            raise ValueError("negative value")
        return row.value


class FinalizeValue(DocumentStep):
    output_column = "finalized"

    def process(self, row):
        assert row.validated >= 0
        return row.validated * 10


def test_document_rejection_preserves_other_rows_and_skips_later_document_steps():
    source = pd.DataFrame({"document_id": ["good", "bad"], "value": [2, -1]})
    result = DataFramePipeline((ValidatePositive(), FinalizeValue())).run(source)
    assert result.document_id.tolist() == ["good", "bad"]
    assert result.reason.tolist() == ["", "negative value"]
    assert result.loc[0, "finalized"] == 20
    assert pd.isna(result.loc[1, "finalized"])
    assert list(source.columns) == ["document_id", "value"]
