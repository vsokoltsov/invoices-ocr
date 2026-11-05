from dataclasses import replace
from unittest.mock import Mock

import pandas as pd
import pytest

from app.aggregates.bert.params import BertPytorchParams, DatasetName
from app.extraction.bert.pytorch.pipeline import BertPytorchPipeline


def test_models_receive_the_same_fixed_dataframes_before_task_filtering(
    monkeypatch, tmp_path
):
    pipeline = BertPytorchPipeline()
    params = BertPytorchParams(
        dataset_path="unused",
        result_filename=str(tmp_path / "history.csv"),
        model_name="unused",
        max_length_tokens=32,
        random_seed=42,
        batch_size=2,
        num_epoch=1,
        learning_rate=0.001,
        weight_decay=0,
        dataset_type=DatasetName.DOCILE,
        checkpoint_dir=str(tmp_path / "models"),
        diagnostic_dir=str(tmp_path / "diagnostics"),
    )
    frame = pd.DataFrame({"document_id": [f"doc-{i}" for i in range(30)]})
    monkeypatch.setattr(pipeline, "load_frame", lambda params: frame)
    runs = []

    def run_model(resolved, train_df, validation_df, test_df):
        runs.append((resolved, (train_df, validation_df, test_df)))
        return "trained"

    monkeypatch.setattr(pipeline, "_run_model", run_model)
    split = Mock(wraps=pipeline.split_records)
    monkeypatch.setattr(pipeline, "split_records", split)
    assert pipeline.run(params) == {"body": "trained", "items": "trained"}
    split.assert_called_once()
    for first, second in zip(runs[0][1], runs[1][1], strict=True):
        assert first is second
    partitions = [set(partition.document_id) for partition in runs[0][1]]
    assert set.union(*partitions) == set(frame.document_id)
    assert not (
        partitions[0] & partitions[1]
        or partitions[0] & partitions[2]
        or partitions[1] & partitions[2]
    )
    for (resolved, _), target in zip(runs, ("body", "items"), strict=True):
        assert resolved.checkpoint_dir == str(tmp_path / "models" / target)
        assert resolved.diagnostic_dir == str(tmp_path / "diagnostics" / target)
        assert resolved.result_filename == str(tmp_path / f"history_{target}.csv")
        assert resolved.line_items_only == (target == "items")
    runs.clear()
    assert pipeline.run(replace(params, body_only=True)) == "trained"
    assert len(runs) == 1
    assert runs[0][0].result_filename == params.result_filename
    body_partitions = runs[0][1]
    runs.clear()
    pipeline.run(replace(params, line_items_only=True))
    assert len(runs) == 1
    for first, second in zip(runs[0][1], body_partitions, strict=True):
        pd.testing.assert_frame_equal(first, second)
    with pytest.raises(ValueError, match="cannot be used together"):
        pipeline.run(replace(params, body_only=True, line_items_only=True))
