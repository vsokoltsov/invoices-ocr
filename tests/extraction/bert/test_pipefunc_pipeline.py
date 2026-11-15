import json
from copy import deepcopy
from dataclasses import replace
from unittest.mock import Mock

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel

from app.aggregates.bert.params import BertPytorchParams
from app.extraction.bert import model as model_module
from app.extraction.bert import publisher as publisher_module
from app.extraction.bert.artifacts import BertArtifacts
from app.extraction.bert.checkpoint import BertPytorchCheckpoint
from app.extraction.bert.collation import InvoiceDataCollator
from app.extraction.bert.datasets import InvoiceDataset
from app.pipelines.bert import pipeline as pipeline_module
from app.pipelines.bert.builder import InvoicesBuilder
from app.pipelines.bert.evaluate import EvaluateTraining
from app.pipelines.bert.pipeline import BertTrainingPipeline
from app.pipelines.bert.split import DatasetsSplit
from app.pipelines.bert.training import TrainingResult


@pytest.fixture
def params(tmp_path):
    return BertPytorchParams(
        dataset_path=str(tmp_path / "invoices.parquet"),
        result_filename=str(tmp_path / "history.csv"),
        model_name="local-tiny",
        max_length_tokens=16,
        random_seed=42,
        batch_size=4,
        num_epoch=1,
        learning_rate=0.001,
        weight_decay=0.01,
        checkpoint_dir=str(tmp_path / "checkpoint"),
        diagnostic_dir=None,
    )


def make_frame():
    annotation = {
        "value": "Acme Corporation",
        "source_values": ["Acme Corporation"],
        "pages": [0],
        "bbox": [[0.1, 0.1, 0.6, 0.2]],
    }
    rows = []
    for index in range(12):
        rows.append(
            {
                "document_id": f"doc-{index}",
                "split": "train" if index < 10 else "val",
                "ocr_text": "Acme Corporation",
                "text_with_boxes": [
                    {
                        "text": "Acme",
                        "start": 0,
                        "end": 4,
                        "page": 0,
                        "bbox": [0.1, 0.1, 0.25, 0.2],
                    },
                    {
                        "text": "Corporation",
                        "start": 5,
                        "end": 16,
                        "page": 0,
                        "bbox": [0.3, 0.1, 0.6, 0.2],
                    },
                ],
                "json_data": json.dumps(
                    {
                        "invoice": {"seller_name": annotation},
                        "items": [
                            {"line_item_id": 1, "fields": {"description": annotation}}
                        ],
                    }
                ),
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("line_items_only", [False, True])
def test_graph_trains_reloads_evaluates_and_publishes(
    params, fast_tokenizer, monkeypatch, tmp_path, line_items_only
):
    params = replace(params, line_items_only=line_items_only)
    make_frame().to_parquet(params.dataset_path, index=False)
    tokenizer_factory = pipeline_module.AutoTokenizer.from_pretrained

    def load_tokenizer(name, **kwargs):
        if name == "local-tiny":
            return fast_tokenizer
        return tokenizer_factory(name, **kwargs)

    monkeypatch.setattr(
        pipeline_module.AutoTokenizer, "from_pretrained", load_tokenizer
    )
    encoder = BertModel(
        BertConfig(
            vocab_size=len(fast_tokenizer),
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
        )
    )
    monkeypatch.setattr(
        model_module.AutoModel, "from_pretrained", lambda **kw: deepcopy(encoder)
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    models, initial = [], []

    def create_model(*args, **kwargs):
        model = model_module.BertFieldExtractor(*args, **kwargs)
        models.append(model)
        initial.append(model.classifier.weight.detach().clone())
        return model

    monkeypatch.setattr(pipeline_module, "BertFieldExtractor", create_model)
    task = Mock()
    task.upload_artifact.return_value = True
    task_init = Mock(return_value=task)
    monkeypatch.setattr(publisher_module.Task, "init", task_init)
    output_model = Mock(id="published-model")
    monkeypatch.setattr(
        publisher_module, "OutputModel", Mock(return_value=output_model)
    )
    result = BertTrainingPipeline(params).run()
    assert result == "published-model"
    directory = tmp_path / "checkpoint"
    history_path = tmp_path / "history.csv"
    history = pd.read_csv(history_path)
    assert history["epoch"].tolist() == [1]
    assert not torch.equal(initial[0], models[0].classifier.weight)
    metrics = json.loads((directory / "test_metrics.json").read_text())
    assert 0 <= metrics["f1"] <= 1
    entity = "ITEM_DESCRIPTION" if line_items_only else "SELLER_NAME"
    assert metrics["per_entity"][entity]["support"] == 2
    manifest = json.loads((directory / "split_manifest.json").read_text())
    expected_split = DatasetsSplit.create(make_frame(), params.random_seed)
    assert {key for key, name in manifest.items() if name == "test"} == set(
        expected_split.test.document_id
    )
    for split in ("train", "val", "test"):
        assert pd.read_csv(directory / f"{split}_rejected.csv").empty
    saved_model, tokenizer, schema, snapshot = BertPytorchCheckpoint(
        BertArtifacts(directory)
    ).load(torch.device("cpu"))
    assert snapshot["max_length"] == 16
    assert snapshot["decoder"] == "ocr_word"
    assert (
        schema.field_to_entity
        == json.loads((directory / "config.json").read_text())["field_to_entity"]
    )
    assert not saved_model.training
    # Evaluation must work using the saved model after the training model changes.
    with torch.no_grad():
        models[0].classifier.weight.fill_(float("nan"))
    test_records = json.loads((directory / "test.json").read_text())["records"]
    test_loader = DataLoader(
        InvoiceDataset(
            test_records,
            tokenizer,
            schema.label2id,
            schema.field_to_entity,
            max_length=16,
            include_decoder_metadata=True,
        ),
        batch_size=4,
        collate_fn=InvoiceDataCollator(tokenizer),
    )
    repeated = EvaluateTraining(
        BertArtifacts(
            tmp_path / "repeated", checkpoint_dir=directory, history_path=history_path
        ),
        torch.device("cpu"),
    ).eval(TrainingResult(directory, history_path), {"test": test_loader})
    assert json.loads(repeated.metrics_path.read_text()) == metrics
    task_init.assert_called_once()
    output_model.update_weights_package.assert_called_once()
    assert all(
        not call.kwargs["async_enable"] and not call.kwargs["auto_delete_file"]
        for call in output_model.update_weights_package.call_args_list
    )
    task.close.assert_called_once()
    task.mark_failed.assert_not_called()


def test_split_is_repeatable_disjoint_and_ignores_source_assignments():
    frame = make_frame()
    first = DatasetsSplit.create(frame, 42)
    second = DatasetsSplit.create(frame, 42)
    assert (len(first.train), len(first.val), len(first.test)) == (9, 1, 2)
    for name in ("train", "val", "test"):
        pd.testing.assert_frame_equal(getattr(first, name), getattr(second, name))
    assert set(first.train.document_id).isdisjoint(first.val.document_id)
    assert set(first.train.document_id).isdisjoint(first.test.document_id)
    random = DatasetsSplit.create(frame.drop(columns="split"), 42)
    for name in ("train", "val", "test"):
        assert (
            getattr(first, name).document_id.tolist()
            == getattr(random, name).document_id.tolist()
        )


@pytest.mark.parametrize("failure", ["empty", "duplicate", "null", "missing_columns"])
def test_bad_inputs_fail_before_loading_models(params, monkeypatch, failure):
    frame = make_frame()
    if failure == "empty":
        frame = frame.iloc[:0]
    elif failure == "duplicate":
        frame.loc[1, "document_id"] = frame.loc[0, "document_id"]
    elif failure == "null":
        frame.loc[0, "document_id"] = None
    else:
        frame = frame.drop(columns="ocr_text")
    frame.to_parquet(params.dataset_path, index=False)
    factory = Mock()
    monkeypatch.setattr(pipeline_module.AutoTokenizer, "from_pretrained", factory)
    with pytest.raises(ValueError):
        BertTrainingPipeline(params).run()
    factory.assert_not_called()


def test_builder_saves_rejections_and_refuses_empty_partitions(tmp_path):
    frame = make_frame()
    frame["document_id"] = [f"doc-{i}" for i in range(len(frame))]
    split = DatasetsSplit.create(frame, 42)
    builder = Mock()
    builder.build.return_value = (
        [],
        [{"row_index": 0, "issues": [{"status": "missing"}]}],
    )
    with pytest.raises(ValueError, match="No accepted records in train"):
        InvoicesBuilder(builder, BertArtifacts(tmp_path)).build(split)
    assert len(pd.read_csv(tmp_path / "train_rejected.csv")) == 1
