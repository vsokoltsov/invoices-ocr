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


@pytest.mark.parametrize("model", ["bert", "layoutlm"])
def test_partition_summary_counts_documents_instead_of_pages(capsys, model):
    from app.datasets.split import DatasetsSplit
    from app.pipelines.bert.preprocess import BertPreprocess
    from app.pipelines.layoutlm.preprocess import LayoutLMPreprocess

    source = pd.DataFrame({"document_id": ["good", "bad", "bad-two"]})
    accepted = [{"document_id": "good", "page": 0}, {"document_id": "good", "page": 1}]
    preprocessor, artifacts = Mock(), Mock()
    if model == "bert":
        issues = [
            {"status": "missing_span"},
            {"status": "missing_span"},
            {"status": "overlap"},
        ]
        failures = [
            {"document_id": "bad", "issues": issues},
            {"document_id": "bad-two", "issues": [{"status": "missing_span"}]},
        ]
        preprocessor.run.return_value = (accepted, failures)
        step = BertPreprocess(preprocessor, artifacts)
        reason = "missing_span"
    else:
        failures = pd.DataFrame(
            {
                "document_id": ["bad", "bad-two"],
                "reason": [
                    "annotation_without_assigned_words: SELLER_NAME (annotation_id=0); annotation_without_assigned_words: CLIENT_NAME (annotation_id=1)",
                    "annotation_without_assigned_words: INVOICE_DATE (annotation_id=2)",
                ],
            }
        )
        preprocessor.run.return_value = (accepted, failures)
        step = LayoutLMPreprocess(preprocessor, artifacts)
        reason = "annotation_without_assigned_words"
    partitions, outputs = [], []
    for suffix in ("", "-val", "-test"):
        partitions.append(source.assign(document_id=source.document_id + suffix))
        split_records = [
            {**record, "document_id": f"{record['document_id']}{suffix}"}
            for record in accepted
        ]
        if isinstance(failures, pd.DataFrame):
            split_failures = failures.assign(document_id=failures.document_id + suffix)
        else:
            split_failures = [
                {**failure, "document_id": f"{failure['document_id']}{suffix}"}
                for failure in failures
            ]
        outputs.append((split_records, split_failures))
    preprocessor.run.side_effect = outputs
    result = step.run(DatasetsSplit(*partitions))
    output = capsys.readouterr().out
    assert "train: 3 documents; accepted 1; rejected 2 (66.7%);" in output
    assert "total: 9 documents; accepted 3; rejected 6 (66.7%);" in output
    # Repeated issues and appearances in reports do not multiply document counts.
    assert f"  {reason}: 6" in output
    assert output.count("Top rejection reasons") == 1
    assert len(result.records.train) == 2
    assert artifacts.save_rejections.call_count == 3


def test_empty_partition_prints_rejection_summary_before_raising(capsys):
    from app.datasets.split import DatasetsSplit
    from app.pipelines.layoutlm.preprocess import LayoutLMPreprocess

    source = pd.DataFrame({"document_id": ["bad"]})
    failures = pd.DataFrame(
        {"document_id": ["bad"], "reason": ["Document has no OCR words"]}
    )
    preprocessor, artifacts = Mock(), Mock()
    preprocessor.run.return_value = ([], failures)
    step = LayoutLMPreprocess(preprocessor, artifacts)
    with pytest.raises(ValueError, match="No accepted pages in train"):
        step.run(DatasetsSplit(source, source, source))
    output = capsys.readouterr().out
    assert "train: 1 documents; accepted 0; rejected 1 (100.0%); pages 0" in output
    assert "Document has no OCR words: 1" in output
    artifacts.save_rejections.assert_called_once_with("train", failures)


def test_reason_summary_is_limited_to_three_categories(capsys):
    from app.pipelines.layoutlm.preprocess import LayoutLMPreprocess

    step = LayoutLMPreprocess(Mock(), Mock())
    failures = pd.DataFrame(
        {
            "document_id": ["a", "b", "c", "d"],
            "reason": ["a: details", "b: details", "c: details", "d: details"],
        }
    )
    step.print_rejection_reasons(failures)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 4
    assert lines[1:] == ["  a: 1", "  b: 1", "  c: 1"]
