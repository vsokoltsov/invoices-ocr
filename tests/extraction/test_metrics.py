from dataclasses import asdict

import pytest
import torch
from seqeval.scheme import IOB2

from app.extraction.bert.evaluation import BertEvaluation
from app.extraction.layoutlm.evaluation import LayoutLMEvaluation
from app.extraction.metrics import BioEntityScorer, LossMetrics, WindowMetrics
from app.extraction.schema import InvoiceLabelSchema
from app.extraction.word_metrics import ValidationMetricsResult


def test_loss_metrics_weight_supervised_tokens_and_ignore_masked_batches():
    metrics = LossMetrics()
    logits = torch.zeros(1, 3, 3)
    metrics.update(logits, torch.tensor([[1, -100, -100]]), torch.tensor(0.25))
    metrics.update(logits, torch.tensor([[1, 2, 0]]), torch.tensor(0.75))
    metrics.update(logits, torch.full((1, 3), -100), torch.tensor(float("nan")))
    assert metrics.compute().loss == pytest.approx(0.625)
    metrics.reset()
    with pytest.raises(ValueError, match="no supervised tokens"):
        metrics.compute()


def test_bio_scorers_preserve_default_and_strict_boundary_interpretations():
    actual = [["B-NAME", "I-NAME"]]
    predicted = [["I-NAME", "I-NAME"]]
    default = BioEntityScorer().score(actual, predicted)
    strict = BioEntityScorer(mode="strict", scheme=IOB2).score(actual, predicted)
    assert default.f1 == 1
    assert strict.f1 == 0
    assert default.per_entity["NAME"]["support"] == 1
    assert isinstance(default.per_entity["NAME"]["support"], int)


@pytest.mark.parametrize("predicted, entity_f1", [([1, 2, 0], 1.0), ([2, 2, 0], 0.0)])
def test_bert_and_layoutlm_use_identical_window_reports(predicted, entity_f1):
    schema = InvoiceLabelSchema({"invoice.seller_name": "NAME"})
    reports = []
    evaluators = (
        BertEvaluation(
            model=torch.nn.Identity(),
            metrics=WindowMetrics(schema.id2label),
            criterion=torch.nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
        ),
        LayoutLMEvaluation(
            model=torch.nn.Identity(),
            metrics=WindowMetrics(schema.id2label),
            criterion=torch.nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
            schema=schema,
        ),
    )
    for evaluator in evaluators:
        evaluator.reset()
        # Continuation subtokens and padding have no labels. A fully masked
        # window must not contribute its NaN loss or an empty entity sequence.
        labels = torch.tensor([[1, -100, 2, 0, -100], [-100] * 5])
        ids = torch.tensor([[predicted[0], 0, predicted[1], predicted[2], 0], [0] * 5])
        logits = torch.nn.functional.one_hot(ids, schema.num_labels).float()
        evaluator.update({}, logits, labels, torch.tensor(0.25))
        report = evaluator.compute(
            torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.empty(0)))
        )
        assert report.entity_f1 == entity_f1
        assert report.word_metrics.field_word_micro_f1 == 1.0
        assert report.word_metrics.per_field["NAME"].support == 2
        assert report.metric_contract == "ocr_word_window_strict_iob2_v1"
        assert ValidationMetricsResult.from_dict(asdict(report)) == report
        reports.append(asdict(report))
    assert reports[0] == reports[1]


def test_shared_metrics_do_not_join_entities_across_windows():
    metrics = WindowMetrics({0: "O", 1: "B-NAME", 2: "I-NAME"})
    labels = torch.tensor([[1], [1]])
    metrics.update(
        torch.nn.functional.one_hot(labels, 3).float(), labels, torch.tensor(0.1)
    )
    result = metrics.compute()
    assert result.entity_f1 == 1
    assert result.entity_report["NAME"]["support"] == 2
