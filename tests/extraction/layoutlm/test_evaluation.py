import json
import math

import torch
from torch.utils.data import DataLoader

from app.extraction.layoutlm.artifacts import LayoutLMArtifacts
from app.extraction.layoutlm.builder import InvoiceBuilder
from app.extraction.layoutlm.converter import AttachmentToImageConverter
from app.extraction.layoutlm.dataset import InvoiceDataset
from app.extraction.layoutlm.model import LayoutLMFieldExtractor
from app.extraction.layoutlm.schema import InvoiceLabelSchema
from app.pipelines.layoutlm.evaluate import EvaluateTraining
from app.pipelines.layoutlm.training import TrainingResult


def test_test_step_loads_saved_weights_without_live_trainer(
    document_row, layout_tokenizer, image_processor, layout_backbone, tmp_path
):
    schema = InvoiceLabelSchema()
    pages = InvoiceBuilder(schema.field_to_entity).prepare_document(document_row)
    dataset = InvoiceDataset(
        pages,
        layout_tokenizer,
        image_processor,
        AttachmentToImageConverter(
            LayoutLMArtifacts(output_dir=tmp_path, images_path=tmp_path / "images")
        ),
        max_length=8,
    )
    model = LayoutLMFieldExtractor(
        "local-tiny", schema.num_labels, schema.id2label, schema.label2id
    )
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": "local-tiny",
            "field_to_entity": schema.field_to_entity,
            "labels": list(schema.labels),
            "fields": list(schema.field_to_entity),
            "epoch": 3,
        },
        checkpoint_path,
    )
    # The in-memory classifier cannot be used to produce valid metrics now.
    with torch.no_grad():
        model.classifier.weight.fill_(float("nan"))
    result = EvaluateTraining(
        LayoutLMArtifacts(tmp_path / "reports", checkpoint_path=checkpoint_path),
        torch.device("cpu"),
    ).eval(
        TrainingResult(checkpoint_path, tmp_path / "history.csv"),
        {"test": DataLoader(dataset, batch_size=2)},
    )
    assert result.checkpoint_path == checkpoint_path
    metrics = json.loads(result.metrics_path.read_text())
    assert math.isfinite(metrics["word_metrics"]["loss"])
    assert metrics["entity_report"]["SELLER_NAME"]["support"] == 1
