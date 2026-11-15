import math

import pandas as pd
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from app.extraction.layoutlm.artifacts import LayoutLMArtifacts
from app.extraction.layoutlm.builder import InvoiceBuilder
from app.extraction.layoutlm.converter import AttachmentToImageConverter
from app.extraction.layoutlm.dataset import InvoiceDataset
from app.extraction.layoutlm.metrics import EpochMetrics
from app.extraction.layoutlm.model import LayoutLMFieldExtractor
from app.extraction.layoutlm.schema import InvoiceLabelSchema
from app.extraction.layoutlm.training import OCRInvoiceTraining


def test_pdf_annotations_to_training_and_reload(
    document_row, layout_tokenizer, image_processor, layout_backbone, tmp_path
):
    torch.manual_seed(42)
    schema = InvoiceLabelSchema()
    pages, rejected = InvoiceBuilder(schema.field_to_entity, schema.label2id).build(
        pd.DataFrame([vars(document_row)])
    )
    assert rejected.empty
    dataset = InvoiceDataset(
        pages,
        layout_tokenizer,
        image_processor,
        AttachmentToImageConverter(
            LayoutLMArtifacts(output_dir=tmp_path, images_path=tmp_path / "cache")
        ),
        max_length=8,
    )
    model = LayoutLMFieldExtractor(
        "local-tiny", schema.num_labels, schema.id2label, schema.label2id
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = DataLoader(dataset, batch_size=2)
    trainer = OCRInvoiceTraining(
        model=model,
        metrics=EpochMetrics(schema.labels, list(schema.field_to_entity)),
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(ignore_index=-100),
        train_loader=loader,
        val_loader=loader,
        device=torch.device("cpu"),
        scheduler=LambdaLR(optimizer, lr_lambda=lambda _: 1.0),
        schema=schema,
        epochs=1,
        artifacts=LayoutLMArtifacts(tmp_path / "results"),
    )
    before = model.classifier.weight.detach().clone()
    history = trainer.train()
    assert math.isfinite(history.iloc[0]["validation_loss"])
    assert not torch.equal(before, model.classifier.weight)
    checkpoint = torch.load(trainer.artifacts.checkpoint_path, weights_only=True)
    assert checkpoint["max_length"] == 8
    assert checkpoint["model_name"] == "local-tiny"
    restored = LayoutLMFieldExtractor(
        checkpoint["model_name"],
        len(checkpoint["labels"]),
        schema.id2label,
        schema.label2id,
    )
    restored.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    restored.eval()
    batch = next(iter(loader))
    inputs = {
        key: batch[key]
        for key in ("input_ids", "attention_mask", "bbox", "pixel_values")
    }
    with torch.inference_mode():
        torch.testing.assert_close(model(**inputs), restored(**inputs))
