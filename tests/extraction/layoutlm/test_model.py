import torch

from app.extraction.layoutlm.model import LayoutLMFieldExtractor


def test_real_tiny_backbone_excludes_visual_tokens_and_receives_gradients(
    layout_backbone,
):
    backbone = layout_backbone
    args = {
        "model_name": "local-tiny",
        "num_labels": 3,
        "id2label": {0: "O", 1: "B-NAME", 2: "I-NAME"},
        "label2id": {"O": 0, "B-NAME": 1, "I-NAME": 2},
    }
    model = LayoutLMFieldExtractor(**args)
    model.eval()
    inputs = {
        "input_ids": torch.tensor([[0, 5, 6, 2]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "bbox": torch.tensor(
            [
                [
                    [0, 0, 0, 0],
                    [10, 10, 20, 20],
                    [20, 10, 30, 20],
                    [1000, 1000, 1000, 1000],
                ]
            ]
        ),
        "pixel_values": torch.randn(1, 3, 16, 16),
    }
    logits = model(**inputs)
    assert logits.shape == (1, 4, 3)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 3), torch.tensor([-100, 1, 2, -100])
    )
    loss.backward()
    assert model.classifier.weight.grad is not None
    assert backbone.patch_embed.proj.weight.grad is not None
    assert backbone.patch_embed.proj.weight.grad.abs().sum() > 0
