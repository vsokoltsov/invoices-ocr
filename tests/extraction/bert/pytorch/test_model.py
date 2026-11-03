import torch
from transformers import BertConfig, BertModel

from app.extraction.bert.pytorch import model as model_module


def test_model_returns_token_logits_and_backpropagates(monkeypatch):
    encoder = BertModel(
        BertConfig(
            vocab_size=20,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
        )
    )
    monkeypatch.setattr(
        model_module.AutoModel, "from_pretrained", lambda **kwargs: encoder
    )
    model = model_module.BertFieldExtractor("offline-bert", num_labels=3)
    ids = torch.tensor([[1, 2, 3]])
    logits = model(ids, torch.ones_like(ids))
    assert logits.shape == (1, 3, 3)
    torch.nn.functional.cross_entropy(
        logits.reshape(-1, 3), torch.tensor([0, 1, 2])
    ).backward()
    assert model.classifier.weight.grad is not None
    assert encoder.embeddings.word_embeddings.weight.grad is not None
