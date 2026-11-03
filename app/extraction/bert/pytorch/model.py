from torch import Tensor, nn
from transformers import AutoModel


class BertFieldExtractor(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()

        self.bert = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name)
        self.classifier = nn.Linear(
            in_features=self.bert.config.hidden_size,
            out_features=num_labels,
        )

    # ty: ignore[invalid-mutable-override]
    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        hid = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

        return self.classifier(hid)
