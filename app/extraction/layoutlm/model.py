from torch import nn
from transformers import AutoModel


class LayoutLMFieldExtractor(nn.Module):
    """Predict one BIO label per text token using text, boxes, and page images."""

    def __init__(self, model_name, num_labels, id2label, label2id):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )
        self.dropout = nn.Dropout(self.backbone.config.hidden_dropout_prob)
        self.classifier = nn.Linear(
            self.backbone.config.hidden_size,
            num_labels,
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        bbox,
        pixel_values,
    ):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
            return_dict=True,
        )

        # Visual tokens follow text tokens in the backbone output.
        text_hidden = outputs.last_hidden_state[:, : input_ids.shape[1]]

        return self.classifier(self.dropout(text_hidden))
