from torch import nn
from transformers import AutoModel


class LayoutLMFieldExtractor(nn.Module):
    """Predict one BIO label per text token using text, boxes, and page images."""

    def __init__(self, model_name, num_labels, id2label, label2id, encoder=None):
        """Create a token classifier from pretrained or supplied backbone weights.

        Args:
            model_name: Pretrained backbone name or path when encoder is absent.
            num_labels: Number of BIO output labels.
            id2label: Output ID to BIO label mapping.
            label2id: BIO label to output ID mapping.
            encoder: Optional backbone reconstructed from checkpoint configuration.
                Supplying it avoids fetching pretrained weights during restoration.
        """
        super().__init__()

        self.backbone = (
            encoder
            if encoder is not None
            else AutoModel.from_pretrained(
                model_name,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
            )
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
