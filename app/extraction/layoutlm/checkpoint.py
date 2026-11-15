from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoModel, PretrainedConfig

from app.extraction.layoutlm.artifacts import LayoutLMArtifacts
from app.extraction.layoutlm.model import LayoutLMFieldExtractor
from app.extraction.schema import InvoiceLabelSchema


@dataclass
class LayoutLMCheckpoint:
    """Write LayoutLM inference weights using fixed reconstruction metadata.

    Attributes:
        artifacts: Storage component that owns checkpoint and processor paths.
        config: Backbone configuration used for offline reconstruction.
        schema: BIO labels and field mappings associated with the weights.
        fields: Field order used by the word metrics.
        max_length: Token window length used during training.

    The checkpoint excludes optimizer and scheduler state. Tokenizer and image
    processor settings are stored in the run's processor directory by the pipeline.
    """

    artifacts: LayoutLMArtifacts
    config: PretrainedConfig
    schema: InvoiceLabelSchema
    fields: list[str]
    max_length: int

    def save(self, state: dict[str, torch.Tensor], *, epoch: int, score: float) -> None:
        """Write a selected snapshot using the configured reconstruction metadata.

        Args:
            state: Detached CPU weights supplied by the shared training loop.
            epoch: One-based epoch associated with these weights.
            score: Validation entity F1 used for checkpoint selection.
        """
        self.artifacts.save_checkpoint(
            {
                "model_state_dict": state,
                "model_name": self.config._name_or_path,
                "model_config": self.config.to_dict(),
                "max_length": self.max_length,
                "labels": list(self.schema.labels),
                "fields": list(self.fields),
                "field_to_entity": dict(self.schema.field_to_entity),
                "epoch": int(epoch),
                "val_entity_f1": float(score),
            }
        )


@dataclass
class LayoutLMCheckpointLoader:
    """Restore LayoutLM weights without requiring training configuration.

    Attributes:
        artifacts: Storage containing the checkpoint file.
        device: Device receiving the restored model.
    """

    artifacts: LayoutLMArtifacts
    device: torch.device

    def load(self) -> tuple[LayoutLMFieldExtractor, InvoiceLabelSchema, dict]:
        """Restore the saved classifier and verify its label schema.

        Returns:
            Model in evaluation mode, restored schema, and checkpoint metadata.

        Raises:
            FileNotFoundError: The checkpoint file is missing.
            ValueError: Saved labels differ from the saved field mapping.

        New checkpoints contain the backbone configuration and load offline.
        Older checkpoints without it use the saved pretrained model name, which
        can require a local Hugging Face cache or network access.
        """
        checkpoint = self.artifacts.load_checkpoint()
        schema = InvoiceLabelSchema(checkpoint["field_to_entity"])
        if tuple(checkpoint["labels"]) != schema.labels:
            raise ValueError("Checkpoint labels do not match its field schema")
        encoder = None
        # Keep existing best.pt files readable while new runs restore offline.
        if "model_config" in checkpoint:
            config_values = dict(checkpoint["model_config"])
            model_type = config_values.pop("model_type")
            config = AutoConfig.for_model(model_type, **config_values)
            encoder = AutoModel.from_config(config)
        model = LayoutLMFieldExtractor(
            model_name=checkpoint["model_name"],
            num_labels=schema.num_labels,
            id2label=schema.id2label,
            label2id=schema.label2id,
            encoder=encoder,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device).eval()
        return model, schema, checkpoint
