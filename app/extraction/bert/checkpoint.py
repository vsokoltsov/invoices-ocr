from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, PretrainedConfig

from app.extraction.bert.artifacts import BertArtifacts
from app.extraction.bert.model import BertFieldExtractor
from app.extraction.schema import InvoiceLabelSchema


@dataclass
class BertEvaluationRecords:
    """Save fixed evaluation inputs with changing checkpoint metadata.

    Attributes:
        artifacts: Storage for validation and test JSON files.
        schema: Field mapping used to label the records.
        records: Evaluation records keyed by validation or test split.
        frame: Source documents with document_id and optional text_with_boxes.
        max_length: Token window length needed to reproduce evaluation.
        decoder: Prediction decoder identifier saved with the records.
    """

    artifacts: BertArtifacts
    schema: InvoiceLabelSchema
    records: dict[str, list[dict]]
    frame: pd.DataFrame
    max_length: int
    decoder: str = "token_argmax"

    def save(self, *, epoch: int, score: float | None = None) -> dict[str, dict]:
        """Write configured splits with the selected epoch and score.

        Args:
            epoch: One-based epoch associated with the saved weights.
            score: Optional validation entity F1 for the selected checkpoint.

        Returns:
            Saved payloads keyed by split name.

        Raises:
            ValueError: A split is unsupported or OCR offsets do not match text.
        """
        return {
            split: self._save_split(split, records, epoch=epoch, score=score)
            for split, records in self.records.items()
        }

    def _save_split(self, split, records, *, epoch, score):
        """Write one split after checking original OCR character offsets.

        Args:
            split: Either validation or test.
            records: Configured labeled records for this split.
            epoch: Epoch associated with the saved weights.
            score: Optional checkpoint selection score.

        Returns:
            Payload written to the split JSON file.

        Raises:
            ValueError: The split or an OCR word offset is invalid.
        """
        if split not in {"validation", "test"}:
            raise ValueError("Split must be validation or test")
        words = (
            dict(
                zip(
                    self.frame["document_id"],
                    self.frame["text_with_boxes"],
                    strict=True,
                )
            )
            if "text_with_boxes" in self.frame
            else {}
        )
        saved_records = []
        for record in records:
            saved = dict(record)
            if words:
                saved["text_with_boxes"] = [
                    {
                        "start": int(word["start"]),
                        "end": int(word["end"]),
                        "text": word["text"],
                    }
                    for word in words[record["document_id"]]
                ]
                for word in saved["text_with_boxes"]:
                    if saved["text"][word["start"] : word["end"]] != word["text"]:
                        raise ValueError(
                            "OCR word offsets do not match validation text"
                        )
            saved_records.append(saved)
        payload = {
            "epoch": int(epoch),
            "max_length": int(self.max_length),
            "field_to_entity": self.schema.field_to_entity,
            "records": saved_records,
            "decoder": self.decoder,
        }
        if score is not None:
            payload["val_entity_f1"] = float(score)
        self.artifacts.write_json(self.artifacts.records_path(split), payload)
        return payload


@dataclass
class BertPytorchCheckpoint:
    """Write a BERT inference package using fixed run configuration.

    Attributes:
        artifacts: Checkpoint destination and filesystem operations.
        encoder_config: Backbone architecture needed for offline restoration.
        tokenizer: Fast tokenizer saved alongside model weights.
        evaluation_records: Configured schema, inputs, and decoding settings.

    This package supports inference and debugging, not optimizer-state recovery.
    Files are written separately; an interrupted write can leave a partial package.
    """

    artifacts: BertArtifacts
    encoder_config: PretrainedConfig
    tokenizer: Any
    evaluation_records: BertEvaluationRecords

    def save(self, state: dict[str, torch.Tensor], *, epoch: int, score: float) -> None:
        """Save a selected snapshot without passing fixed run inputs again.

        Args:
            state: Detached CPU model weights supplied by the training loop.
            epoch: One-based epoch that produced these weights.
            score: Validation entity F1 used for selection.
        """
        self.artifacts.save_pretrained(self.artifacts.encoder_dir, self.encoder_config)
        self.artifacts.save_pretrained(self.artifacts.tokenizer_dir, self.tokenizer)
        self.artifacts.save_weights(self.artifacts.weights_path, state)
        self.evaluation_records.save(epoch=epoch, score=score)


@dataclass
class BertCheckpointLoader:
    """Restore a BERT package without requiring training inputs.

    Attributes:
        artifacts: Storage containing the checkpoint package.
        device: Device receiving the reconstructed model.
    """

    artifacts: BertArtifacts
    device: torch.device

    def load(self):
        """Reconstruct model, tokenizer, and schema from local checkpoint files.

        Reads encoder configuration and tokenizer locally without downloading them.
        Weights are loaded with weights_only=True through the artifact store.

        Returns:
            Tuple of model in evaluation mode, fast tokenizer, label schema, and saved
            validation payload containing epoch, length, records, and decoder settings.

        Raises:
            FileNotFoundError: A required checkpoint component is absent.
            ValueError: The restored tokenizer is missing or is not fast.
        """
        self.artifacts.validate_checkpoint()
        payload = self.artifacts.load_records("validation")
        schema = InvoiceLabelSchema(field_to_entity=payload["field_to_entity"])
        # Rebuild the architecture locally; model.pt supplies the trained weights.
        config = AutoConfig.from_pretrained(
            self.artifacts.encoder_dir, local_files_only=True
        )
        model = BertFieldExtractor(
            model_name="",
            num_labels=schema.num_labels,
            encoder=AutoModel.from_config(config),
        )
        model.load_state_dict(self.artifacts.load_weights(self.artifacts.weights_path))
        model.to(self.device).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            self.artifacts.tokenizer_dir, use_fast=True, local_files_only=True
        )
        if tokenizer is None or not tokenizer.is_fast:
            raise ValueError("Checkpoint requires a fast tokenizer")
        return model, tokenizer, schema, payload
