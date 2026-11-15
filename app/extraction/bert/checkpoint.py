from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from app.extraction.bert.artifacts import BertArtifacts
from app.extraction.bert.model import BertFieldExtractor
from app.extraction.bert.schema import InvoiceLabelSchema


@dataclass
class BertPytorchCheckpoint:
    """Store best weights and exact evaluation inputs for offline debugging."""

    artifacts: BertArtifacts

    def save(
        self,
        model,
        tokenizer,
        schema,
        records,
        frame,
        *,
        epoch,
        max_length,
        test_records=None,
        decoder="token_argmax",
    ):
        self.artifacts.save_pretrained(self.artifacts.encoder_dir, model.bert.config)
        self.artifacts.save_pretrained(self.artifacts.tokenizer_dir, tokenizer)
        self.artifacts.save_weights(
            self.artifacts.weights_path,
            {key: value.detach().cpu() for key, value in model.state_dict().items()},
        )
        self.save_records(
            "validation",
            records,
            frame,
            schema,
            epoch=epoch,
            max_length=max_length,
            decoder=decoder,
        )
        if test_records is not None:
            self.save_records(
                "test",
                test_records,
                frame,
                schema,
                epoch=epoch,
                max_length=max_length,
                decoder=decoder,
            )

    def save_records(
        self,
        split,
        records,
        frame,
        schema,
        *,
        epoch,
        max_length,
        decoder="token_argmax",
    ):
        """Save evaluation inputs with their original OCR word offsets."""
        if split not in {"validation", "test"}:
            raise ValueError("Split must be validation or test")
        words = (
            dict(zip(frame["document_id"], frame["text_with_boxes"], strict=True))
            if "text_with_boxes" in frame
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
            "max_length": int(max_length),
            "field_to_entity": schema.field_to_entity,
            "records": saved_records,
            "decoder": decoder,
        }
        self.artifacts.write_json(self.artifacts.records_path(split), payload)
        return payload

    def load(self, device: torch.device):
        """Load local artifacts without fetching a pretrained model or dataset."""
        self.artifacts.validate_checkpoint()
        payload = self.artifacts.load_records("validation")
        schema = InvoiceLabelSchema(field_to_entity=payload["field_to_entity"])
        config = AutoConfig.from_pretrained(
            self.artifacts.encoder_dir, local_files_only=True
        )
        model = BertFieldExtractor(
            model_name="",
            num_labels=schema.num_labels,
            encoder=AutoModel.from_config(config),
        )
        model.load_state_dict(self.artifacts.load_weights(self.artifacts.weights_path))
        model.to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            self.artifacts.tokenizer_dir, use_fast=True, local_files_only=True
        )
        if tokenizer is None or not tokenizer.is_fast:
            raise ValueError("Checkpoint requires a fast tokenizer")
        return model, tokenizer, schema, payload
