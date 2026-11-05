import json
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from app.extraction.bert.pytorch.model import BertFieldExtractor
from app.extraction.bert.pytorch.schema import InvoiceLabelSchema


@dataclass
class BertPytorchCheckpoint:
    """Store best weights and exact evaluation inputs for offline debugging."""

    directory: Path

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
    ):
        self.directory.mkdir(parents=True, exist_ok=True)
        model.bert.config.save_pretrained(self.directory / "encoder")
        tokenizer.save_pretrained(self.directory / "tokenizer")
        torch.save(
            {key: value.detach().cpu() for key, value in model.state_dict().items()},
            self.directory / "model.pt",
        )
        self.save_records(
            "validation", records, frame, schema, epoch=epoch, max_length=max_length
        )
        if test_records is not None:
            self.save_records(
                "test", test_records, frame, schema, epoch=epoch, max_length=max_length
            )

    def save_records(self, split, records, frame, schema, *, epoch, max_length):
        """Save evaluation inputs with their original OCR word offsets."""
        if split not in {"validation", "test"}:
            raise ValueError("Split must be validation or test")
        self.directory.mkdir(parents=True, exist_ok=True)
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
        }
        (self.directory / f"{split}.json").write_text(
            json.dumps(
                payload, ensure_ascii=False, default=lambda value: value.tolist()
            ),
            encoding="utf-8",
        )
        return payload

    def load(self, device: torch.device):
        """Load local artifacts without fetching a pretrained model or dataset."""
        for name in ("model.pt", "validation.json", "encoder", "tokenizer"):
            if not (self.directory / name).exists():
                raise FileNotFoundError(
                    f"Missing checkpoint artifact: {self.directory / name}"
                )
        payload = json.loads(
            (self.directory / "validation.json").read_text(encoding="utf-8")
        )
        schema = InvoiceLabelSchema(field_to_entity=payload["field_to_entity"])
        config = AutoConfig.from_pretrained(
            self.directory / "encoder", local_files_only=True
        )
        model = BertFieldExtractor(
            model_name="",
            num_labels=schema.num_labels,
            encoder=AutoModel.from_config(config),
        )
        model.load_state_dict(
            torch.load(
                self.directory / "model.pt", map_location="cpu", weights_only=True
            )
        )
        model.to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            self.directory / "tokenizer", use_fast=True, local_files_only=True
        )
        if tokenizer is None or not tokenizer.is_fast:
            raise ValueError("Checkpoint requires a fast tokenizer")
        return model, tokenizer, schema, payload
