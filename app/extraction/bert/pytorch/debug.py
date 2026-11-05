import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from app.aggregates.bert.params import (
    BertPytorchDebugParams,
    BertPytorchParams,
    DatasetName,
)
from app.extraction.bert.pytorch.checkpoint import BertPytorchCheckpoint
from app.extraction.bert.pytorch.decoding import OCRWordDecoder
from app.extraction.bert.pytorch.inspector import EntityErrorInspector
from app.extraction.bert.pytorch.pipeline import BertPytorchPipeline


@dataclass
class BertPytorchDebugPipeline:
    """Compare decoders on validation or held-out test inputs with fixed weights."""

    @staticmethod
    def _test_payload(params, checkpoint_dir, output_dir, schema, validation):
        path = checkpoint_dir / "test.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if any(
                payload[key] != validation[key]
                for key in ("epoch", "max_length", "field_to_entity")
            ):
                raise ValueError(
                    "Saved test metadata does not match the validation checkpoint"
                )
            validation_ids = {r.get("document_id") for r in validation["records"]}
            if validation_ids & {r.get("document_id") for r in payload["records"]}:
                raise ValueError("Test and validation document IDs overlap")
            return payload
        if not params.dataset_path:
            raise ValueError(
                "No saved test split; pass --dataset-path and the original --random-seed"
            )
        training_params = BertPytorchParams(
            dataset_path=params.dataset_path,
            result_filename="unused",
            model_name="unused",
            max_length_tokens=validation["max_length"],
            random_seed=params.random_seed,
            batch_size=1,
            num_epoch=1,
            learning_rate=0,
            weight_decay=0,
            dataset_type=DatasetName.DOCILE,
            line_items_only=all(
                path.startswith("items.") for path in schema.field_to_entity
            ),
        )
        frame = BertPytorchPipeline.load_frame(training_params)
        records, _, rebuilt_schema = BertPytorchPipeline.prepare_records(
            frame, DatasetName.DOCILE, line_items_only=training_params.line_items_only
        )
        if rebuilt_schema.field_to_entity != schema.field_to_entity:
            raise ValueError("Rebuilt label schema does not match the checkpoint")
        _, rebuilt_validation, test = BertPytorchPipeline.split_records(
            records, frame, training_params
        )
        # Compare supervised inputs, not only IDs, before trusting reconstructed membership.
        keys = ("document_id", "text", "entities", "ignored_regions")

        def signature(items):
            return json.dumps(
                [{key: record.get(key) for key in keys} for record in items],
                sort_keys=True,
                default=lambda value: value.tolist(),
            )

        if signature(rebuilt_validation) != signature(validation["records"]):
            raise ValueError(
                "Rebuilt validation split differs from checkpoint; check dataset, seed, and preparation code"
            )
        return BertPytorchCheckpoint(output_dir).save_records(
            "test",
            test,
            frame,
            schema,
            epoch=validation["epoch"],
            max_length=validation["max_length"],
        )

    def run(self, params: BertPytorchDebugParams) -> Path:
        checkpoint_dir = Path(params.checkpoint_dir).expanduser().resolve()
        output_dir = Path(params.output_dir).expanduser().resolve()
        if output_dir == checkpoint_dir or checkpoint_dir in output_dir.parents:
            raise ValueError("Debug output must be outside the checkpoint directory")
        if params.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("Device must be auto, cpu, cuda, or mps")
        if params.split not in {"validation", "test"}:
            raise ValueError("Split must be validation or test")
        device = torch.device(
            params.device
            if params.device != "auto"
            else "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        model, tokenizer, schema, payload = BertPytorchCheckpoint(checkpoint_dir).load(
            device
        )
        if params.split == "test":
            payload = self._test_payload(
                params, checkpoint_dir, output_dir, schema, payload
            )
        records = payload["records"]
        if not records or any("text_with_boxes" not in record for record in records):
            raise ValueError(
                "OCR-word comparison requires saved DocILE validation words or test words"
            )
        reports = []
        for name, decoder in (
            ("token_argmax", None),
            ("ocr_word", OCRWordDecoder(schema.label2id)),
        ):
            report, _ = EntityErrorInspector(
                model=model,
                tokenizer=tokenizer,
                label_schema=schema,
                device=device,
                max_length=payload["max_length"],
                word_decoder=decoder,
            ).run(records, output_dir / name, epoch=payload["epoch"])
            report.index.name = "entity"
            report["decoder"] = name
            report["split"] = params.split
            reports.append(report.reset_index())
        comparison = pd.concat(reports, ignore_index=True)
        result = output_dir / "decoder_comparison.csv"
        comparison.to_csv(result, index=False)
        return result
