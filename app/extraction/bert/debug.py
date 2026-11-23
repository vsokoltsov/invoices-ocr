import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from app.aggregates.bert.params import BertPytorchDebugParams
from app.datasets.source import DatasetSource
from app.datasets.split import DatasetsSplit
from app.extraction.bert.artifacts import BertArtifacts
from app.extraction.bert.checkpoint import BertCheckpointLoader, BertEvaluationRecords
from app.extraction.bert.decoding import OCRWordDecoder
from app.extraction.bert.inspector import EntityErrorInspector
from app.extraction.bert.item_preprocess import ItemPreprocessor
from app.extraction.bert.preprocess import InvoicePreprocessor


@dataclass
class BertPytorchDebugPipeline:
    """Compare decoders on validation or held-out test inputs with fixed weights."""

    @staticmethod
    def _test_payload(
        params,
        checkpoint_artifacts: BertArtifacts,
        reports: BertArtifacts,
        schema,
        validation,
    ):
        if checkpoint_artifacts.has_records("test"):
            payload = checkpoint_artifacts.load_records("test")
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
        frame = (
            DatasetSource(params.dataset_path)
            .resolve("preprocessed.parquet")
            .read_parquet()
        )
        preprocessor = (
            ItemPreprocessor(schema)
            if all(path.startswith("items.") for path in schema.field_to_entity)
            else InvoicePreprocessor(schema)
        )
        records, _ = preprocessor.run(frame)
        if checkpoint_artifacts.has_manifest():
            membership = checkpoint_artifacts.load_manifest()
            by_id = {record["document_id"]: record for record in records}
            rebuilt_validation = [
                by_id[document_id]
                for document_id, split in membership.items()
                if split == "validation" and document_id in by_id
            ]
            test = [
                by_id[document_id]
                for document_id, split in membership.items()
                if split == "test" and document_id in by_id
            ]
        else:
            partitions = DatasetsSplit.create(frame, params.random_seed)
            rebuilt_validation, _ = preprocessor.run(partitions.val)
            test, _ = preprocessor.run(partitions.test)
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
        return BertEvaluationRecords(
            artifacts=reports,
            schema=schema,
            records={"test": test},
            frame=frame,
            max_length=validation["max_length"],
            decoder=validation.get("decoder", "token_argmax"),
        ).save(epoch=validation["epoch"])["test"]

    def run(self, params: BertPytorchDebugParams) -> Path:
        artifacts = BertArtifacts(Path(params.checkpoint_dir))
        reports = artifacts.debug_reports(Path(params.output_dir))
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
        model, tokenizer, schema, payload = BertCheckpointLoader(
            artifacts, device
        ).load()
        if params.split == "test":
            payload = self._test_payload(params, artifacts, reports, schema, payload)
        records = payload["records"]
        if not records or any("text_with_boxes" not in record for record in records):
            raise ValueError(
                "OCR-word comparison requires saved DocILE validation words or test words"
            )
        comparisons = []
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
            ).run(records, reports.decoder_reports(name), epoch=payload["epoch"])
            report.index.name = "entity"
            report["decoder"] = name
            report["split"] = params.split
            comparisons.append(report.reset_index())
        return reports.save_comparison(pd.concat(comparisons, ignore_index=True))
