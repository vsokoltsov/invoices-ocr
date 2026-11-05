from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from app.aggregates.bert.params import BertPytorchDebugParams
from app.extraction.bert.pytorch.checkpoint import BertPytorchCheckpoint
from app.extraction.bert.pytorch.decoding import OCRWordDecoder
from app.extraction.bert.pytorch.inspector import EntityErrorInspector


@dataclass
class BertPytorchDebugPipeline:
    """Compare token argmax and OCR-word decoding on saved validation inputs."""

    def run(self, params: BertPytorchDebugParams) -> Path:
        checkpoint_dir = Path(params.checkpoint_dir).expanduser().resolve()
        output_dir = Path(params.output_dir).expanduser().resolve()
        if output_dir == checkpoint_dir or checkpoint_dir in output_dir.parents:
            raise ValueError("Debug output must be outside the checkpoint directory")
        if params.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("Device must be auto, cpu, cuda, or mps")
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
        records = payload["records"]
        if not records or any("text_with_boxes" not in record for record in records):
            raise ValueError(
                "OCR-word comparison requires saved DocILE validation words"
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
            reports.append(report.reset_index())
        comparison = pd.concat(reports, ignore_index=True)
        result = output_dir / "decoder_comparison.csv"
        comparison.to_csv(result, index=False)
        return result
