import os
import random
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)

from app.aggregates.bert import BertTrainingParams
from app.extraction.bert.alignment import DocumentAligner, FinancialSummaryAligner
from app.extraction.bert.candidates import CandidatesSelector
from app.extraction.bert.dataset import HuggingFaceDatasetBuilder
from app.extraction.bert.encoding import TokenLabelBatchEncoder
from app.extraction.bert.formatter import DocumentOutputFormatter
from app.extraction.bert.inference import DocumentModelPredictor
from app.extraction.bert.metrics import TokenClassificationMetrics
from app.extraction.bert.models import InvoiceTarget
from app.extraction.bert.operations import DocumentModelTrainer


def download_and_extract_zip(
    *,
    url: str,
    archive_path: Path,
    destination: Path,
) -> None:
    archive_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    with requests.get(
        url,
        stream=True,
        timeout=(10, 300),
    ) as response:
        response.raise_for_status()

        with archive_path.open("wb") as file:
            for chunk in response.iter_content(
                chunk_size=1024 * 1024,
            ):
                if chunk:
                    file.write(chunk)

    with ZipFile(archive_path) as archive:
        archive.extractall(destination)

ENTITY_TYPES = [
    "INVOICE_NUMBER",
    "INVOICE_DATE",
    "INVOICE_DUE_DATE",
    "SELLER_NAME",
    "SELLER_ADDRESS",
    "CLIENT_NAME",
    "CLIENT_ADDRESS",
    "TAX",
    "DISCOUNT",
    "TOTAL",
    "PAYMENT_DUE_DATE",
    "BANK_NAME",
    "ACCOUNT_NUMBER",
    "PAYMENT_METHOD",
]

LABELS = ["O"]


@dataclass
class BertTrainingPipeline:
    params: BertTrainingParams
    
    def run(self):
        random_state = 42
        model_name = "google-bert/bert-base-uncased"
        invoices_path = Path(os.path.join(self.params.data_path, "invoices"))
        for entity in ENTITY_TYPES:
            LABELS.extend([
                f"B-{entity}",
                f"I-{entity}",
            ])

        label2id = {
            label: index
            for index, label in enumerate(LABELS)
        }

        id2label = {
            index: label
            for label, index in label2id.items()
        }
        output_dir = Path(
            self.params.debug_artifacts_path,
            'bert-invoice-cli'
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        SEED = 42

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        set_seed(SEED)

        def detect_device() -> torch.device:
            if torch.cuda.is_available():
                return torch.device("cuda")

            if torch.backends.mps.is_available():
                return torch.device("mps")

            return torch.device("cpu")


        device = detect_device()

        print("PyTorch:", torch.__version__)
        print("Device:", device)
        print("CUDA:", torch.cuda.is_available())
        print("MPS available:", torch.backends.mps.is_available())

        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))

        dfs = []

        print("====== 1. Prepare dataframe ======")
        for csv_path in sorted(invoices_path.rglob("*.csv")):
            image_dir = csv_path.parent / csv_path.stem

            if not image_dir.is_dir():
                print(f"Skipping: image directory not found for {csv_path}")
                continue

            df_part = pd.read_csv(csv_path)

            if "File Name" not in df_part.columns:
                print(f"Skipping: 'File Name' column not found in {csv_path}")
                continue

            # Keep provenance because folder names repeat in different batches.
            df_part["subdir"] = csv_path.stem
            df_part["source_group"] = str(csv_path.parent.relative_to(invoices_path))
            df_part["csv_path"] = str(csv_path)

            df_part["path"] = df_part["File Name"].map(
                lambda filename: str(image_dir / filename)
            )

            dfs.append(df_part)

        if not dfs:
            raise RuntimeError("No valid CSV files were found")

        df = pd.concat(dfs, ignore_index=True)

        # Normalize column names.
        df.columns = (
            df.columns
            .str.strip()
            .str.replace(r"(?<=[a-z0-9])(?=[A-Z])", "_", regex=True)
            .str.replace(r"[^a-zA-Z0-9]+", "_", regex=True)
            .str.strip("_")
            .str.lower()
        )

        df.reset_index(drop=True, inplace=True)
        print(df.shape)


        print("====== 2. Split dataframe to train / val / split ======")
        stratify_values = (
            df["subdir"]
                if (
                    "subdir" in df.columns
                    and df[
                        "subdir"
                    ].value_counts().min() >= 2
                )
                else None
        )

        train_frame, temporary_frame = (
            train_test_split(
                df,
                test_size=0.20,
                random_state=42,
                shuffle=True,
                stratify=stratify_values,
            )
        )

        temporary_stratify = (
            temporary_frame["subdir"]
            if stratify_values is not None
            else None
        )

        validation_frame, test_frame = (
            train_test_split(
                temporary_frame,
                test_size=0.50,
                random_state=42,
                shuffle=True,
                stratify=temporary_stratify,
            )
        )
        # def select_fully_aligned(
        #     frame: pd.DataFrame,
        #     report: AlignmentReport,
        # ) -> tuple[pd.DataFrame, pd.DataFrame]:
        #     accepted_indexes = set(
        #         report.documents.loc[
        #             report.documents[
        #                 "alignment_coverage"
        #             ] == 1.0,
        #             "row_index",
        #         ]
        #     )

        #     accepted = frame.loc[
        #         frame.index.isin(accepted_indexes)
        #     ].copy()

        #     rejected = frame.loc[
        #         ~frame.index.isin(accepted_indexes)
        #     ].copy()

        #     return accepted, rejected


        print("====== 4. Initialize classes ======")
        candidate_selector = CandidatesSelector()

        financial_summary_aligner = (
            FinancialSummaryAligner()
        )

        document_aligner = DocumentAligner(
            financial_summary_aligner=(
                financial_summary_aligner
            ),
            selector=candidate_selector,
        )

        print("====== 4. Initialize tokenizer ======")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
        )

        assert tokenizer is not None, "Model tokenizer is not defined"

        print("====== 5. Initialize batch encoder ======")

        batch_encoder = TokenLabelBatchEncoder(
            tokenizer=tokenizer,
            aligner=document_aligner,
            label2id=label2id,
            max_length=512,
            window_stride=64,
        )

        print("====== 6. Build dataframes ======")
        dataset_builder = (
            HuggingFaceDatasetBuilder(
                batch_encoder=batch_encoder,
                batch_size=16,
            )
        )

        dataset_bundle = dataset_builder.build(
            train=train_frame,
            validation=validation_frame,
            test=test_frame,
        )
        
        print("====== 7. Initialize models ======")
        model_metrics = (
            TokenClassificationMetrics(
                id2label=id2label
            )
        )

        model, loading_info = (
            AutoModelForTokenClassification
            .from_pretrained(
                model_name,
                num_labels=len(LABELS),
                id2label=id2label,
                label2id=label2id,
                output_loading_info=True,
                ignore_mismatched_sizes=True,
            )
        )

        print("====== 8. Initialize training arguments ======")

        training_arguments = TrainingArguments(
            output_dir=str(output_dir),

            num_train_epochs=5,
            learning_rate=2e-5,
            weight_decay=0.01,

            per_device_train_batch_size=4,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=4,
            # per_device_train_batch_size=16,
            # per_device_eval_batch_size=16,

            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=20,

            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,

            fp16=torch.cuda.is_available(),

            report_to=[],
            seed=random_state,
            data_seed=random_state,
            dataloader_num_workers=0,
        )
        print("Trainer device:", training_arguments.device)

        print("====== 9. Initialize trainer ======")
        document_model_trainer = (
            DocumentModelTrainer(
                model=model,
                tokenizer=tokenizer,
                datasets=(
                    dataset_bundle.tokenized
                ),
                training_arguments=(
                    training_arguments
                ),
                metrics=model_metrics,
            )
        )

        print("====== 10. Train model ======")
        training_result = (
            document_model_trainer.train()
        )

        training_history = pd.DataFrame(
            data=document_model_trainer
            .trainer
            .state
            .log_history
        )

        print("====== 11. Save training results ======")
        document_model_trainer.save(output_directory=str(Path(output_dir / "bert_cli")))
        training_result.to_csv(Path(output_dir / "bert_training_result.csv"))
        training_history.to_csv(Path(output_dir / "bert_training_history.csv"))
        
        print("====== 12. Validate model results ======")
        validation_metrics = (
            document_model_trainer
            .evaluate_validation()
        )

        validation_metrics_frame = pd.DataFrame([
            validation_metrics
        ])

        validation_metrics_frame.to_csv(Path(output_dir / "bert_validation_metrics.csv"))

        print("====== 13. Evaluate model results ======")
        test_metrics = (
            document_model_trainer
            .evaluate_test()
        )

        test_metrics_frame = pd.DataFrame([
            test_metrics
        ])

        test_metrics_frame.to_csv(Path(output_dir / "bert_test_metrics.csv"))

        print("====== 14. Evaluate predictions ======")
        document_predictor = (
            DocumentModelPredictor(
                model=(
                    document_model_trainer.model
                ),
                tokenizer=tokenizer,
                id2label=id2label,
                max_length=512,
                window_stride=64,
            )
        )

        document_output_formatter = (
            DocumentOutputFormatter()
        )

        sample_row = test_frame.iloc[111]

        sample_actual = (
            InvoiceTarget.model_validate_json(
                sample_row["json_data"]
            )
        )

        sample_entities = (
            document_predictor.predict(
                sample_row["ocred_text"]
            )
        )

        sample_prediction = (
            document_output_formatter.format(
                sample_entities
            )
        )

        sample_entities_frame = pd.DataFrame([
            {
                "entity": entity.entity.value,
                "text": entity.text,
                "start": entity.start,
                "end": entity.end,
                "confidence": entity.confidence,
            }
            for entity in sample_entities
        ])

        (output_dir / "sample_actual.json").write_text(
            sample_actual.model_dump_json(
                exclude_none=True,
                indent=2,
            ),
            encoding="utf-8",
        )

        (output_dir / "sample_prediction.json").write_text(
            sample_prediction.to_json(),
            encoding="utf-8",
        )

        sample_entities_frame.to_csv(
            output_dir / "sample_entities.csv",
            index=False,
        )

        print(f"====== Training is done. Artifacts are stored in {output_dir} ======")





