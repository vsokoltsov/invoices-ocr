from collections import Counter
from collections.abc import Callable, Mapping

import pandas as pd
import pytest

from app.extraction.bert.dataset import DatasetBundle, HuggingFaceDatasetBuilder
from app.extraction.bert.encoding import TokenLabelBatchEncoder


@pytest.fixture
def dataset_record_factory(target_factory) -> Callable[..., dict[str, str]]:
    """Create realistic source records for Hugging Face dataset split fixtures."""

    def record(
        *,
        file_name: str,
        invoice_number: str,
        seller_name: str,
    ) -> dict[str, str]:
        target = target_factory(
            invoice={
                "invoice_number": invoice_number,
                "seller_name": seller_name,
            }
        )
        return {
            "file_name": file_name,
            "ocred_text": (
                f"Invoice no: {invoice_number} Seller: {seller_name}"
            ),
            "json_data": target.model_dump_json(),
            "source_group": "batch_1",
        }

    return record


@pytest.fixture
def dataset_frames(
    dataset_record_factory: Callable[..., dict[str, str]],
) -> dict[str, pd.DataFrame]:
    """Provide realistic split frames with source-only columns to discard."""

    return {
        "train": pd.DataFrame(
            [
                dataset_record_factory(
                    file_name="train-1.jpg",
                    invoice_number="1001",
                    seller_name="Acme",
                ),
                dataset_record_factory(
                    file_name="train-2.jpg",
                    invoice_number="1002",
                    seller_name="Corporation",
                ),
            ],
            index=pd.Index([10, 20]),
        ),
        "validation": pd.DataFrame(
            [
                dataset_record_factory(
                    file_name="validation-1.jpg",
                    invoice_number="2001",
                    seller_name="Acme",
                ),
            ],
            index=pd.Index([30]),
        ),
        "test": pd.DataFrame(
            [
                dataset_record_factory(
                    file_name="test-1.jpg",
                    invoice_number="3001",
                    seller_name="Corporation",
                ),
            ],
            index=pd.Index([40]),
        ),
    }


@pytest.fixture
def dataset_builder(
    batch_encoder: TokenLabelBatchEncoder,
) -> HuggingFaceDatasetBuilder:
    """Build datasets with the production token-label encoder and tiny batches."""

    return HuggingFaceDatasetBuilder(
        batch_encoder=batch_encoder,
        batch_size=1,
    )


@pytest.fixture
def dataset_bundle(
    dataset_builder: HuggingFaceDatasetBuilder,
    dataset_frames: dict[str, pd.DataFrame],
) -> DatasetBundle:
    """Create raw and tokenized Hugging Face datasets from all split fixtures."""

    return dataset_builder.build(
        train=dataset_frames["train"],
        validation=dataset_frames["validation"],
        test=dataset_frames["test"],
    )


def test_builder_creates_expected_raw_splits(
    dataset_bundle: DatasetBundle,
    dataset_frames: dict[str, pd.DataFrame],
) -> None:
    """Create one raw Dataset per input split without preserving DataFrame indexes."""

    assert list(dataset_bundle.raw) == ["train", "validation", "test"]

    for split_name, frame in dataset_frames.items():
        dataset = dataset_bundle.raw[split_name]

        assert len(dataset) == len(frame)
        assert dataset.column_names == ["ocred_text", "json_data"]
        assert dataset["ocred_text"] == frame["ocred_text"].tolist()
        assert dataset["json_data"] == frame["json_data"].tolist()


def test_builder_removes_source_columns_from_tokenized_splits(
    dataset_bundle: DatasetBundle,
) -> None:
    """Keep only model inputs and labels after mapping the batch encoder."""

    expected_columns = {
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "labels",
    }

    for dataset in dataset_bundle.tokenized.values():
        assert set(dataset.column_names) == expected_columns
        assert "ocred_text" not in dataset.column_names
        assert "json_data" not in dataset.column_names


def test_builder_keeps_token_inputs_and_labels_aligned(
    dataset_bundle: DatasetBundle,
) -> None:
    """Store equally sized inputs and labels for every tokenized dataset row."""

    for dataset in dataset_bundle.tokenized.values():
        for row in dataset:
            assert len(row["input_ids"]) == len(row["attention_mask"])
            assert len(row["input_ids"]) == len(row["token_type_ids"])
            assert len(row["input_ids"]) == len(row["labels"])


def test_builder_applies_real_encoder_to_every_document(
    dataset_bundle: DatasetBundle,
    dataset_frames: dict[str, pd.DataFrame],
    label2id: Mapping[str, int],
) -> None:
    """Produce invoice-number and seller-name BIO labels for every split document."""

    for split_name, frame in dataset_frames.items():
        labels = [
            label_id
            for row in dataset_bundle.tokenized[split_name]
            for label_id in row["labels"]
            if label_id != -100
        ]
        counts = Counter(labels)

        assert counts[label2id["B-INVOICE_NUMBER"]] == len(frame)
        assert counts[label2id["B-SELLER_NAME"]] == len(frame)


def test_builder_returns_more_rows_when_encoder_creates_overflow_chunks(
    batch_encoder: TokenLabelBatchEncoder,
    target_factory,
) -> None:
    """Expand a split when one long document requires several BERT windows."""

    long_text = " ".join([*["prefix"] * 40, "40108666"])
    target = target_factory(
        invoice={"invoice_number": "40108666"},
    )
    frame = pd.DataFrame(
        {
            "ocred_text": [long_text],
            "json_data": [target.model_dump_json()],
        }
    )
    builder = HuggingFaceDatasetBuilder(
        batch_encoder=TokenLabelBatchEncoder(
            tokenizer=batch_encoder.tokenizer,
            aligner=batch_encoder.aligner,
            label2id=batch_encoder.label2id,
            max_length=8,
            window_stride=2,
        ),
        batch_size=1,
    )

    bundle = builder.build(
        train=frame,
        validation=frame.iloc[0:0],
        test=frame.iloc[0:0],
    )

    assert len(bundle.raw["train"]) == 1
    assert len(bundle.tokenized["train"]) > 1
