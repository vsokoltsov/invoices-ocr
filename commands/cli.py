import argparse
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

from app.aggregates.bert.params import (
    BertPytorchDebugParams,
    BertPytorchParams,
    BertTrainingParams,
    DatasetName,
)
from app.config import (
    BertConfig,
    BertPytorchConfig,
    BertPytorchDebugConfig,
    DocileInvoicesConfig,
    KaggleInvoicesConfig,
    Settings,
    get_settings,
)
from app.datasets.docile.downloader import DocileDatasetDownloader
from app.datasets.docile.preparation import DocileInvoicesPreparation
from app.datasets.kaggle_invoices.downloader import InvoiceDatasetDownloader
from app.datasets.kaggle_invoices.preparation import KaggleInvoicesPreparation
from app.extraction.bert.pipeline import BertTrainingPipeline
from app.extraction.bert.pytorch.debug import BertPytorchDebugPipeline
from app.extraction.bert.pytorch.pipeline import BertPytorchPipeline


def option_name(field_name: str) -> str:
    return "--" + field_name.lower().replace("_", "-")


def add_config_options(
    parser: argparse.ArgumentParser,
    config: BertConfig
    | BertPytorchConfig
    | BertPytorchDebugConfig
    | KaggleInvoicesConfig
    | DocileInvoicesConfig,
    *,
    exclude: tuple[str, ...] = (),
) -> None:
    for name, field in type(config).model_fields.items():
        if name in exclude:
            continue
        default = getattr(config, name)
        if isinstance(default, bool):
            parser.add_argument(
                option_name(name),
                dest=name,
                action=argparse.BooleanOptionalAction,
                help=field.description,
                default=default,
            )
        else:
            parser.add_argument(
                option_name(name),
                dest=name,
                type=type(default),
                choices=list(type(default)) if isinstance(default, StrEnum) else None,
                help=field.description,
                default=default,
                metavar=option_name(name).removeprefix("--"),
            )


def train_bert(
    args: argparse.Namespace,
    settings: Settings,
) -> None:
    params = BertTrainingParams(
        data_path=args.data_path,
        debug_artifacts_path=args.debug_artifacts_path,
        random_seed=args.random_seed,
        model_name=args.MODEL_NAME,
        max_length_tokens=args.MAX_LENGTH_TOKENS,
        model_window_stride=args.MODEL_WINDOW_STRIDE,
        batch_size=args.BATCH_SIZE,
        num_epoch=args.NUM_EPOCH,
        learning_rate=args.LEARNING_RATE,
        weight_decay=args.WEIGHT_DECAY,
        gradient_accumulation_steps=args.GRADIENT_ACCUMULATION_STEPS,
        per_device_train_batch_size=args.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=args.PER_DEVICE_EVAL_BATCH_SIZE,
        eval_strategy=args.EVAL_STRATEGY,
        save_strategy=args.SAVE_STRATEGY,
        logging_strategy=args.LOGGING_STRATEGY,
        logging_steps=args.LOGGING_STEPS,
        load_best_model_at_end=args.LOAD_BEST_MODEL_AT_END,
        metric_for_best_model=args.METRIC_FOR_BEST_MODEL,
        greater_is_better=args.GREATER_IS_BETTER,
        dataloader_num_workers=args.DATALOADER_NUM_WORKERS,
    )
    BertTrainingPipeline(params=params).run()


def train_bert_pytorch(args: argparse.Namespace, settings: Settings) -> None:
    params = BertPytorchParams(
        dataset_path=args.DATASET_PATH,
        result_filename=args.RESULT_FILENAME,
        random_seed=args.random_seed,
        model_name=args.MODEL_NAME,
        max_length_tokens=args.MAX_LENGTH_TOKENS,
        batch_size=args.BATCH_SIZE,
        num_epoch=args.NUM_EPOCH,
        learning_rate=args.LEARNING_RATE,
        weight_decay=args.WEIGHT_DECAY,
        dataset_type=DatasetName(args.dataset_name),
        diagnostic_dir=args.DIAGNOSTIC_DIR,
        checkpoint_dir=args.CHECKPOINT_DIR,
        line_items_only=args.LINE_ITEMS_ONLY,
    )
    BertPytorchPipeline().run(params)


def debug_bert_pytorch(args: argparse.Namespace, settings: Settings) -> None:
    path = BertPytorchDebugPipeline().run(
        BertPytorchDebugParams(
            checkpoint_dir=args.CHECKPOINT_DIR,
            output_dir=args.OUTPUT_DIR,
            device=args.DEVICE,
            split=args.SPLIT,
            dataset_path=args.DATASET_PATH,
            random_seed=args.RANDOM_SEED,
        )
    )
    print(path)


def download_kaggle_invoices(args: argparse.Namespace, settings: Settings) -> None:
    downloader = InvoiceDatasetDownloader(
        destination=args.DESTINATION_PATH,
        archive_path=args.ARCHIVE_PATH,
    )
    path = KaggleInvoicesPreparation(
        downloader=downloader,
        output_path=Path(args.OUTPUT_PATH),
    ).run()
    print(path)


def download_docile(args: argparse.Namespace, settings: Settings) -> None:
    downloader = DocileDatasetDownloader(
        token=args.TOKEN or settings.DOCILE.TOKEN or settings.DOCILE_TOKEN,
        target_dir=args.DESTINATION_PATH,
        unzip=True,
        minimum_archive_bytes=args.MINIMUM_ARCHIVE_BYTES,
    )
    path = DocileInvoicesPreparation(
        downloader=downloader,
        output_path=Path(args.OUTPUT_PATH),
        split=args.SPLIT,
        use_snapped_boxes=args.USE_SNAPPED_BOXES,
        corrections_path=args.CORRECTIONS_PATH or None,
    ).run()
    print(path)


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m commands.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train a model")
    models = train.add_subparsers(dest="model", required=True)
    bert = models.add_parser(
        "bert-transformers",
        help="Train BERT with the Transformers Trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    bert.add_argument("--data-path", default=settings.DATA_PATH)
    bert.add_argument(
        "--debug-artifacts-path",
        default=settings.DEBUG_ARTIFACTS_PATH,
    )
    bert.add_argument(
        "--random-seed",
        type=int,
        default=settings.RANDOM_SEED,
    )
    add_config_options(bert, settings.BERT)
    bert.set_defaults(handler=train_bert)

    pytorch = models.add_parser(
        "bert-pytorch",
        help="Train BERT with a plain PyTorch training loop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pytorch.add_argument("--random-seed", type=int, default=settings.RANDOM_SEED)
    pytorch.add_argument(
        "--dataset-name",
        type=DatasetName,
        choices=list(DatasetName),
        default=settings.DATASET_NAME,
        help="Select dataset-specific loading, labels, and span preparation.",
    )
    add_config_options(pytorch, settings.BERT_PYTORCH)
    pytorch.set_defaults(handler=train_bert_pytorch)

    debug = commands.add_parser("debug", help="Inspect saved model predictions")
    debug_models = debug.add_subparsers(dest="model", required=True)
    debug_pytorch = debug_models.add_parser(
        "bert-pytorch",
        help="Compare token and OCR-word decoders without training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_config_options(debug_pytorch, settings.BERT_PYTORCH_DEBUG)
    debug_pytorch.set_defaults(handler=debug_bert_pytorch)

    datasets = commands.add_parser("datasets", help="Download and prepare datasets")
    dataset_commands = datasets.add_subparsers(dest="dataset_command", required=True)
    download = dataset_commands.add_parser(
        "download", help="Download a dataset and save a prepared dataframe"
    )
    dataset_names = download.add_subparsers(dest="dataset", required=True)
    kaggle = dataset_names.add_parser(
        "kaggle-invoices",
        help="Download and prepare the Kaggle invoices dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_config_options(kaggle, settings.KAGGLE_INVOICES)
    kaggle.set_defaults(handler=download_kaggle_invoices)

    docile = dataset_names.add_parser(
        "docile",
        help="Download and prepare annotated DocILE invoices",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_config_options(docile, settings.DOCILE, exclude=("TOKEN",))
    docile.add_argument(
        "--token",
        dest="TOKEN",
        default=None,
        help="Download token; defaults to the DOCILE_TOKEN environment setting.",
    )
    docile.set_defaults(handler=download_docile)
    return parser


def main(argv: Sequence[str] | None = None):
    settings = get_settings()
    args = build_parser(settings).parse_args(argv)
    args.handler(args, settings)


if __name__ == "__main__":
    main()
