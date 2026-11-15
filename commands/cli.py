import argparse
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

from app.aggregates.bert.params import (
    BertPytorchDebugParams,
    BertPytorchParams,
)
from app.aggregates.layoutlm.params import LayoutLMTrainingParams
from app.config import (
    BertPytorchConfig,
    BertPytorchDebugConfig,
    DocileInvoicesConfig,
    LayoutLMConfig,
    Settings,
    get_settings,
)
from app.datasets.docile.downloader import DocileDatasetDownloader
from app.datasets.docile.preparation import DocileInvoicesPreparation
from app.extraction.bert.debug import BertPytorchDebugPipeline
from app.pipelines.bert.pipeline import BertTrainingPipeline


def option_name(field_name: str) -> str:
    return "--" + field_name.lower().replace("_", "-")


def add_config_options(
    parser: argparse.ArgumentParser | argparse._MutuallyExclusiveGroup,
    config: BertPytorchConfig
    | BertPytorchDebugConfig
    | LayoutLMConfig
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
        diagnostic_dir=args.DIAGNOSTIC_DIR,
        checkpoint_dir=args.CHECKPOINT_DIR,
        line_items_only=args.LINE_ITEMS_ONLY,
    )
    BertTrainingPipeline(params=params).run()


def train_layoutlm(args: argparse.Namespace, settings: Settings) -> None:
    from app.pipelines.layoutlm.pipeline import LayoutLMTrainingPipeline

    config = LayoutLMConfig(
        _env_file=None,
        **{name: getattr(args, name) for name in LayoutLMConfig.model_fields},
    )
    params = LayoutLMTrainingParams(
        **{name.lower(): getattr(config, name) for name in LayoutLMConfig.model_fields}
    )
    LayoutLMTrainingPipeline(params=params).run()


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


def cache_docile_images(args: argparse.Namespace, settings: Settings) -> None:
    from app.datasets.docile.images import DocileImageCache

    path = DocileImageCache(
        dataset_path=Path(args.dataset_path),
        images_path=Path(args.images_path),
    ).run()
    print(path)


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m commands.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train a model")
    models = train.add_subparsers(dest="model", required=True)
    pytorch = models.add_parser(
        "bert-pytorch",
        help="Train BERT with a plain PyTorch training loop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pytorch.add_argument("--random-seed", type=int, default=settings.RANDOM_SEED)
    add_config_options(pytorch, settings.BERT_PYTORCH)
    pytorch.set_defaults(handler=train_bert_pytorch)

    layoutlm = models.add_parser(
        "layoutlm",
        help="Train LayoutLMv3 with page images and OCR word boxes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_config_options(layoutlm, settings.LAYOUTLM)
    layoutlm.set_defaults(handler=train_layoutlm)

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

    cache_images = dataset_commands.add_parser(
        "cache-images", help="Cache PDF page images for a downloaded dataset"
    )
    cache_datasets = cache_images.add_subparsers(dest="dataset", required=True)
    docile_images = cache_datasets.add_parser(
        "docile",
        help="Cache DocILE page images at the LayoutLM training resolution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    docile_images.add_argument(
        "--dataset-path",
        default=settings.DOCILE.OUTPUT_PATH,
        help="Prepared DocILE .parquet or .csv file containing document_id and pdf_path.",
    )
    docile_images.add_argument(
        "--images-path",
        default=settings.LAYOUTLM.IMAGES_PATH,
        help="LayoutLM image cache directory; defaults to LAYOUTLM_IMAGES_PATH.",
    )
    docile_images.set_defaults(handler=cache_docile_images)
    return parser


def main(argv: Sequence[str] | None = None):
    settings = get_settings()
    args = build_parser(settings).parse_args(argv)
    args.handler(args, settings)


if __name__ == "__main__":
    main()
