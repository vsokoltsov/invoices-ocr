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
    ClearMLPipelineConfig,
    DocileInvoicesConfig,
    LayoutLMConfig,
    Settings,
    get_settings,
)
from app.datasets.docile.downloader import DocileDataset, DocileDatasetDownloader
from app.datasets.docile.enums import DocileSplit
from app.datasets.docile.loader import DocileDatasetLoader
from app.datasets.docile.preprocess import DocileSource
from app.extraction.bert.debug import BertPytorchDebugPipeline
from app.extraction.preprocess import InvoicePreprocess
from app.pipelines.bert.pipeline import BertTrainingPipeline


def option_name(field_name: str) -> str:
    return "--" + field_name.lower().replace("_", "-")


def add_config_options(
    parser: argparse.ArgumentParser | argparse._MutuallyExclusiveGroup,
    config: BertPytorchConfig
    | BertPytorchDebugConfig
    | LayoutLMConfig
    | DocileInvoicesConfig
    | ClearMLPipelineConfig,
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


def bert_params(args: argparse.Namespace) -> BertPytorchParams:
    """Resolve fixed BERT settings from training or pipeline CLI arguments.

    Args:
        args: Parsed CLI options with model settings.

    Returns:
        Validated CLI selections as BERT parameters.
    """
    return BertPytorchParams(
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


def train_bert(args: argparse.Namespace, settings: Settings) -> None:
    BertTrainingPipeline(params=bert_params(args)).run()


def layoutlm_params(args: argparse.Namespace) -> LayoutLMTrainingParams:
    """Resolve validated LayoutLM settings for either execution backend.

    Args:
        args: Parsed model settings.

    Returns:
        LayoutLM training parameters.
    """
    config = LayoutLMConfig(
        _env_file=None,
        **{name: getattr(args, name) for name in LayoutLMConfig.model_fields},
    )
    return LayoutLMTrainingParams(
        **{name.lower(): getattr(config, name) for name in LayoutLMConfig.model_fields}
    )


def train_layoutlm(args: argparse.Namespace, settings: Settings) -> None:
    from app.pipelines.layoutlm.pipeline import LayoutLMTrainingPipeline

    LayoutLMTrainingPipeline(params=layoutlm_params(args)).run()


def run_clearml_pipeline(args: argparse.Namespace, settings: Settings) -> None:
    """Run the selected ClearML pipeline with serialized model settings.

    Args:
        args: Model and orchestration CLI options.
        settings: Loaded application settings; model values are already resolved.
    """
    from app.aggregates.pipeline import ClearMLPipelineParams
    from app.pipelines.clearml.pipeline import (
        BertClearMLPipeline,
        LayoutLMClearMLPipeline,
    )

    execution = ClearMLPipelineParams(
        project=args.PROJECT, execution=args.EXECUTION, queue=args.QUEUE
    )
    factories = {
        "bert": lambda: BertClearMLPipeline(
            bert_params(args), execution, mode=args.mode
        ),
        "layoutlm": lambda: LayoutLMClearMLPipeline(layoutlm_params(args), execution),
    }
    controller = factories[args.model]().run()
    print(controller.id)


def debug_bert(args: argparse.Namespace, settings: Settings) -> None:
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
    """Download source archives without extracting or preprocessing invoices.

    Args:
        args: Download token, destination, split, and validation settings.
        settings: Application settings used as a token fallback.
    """
    downloader = DocileDatasetDownloader(
        token=args.TOKEN or settings.DOCILE.TOKEN or settings.DOCILE_TOKEN,
        dataset=DocileDataset.TEST
        if args.SPLIT == DocileSplit.TEST
        else DocileDataset.ANNOTATED_TRAINVAL,
        target_dir=args.DESTINATION_PATH,
        minimum_archive_bytes=args.MINIMUM_ARCHIVE_BYTES,
    )
    print(downloader.download().resolve())


def load_docile(args: argparse.Namespace, settings: Settings) -> None:
    """Store original DocILE records in raw Parquet without preprocessing.

    Args:
        args: Local archive directory, raw output path, and source split.
        settings: Application settings; CLI defaults have already been resolved.
    """
    path = DocileDatasetLoader(args.source_path).save(args.OUTPUT_PATH, args.SPLIT)
    print(path)


def preprocess_model(args: argparse.Namespace, settings: Settings) -> None:
    """Preprocess a raw DocILE Parquet snapshot without accessing source JSON files.

    Args:
        args: Raw Parquet path, processed output path, and preprocessing settings.
        settings: Application settings; CLI defaults have already been resolved.
    """
    path = InvoicePreprocess(
        source=DocileSource(
            dataset_path=Path(args.source_path),
            use_snapped_boxes=args.USE_SNAPPED_BOXES,
            corrections_path=args.CORRECTIONS_PATH or None,
        ),
        output_path=Path(args.OUTPUT_PATH),
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
    bert = models.add_parser(
        "bert",
        help="Train BERT with a plain PyTorch training loop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    bert.add_argument("--random-seed", type=int, default=settings.RANDOM_SEED)
    add_config_options(bert, settings.BERT_PYTORCH)
    bert.set_defaults(handler=train_bert)

    layoutlm = models.add_parser(
        "layoutlm",
        help="Train LayoutLMv3 with page images and OCR word boxes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_config_options(layoutlm, settings.LAYOUTLM)
    layoutlm.set_defaults(handler=train_layoutlm)

    from app.aggregates.pipeline import BertRunMode

    pipeline = commands.add_parser(
        "pipeline", help="Run training as separate ClearML stage tasks"
    )
    pipeline_models = pipeline.add_subparsers(dest="model", required=True)
    bert_pipeline = pipeline_models.add_parser(
        "bert", help="Train invoice fields, line items, or both"
    )
    bert_pipeline.add_argument("--random-seed", type=int, default=settings.RANDOM_SEED)
    add_config_options(
        bert_pipeline, settings.BERT_PYTORCH, exclude=("LINE_ITEMS_ONLY",)
    )
    bert_pipeline.set_defaults(LINE_ITEMS_ONLY=False)
    bert_pipeline.add_argument(
        "--mode",
        type=BertRunMode,
        choices=list(BertRunMode),
        default=settings.CLEARML_PIPELINE.BERT_MODE,
    )
    layout_pipeline = pipeline_models.add_parser(
        "layoutlm", help="Train LayoutLM through ClearML stage tasks"
    )
    add_config_options(layout_pipeline, settings.LAYOUTLM)
    for model_parser in (bert_pipeline, layout_pipeline):
        add_config_options(
            model_parser, settings.CLEARML_PIPELINE, exclude=("BERT_MODE",)
        )
        model_parser.set_defaults(handler=run_clearml_pipeline)

    debug = commands.add_parser("debug", help="Inspect saved model predictions")
    debug_models = debug.add_subparsers(dest="model", required=True)
    debug_bert_parser = debug_models.add_parser(
        "bert",
        help="Compare token and OCR-word decoders without training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_config_options(debug_bert_parser, settings.BERT_PYTORCH_DEBUG)
    debug_bert_parser.set_defaults(handler=debug_bert)

    preprocess = commands.add_parser(
        "preprocess", help="Preprocess source documents for a model"
    )
    preprocess_models = preprocess.add_subparsers(dest="model", required=True)
    for model_name, model_config in (
        ("bert", settings.BERT_PYTORCH),
        ("layoutlm", settings.LAYOUTLM),
    ):
        model_preprocess = preprocess_models.add_parser(
            model_name,
            help=f"Preprocess local DocILE documents for {model_name}",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        add_config_options(
            model_preprocess,
            settings.DOCILE,
            exclude=(
                "TOKEN",
                "DESTINATION_PATH",
                "MINIMUM_ARCHIVE_BYTES",
                "OUTPUT_PATH",
                "SPLIT",
            ),
        )
        model_preprocess.add_argument(
            "--source-path",
            default=settings.DOCILE.OUTPUT_PATH,
            help="Raw DocILE Parquet file written by datasets load docile.",
        )
        model_preprocess.add_argument(
            "--output-path",
            dest="OUTPUT_PATH",
            default=model_config.DATASET_PATH,
            help="Exported invoice Parquet or CSV; defaults to the model dataset path.",
        )
        model_preprocess.set_defaults(handler=preprocess_model)

    datasets = commands.add_parser(
        "datasets", help="Download archives, load raw Parquet, and cache page images"
    )
    dataset_commands = datasets.add_subparsers(dest="dataset_command", required=True)
    download = dataset_commands.add_parser(
        "download", help="Download source archives without preprocessing"
    )
    dataset_names = download.add_subparsers(dest="dataset", required=True)
    docile = dataset_names.add_parser(
        "docile",
        help="Download DocILE source archives",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_config_options(
        docile,
        settings.DOCILE,
        exclude=("TOKEN", "OUTPUT_PATH", "USE_SNAPPED_BOXES", "CORRECTIONS_PATH"),
    )
    docile.add_argument(
        "--token",
        dest="TOKEN",
        default=None,
        help="Download token; defaults to the DOCILE_TOKEN environment setting.",
    )
    docile.set_defaults(handler=download_docile)

    load = dataset_commands.add_parser(
        "load", help="Load local source data into raw Parquet"
    )
    load_datasets = load.add_subparsers(dest="dataset", required=True)
    docile_load = load_datasets.add_parser(
        "docile",
        help="Extract local DocILE files and save unchanged OCR and annotations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_config_options(
        docile_load,
        settings.DOCILE,
        exclude=(
            "TOKEN",
            "DESTINATION_PATH",
            "MINIMUM_ARCHIVE_BYTES",
            "USE_SNAPPED_BOXES",
            "CORRECTIONS_PATH",
        ),
    )
    docile_load.add_argument(
        "--source-path",
        default=settings.DOCILE.DESTINATION_PATH,
        help="Directory containing downloaded ZIP archives or extracted original DocILE files.",
    )
    docile_load.set_defaults(handler=load_docile)

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
        help="Raw or preprocessed DocILE file containing document_id and pdf_path.",
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
