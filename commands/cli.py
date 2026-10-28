import argparse
from collections.abc import Sequence

from app.aggregates.bert import BertTrainingParams
from app.config import BertConfig, Settings, get_settings
from app.extraction.bert.pipeline import BertTrainingPipeline


def option_name(field_name: str) -> str:
    return "--" + field_name.lower().replace("_", "-")

def add_bert_options(
    parser: argparse.ArgumentParser,
    config: BertConfig,
) -> None:
    for name, field in BertConfig.model_fields.items():
        default = getattr(config, name)
        argument_options = {
            "help": field.description,
            "default": default,
            "metavar": option_name(name).removeprefix("--"),
        }
        if isinstance(default, bool):
            parser.add_argument(
                option_name(name),
                dest=name,
                action=argparse.BooleanOptionalAction,
                **argument_options,
            )
        else:
            parser.add_argument(
                option_name(name),
                dest=name,
                type=type(default),
                **argument_options,
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

def build_parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m commands.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train a model")
    models = train.add_subparsers(dest="model", required=True)
    bert = models.add_parser(
        "bert",
        help="Train the BERT token classifier",
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
    add_bert_options(bert, settings.BERT)
    bert.set_defaults(handler=train_bert)
    return parser

def main(argv: Sequence[str] | None = None):
    settings = get_settings()
    args = build_parser(settings).parse_args(argv)
    args.handler(args, settings)


if __name__ == '__main__':
    main()

