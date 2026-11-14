# Invoice extraction

See [CLI](#cli) for all commands, options, and full invocation examples.

Run the Transformers Trainer pipeline:

```sh
uv run python -m commands.cli train bert-transformers --help
```

Download and prepare the Kaggle dataset first:

```sh
uv run python -m commands.cli datasets download kaggle-invoices \
  --destination-path ./data/raw/kaggle-invoices \
  --archive-path ./data/raw/invoices-ocr.zip \
  --output-path ./data/raw/kaggle-invoices.csv
```

This command downloads or reuses the archive, extracts and combines its CSV
batches, saves the prepared dataset, and prints its absolute CSV path.
Dataset defaults use the `KAGGLE_INVOICES_` settings in `.env.sample`.

Train the plain PyTorch pipeline using that prepared CSV:

```sh
uv run python -m commands.cli train bert-pytorch \
  --dataset-name kaggle_invoices \
  --dataset-path ./data/raw/kaggle-invoices.csv \
  --result-filename ./data/interim/bert-experiment \
  --num-epoch 5
```

All fields in `BertPytorchConfig` are available as CLI options. For example,
`MAX_LENGTH_TOKENS` becomes `--max-length-tokens`. Defaults come from the
`BERTPYTORCH_` environment variables shown in `.env.sample`. `--random-seed`
uses the shared `RANDOM_SEED` setting unless overridden.

The training pipeline reads the prepared CSV and fails before model loading if
the file is missing or lacks `ocred_text` and `json_data` columns. It builds
validated span records, trains on an 80/10/10 document split, restores the
checkpoint with the best validation entity F1, and evaluates it on the test
split. It does not download or extract datasets.

Training history uses `--result-filename`, with `.csv` enforced by the pipeline.
The example writes `./data/interim/bert-experiment.csv`. Supplying a different
extension replaces it with `.csv`. Missing output directories are created.

Validation diagnostics are enabled by default in `./data/interim/bert_diagnostics`.
Use `--diagnostic-dir ./data/interim/my-run` or `BERTPYTORCH_DIAGNOSTIC_DIR` to
choose another location. Pass `--diagnostic-dir ""` to disable diagnostics and
skip inspector creation. Each `epoch_001`, `epoch_002`, etc. directory contains
per-entity metrics and false-positive/false-negative CSV files. Inspection adds
one validation inference pass per epoch.

Focused token-debug exports search validation records first, then training
records. The `record_split` column identifies the source. Training debug tokens
do not contribute to validation metrics or false-positive/false-negative tables.
The final evaluation split is not searched for development diagnostics.

Run local checks:

```sh
uv run pytest
uv run ruff check .
uv run ty check .
```

Prepare annotated DocILE data using `DOCILE_TOKEN` from the environment or `.env`:

```sh
uv run python -m commands.cli datasets download docile \
  --destination-path ./data/raw/docile \
  --output-path ./data/raw/docile.parquet

uv run python -m commands.cli train bert-pytorch \
  --dataset-name docile \
  --dataset-path ./data/raw/docile.parquet \
  --result-filename ./data/interim/docile-bert
```

The download command reuses extracted files or a cached archive and prints the
absolute output path. `--split` accepts `train`, `val`, `trainval` (default), or
`test`. Loading a split requires its manifest, OCR files, and annotations locally.
`--no-use-snapped-boxes` selects the original OCR boxes. The default export uses
Parquet; an output path ending in `.csv` produces CSV instead. Annotations use
JSON strings in both formats to preserve empty sections and variable metadata.
Parquet preserves nested OCR words; CSV stores those words as JSON strings too.

`--dataset-name` accepts `kaggle_invoices` or `docile`, defaults to `DATASET_NAME`,
and becomes `BertPytorchParams.dataset_type`. DocILE training requires
`document_id`, `ocr_text`, `text_with_boxes`, and `json_data` columns. Existing
Parquet files with nested annotation structures also work. DocILE uses boxes to
assign word labels, resolves names inside addresses, and rejects documents with
missing spans, unresolved overlaps, or invalid offsets. The selected fields are
the nine Kaggle fields plus due date, payment terms, and `AMOUNT_DUE`
(`other_fields.amount_due`); line items are excluded. `AMOUNT_DUE` remains
separate from `TOTAL` (`subtotal.total`).
Where both annotations select the same OCR word, `TOTAL` takes priority.
`AMOUNT_DUE` is supervised at distinct locations; an occurrence fully covered by
`TOTAL` is intentionally suppressed rather than treated as missing supervision.
Accepted records store `amount_due_suppressed_words`, also summed in the
record-preparation log. Other unresolved overlaps still cause rejection.

When official split metadata is present, 10% of accepted official training
documents are used for tuning; official validation documents are used only for
the final evaluation. This final evaluation is not the official hidden test set.
Without split metadata, the pipeline uses the seeded 80/10/10 document split.
Preparing only one official split is useful for inspection, but training needs
both train and validation documents. Training never downloads a dataset.

For a custom dataset with verified annotation corrections, pass a manifest when
preparing the data:

```sh
uv run python -m commands.cli datasets download docile \
  --destination-path ./data/raw/docile \
  --output-path ./data/raw/docile-corrected.parquet \
  --corrections-path ./config/docile_corrections.json
```

`DOCILE_CORRECTIONS_PATH` sets the default manifest; it is empty by default to
preserve original labels. The included manifest reclassifies two reviewed
remittance-address annotations in document `26f0e3a78e26450b9b47dd81` as
`vendor_address`. It does not change the truncated logo annotation.

Rules select an annotation by document ID, zero-based page, original type, exact
text, and optionally its exact normalized box. Each applicable rule must match
exactly one original annotation. Missing, ambiguous, or conflicting rules fail.
Only the field type is changed; original annotation files are never written.
Rules for documents outside the selected split remain unused.

Corrections are applied before field conversion. Re-running preparation rebuilds
the output even when extracted data and an output file already exist. Corrected
exports include an `annotation_corrections` column containing JSON audit entries
with selectors, replacement types, and review reasons. Existing prepared files
are not patched automatically. Train using the newly prepared path; preserve
uncorrected data for official benchmark evaluation.

## CLI

Run commands from the project root with `uv run python -m commands.cli`.
Set `DATA_PATH` and `DEBUG_ARTIFACTS_PATH` in the environment or `.env`, as shown
in `.env.sample`; settings initialization requires both values for every command.
Paths in the examples are relative to the project root.

Each example below includes every compatible command-specific option with a
representative value; mutually exclusive alternatives are listed separately.
Options are optional and override their environment or configuration
defaults. Boolean options take no value: use `--option` to enable a setting and
`--no-option` to disable it. `--help` (or `-h`) shows help without running the
command:

```sh
uv run python -m commands.cli --help
uv run python -m commands.cli datasets download docile --help
```

### Download and prepare Kaggle invoices

Download or reuse the archive, extract the dataset, combine its CSV batches,
and write a prepared CSV. The command prints the absolute output path.
Defaults use the `KAGGLE_INVOICES_` environment prefix.

```sh
uv run python -m commands.cli datasets download kaggle-invoices \
  --destination-path ./data/raw/kaggle-invoices \
  --archive-path ./data/raw/invoices-ocr.zip \
  --output-path ./data/raw/kaggle-invoices.csv
```

| Option | Meaning |
| --- | --- |
| `--destination-path` | Directory for extracted files. |
| `--archive-path` | Downloaded ZIP path; an existing archive is reused. |
| `--output-path` | Prepared CSV path. |

### Download and prepare DocILE

Download annotated documents or reuse cached data, then prepare the selected
split as Parquet or CSV. The command prints the absolute output path. Defaults
use the `DOCILE_` environment prefix. Obtain a DocILE access token first; supply
it through `DOCILE_TOKEN` in the environment or `.env`, or through `--token`.
Omit `--token` from the example to use the environment setting.

```sh
uv run python -m commands.cli datasets download docile \
  --token YOUR_DOCILE_TOKEN \
  --destination-path ./data/raw/docile \
  --output-path ./data/raw/docile.parquet \
  --split trainval \
  --use-snapped-boxes \
  --corrections-path "" \
  --minimum-archive-bytes 100000000
```

| Option | Meaning |
| --- | --- |
| `--token` | Access token; defaults to `DOCILE_TOKEN`. |
| `--destination-path` | Directory for dataset archives and extracted files. |
| `--output-path` | Prepared dataset path ending in `.parquet` or `.csv`. |
| `--split` | `train`, `val`, `trainval` (default), or `test`. |
| `--use-snapped-boxes` / `--no-use-snapped-boxes` | Use OCR boxes snapped to text when available (default), or original OCR boxes. |
| `--corrections-path` | JSON correction manifest; `""` keeps original annotations. See the correction workflow above. |
| `--minimum-archive-bytes` | Minimum accepted archive size in bytes; must be nonnegative. |

### Cache DocILE page images

After downloading and preparing DocILE, render its PDF pages into the image
cache used by LayoutLM training:

```sh
uv run python -m commands.cli datasets cache-images docile \
  --dataset-path ./data/raw/docile.parquet \
  --images-path ./data/interim/docile_images
```

| Option | Meaning |
| --- | --- |
| `--dataset-path` | Prepared `.parquet` or `.csv` containing `document_id` and `pdf_path`; defaults to `DOCILE_OUTPUT_PATH`. |
| `--images-path` | Cache directory; defaults to `LAYOUTLM_IMAGES_PATH`. |

The command renders every page, including pages without selected field labels,
at 150 DPI. Images use the existing converter's
`dpi_150/<document_id>/page_0000.png` naming scheme and are reused by training.
Existing images are loaded without rendering them again, so interrupted runs can
be restarted. The source PDFs must be available at their stored paths. A failed
document stops the command with its ID and PDF path; completed cache entries remain.
The command prints the absolute cache directory when it finishes. It does not
download data, run OCR, or load a model.

### Train BERT with plain PyTorch

Train from an existing prepared Kaggle CSV or DocILE CSV/Parquet. The command
saves training history, the best checkpoint, and optional validation diagnostics.
It does not download data. Defaults use the `BERTPYTORCH_` environment prefix,
except `--dataset-name` uses `DATASET_NAME` and `--random-seed` uses `RANDOM_SEED`.

```sh
uv run python -m commands.cli train bert-pytorch \
  --dataset-name docile \
  --dataset-path ./data/raw/docile.parquet \
  --result-filename ./data/interim/docile-bert \
  --diagnostic-dir ./data/interim/bert_diagnostics \
  --checkpoint-dir ./data/interim/bert_pytorch_checkpoint \
  --model-name google-bert/bert-base-uncased \
  --batch-size 16 \
  --num-epoch 5 \
  --max-length-tokens 512 \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --random-seed 42 \
  --no-body-only
```

| Option | Meaning |
| --- | --- |
| `--dataset-name` | `kaggle_invoices` or `docile`; selects dataset loading and label preparation. |
| `--dataset-path` | Existing prepared dataset file. |
| `--result-filename` | Training-history path; its extension is replaced with `.csv`. |
| `--diagnostic-dir` | Per-epoch diagnostic output directory; `""` disables diagnostics. |
| `--checkpoint-dir` | Directory for best weights, tokenizer, label schema, and saved evaluation inputs. |
| `--model-name` | Hugging Face checkpoint used to initialize the model. |
| `--batch-size` | Documents per dataset-encoding batch. |
| `--num-epoch` | Number of training epochs. |
| `--max-length-tokens` | Maximum tokens per document window. |
| `--learning-rate` | Optimizer learning rate. |
| `--weight-decay` | Optimizer weight-decay coefficient. |
| `--random-seed` | Seed for reproducible data splits and training. |
| `--body-only` / `--no-body-only` | Train invoice fields without the line-item model, or disable this restriction (default). |
| `--line-items-only` / `--no-line-items-only` | Train only DocILE item descriptions, quantities, gross unit prices, and gross amounts, or disable this restriction (default). Requires `--dataset-name docile` when enabled. |

Replace the example's `--no-body-only` with `--body-only`, `--line-items-only`,
or `--no-line-items-only` to select another mode. Argparse places both options,
including their `--no-` forms, in a mutually exclusive group, so pass at most one
of these four flags. Both settings default to false. To use Kaggle data,
set `--dataset-name kaggle_invoices` and its prepared CSV path.

### Train BERT with Transformers Trainer

Run the Transformers training pipeline on extracted invoice CSV batches and
their image directories. `--data-path` must contain an `invoices` directory.
Artifacts are saved under `<debug-artifacts-path>/bert-invoice-cli`.
Configuration options use the `BERT_` environment prefix; the three path/seed
options use `DATA_PATH`, `DEBUG_ARTIFACTS_PATH`, and `RANDOM_SEED`.

The parser accepts every option shown below. The current pipeline uses
`--data-path` and `--debug-artifacts-path`, but hard-codes its model, seed,
encoding settings, and Trainer settings instead of reading the corresponding
parameters. Changing those CLI options currently has no effect. In particular,
the actual training/evaluation batch sizes are 4/8 and gradient accumulation is
4. Use `train bert-pytorch` when you need configurable training.

```sh
uv run python -m commands.cli train bert-transformers \
  --data-path ./data/raw \
  --debug-artifacts-path ./data/interim \
  --random-seed 42 \
  --model-name google-bert/bert-base-uncased \
  --max-length-tokens 512 \
  --model-window-stride 64 \
  --batch-size 64 \
  --num-epoch 5 \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --gradient-accumulation-steps 1 \
  --per-device-train-batch-size 16 \
  --per-device-eval-batch-size 16 \
  --eval-strategy epoch \
  --save-strategy epoch \
  --logging-strategy steps \
  --logging-steps 20 \
  --load-best-model-at-end \
  --metric-for-best-model f1 \
  --greater-is-better \
  --dataloader-num-workers 0
```

The following meanings describe the accepted configuration parameters, subject
to the pipeline limitation above.

| Option | Meaning |
| --- | --- |
| `--data-path` | Parent of the extracted `invoices` directory. |
| `--debug-artifacts-path` | Parent directory for training artifacts. |
| `--random-seed` | Random seed. |
| `--model-name` | Initial Hugging Face checkpoint. |
| `--max-length-tokens` | Maximum tokens per document window. |
| `--model-window-stride` | Overlapping tokens between consecutive windows. |
| `--batch-size` | Documents per dataset-encoding batch. |
| `--num-epoch` | Number of training epochs. |
| `--learning-rate` | Optimizer learning rate. |
| `--weight-decay` | Optimizer weight-decay coefficient. |
| `--gradient-accumulation-steps` | Batches accumulated before an optimizer update. |
| `--per-device-train-batch-size` | Training windows per device per batch. |
| `--per-device-eval-batch-size` | Evaluation windows per device per batch. |
| `--eval-strategy` | Evaluation schedule, for example `epoch` or `steps`. |
| `--save-strategy` | Checkpoint schedule, for example `epoch` or `steps`. |
| `--logging-strategy` | Logging schedule, for example `steps` or `epoch`. |
| `--logging-steps` | Steps between logs for step-based logging. |
| `--load-best-model-at-end` / `--no-load-best-model-at-end` | Enable (default) or disable restoring the best checkpoint. |
| `--metric-for-best-model` | Metric for checkpoint selection. |
| `--greater-is-better` / `--no-greater-is-better` | Prefer a larger (default) or smaller checkpoint-selection metric. |
| `--dataloader-num-workers` | Data loader worker processes. |

### Train LayoutLMv3

The command runs `LayoutLMTrainingPipeline(params).run()` from
`app/pipelines/layoutlm/pipeline.py`. It splits documents randomly into 80/10/10
partitions, prepares page labels, builds image/text windows, trains, restores the
best validation-entity-F1 checkpoint, and evaluates the held-out test split.
It saves `history.csv`, `test_metrics.json`, `config.json`, `split_manifest.csv`,
per-split rejection CSVs, and tokenizer/image-processor settings under
`--output-dir`. The checkpoint and image cache use their own configured paths.

The random document split does not enforce template separation or preserve
DocILE's official split assignments. Use the same saved document membership
when comparing experiments. To list options:

```sh
uv run python -m commands.cli train layoutlm --help
```

All defaults use the `LAYOUTLM_` environment prefix and can be set in `.env`.
Each setting is also exposed as a CLI option:

```sh
uv run python -m commands.cli train layoutlm \
  --dataset-path ./data/raw/docile.parquet \
  --model-name microsoft/layoutlmv3-base \
  --output-dir ./data/interim/layoutlm_experiment \
  --checkpoint-path ./data/interim/layoutlm_experiment/best.pt \
  --images-path ./data/interim/docile_images \
  --num-epoch 20 \
  --batch-size 16 \
  --max-length-tokens 512 \
  --learning-rate 3e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.1 \
  --min-word-coverage 0.5 \
  --dataloader-num-workers 0 \
  --random-seed 42 \
  --device auto
```

| Option | Meaning |
| --- | --- |
| `--dataset-path` | Prepared DocILE data with OCR words, boxes, annotations, and PDF paths. |
| `--model-name` | Initial Hugging Face LayoutLMv3 checkpoint. |
| `--output-dir` | Directory for history and reports. |
| `--checkpoint-path` | Best-validation-F1 checkpoint file; independent of `--output-dir`. |
| `--images-path` | Directory for cached PDF page images. |
| `--num-epoch` | Positive number of training epochs. |
| `--batch-size` | Token windows per training or evaluation batch. |
| `--max-length-tokens` | Window length including special tokens, between 4 and 512. |
| `--learning-rate` | Positive peak AdamW learning rate. |
| `--weight-decay` | Nonnegative weight decay for the optimizer's decay parameter groups. |
| `--warmup-ratio` | Fraction of optimizer steps for warmup, from 0 inclusive to 1 exclusive. |
| `--min-word-coverage` | Minimum annotation coverage of an OCR word, greater than 0 and at most 1. |
| `--dataloader-num-workers` | Nonnegative number of data loader worker processes. |
| `--random-seed` | Seed for data splitting and training. |
| `--device` | `auto`, `cpu`, `cuda`, or `mps`. |

### Inspect a saved PyTorch checkpoint

Compare token-argmax and OCR-word decoding with fixed model weights. This command
does not train. It requires a DocILE checkpoint with OCR words in its saved
evaluation records and writes `decoder_comparison.csv` plus per-decoder reports.
Defaults use the `BERT_PYTORCH_DEBUG_` environment prefix.

```sh
uv run python -m commands.cli debug bert-pytorch \
  --checkpoint-dir ./data/interim/bert_pytorch_checkpoint \
  --output-dir ./data/interim/bert_decoder_debug \
  --device auto \
  --split validation \
  --dataset-path "" \
  --random-seed 42
```

| Option | Meaning |
| --- | --- |
| `--checkpoint-dir` | Saved model, tokenizer, label schema, and evaluation inputs. |
| `--output-dir` | Report directory; must be outside the checkpoint directory. |
| `--device` | `auto`, `cpu`, `cuda`, or `mps`. |
| `--split` | `validation` (default) or `test`. |
| `--dataset-path` | Original prepared DocILE dataset, needed to rebuild test inputs if the checkpoint has no saved test records; otherwise leave empty. |
| `--random-seed` | Original training split seed when rebuilding test inputs. |
