# CLI reference

[Back to OCR](../README.md)

Run commands from the project root with `uv run python -m commands.cli`.
Use `.env.sample` as a reference for environment settings.
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

## Download and prepare DocILE

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
| `--corrections-path` | JSON correction manifest; `""` keeps original annotations. See [Annotation corrections](#annotation-corrections). |
| `--minimum-archive-bytes` | Minimum accepted archive size in bytes; must be nonnegative. |

## Cache DocILE page images

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

## Train BERT with plain PyTorch

Train one model from prepared DocILE Parquet through the PipeFunc pipeline
in `app/pipelines/bert/pipeline.py`. The pipeline splits documents randomly into
80/10/10 partitions, prepares records, trains, reloads the best checkpoint,
evaluates the test partition, and publishes to ClearML. It saves training history,
checkpoint files, test metrics, split membership, and optional validation diagnostics.
Load ClearML credentials with `uv run --env-file .env` when using a `.env` file.
Defaults use the `BERTPYTORCH_` environment prefix, except `--random-seed` uses
`RANDOM_SEED`.

```sh
uv run python -m commands.cli train bert-pytorch \
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
  --no-line-items-only
```

| Option | Meaning |
| --- | --- |
| `--dataset-path` | Existing prepared dataset file. |
| `--result-filename` | Training-history path; its extension is replaced with `.csv`. |
| `--diagnostic-dir` | Per-epoch diagnostic output directory; `""` disables diagnostics. |
| `--checkpoint-dir` | Directory for best weights, tokenizer, label schema, and saved evaluation inputs. |
| `--model-name` | Hugging Face checkpoint used to initialize the model. |
| `--batch-size` | Token windows per training or evaluation batch. |
| `--num-epoch` | Number of training epochs. |
| `--max-length-tokens` | Maximum tokens per document window. |
| `--learning-rate` | Optimizer learning rate. |
| `--weight-decay` | Optimizer weight-decay coefficient. |
| `--random-seed` | Seed for reproducible data splits and training. |
| `--line-items-only` / `--no-line-items-only` | Train only DocILE item descriptions, quantities, gross unit prices, and gross amounts, or train invoice fields (default). |

Use `--line-items-only` to train an item model in a separate run. The default
trains invoice fields using the shared field-to-entity mapping.

## Train LayoutLMv3

The command runs `LayoutLMTrainingPipeline(params).run()` from
`app/pipelines/layoutlm/pipeline.py`. It splits documents randomly into 80/10/10
partitions, prepares page labels, builds image/text windows, trains, restores the
best validation-entity-F1 checkpoint, and evaluates the held-out test split.
Evaluation reloads the checkpoint from disk. After evaluation succeeds, the
publication step uploads the model and run artifacts to ClearML and the pipeline
returns the ClearML model ID. Configure ClearML credentials before running it.
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

## ClearML training pipelines

The `pipeline` commands create a ClearML pipeline controller with four distinct
worker tasks per model: **prepare → train → evaluate → publish**. They reuse the
same preparation, trainers, best-checkpoint selection, and scoring as the local
PipeFunc commands. Preparation saves its records beside the run outputs in `<output directory>-prepared/`,
so intermediate datasets are excluded from BERT checkpoint ZIP packages.
Workers exchange that directory and checkpoint/metric paths; live models and
PyTorch dataloaders are never passed between tasks.

The controller runs on the machine where you launch the command. By default,
ClearML runs each stage in a separate local subprocess using the current Python
environment. There is no agent requirement for this mode. Load the existing
`CLEARML_API_HOST`, `CLEARML_WEB_HOST`, `CLEARML_FILES_HOST`,
`CLEARML_API_ACCESS_KEY`, and `CLEARML_API_SECRET_KEY` environment variables with
`uv run --env-file .env`. A `clearml-init` call is unnecessary when these are set.

### BERT model selection

`--mode model` trains invoice fields, `--mode line-items` trains item fields,
and `--mode both` trains **two separate models**. Both branches use the same
source data, row order, and split seed. Their preparation can reject different
documents for their respective labels. The second branch waits for publication
of the first, so both do not compete for a single GPU.

```sh
uv run --env-file .env python -m commands.cli pipeline bert \
  --mode both \
  --project ocr \
  --execution local \
  --queue default \
  --dataset-path ./data/raw/docile.parquet \
  --result-filename ./data/interim/bert/history \
  --diagnostic-dir ./data/interim/bert/diagnostics \
  --checkpoint-dir ./data/interim/bert/checkpoints \
  --model-name google-bert/bert-base-uncased \
  --batch-size 16 \
  --num-epoch 5 \
  --max-length-tokens 512 \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --random-seed 42
```

In `both` mode, `model/` and `items/` are added below the checkpoint and diagnostic
directories. History files go into sibling `model/` and `items/` directories.
For example, `--result-filename ./runs/history` produces
`./runs/model/history.csv` and `./runs/items/history.csv`. Single-model modes use
the specified paths directly. `--diagnostic-dir ""` disables diagnostics in either
mode. The `--line-items-only` switch belongs to the local `train` command; the
ClearML command uses `--mode` instead.

### LayoutLM

Cache the DocILE images first, then run:

```sh
uv run --env-file .env python -m commands.cli pipeline layoutlm \
  --project ocr \
  --execution local \
  --queue default \
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

### Orchestration settings

| Option | Environment variable | Meaning |
| --- | --- | --- |
| `--project` | `CLEARML_PIPELINE_PROJECT` | ClearML project; default `ocr`. |
| `--execution` | `CLEARML_PIPELINE_EXECUTION` | `local` subprocesses (default), or `agent` queue execution. |
| `--queue` | `CLEARML_PIPELINE_QUEUE` | Agent execution queue; default `default`. |
| `--mode` (BERT) | `CLEARML_PIPELINE_BERT_MODE` | `model` (default), `line-items`, or `both`. |

All model hyperparameters retain their existing `BERTPYTORCH_` or `LAYOUTLM_`
environment defaults and CLI overrides. BERT's split seed uses `RANDOM_SEED`.
Each command prints its ClearML pipeline task ID after successful completion.
A failed stage stops the pipeline and causes the command to fail. Steps are not
cached across runs, so an existing checkpoint cannot silently replace new training.
Completed best checkpoints remain available if a later stage fails.

To use agents, pass `--execution agent --queue YOUR_QUEUE`. Start a ClearML agent
consuming that queue separately. Agents must run the committed project code with
its dependencies installed. **All workers must see the same absolute paths** for
prepared data, source PDFs, image caches, and output directories; use one machine
or shared storage mounted at identical paths. This implementation exchanges paths,
so it does not transfer the dataset and checkpoints between unrelated worker disks.
The controller resolves the configured input and output locations to absolute paths.
Stage environments must also have access to model downloads or a pretrained cache.

The pipeline appears in ClearML's Pipelines view. Open each publish task to find
its registered model, training history, test metrics, and configuration artifacts.
The publisher uses that task directly and leaves task completion to ClearML.
For SDK execution details, see the
[ClearML PipelineController documentation](https://clear.ml/docs/latest/docs/pipelines/pipelines_sdk_tasks/).

## Inspect a saved PyTorch checkpoint

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

## PyTorch training outputs and diagnostics

The PyTorch pipeline reads prepared DocILE documents, builds validated span
records, restores the best validation-entity-F1 checkpoint, and evaluates the test
partition. It uses the same seeded 80/10/10 document split as LayoutLM. Training
does not download or extract datasets.

Training history uses `--result-filename`, with `.csv` enforced by the pipeline.
The example writes `./data/interim/docile-bert.csv`. Supplying a different
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

## DocILE exports

The download command reuses extracted files or a cached archive and prints the
absolute output path. `--split` accepts `train`, `val`, `trainval` (default), or
`test`. Loading a split requires its manifest, OCR files, and annotations locally.
`--no-use-snapped-boxes` selects the original OCR boxes. The default export uses
Parquet; an output path ending in `.csv` produces CSV instead. Annotations use
JSON strings in both formats to preserve empty sections and variable metadata.
Parquet preserves nested OCR words; CSV stores those words as JSON strings too.

## DocILE labels and splits

BERT training requires `document_id`, `ocr_text`, `text_with_boxes`, and
`json_data` columns. It assigns word labels using annotation boxes, resolves
names inside addresses, and rejects documents with missing spans, unresolved
overlaps, or invalid offsets. Invoice-field training uses the shared
`FIELD_TO_ENTITY` mapping. Item training uses a separate item-field mapping.

Documents are split randomly into approximately 80/10/10 partitions before
record preparation. The source dataset's `split` column does not affect these
partitions. `split_manifest.json` records document membership for reproducibility.
Use the same prepared data, row order, and seed when comparing runs with LayoutLM.

## Annotation corrections

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
