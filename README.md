# OCR

## Context

![An invoice image or PDF page passes through OCR to extract text and bounding boxes, then through field extraction to produce invoice JSON.](docs/assets/invoice-processing.png)

OCR explores how to turn invoice images and PDF pages into structured data.
An OCR engine, such as PaddleOCR, extracts text and its bounding boxes. A field
extraction model identifies values such as invoice numbers, dates, seller and
customer details, and amounts. These values can then be assembled into an
invoice JSON document.

The repository focuses on dataset preparation, training, and evaluation for
invoice field extraction with BERT and LayoutLMv3. It uses the DocILE dataset for document preparation, cached PDF page rendering,
model diagnostics, and ClearML artifact publication. Training can use OCR text and boxes already supplied by a dataset;
the diagram shows the overall document-processing workflow.

## CLI

Run commands from the project root after installing dependencies with `uv sync`
and configuring `.env` from [`.env.sample`](.env.sample).

| Command | Purpose |
| --- | --- |
| `uv run python -m commands.cli datasets download docile` | Download original DocILE ZIP archives. |
| `uv run python -m commands.cli datasets load docile` | Save original DocILE OCR and annotations to raw Parquet. |
| `uv run python -m commands.cli datasets upload docile --name NAME` | Publish raw Parquet and PDFs to ClearML. |
| `uv run python -m commands.cli datasets upload docile-preprocessed --name NAME --parent-id ID` | Publish an existing processed Parquet as a child dataset. |
| `uv run python -m commands.cli datasets upload docile-images --name NAME --parent-id ID` | Publish cached images as a child dataset. |
| `uv run python -m commands.cli datasets download clearml --dataset-id ID --destination-path PATH` | Download a dataset with its inherited files. |
| `uv run python -m commands.cli preprocess bert` | Export preprocessed DocILE documents to the BERT dataset path. |
| `uv run python -m commands.cli preprocess layoutlm` | Export preprocessed DocILE documents to the LayoutLM dataset path. |
| `uv run python -m commands.cli datasets cache-images docile` | Cache PDF page images for LayoutLM training. |
| `uv run python -m commands.cli train bert` | Train and evaluate BERT with the PyTorch training loop. |
| `uv run python -m commands.cli train layoutlm` | Train LayoutLMv3, evaluate its best checkpoint, and publish artifacts to ClearML. |
| `uv run --env-file .env python -m commands.cli pipeline bert` | Run BERT preparation, training, evaluation, and publication as separate ClearML tasks. |
| `uv run --env-file .env python -m commands.cli pipeline layoutlm` | Run LayoutLM preparation, training, evaluation, and publication as separate ClearML tasks. |
| `uv run python -m commands.cli debug bert` | Compare token and OCR-word decoding for a saved checkpoint. |

See the [full CLI reference](https://github.com/vsokoltsov/invoices-ocr/blob/main/docs/cli.md)
for every option, complete invocation examples, environment settings, and dataset
and training details.

See the [metrics guide](docs/metrics.md) for score definitions and illustrated
examples of how to interpret model results.

See [preprocessing](docs/preprocessing.md) for the shared and model-specific
PipeFunc steps and the download-to-training workflow.

## Development

Run the local checks:

```sh
uv run pytest
uv run ruff check .
uv run ty check .
```

See [ClearML datasets](docs/clearml_datasets.md) for versioned raw data, image caches,
and portable PDF paths.
