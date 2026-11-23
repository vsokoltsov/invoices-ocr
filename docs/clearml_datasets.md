# ClearML datasets

[Back to OCR](../README.md) · [CLI reference](cli.md)

Preprocessing accepts local Parquet files or explicit ClearML input references.
Local inputs do not need ClearML credentials. Remote inputs are resolved by the
source loader before the same preprocessing steps run. Storage is optional.
Upload commands store files that already exist. They do not load DocILE source
JSON or run preprocessing.

Store three related datasets:

- Raw: original OCR and annotations in Parquet, with PDFs.
- Preprocessed: the existing processed Parquet, with the raw dataset as parent.
- Images: the existing PDF page cache, with either dataset as parent.

Use the preprocessed dataset as the image dataset's parent to download all three
parts together. ClearML inherits parent files without uploading them again.
See [ClearML Data](https://clear.ml/docs/latest/docs/clearml_data/).

```mermaid
flowchart LR
    A[Local raw Parquet and PDFs] --> B[Local preprocessing]
    B --> C[Local processed Parquet]
    A --> D[Local image caching]
    D --> E[Local PNG cache]
    A -->|optional upload| R[Raw dataset]
    C -->|optional upload| P[Preprocessed dataset]
    E -->|optional upload| I[Image dataset]
    R -->|parent ID| P
    P -->|parent ID| I
```

## Prepare local files

The existing commands keep their local file interface:

```sh
uv run --env-file .env python -m commands.cli datasets load docile \
  --source-path ./data/raw/docile \
  --output-path ./data/raw/docile.parquet --split trainval

uv run python -m commands.cli preprocess bert \
  --source-path ./data/raw/docile.parquet \
  --output-path ./data/interim/docile.parquet

uv run python -m commands.cli datasets cache-images docile \
  --dataset-path ./data/raw/docile.parquet \
  --images-path ./data/interim/docile_images
```

Download DocILE archives first if they are not present. `preprocess layoutlm`
can replace `preprocess bert`; both write the same shared invoice schema.
These operations do not upload data. You can train directly from their outputs.

## Load directly from ClearML

Use `clearml://<dataset-id>/<member>` in existing input path options. Use the
exact finalized dataset ID. The member is `raw.parquet`, `preprocessed.parquet`,
or `images`, matching the upload layout. ClearML credentials must be available
on the machine running the command.

### Raw Parquet for preprocessing

```sh
uv run --env-file .env python -m commands.cli preprocess bert \
  --source-path "clearml://$RAW_ID/raw.parquet" \
  --output-path ./data/interim/docile.parquet
```

Use `preprocess layoutlm` for the same shared invoice export. The source loader
retrieves raw Parquet and PDFs through the ClearML SDK cache and resolves PDF
references in memory. The preprocessing steps still operate on a DataFrame;
they do not upload files or look up dataset IDs themselves. Output remains a
local Parquet or CSV file.

### Preprocessed Parquet for training

```sh
uv run --env-file .env python -m commands.cli train bert \
  --dataset-path "clearml://$BERT_ID/preprocessed.parquet"

uv run --env-file .env python -m commands.cli train layoutlm \
  --dataset-path "clearml://$LAYOUTLM_ID/preprocessed.parquet" \
  --images-path "clearml://$IMAGES_ID/images"
```

The image and preprocessed datasets can be siblings under the raw dataset.
Use matching dataset versions and the same document split. A combined child
that inherits both members can use the same ID for both options.

`pipeline bert`, `pipeline layoutlm`, and the BERT debug dataset input also
accept these references. Controllers retain the URI. Each worker resolves it
on its own machine. Preparation and training output directories must still be
shared when ClearML stages run on different workers.

For configuration through `.env`, existing input settings accept references:

```dotenv
BERTPYTORCH_DATASET_PATH=clearml://<bert-dataset-id>/preprocessed.parquet
LAYOUTLM_DATASET_PATH=clearml://<layoutlm-dataset-id>/preprocessed.parquet
LAYOUTLM_IMAGES_PATH=clearml://<image-dataset-id>/images
```

Keep `DOCILE_OUTPUT_PATH` as a local destination for `datasets load docile`.
Pass a raw reference with `preprocess --source-path` instead.

### Render a new cache from stored PDFs

```sh
uv run --env-file .env python -m commands.cli datasets cache-images docile \
  --dataset-path "clearml://$RAW_ID/raw.parquet" \
  --images-path ./data/interim/docile_images
```

`--images-path` is a local output directory for this command. For training,
the same option can select the existing ClearML image cache as shown above.

### Python and notebooks

Use the same resolver when working directly with a DataFrame:

```python
from app.datasets.source import DatasetSource

source = DatasetSource(
    "clearml://<processed-dataset-id>/preprocessed.parquet"
).resolve("preprocessed.parquet")
documents = source.read_parquet()

images = DatasetSource(
    "clearml://<image-dataset-id>/images"
).resolve("images").path
page_path = images / "dpi_150" / document_id / "page_0000.png"
```

Replace the IDs and set `document_id` to a document in your dataset. Load the
same ClearML credentials used by the CLI. Use `resolve("raw.parquet")` with a raw
dataset reference when original OCR and annotations are required.
These references are handled by `DatasetSource`, not by pandas directly.

### Cache and file paths

The resolver asks ClearML for the selected member and its required files:

- Parquet inputs: the selected Parquet and inherited PDFs.
- Image inputs: PNG files under `images/dpi_150/`.

ClearML downloads archive chunks, so a chunk can contain additional files even
when only selected members are requested. The SDK reuses its local cache.

Cached Parquet files remain unchanged. Stored relative PDF paths become local
absolute paths only in the loaded DataFrame. Cached images are read-only inputs;
newly rendered fallback pages go to `<LAYOUTLM_OUTPUT_DIR>/images` rather than
into the ClearML cache. Local image paths retain their existing behavior.
A missing or unreadable cached image is not repaired remotely. A missing page
can be rendered from the downloaded PDF into the local fallback cache.

An omitted member uses the context default: raw Parquet for preprocessing and
image rendering, preprocessed Parquet for training, and `images` for a training
image source. Explicit member names are clearer in scripts.

Local paths continue to work. The explicit `datasets download clearml` command
also remains available when you want a writable bundle in a chosen directory.

## Upload existing files

Configure ClearML credentials in `.env` or `clearml.conf` as for training.
Upload settings use `CLEARML_DATASET_PROJECT` (default: `ocr`) and
`CLEARML_DATASET_OUTPUT_URI` (default: empty, which uses ClearML default storage).
`--project` and `--output-uri` override them.

Upload raw Parquet and its PDFs:

```sh
uv run --env-file .env python -m commands.cli datasets upload docile \
  --dataset-path ./data/raw/docile.parquet \
  --project ocr --name docile-trainval-raw
```

The command prints the finalized raw dataset ID. Set `RAW_ID` to that ID in your
shell. SDK progress and connection logs can also appear. All referenced PDFs
must exist. The command uploads the Parquet and those PDFs; it does not upload
ZIP archives. Source Parquet, OCR, annotations, and PDF files remain unchanged.

Upload the preprocessed Parquet that you created locally:

```sh
uv run --env-file .env python -m commands.cli datasets upload docile-preprocessed \
  --dataset-path ./data/interim/docile.parquet \
  --parent-id "$RAW_ID" --project ocr --name docile-trainval-preprocessed
```

Set `PREPROCESSED_ID` to the printed ID. This child adds `preprocessed.parquet`
and inherits `raw.parquet` and `pdfs/`. It checks that the parent contains the
PDFs for its document IDs. It preserves processed fields, words, and annotations.
Only PDF references change in the uploaded copy. No preprocessing runs here.
The default input path is `BERTPYTORCH_DATASET_PATH`; use `--dataset-path` when
your processed file is elsewhere.

Upload the existing page cache:

```sh
uv run --env-file .env python -m commands.cli datasets upload docile-images \
  --parent-id "$PREPROCESSED_ID" \
  --images-path ./data/interim/docile_images \
  --project ocr --name docile-trainval-images
```

Set `IMAGES_ID` to the printed ID. This child uploads PNG files and inherits the
raw and preprocessed data. You can use `RAW_ID` as its parent if you only want
raw data and images. Use the cache rendered from that raw dataset version.
The image upload checks that every PDF page has a readable PNG and rejects
missing or extra pages. It reads the parent's files through the ClearML cache
to check page counts; this can download the parent if it is not cached.

Uploads create finalized dataset versions with dependency links. Use exact IDs
for repeatable runs. Create separate versions for the official `test` split.

## Download and use existing files

```sh
uv run --env-file .env python -m commands.cli datasets download clearml \
  --dataset-id "$IMAGES_ID" --destination-path ./data/docile-bundle

uv run --env-file .env python -m commands.cli pipeline layoutlm \
  --dataset-path ./data/docile-bundle/preprocessed.parquet \
  --images-path ./data/docile-bundle/images
```

The download retrieves the selected dataset and inherited files. The destination
must be new or empty. Existing files are not replaced. ClearML caches the data
before it copies files to the destination; allow disk space for both copies.

The downloaded preprocessed file is ready for the existing training commands.
To run a different preprocessing configuration, pass the downloaded raw file to
the same local command:

```sh
uv run python -m commands.cli preprocess bert \
  --source-path ./data/docile-bundle/raw.parquet \
  --output-path ./data/interim/docile-alternative.parquet
```

## PDF paths

The stored ClearML copies use paths relative to each Parquet file:

```text
docile-bundle/
├── raw.parquet
├── preprocessed.parquet
├── pdfs/
│   └── abc123.pdf
└── images/
    └── dpi_150/
        └── abc123/
            ├── page_0000.png
            └── page_0001.png
```

| Parquet location | Example `pdf_path` |
| --- | --- |
| Stored in ClearML | `pdfs/abc123.pdf` |
| Downloaded to `/workspace/data/docile-bundle` | `/workspace/data/docile-bundle/pdfs/abc123.pdf` |
| Downloaded to `/mnt/invoices` | `/mnt/invoices/pdfs/abc123.pdf` |

Upload converts PDF paths only in a temporary copy. Download converts them to
absolute paths only in the local destination copy. The remote dataset stays
unchanged. The Parquet file and all other values remain available through the
same local interfaces. Direct URI loading instead resolves paths in memory and
leaves the cached Parquet unchanged. No ClearML links enter preprocessing or training.

If you move an already downloaded directory, download it again at the new
location so PDF references point to the new paths.

See [the ClearML Dataset SDK](https://clear.ml/docs/latest/docs/references/sdk/dataset/)
for parent datasets, uploads, finalization, and local copies.
