from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from urllib.parse import quote
from zipfile import ZipFile, is_zipfile

import httpx
from tqdm.std import tqdm

UNLABELED_CHUNK_MAX = 93

BASELINE_VARIANTS = (
    "roberta-base",
    "roberta-ours",
    "layoutlmv3-ours",
    "roberta-pretraining",
    "layoutlmv3-pretraining",
    "detr",
    "roberta-base-with-synthetic-pretraining",
    "roberta-ours-with-synthetic-pretraining",
    "layoutlmv3-ours-with-synthetic-pretraining",
)


DocileDataset = StrEnum(
    "DocileDataset",
    list(
        {
            "ANNOTATED_TRAINVAL": "annotated-trainval",
            "SYNTHETIC": "synthetic",
            "UNLABELED": "unlabeled",
            "UNLABELED_ANNOTATIONS": "unlabeled-annotations",
            "BASELINES": "baselines",
            "BASELINES_20230315": "baselines-20230315",
            # synthetic-chunk-0 through synthetic-chunk-9
            **{
                f"SYNTHETIC_CHUNK_{index}": f"synthetic-chunk-{index}"
                for index in range(10)
            },
            # unlabeled-chunk-00 through unlabeled-chunk-93
            **{
                f"UNLABELED_CHUNK_{index:02d}": f"unlabeled-chunk-{index:02d}"
                for index in range(UNLABELED_CHUNK_MAX + 1)
            },
            # unlabeled-chunk-ocr-only-00 through ...-93
            **{
                f"UNLABELED_CHUNK_OCR_ONLY_{index:02d}": (
                    f"unlabeled-chunk-ocr-only-{index:02d}"
                )
                for index in range(UNLABELED_CHUNK_MAX + 1)
            },
            # Individual archives from either baseline selection.
            **{
                f"{base}-{variant}".replace("-", "_").upper(): (f"{base}-{variant}")
                for base in ("baselines", "baselines-20230315")
                for variant in BASELINE_VARIANTS
            },
        }.items()
    ),
)


@dataclass(init=False)
class DocileDatasetDownloader:
    token: str = field(repr=False)
    dataset: DocileDataset = DocileDataset.ANNOTATED_TRAINVAL
    target_dir: Path = Path("data/docile")
    unzip: bool = False
    without_pdfs: bool = False
    minimum_archive_bytes: int = 100_000_000

    def __init__(
        self,
        token: str,
        dataset: DocileDataset | str = DocileDataset.ANNOTATED_TRAINVAL,
        target_dir: str | Path = "data/docile",
        unzip: bool = False,
        without_pdfs: bool = False,
        minimum_archive_bytes: int = 100_000_000,
    ) -> None:
        self.token = token
        self.dataset = DocileDataset(dataset)
        self.target_dir = Path(target_dir).expanduser()
        self.unzip = unzip
        self.without_pdfs = without_pdfs
        self.minimum_archive_bytes = minimum_archive_bytes
        self.__post_init__()

    def __post_init__(self):
        # Also validates callers passing a plain string.
        self.dataset = DocileDataset(self.dataset)
        self.target_dir = Path(self.target_dir).expanduser()

        if not self.token.strip():
            raise ValueError("A download token is required.")

        if self.minimum_archive_bytes < 0:
            raise ValueError("minimum_archive_bytes cannot be negative.")

        if self.without_pdfs and self.dataset != DocileDataset.UNLABELED:
            raise ValueError("without_pdfs is supported only for dataset='unlabeled'.")

    def archive_names(self) -> list[str]:
        """Expand the selected dataset into downloadable archive names."""
        if self.dataset == DocileDataset.UNLABELED:
            prefix = (
                "unlabeled-chunk-ocr-only" if self.without_pdfs else "unlabeled-chunk"
            )

            return ["unlabeled-annotations"] + [
                f"{prefix}-{index:02d}" for index in range(UNLABELED_CHUNK_MAX + 1)
            ]

        if self.dataset in {
            DocileDataset.BASELINES,
            DocileDataset.BASELINES_20230315,
        }:
            return [f"{self.dataset.value}-{variant}" for variant in BASELINE_VARIANTS]

        return [self.dataset.value]

    def download_urls(self) -> list[str]:
        """Return URLs containing the token, equivalent to --show-urls."""
        base_url = "https://docile-dataset-rossum.s3.eu-west-1.amazonaws.com"
        token = quote(self.token, safe="")

        return [f"{base_url}/{token}/{name}.zip" for name in self.archive_names()]

    def _validate_archive(self, path: Path) -> None:
        if path.stat().st_size < self.minimum_archive_bytes:
            raise ValueError(f"{path.name} is smaller than the expected minimum size.")

        if not is_zipfile(path):
            raise ValueError(f"{path.name} is not a valid ZIP archive.")

    def _download_archive(
        self,
        client: httpx.Client,
        name: str,
        url: str,
    ) -> Path:
        archive_path = self.target_dir / f"{name}.zip"

        if archive_path.is_file():
            self._validate_archive(archive_path)
            return archive_path

        temporary_path = archive_path.with_suffix(".zip.part")

        try:
            with client.stream("GET", url) as response:
                response.raise_for_status()

                content_length = response.headers.get("Content-Length")
                total_bytes = int(content_length) if content_length else None

                with (
                    temporary_path.open("wb") as output,
                    tqdm(
                        total=total_bytes,
                        desc=name,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as progress,
                ):
                    for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                        output.write(chunk)
                        progress.update(len(chunk))

            self._validate_archive(temporary_path)
            temporary_path.replace(archive_path)

        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            request_id = exc.response.headers.get("x-amz-request-id", "unknown")

            raise RuntimeError(
                f"Download failed for {name}.zip: HTTP {status}. "
                f"S3 request ID: {request_id}"
            ) from None

        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"Download timed out for {name}.zip ({type(exc).__name__})."
            ) from None

        except httpx.RequestError as exc:
            raise RuntimeError(
                f"Connection failed for {name}.zip ({type(exc).__name__})."
            ) from None

        finally:
            temporary_path.unlink(missing_ok=True)

        return archive_path

    def download(self) -> Path:
        """Download archives; optionally extract and delete each ZIP."""
        self.target_dir.mkdir(parents=True, exist_ok=True)

        with httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(120.0, connect=30.0),
        ) as client:
            for name, url in zip(
                self.archive_names(),
                self.download_urls(),
            ):
                archive_path = self._download_archive(client, name, url)

                if self.unzip:
                    with ZipFile(archive_path) as archive:
                        archive.extractall(self.target_dir)

                    archive_path.unlink()

        return self.target_dir
