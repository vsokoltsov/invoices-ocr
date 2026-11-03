from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import httpx
from tqdm import tqdm

KAGGLE_INVOICES_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "osamahosamabdellatif/high-quality-invoice-images-for-ocr"
)


@dataclass(init=False)
class InvoiceDatasetDownloader:
    destination: Path
    archive_path: Path

    def __init__(
        self,
        destination: str | Path = "/invoices",
        archive_path: str | Path = "/invoices-ocr.zip",
    ):
        self.destination = Path(destination).expanduser()
        self.archive_path = Path(archive_path).expanduser()

    def download(self) -> Path:
        """Reuse an existing archive or download it with progress reporting."""

        if self.archive_path.is_file():
            return self.archive_path

        self.archive_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.archive_path.with_suffix(".zip.part")

        try:
            with httpx.stream(
                "GET",
                KAGGLE_INVOICES_URL,
                follow_redirects=True,
                timeout=httpx.Timeout(120.0, connect=30.0),
            ) as response:
                response.raise_for_status()

                content_length = response.headers.get("Content-Length")
                total_bytes = int(content_length) if content_length else None

                with (
                    temporary_path.open("wb") as file,
                    tqdm(
                        total=total_bytes,
                        desc="Downloading invoices",
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as progress,
                ):
                    for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                        file.write(chunk)
                        progress.update(len(chunk))

            temporary_path.replace(self.archive_path)
        finally:
            temporary_path.unlink(missing_ok=True)

        return self.archive_path

    def extract(self) -> Path:
        """Extract the downloaded ZIP, overwriting files with matching names."""
        self.destination.mkdir(parents=True, exist_ok=True)

        with ZipFile(self.archive_path) as archive:
            archive.extractall(self.destination)

        return self.destination

    def download_and_extract(self) -> Path:
        self.download()
        return self.extract()
