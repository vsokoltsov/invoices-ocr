from dataclasses import dataclass
from pathlib import Path

from app.datasets.kaggle_invoices.downloader import InvoiceDatasetDownloader
from app.datasets.kaggle_invoices.loader import InvoiceDatasetLoader


@dataclass
class KaggleInvoicesPreparation:
    downloader: InvoiceDatasetDownloader
    output_path: Path

    def run(self) -> Path:
        """Download, extract, and combine invoice batches into one CSV.

        Return the absolute output path. Replace any supplied output extension
        with .csv and overwrite an existing CSV at that path.
        """
        dataset_directory = self.downloader.download_and_extract().resolve()
        frame = InvoiceDatasetLoader(dataset_directory).load()
        output_path = self.output_path.expanduser().with_suffix(".csv").resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
        return output_path
