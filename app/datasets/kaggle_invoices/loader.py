from dataclasses import dataclass
import pandas as pd
from pathlib import Path

@dataclass
class InvoiceDatasetLoader:
    invoices_path: Path

    def __post_init__(self):
        self.invoices_path = Path(self.invoices_path).expanduser()

    def load(self) -> pd.DataFrame:
        dfs = []
        for csv_path in sorted(self.invoices_path.rglob("*.csv")):
            image_dir = csv_path.parent / csv_path.stem

            if not image_dir.is_dir():
                print(f"Skipping: image directory not found for {csv_path}")
                continue

            df_part = pd.read_csv(csv_path)

            if "File Name" not in df_part.columns:
                print(f"Skipping: 'File Name' column not found in {csv_path}")
                continue

            # Keep provenance because folder names repeat in different batches.
            df_part["subdir"] = csv_path.stem
            df_part["source_group"] = str(csv_path.parent.relative_to(self.invoices_path))
            df_part["csv_path"] = str(csv_path)

            df_part["path"] = df_part["File Name"].map(
                lambda filename: str(image_dir / filename)
            )

            dfs.append(df_part)

        if not dfs:
            raise RuntimeError("No valid CSV files were found")

        df = pd.concat(dfs, ignore_index=True)

        # Normalize column names.
        df.columns = (
            df.columns
            .str.strip()
            .str.replace(r"(?<=[a-z0-9])(?=[A-Z])", "_", regex=True)
            .str.replace(r"[^a-zA-Z0-9]+", "_", regex=True)
            .str.strip("_")
            .str.lower()
        )
        return df
