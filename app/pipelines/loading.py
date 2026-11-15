from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DocileDataset:
    """Load prepared DocILE documents before model initialization.

    Attributes:
        path: Prepared Parquet dataset path.
        required_columns: Columns needed by the selected model.
    """

    path: str | Path
    required_columns: frozenset[str]

    def load(self) -> pd.DataFrame:
        """Read documents and validate inputs needed by document splitting.

        Returns:
            Source documents without changing JSON or OCR representations.

        Raises:
            ValueError: Columns are missing, document IDs are invalid, or fewer
                than six documents are available for an 80/10/10 split.
        """
        frame = pd.read_parquet(self.path)
        if missing := self.required_columns - set(frame.columns):
            raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
        if frame["document_id"].isna().any() or frame["document_id"].duplicated().any():
            raise ValueError("Document IDs must be present and unique before splitting")
        if len(frame) < 6:
            raise ValueError(
                "At least six documents are required for an 80/10/10 split"
            )
        return frame
