from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetsSplit:
    """Carry document partitions through preparation and training.

    Attributes:
        train: Training documents or their prepared records.
        val: Validation documents or their prepared records.
        test: Held-out documents or their prepared records.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    @classmethod
    def create(cls, df: pd.DataFrame, random_seed: int) -> "DatasetsSplit":
        """Split documents randomly into approximately 80/10/10 partitions.

        Args:
            df: Source documents with unique document_id values.
            random_seed: Seed used for both random split operations.

        Returns:
            Disjoint document partitions with source columns preserved.

        Raises:
            ValueError: Document IDs are duplicated or too few documents are
                available for the requested split sizes.
        """

        if df["document_id"].duplicated().any():
            raise ValueError("Document IDs must be unique before splitting")

        train_df, remaining_df = train_test_split(
            df,
            test_size=0.2,
            shuffle=True,
            random_state=random_seed,
        )

        val_df, test_df = train_test_split(
            remaining_df,
            test_size=0.5,
            shuffle=True,
            random_state=random_seed,
        )
        return cls(train=train_df, val=val_df, test=test_df)
