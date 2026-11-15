from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd
from torch import nn

from app.extraction.results import TrainingResult


class TrainingRunner(Protocol):
    """Expose training execution independently of model-specific construction."""

    def train(self) -> pd.DataFrame:
        """Run training and persist best checkpoints through configured callbacks.

        Returns:
            Completed epoch history.
        """
        ...


class TrainingArtifacts(Protocol):
    """Expose the output paths returned by a completed training step."""

    @property
    def checkpoint_location(self) -> Path:
        """Return the best-checkpoint file or directory."""
        ...

    @property
    def history_path(self) -> Path:
        """Return the epoch-history CSV path."""
        ...


@dataclass
class TrainingRun:
    """Execute a configured trainer and release its model from the accelerator.

    Attributes:
        trainer: Trainer configured with model-specific metrics and checkpoints.
        model: Model returned to the CPU after training success or failure.
        artifacts: Saved checkpoint and history locations.
    """

    trainer: TrainingRunner
    model: nn.Module
    artifacts: TrainingArtifacts

    def run(self) -> TrainingResult:
        """Run training without writing an additional checkpoint afterwards.

        Returns:
            Best-checkpoint and epoch-history locations.
        """
        try:
            self.trainer.train()
        finally:
            self.model.to("cpu")
        return TrainingResult(
            self.artifacts.checkpoint_location, self.artifacts.history_path
        )
