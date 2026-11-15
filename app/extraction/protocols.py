from typing import Protocol, TypeVar

import pandas as pd
import torch
from torch.utils.data import DataLoader


class LossResult(Protocol):
    """Expose the loss consumed by the shared epoch loop."""

    @property
    def loss(self) -> float:
        """Return the mean supervised-token loss.

        Returns:
            Epoch loss as a Python float.
        """
        ...


class ValidationResult(LossResult, Protocol):
    """Expose validation loss, checkpoint score, and history values."""

    @property
    def score(self) -> float:
        """Return the score maximized during checkpoint selection.

        Returns:
            Finite validation score; larger values indicate better results.
        """
        ...

    @property
    def scalars(self) -> dict[str, float]:
        """Return numeric values to include in training history.

        Returns:
            Metric names without the validation prefix mapped to Python floats.
        """
        ...


ResultT = TypeVar("ResultT", bound=LossResult, covariant=True)


class MetricAccumulator(Protocol[ResultT]):
    """Collect detached batch statistics and produce a typed epoch result."""

    ignore_index: int

    def reset(self) -> None:
        """Discard statistics from the previous epoch."""
        ...

    def update(
        self, logits: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor, /
    ) -> None:
        """Accumulate one supervised batch.

        Args:
            logits: Raw token scores.
            labels: Target IDs with ignored positions masked.
            loss: Scalar mean loss for the supervised tokens.
        """
        ...

    def compute(self) -> ResultT:
        """Build an epoch result from accumulated statistics.

        Returns:
            Typed result containing at least the mean loss.
        """
        ...


class Evaluator(Protocol[ResultT]):
    """Define the evaluation operation required by a trainer."""

    def evaluate(self, loader: DataLoader, epoch: int) -> ResultT:
        """Evaluate a loader with fixed model weights.

        Args:
            loader: Batches to score.
            epoch: Epoch number used in progress output.

        Returns:
            Typed evaluation result.
        """
        ...


class BatchAdapter(Protocol):
    """Select and transfer model inputs without passing metadata to the model."""

    def __call__(self, batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
        """Build keyword arguments for model inference.

        Args:
            batch: Batch tensors and metadata.
            device: Execution device.

        Returns:
            Only tensors accepted by the model.
        """
        ...


class Schedule(Protocol):
    """Advance a learning-rate policy after an optimizer update."""

    def step(self) -> None:
        """Advance the schedule after one completed optimizer update."""
        ...


class Checkpoint(Protocol):
    """Persist selected weights using fixed constructor configuration."""

    def save(self, state: dict[str, torch.Tensor], *, epoch: int, score: float) -> None:
        """Persist an improved model snapshot.

        Args:
            state: Detached CPU model weights.
            epoch: One-based epoch that produced the weights.
            score: Validation score used for best-model selection.
        """
        ...


class HistoryArtifacts(Protocol):
    """Persist the complete epoch history."""

    def save_history(self, history: pd.DataFrame) -> None:
        """Write the full history accumulated so far.

        Args:
            history: One row per completed epoch.
        """
        ...


class PredictionDecoder(Protocol):
    """Convert raw token scores into label IDs using optional batch metadata."""

    def decode(self, logits: torch.Tensor, batch: dict) -> torch.Tensor:
        """Decode token scores for one batch.

        Args:
            logits: Scores with shape [batch, tokens, labels].
            batch: Input batch including any decoder metadata.

        Returns:
            Predicted token IDs with shape [batch, tokens].
        """
        ...
