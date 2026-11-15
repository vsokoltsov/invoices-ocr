import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Generic, TypeVar

import pandas as pd
import structlog
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from app.extraction.batches import token_loss
from app.extraction.protocols import (
    BatchAdapter,
    Checkpoint,
    Evaluator,
    HistoryArtifacts,
    LossResult,
    MetricAccumulator,
    Schedule,
    ValidationResult,
)

logger = structlog.get_logger()
TrainT = TypeVar("TrainT", bound=LossResult)
ValidationT = TypeVar("ValidationT", bound=ValidationResult)


class ConstantSchedule:
    """Keep the configured learning rate constant throughout training."""

    def step(self) -> None:
        """Leave optimizer learning rates unchanged."""
        pass


class TrainingCallback:
    """Provide optional hooks for epoch results and improved weights.

    Default hooks have no side effects. Override only the events a callback needs.
    """

    def on_epoch(
        self, epoch: int, validation: ValidationResult, history: pd.DataFrame
    ) -> None:
        """Handle a completed validation epoch.

        Args:
            epoch: One-based epoch number.
            validation: Metrics from the completed validation pass.
            history: Full history including the current epoch.
        """
        pass

    def on_best(self, epoch: int, score: float, state: dict[str, torch.Tensor]) -> None:
        """Handle a strict improvement in the validation score.

        Args:
            epoch: One-based epoch number.
            score: New best validation score.
            state: Detached, cloned model weights on the CPU; treat as read-only.
        """
        pass


@dataclass
class BestCheckpointWriter(TrainingCallback):
    """Delegate improved snapshots to a configured model-specific checkpoint.

    Attributes:
        checkpoint: Serializer with fixed architecture and preprocessing metadata.
    """

    checkpoint: Checkpoint

    def on_best(self, epoch: int, score: float, state: dict[str, torch.Tensor]) -> None:
        """Save only the snapshot selected by the shared training loop.

        Args:
            epoch: One-based epoch that produced the improved weights.
            score: New best validation score.
            state: Detached CPU weight snapshot.
        """
        self.checkpoint.save(state, epoch=epoch, score=score)


@dataclass
class HistoryWriter(TrainingCallback):
    """Persist history after every completed validation epoch.

    Attributes:
        artifacts: Storage component responsible for the history destination.
    """

    artifacts: HistoryArtifacts

    def on_epoch(
        self, epoch: int, validation: ValidationResult, history: pd.DataFrame
    ) -> None:
        """Save the complete history through the artifact store.

        Args:
            epoch: Completed epoch; unused by this writer.
            validation: Validation result; unused by this writer.
            history: Full history including the completed epoch.
        """
        self.artifacts.save_history(history)


@dataclass(kw_only=True)
class TokenClassificationTraining(ABC, Generic[TrainT, ValidationT]):
    """Train token classifiers and restore weights with the best validation score.

    Subclasses provide metrics, evaluation, scheduling, and persistence hooks.
    The loop does not choose model-specific paths or checkpoint formats.

    Attributes:
        model: Module returning raw token logits.
        optimizer: Optimizer bound to the model parameters.
        criterion: Scalar token loss with the required ignore index.
        train_loader: Training batches.
        val_loader: Validation batches.
        device: Device used for model execution.
        epochs: Positive number of epochs.
        batch_adapter: Model-specific selection and transfer of batch tensors.
    """

    model: nn.Module
    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    train_loader: DataLoader
    val_loader: DataLoader
    device: torch.device
    epochs: int
    batch_adapter: ClassVar[BatchAdapter]

    def __post_init__(self) -> None:
        """Validate the configured training duration.

        Raises:
            ValueError: The number of epochs is less than one.
        """
        if self.epochs < 1:
            raise ValueError("epochs must be positive")

    @property
    @abstractmethod
    def training_metrics(self) -> MetricAccumulator[TrainT]:
        """Return the accumulator used for training batches.

        Returns:
            Accumulator whose ignore index matches the loss function.
        """
        ...

    @property
    @abstractmethod
    def schedule(self) -> Schedule:
        """Return the policy advanced after each optimizer update.

        Returns:
            Learning-rate schedule or a constant policy.
        """
        ...

    @abstractmethod
    def make_evaluator(self) -> Evaluator[ValidationT]:
        """Build an evaluator with the current model and decoding configuration.

        Returns:
            Evaluator for this model and its validation result type.
        """
        ...

    def callbacks(self) -> tuple[TrainingCallback, ...]:
        """Build the ordered callbacks used for one training run.

        Returns:
            Callbacks for epoch completion and best-weight events; empty by default.
        """
        return ()

    def train_one_epoch(self, epoch: int) -> TrainT:
        """Optimize one epoch while skipping batches without supervised labels.

        Each optimizer update clips the gradient norm to 1 and advances the schedule.
        Ignored batches do not update parameters, metrics, or the schedule.

        Args:
            epoch: One-based epoch number for progress output.

        Returns:
            Accumulated training metrics.

        Raises:
            FloatingPointError: The training loss is not finite.
            RuntimeError: Gradient clipping detects nonfinite gradients.
            ValueError: The metric accumulator has no supervised observations.
        """
        self.model.train()
        metrics = self.training_metrics
        metrics.reset()
        progress = tqdm(
            self.train_loader, position=1, desc=f"Train epoch {epoch}", leave=False
        )
        for batch in progress:
            labels = batch["labels"].to(self.device, non_blocking=True)
            # Empty supervision must not consume an optimizer or scheduler step.
            if not (labels != metrics.ignore_index).any():
                continue
            inputs = self.batch_adapter(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(**inputs)
            loss = token_loss(logits, labels, self.criterion, "Training")
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0, error_if_nonfinite=True
            )
            self.optimizer.step()
            self.schedule.step()
            metrics.update(logits, labels, loss)
        return metrics.compute()

    def eval_one_epoch(
        self, epoch: int, external_loader: DataLoader | None = None
    ) -> ValidationT:
        """Evaluate validation data or a supplied external loader.

        Args:
            epoch: Epoch number for progress output.
            external_loader: Optional loader that replaces the configured validation loader.

        Returns:
            Model-specific validation metrics.
        """
        loader = external_loader if external_loader is not None else self.val_loader
        return self.make_evaluator().evaluate(loader, epoch)

    def train(self) -> pd.DataFrame:
        """Run all epochs, call persistence hooks, and restore the best weights.

        Weights are copied to the CPU only on a strict score improvement. Callbacks
        receive those weights for optional persistence. If training fails, the error
        propagates; restoration occurs only after the loop completes successfully.

        Returns:
            DataFrame with epoch, training loss, and prefixed validation scalars.

        Raises:
            FloatingPointError: A validation selection score is not finite.
        """
        self.model.to(self.device)
        best_score = -math.inf
        best_state = None
        history = []
        callbacks = self.callbacks()
        progress = tqdm(range(1, self.epochs + 1), position=0, leave=True)
        for epoch in progress:
            progress.set_description(f"Train model {epoch}/{self.epochs}")
            training = self.train_one_epoch(epoch)
            validation = self.eval_one_epoch(epoch)
            if not math.isfinite(validation.score):
                raise FloatingPointError("Validation score is not finite")
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": training.loss,
                    **{
                        f"validation_{key}": value
                        for key, value in validation.scalars.items()
                    },
                }
            )
            logger.info(
                "Training results",
                epoch=f"{epoch}/{self.epochs}",
                train_loss=training.loss,
                val_loss=validation.loss,
                val_f1=validation.score,
            )
            if validation.score > best_score:
                best_score = float(validation.score)
                # Clone to keep subsequent optimizer updates out of this snapshot.
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                for callback in callbacks:
                    callback.on_best(epoch, best_score, best_state)
            # Persist improved weights before diagnostics or history can fail.
            history_frame = pd.DataFrame(history)
            for callback in callbacks:
                callback.on_epoch(epoch, validation, history_frame)
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return pd.DataFrame(history)
