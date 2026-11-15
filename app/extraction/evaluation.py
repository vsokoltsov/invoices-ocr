from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Generic, TypeVar

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from app.extraction.batches import token_loss
from app.extraction.protocols import BatchAdapter, LossResult

ResultT = TypeVar("ResultT", bound=LossResult)


@dataclass(kw_only=True)
class TokenClassificationEvaluation(ABC, Generic[ResultT]):
    """Run a shared inference loop with model-specific metric collection.

    Attributes:
        model: Module returning raw token logits.
        device: Execution device; the caller must place the model there.
        criterion: Scalar loss function with the required ignore index.
        batch_adapter: Selects tensors passed to the model.
    """

    model: nn.Module
    device: torch.device
    criterion: nn.Module
    batch_adapter: ClassVar[BatchAdapter]

    @property
    @abstractmethod
    def ignore_index(self) -> int:
        """Return the target value excluded from loss and scoring.

        Returns:
            Ignored target label ID.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear accumulated predictions and metrics before evaluation."""
        ...

    @abstractmethod
    def update(
        self,
        batch: dict,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss: torch.Tensor,
    ) -> None:
        """Collect predictions and loss for a supervised batch.

        Args:
            batch: Original input batch, including metadata.
            logits: Model scores on the execution device.
            labels: Target IDs on the execution device.
            loss: Scalar loss for this batch.
        """
        ...

    @abstractmethod
    def compute(self, loader: DataLoader) -> ResultT:
        """Build the final evaluation result.

        Args:
            loader: Evaluated loader; its dataset may supply reconstruction metadata.

        Returns:
            Model-specific metrics containing the mean loss.
        """
        ...

    @torch.inference_mode()
    def evaluate(self, loader: DataLoader, epoch: int) -> ResultT:
        """Evaluate batches without gradients and skip fully ignored targets.

        Sets evaluation mode and leaves it active. This method does not move the
        model or restore its previous training mode.

        Args:
            loader: Batches to score.
            epoch: Epoch number for progress output.

        Returns:
            Final metrics from the model-specific collector.

        Raises:
            ValueError: No supervised tokens are present.
            FloatingPointError: An evaluation loss is NaN or infinite.
        """
        self.model.eval()
        self.reset()
        progress = tqdm(loader, position=1, desc=f"Eval epoch {epoch}", leave=False)
        supervised_tokens = 0
        for batch in progress:
            labels = batch["labels"].to(self.device, non_blocking=True)
            count = int((labels != self.ignore_index).sum().item())
            if count == 0:
                continue
            supervised_tokens += count
            inputs = self.batch_adapter(batch, self.device)
            logits = self.model(**inputs)
            loss = token_loss(logits, labels, self.criterion, "Evaluation")
            self.update(batch, logits, labels, loss)
        if supervised_tokens == 0:
            raise ValueError("Evaluation loader contains no supervised tokens.")
        return self.compute(loader)
