from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TensorBatchAdapter:
    """Select model input tensors and move them to the execution device.

    Attributes:
        keys: Batch keys accepted by the model. Metadata stays on its original device.
    """

    keys: tuple[str, ...]

    def __call__(self, batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
        """Build model keyword arguments from a batch.

        Args:
            batch: Input tensors and optional decoding metadata.
            device: Device used for model execution.

        Returns:
            Selected tensors transferred with nonblocking copies when supported.

        Raises:
            KeyError: A configured model input is absent.
        """
        # Page IDs and OCR offsets belong to decoders, not model.forward().
        return {key: batch[key].to(device, non_blocking=True) for key in self.keys}


def token_loss(
    logits: torch.Tensor, labels: torch.Tensor, criterion: nn.Module, phase: str
) -> torch.Tensor:
    """Flatten token predictions and compute a finite scalar loss.

    Args:
        logits: Raw scores with shape [batch, tokens, labels].
        labels: Target IDs with shape [batch, tokens].
        criterion: Scalar loss function that handles the configured ignore index.
        phase: Training or evaluation name included in failure messages.

    Returns:
        Scalar loss tensor that retains its gradient graph.

    Raises:
        FloatingPointError: The computed loss is NaN or infinite.
    """
    loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
    if not torch.isfinite(loss):
        raise FloatingPointError(f"{phase} loss is not finite")
    return loss


class ArgmaxDecoder:
    """Select token labels independently from raw model scores."""

    def decode(self, logits: torch.Tensor, batch: dict) -> torch.Tensor:
        """Select the highest-scoring label for each token.

        Args:
            logits: Scores with shape [batch, tokens, labels].
            batch: Batch metadata; unused by this decoder.

        Returns:
            Label IDs with shape [batch, tokens] on the logits device.
        """
        return logits.argmax(dim=-1)
