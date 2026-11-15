from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceSelector:
    """Resolve automatic or explicit accelerator requests.

    Attributes:
        requested: auto, cpu, cuda, or mps. Auto prefers CUDA, then MPS, then CPU.
    """

    requested: str = "auto"

    def resolve(self) -> torch.device:
        """Resolve the requested execution device.

        Returns:
            Available torch device.

        Raises:
            ValueError: An explicitly requested accelerator is unavailable.
        """
        available = {
            "cuda": torch.cuda.is_available(),
            "mps": torch.backends.mps.is_available(),
            "cpu": True,
        }
        if self.requested == "auto":
            return torch.device(
                next(name for name, ready in available.items() if ready)
            )
        if self.requested in available and not available[self.requested]:
            raise ValueError(
                f"{self.requested.upper()} was requested but is not available"
            )
        return torch.device(self.requested)
