from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingResult:
    """Identify the saved outputs of a completed training step.

    Attributes:
        checkpoint_path: Model checkpoint file or directory.
        history_path: CSV file containing epoch metrics.
    """

    checkpoint_path: Path
    history_path: Path


@dataclass(frozen=True)
class EvaluationResult:
    """Identify the checkpoint and held-out metrics passed to publication.

    Attributes:
        checkpoint_path: Checkpoint evaluated on the test split.
        metrics_path: JSON file containing the test result.
    """

    checkpoint_path: Path
    metrics_path: Path
