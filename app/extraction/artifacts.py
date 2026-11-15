import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch


@dataclass(frozen=True)
class RunArtifacts(ABC):
    """Own filesystem operations for a run without creating files on construction."""

    output_dir: Path

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "output_dir", Path(self.output_dir).expanduser().resolve()
        )

    def resolve(self, path: Path) -> Path:
        """Resolve relative artifact paths under this run's output directory."""
        path = Path(path).expanduser()
        return (
            path.resolve() if path.is_absolute() else (self.output_dir / path).resolve()
        )

    def prepare(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def require_file(path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Missing artifact file: {path}")

    @staticmethod
    def require_directory(path: Path) -> None:
        if not path.is_dir():
            raise FileNotFoundError(f"Missing artifact directory: {path}")

    def write_json(self, path: Path, payload: dict) -> Path:
        encoded = json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=lambda value: value.tolist(),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding="utf-8")
        return path

    def read_json(self, path: Path) -> dict:
        self.require_file(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def write_csv(
        self, path: Path, frame: pd.DataFrame, *, index_label: str | None = None
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=index_label is not None, index_label=index_label)
        return path

    def save_weights(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def load_weights(self, path: Path) -> dict:
        self.require_file(path)
        return torch.load(path, map_location="cpu", weights_only=True)

    def save_pretrained(self, path: Path, component) -> None:
        path.mkdir(parents=True, exist_ok=True)
        component.save_pretrained(path)

    @property
    def config_path(self) -> Path:
        return self.output_dir / "config.json"

    @property
    def test_metrics_path(self) -> Path:
        return self.output_dir / "test_metrics.json"

    def save_config(self, config: dict) -> Path:
        return self.write_json(self.config_path, config)

    def save_metrics(self, metrics: dict) -> Path:
        return self.write_json(self.test_metrics_path, metrics)

    def load_config(self) -> dict:
        return self.read_json(self.config_path)

    def load_metrics(self) -> dict:
        return self.read_json(self.test_metrics_path)

    def save_rejections(self, split: str, frame: pd.DataFrame) -> Path:
        if split not in {"train", "val", "test"}:
            raise ValueError("Split must be train, val, or test")
        return self.write_csv(self.output_dir / f"{split}_rejected.csv", frame)

    def validate_evaluation(self, checkpoint_path: Path, metrics_path: Path) -> None:
        self.validate_checkpoint_reference(checkpoint_path)
        if Path(metrics_path).resolve() != self.test_metrics_path:
            raise ValueError(
                "Evaluation metrics must belong to the publication directory"
            )

    def validate_checkpoint_reference(self, path: Path) -> None:
        if Path(path).resolve() != self.checkpoint_location:
            raise ValueError("Checkpoint does not match the run artifacts")

    @property
    @abstractmethod
    def checkpoint_location(self) -> Path:
        """Return the model's checkpoint file or directory."""

    @abstractmethod
    def publication_files(self) -> dict[str, Path]:
        """Validate model-specific outputs before publishing.

        Returns:
            Artifact names mapped to local upload paths.

        Raises:
            FileNotFoundError: A required checkpoint or run output is absent.
        """
