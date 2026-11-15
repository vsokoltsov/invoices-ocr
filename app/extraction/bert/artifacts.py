from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.aggregates.bert.params import BertPytorchParams
from app.extraction.artifacts import RunArtifacts


@dataclass(frozen=True)
class BertArtifacts(RunArtifacts):
    """Paths and storage for BERT checkpoints, run results, and diagnostics."""

    checkpoint_dir: Path = Path(".")
    history_path: Path = Path("history.csv")
    diagnostics_dir: Path | None = None

    @classmethod
    def from_params(cls, params: BertPytorchParams) -> "BertArtifacts":
        """Keep configured checkpoint, history, and diagnostic paths independent."""
        return cls(
            output_dir=Path(params.checkpoint_dir),
            history_path=Path(params.result_filename)
            .expanduser()
            .with_suffix(".csv")
            .resolve(),
            diagnostics_dir=Path(params.diagnostic_dir).expanduser().resolve()
            if params.diagnostic_dir
            else None,
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("checkpoint_dir", "history_path"):
            object.__setattr__(self, name, self.resolve(getattr(self, name)))
        if self.diagnostics_dir is not None:
            object.__setattr__(
                self, "diagnostics_dir", self.resolve(self.diagnostics_dir)
            )

    def prepare(self) -> None:
        super().prepare()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        if self.diagnostics_dir is not None:
            self.diagnostics_dir.mkdir(parents=True, exist_ok=True)

    @property
    def checkpoint_location(self) -> Path:
        return self.checkpoint_dir

    @property
    def weights_path(self) -> Path:
        return self.checkpoint_dir / "model.pt"

    @property
    def encoder_dir(self) -> Path:
        return self.checkpoint_dir / "encoder"

    @property
    def tokenizer_dir(self) -> Path:
        return self.checkpoint_dir / "tokenizer"

    @property
    def manifest_path(self) -> Path:
        return self.checkpoint_dir / "split_manifest.json"

    def records_path(self, split: str) -> Path:
        if split not in {"validation", "test"}:
            raise ValueError("Split must be validation or test")
        return self.checkpoint_dir / f"{split}.json"

    def load_records(self, split: str) -> dict:
        return self.read_json(self.records_path(split))

    def has_records(self, split: str) -> bool:
        return self.records_path(split).is_file()

    def has_manifest(self) -> bool:
        return self.manifest_path.is_file()

    def save_manifest(self, manifest: dict) -> None:
        self.write_json(self.manifest_path, manifest)

    def load_manifest(self) -> dict:
        return self.read_json(self.manifest_path)

    def save_history(self, history: pd.DataFrame) -> None:
        self.write_csv(self.history_path, history)

    def validate_checkpoint(self) -> None:
        for path in (
            self.weights_path,
            self.records_path("validation"),
            self.encoder_dir / "config.json",
        ):
            self.require_file(path)
        self.require_directory(self.tokenizer_dir)

    def publication_files(self) -> dict[str, Path]:
        self.validate_checkpoint()
        for path in (self.config_path, self.test_metrics_path, self.history_path):
            self.require_file(path)
        files = {"history.csv": self.history_path}
        if self.has_manifest():
            files["split_manifest.json"] = self.manifest_path
        return files

    def diagnostics(self, epoch: int) -> "BertArtifacts":
        if self.diagnostics_dir is None:
            raise ValueError("Diagnostic output is not configured")
        return BertArtifacts(self.diagnostics_dir / f"epoch_{epoch:03d}")

    def debug_reports(self, output_dir: Path) -> "BertArtifacts":
        output_dir = Path(output_dir).expanduser().resolve()
        if (
            output_dir == self.checkpoint_dir
            or self.checkpoint_dir in output_dir.parents
        ):
            raise ValueError("Debug output must be outside the checkpoint directory")
        return BertArtifacts(output_dir)

    def decoder_reports(self, decoder: str) -> "BertArtifacts":
        if decoder not in {"token_argmax", "ocr_word"}:
            raise ValueError("Unknown decoder")
        return BertArtifacts(self.output_dir / decoder)

    def save_diagnostics(self, report: pd.DataFrame, errors: pd.DataFrame) -> None:
        self.write_csv(self.output_dir / "per_entity.csv", report, index_label="entity")
        self.write_csv(self.output_dir / "entity_errors.csv", errors)
        for status in ("false_positive", "false_negative"):
            self.write_csv(
                self.output_dir / f"{status}.csv",
                errors.loc[errors["status"] == status],
            )

    def save_comparison(self, comparison: pd.DataFrame) -> Path:
        return self.write_csv(self.output_dir / "decoder_comparison.csv", comparison)
