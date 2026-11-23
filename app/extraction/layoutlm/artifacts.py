from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image

from app.aggregates.layoutlm.params import LayoutLMTrainingParams
from app.datasets.source import DatasetSource
from app.extraction.artifacts import RunArtifacts


@dataclass(frozen=True)
class LayoutLMArtifacts(RunArtifacts):
    """Paths and storage for LayoutLM results, processor files, and page images."""

    checkpoint_path: Path = Path("best.pt")
    images_path: Path = Path("images")
    source_images_path: Path | None = None
    diagnostics_dir: Path | None = None

    @classmethod
    def from_params(cls, params: LayoutLMTrainingParams) -> "LayoutLMArtifacts":
        """Keep configured report, checkpoint, and image-cache paths independent."""
        source = DatasetSource(params.images_path).resolve("images")
        return cls(
            output_dir=Path(params.output_dir),
            checkpoint_path=Path(params.checkpoint_path).expanduser().resolve(),
            images_path=source.path
            if source.root is None
            else Path(params.output_dir).expanduser().resolve() / "images",
            source_images_path=source.path if source.root is not None else None,
            diagnostics_dir=Path(params.diagnostic_dir).expanduser().resolve()
            if params.diagnostic_dir
            else None,
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("checkpoint_path", "images_path"):
            object.__setattr__(self, name, self.resolve(getattr(self, name)))
        if self.diagnostics_dir is not None:
            object.__setattr__(
                self, "diagnostics_dir", self.resolve(self.diagnostics_dir)
            )

    def prepare(self) -> None:
        super().prepare()
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.images_path.mkdir(parents=True, exist_ok=True)
        if self.diagnostics_dir is not None:
            self.diagnostics_dir.mkdir(parents=True, exist_ok=True)

    def diagnostics(self, epoch: int) -> "LayoutLMArtifacts":
        """Return storage for one completed validation epoch.

        Args:
            epoch: One-based epoch used in the directory name.

        Returns:
            Storage rooted at diagnostic_dir/epoch_NNN.

        Raises:
            ValueError: Diagnostic output is disabled.
        """
        if self.diagnostics_dir is None:
            raise ValueError("Diagnostic output is not configured")
        return LayoutLMArtifacts(self.diagnostics_dir / f"epoch_{epoch:03d}")

    def save_diagnostics(self, report: pd.DataFrame, errors: pd.DataFrame) -> None:
        """Save strict entity scores and false-positive/false-negative reports.

        Args:
            report: Per-entity scores, indexed by entity name.
            errors: Entity errors with a status column.
        """
        self.write_csv(self.output_dir / "per_entity.csv", report, index_label="entity")
        self.write_csv(self.output_dir / "entity_errors.csv", errors)
        for status in ("false_positive", "false_negative"):
            self.write_csv(
                self.output_dir / f"{status}.csv", errors.loc[errors.status == status]
            )

    @property
    def checkpoint_location(self) -> Path:
        return self.checkpoint_path

    @property
    def history_path(self) -> Path:
        return self.output_dir / "history.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "split_manifest.csv"

    @property
    def processor_dir(self) -> Path:
        return self.output_dir / "processor"

    def save_history(self, history: pd.DataFrame) -> None:
        self.write_csv(self.history_path, history)

    def save_manifest(self, manifest: pd.DataFrame) -> None:
        self.write_csv(self.manifest_path, manifest)

    def save_processor(self, tokenizer, image_processor) -> None:
        self.save_pretrained(self.processor_dir, tokenizer)
        self.save_pretrained(self.processor_dir, image_processor)

    def save_checkpoint(self, payload: dict) -> None:
        self.save_weights(self.checkpoint_path, payload)

    def load_checkpoint(self) -> dict:
        return self.load_weights(self.checkpoint_path)

    def publication_files(self) -> dict[str, Path]:
        files = {
            "test_metrics.json": self.test_metrics_path,
            "history.csv": self.history_path,
            "config.json": self.config_path,
            "split_manifest.csv": self.manifest_path,
        }
        for path in (self.checkpoint_path, *files.values()):
            self.require_file(path)
        self.require_directory(self.processor_dir)
        return {**files, "processor": self.processor_dir}

    def page_image_path(self, document_id: str, page: int, dpi: int) -> Path:
        return self.images_path / f"dpi_{dpi}" / document_id / f"page_{page:04d}.png"

    def load_page_image(
        self, document_id: str, page: int, dpi: int
    ) -> Image.Image | None:
        paths = [self.page_image_path(document_id, page, dpi)]
        if self.source_images_path is not None:
            paths.insert(
                0,
                self.source_images_path
                / f"dpi_{dpi}"
                / document_id
                / f"page_{page:04d}.png",
            )
        for path in paths:
            if path.is_file():
                with Image.open(path) as image:
                    return image.convert("RGB")
        return None

    def save_page_image(
        self, document_id: str, page: int, dpi: int, image: Image.Image
    ) -> None:
        path = self.page_image_path(document_id, page, dpi)
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
