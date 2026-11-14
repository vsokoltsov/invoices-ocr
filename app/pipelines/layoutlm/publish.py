from dataclasses import dataclass
from pathlib import Path

from app.extraction.layoutlm.publisher import ClearMLPublisher

@dataclass
class PublishModel:
    publisher: ClearMLPublisher
    checkpoint_path: str

    def publish(self, output_dir: Path) -> str:
        return self.publisher.run(
            checkpoint_path=Path(self.checkpoint_path),
            output_dir=output_dir
        )