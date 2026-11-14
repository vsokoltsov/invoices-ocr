from dataclasses import dataclass, asdict
import json
from pathlib import Path
from torch.utils.data import DataLoader
from app.aggregates.layoutlm.params import LayoutLMTrainingParams
from app.pipelines.layoutlm.training import TrainingResult

@dataclass
class EvaluateTraining:
    output_dir: Path

    def eval(self, train_result: TrainingResult) -> Path:
        test_metrics = train_result.trainer.eval_one_epoch(
            epoch=1, external_loader=train_result.loader
        )
        (self.output_dir / "test_metrics.json").write_text(
            json.dumps(asdict(test_metrics), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        return self.output_dir
