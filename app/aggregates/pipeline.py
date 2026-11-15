from dataclasses import dataclass
from enum import StrEnum


class PipelineExecution(StrEnum):
    """Select subprocess execution or ClearML-agent queue execution."""

    LOCAL = "local"
    AGENT = "agent"


class BertRunMode(StrEnum):
    """Select the BERT models trained by one ClearML pipeline."""

    MODEL = "model"
    LINE_ITEMS = "line-items"
    BOTH = "both"


@dataclass(frozen=True)
class ClearMLPipelineParams:
    """Configure ClearML orchestration independently of model hyperparameters.

    Attributes:
        project: ClearML project receiving pipeline and worker tasks.
        execution: Local subprocesses or tasks sent to an agent queue.
        queue: Queue consumed by configured ClearML agents in agent mode.
    """

    project: str = "ocr"
    execution: PipelineExecution = PipelineExecution.LOCAL
    queue: str = "default"
