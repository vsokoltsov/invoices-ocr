"""Shared window metrics used by BERT training and evaluation."""

from app.extraction.metrics import WindowMetrics as EvalMetrics
from app.extraction.word_metrics import ValidationMetricsResult as EvalResult

__all__ = ["EvalMetrics", "EvalResult"]
