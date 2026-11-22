"""Shared word metrics used by LayoutLM training and evaluation."""

from app.extraction.word_metrics import (
    EpochMetricsResult,
    ValidationMetricsResult,
    WordMetrics,
)

EpochMetrics = WordMetrics

__all__ = ["EpochMetrics", "EpochMetricsResult", "ValidationMetricsResult"]
