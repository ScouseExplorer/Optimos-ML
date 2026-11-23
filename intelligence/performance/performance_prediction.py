"""Prediction facade backed by evaluation readiness checks until labels are ready."""

from intelligence.evaluation.readiness import PerformancePredictionEngine

__all__ = ["PerformancePredictionEngine"]
