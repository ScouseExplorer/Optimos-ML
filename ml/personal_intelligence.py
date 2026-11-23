"""Compatibility exports; implementation lives in intelligence.performance."""

from intelligence.performance.engagement_intelligence import analyze_personal_performance
from intelligence.personal.personal_intelligence import analyze_personal_baselines

__all__ = ["analyze_personal_performance", "analyze_personal_baselines"]
