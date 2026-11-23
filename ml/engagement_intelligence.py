"""Compatibility exports; implementation lives in intelligence.performance."""

from intelligence.performance.engagement_intelligence import (
    EngagementIntelligenceEngine,
    analyze_personal_performance,
)

__all__ = ["EngagementIntelligenceEngine", "analyze_personal_performance"]
