"""Versioned evaluation summaries for content, personal, recommendation and forecasting components."""

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Sequence


def build_evaluation_report(
	*,
	dataset_version: str,
	component_versions: Mapping[str, str],
	content: Mapping[str, Any],
	performance: Mapping[str, Any],
	personal: Mapping[str, Any],
	recommendations: Mapping[str, Any],
	forecasting: Mapping[str, Any],
) -> Dict[str, Any]:
	if not isinstance(dataset_version, str) or not dataset_version.strip():
		raise ValueError("dataset_version is required")
	return {
		"evaluation_version": "1.0",
		"generated_at": datetime.now(timezone.utc).isoformat(),
		"dataset_version": dataset_version,
		"component_versions": dict(component_versions),
		"content": {
			"status": content.get("status", "unknown"),
			"rubric_version": content.get("rubric_version"),
			"human_assessment_available": bool(content.get("human_assessment_available", False)),
			"notes": content.get("notes", []),
		},
		"performance": {
			"status": performance.get("status", "unknown"),
			"missing_value_proportion": performance.get("missing_value_proportion"),
			"observation_consistency": performance.get("observation_consistency"),
			"notes": performance.get("notes", []),
		},
		"personal": {
			"status": personal.get("status", "unknown"),
			"minimum_observation_count": personal.get("minimum_observation_count"),
			"baseline_stability": personal.get("baseline_stability"),
			"notes": personal.get("notes", []),
		},
		"recommendations": {
			"status": recommendations.get("status", "unknown"),
			"acceptance_count": recommendations.get("acceptance_count"),
			"helpful_count": recommendations.get("helpful_count"),
			"relevance_sample_count": recommendations.get("relevance_sample_count"),
			"notes": recommendations.get("notes", []),
		},
		"forecasting": {
			"status": forecasting.get("status", "not_ready"),
			"baseline_comparison": forecasting.get("baseline_comparison"),
			"prediction_error": forecasting.get("prediction_error"),
			"calibration": forecasting.get("calibration"),
			"drift": forecasting.get("drift"),
			"model_version": forecasting.get("model_version"),
			"notes": forecasting.get("notes", ["No validated forecasting model is enabled."]),
		},
	}
