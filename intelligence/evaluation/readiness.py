"""Readiness gate for forecasts that require comparable performance labels."""

from typing import Any, Dict, Mapping


class PerformancePredictionEngine:
	"""Expose prediction readiness without presenting raw counts as forecasts."""

	REQUIRED_CONTEXT = (
		"published_at",
		"metrics_captured_at",
		"audience_size",
		"impressions",
	)

	def __init__(self, training_metadata: Mapping[str, Any]):
		self.training_metadata = dict(training_metadata)

	def readiness(self) -> Dict[str, Any]:
		context_fields = set(self.training_metadata.get("label_context_fields", []))
		missing_context = [field for field in self.REQUIRED_CONTEXT if field not in context_fields]
		if self.training_metadata.get("target_metric") != "engagement_rate_by_impressions":
			missing_context.append("engagement_rate_by_impressions_target")
		if not self.training_metadata.get("prediction_model_available", False):
			missing_context.append("trained_performance_model")
		if not self.training_metadata.get("model_schema_compatible", False):
			missing_context.append("model_schema_compatible")
		if not self.training_metadata.get("post_level_split_verified", False):
			missing_context.append("post_level_split_verified")
		if not self.training_metadata.get("locked_final_test", False):
			missing_context.append("locked_final_test")
		if self.training_metadata.get("validation_status") != "passed":
			missing_context.append("validation_status_passed")
		status = "ready" if not missing_context else "not_ready"
		return {
			"engine": "performance_prediction",
			"status": status,
			"prediction": None,
			"missing_requirements": missing_context,
			"interpretation": "forecast" if status == "ready" else "no_forecast_available",
		}