"""Responsible forecasting foundations: target validation, dataset splitting and benchmarks."""

from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


TARGET_METRIC = "engagement_rate_by_impressions"


@dataclass(frozen=True)
class ForecastingRecord:
	post_id: str
	published_at: str
	prediction_at: str
	outcome_at: str
	audience_size: Optional[float]
	impressions: Optional[float]
	likes: Optional[float]
	comments: Optional[float]
	shares: Optional[float]
	target: Optional[float]
	author_id: Optional[str]
	duplicate_group_id: str
	quality_status: str


def _optional_nonnegative_number(record: Mapping[str, Any], *names: str) -> Optional[float]:
	for name in names:
		value = record.get(name)
		if value is None:
			continue
		if isinstance(value, bool) or not isinstance(value, (int, float)):
			return None
		number = float(value)
		if not isfinite(number) or number < 0:
			return None
		return number
	return None


def calculate_impression_engagement_rate(record: Mapping[str, Any]) -> Optional[float]:
	"""Calculate engagement / impressions * 100 only with complete positive inputs."""
	engagement = record.get("engagement") if isinstance(record.get("engagement"), Mapping) else {}
	likes = _optional_nonnegative_number(engagement, "likes")
	comments = _optional_nonnegative_number(engagement, "comments")
	shares = _optional_nonnegative_number(engagement, "shares")
	impressions = _optional_nonnegative_number(record, "impressions")
	if impressions is None:
		impressions = _optional_nonnegative_number(engagement, "impressions")
	if likes is None or comments is None or shares is None or impressions is None or impressions <= 0:
		return None
	return (likes + comments + shares) / impressions * 100.0


def validate_forecasting_records(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
	"""Validate whether a row set can participate in a future forecasting task."""
	rows = list(records)
	missing = {
		"post_id": sum(not isinstance(row.get("post_id"), str) or not row["post_id"] for row in rows),
		"published_at": sum(row.get("published_at") is None for row in rows),
		"prediction_at": sum(row.get("prediction_at") is None for row in rows),
		"outcome_at": sum(row.get("outcome_at") is None for row in rows),
		"impressions": sum(calculate_impression_engagement_rate(row) is None for row in rows),
	}
	valid_targets = sum(calculate_impression_engagement_rate(row) is not None for row in rows)
	return {
		"record_count": len(rows),
		"valid_target_count": valid_targets,
		"missing": missing,
		"target_metric": TARGET_METRIC,
		"ready": len(rows) > 0 and valid_targets == len(rows) and not any(missing.values()),
	}


def build_forecasting_dataset(
	records: Sequence[Mapping[str, Any]],
	*,
	prediction_point_hours: int,
	outcome_window_hours: int,
) -> Dict[str, Any]:
	"""Build a frozen dataset contract without enabling training on unsupported labels."""
	if prediction_point_hours < 0 or outcome_window_hours <= prediction_point_hours:
		raise ValueError("Outcome window must exceed the prediction point")

	forecast_rows: List[Dict[str, Any]] = []
	for row in records:
		target = calculate_impression_engagement_rate(row)
		post_id = row.get("post_id")
		published_at = row.get("published_at")
		prediction_at = row.get("prediction_at")
		outcome_at = row.get("outcome_at")
		if not all(isinstance(value, str) and value for value in (post_id, published_at, prediction_at, outcome_at)):
			continue
		if target is None:
			continue
		engagement = row.get("engagement") if isinstance(row.get("engagement"), Mapping) else {}
		audience_size = _optional_nonnegative_number(row, "audience_size")
		if audience_size is None:
			audience_size = _optional_nonnegative_number(engagement, "audience_size")
		forecast_rows.append({
			"post_id": post_id,
			"published_at": published_at,
			"prediction_at": prediction_at,
			"outcome_at": outcome_at,
			"audience_size": audience_size,
			"target": target,
			"author_id": row.get("author_id") or row.get("author_name"),
			"duplicate_group_id": row.get("duplicate_group_id") or f"post-{post_id}",
			"quality_status": row.get("quality_status") or "unclassified",
		})

	return {
		"target_metric": TARGET_METRIC,
		"prediction_point_hours": prediction_point_hours,
		"outcome_window_hours": outcome_window_hours,
		"training_enabled": False,
		"records": forecast_rows,
		"excluded_record_count": len(records) - len(forecast_rows),
	}


def chronological_author_split(
	dataset: Mapping[str, Any],
	*,
	validation_fraction: float = 0.15,
	final_test_fraction: float = 0.15,
) -> Dict[str, Any]:
	"""Create persisted-like train/validation/test membership IDs without touching a final test set."""
	rows = list(dataset.get("records", []))
	if len(rows) < 10:
		raise ValueError("At least 10 rows are required for a three-way split")
	if validation_fraction <= 0 or final_test_fraction <= 0 or validation_fraction + final_test_fraction >= 1:
		raise ValueError("Validation and final-test fractions must be positive and sum to less than one")

	def parsed(value: Any) -> Optional[datetime]:
		if not isinstance(value, str):
			return None
		try:
			return datetime.fromisoformat(value.replace("Z", "+00:00"))
		except ValueError:
			return None

	indexed = []
	for index, row in enumerate(rows):
		date = parsed(row.get("published_at"))
		if date is None:
			raise ValueError("Every forecasting row must have a valid published_at timestamp")
		indexed.append((index, date, row))
	indexed.sort(key=lambda item: (item[1], item[0]))

	validation_start = int(len(indexed) * (1 - validation_fraction - final_test_fraction))
	final_test_start = int(len(indexed) * (1 - final_test_fraction))
	train_rows = [row for _, _, row in indexed[:validation_start]]
	validation_rows = [row for _, _, row in indexed[validation_start:final_test_start]]
	final_test_rows = [row for _, _, row in indexed[final_test_start:]]

	def unique_groups(items: Sequence[Mapping[str, Any]]) -> set:
		return {str(item["duplicate_group_id"]) for item in items}

	def unique_authors(items: Sequence[Mapping[str, Any]]) -> set:
		return {str(item["author_id"]) for item in items if item.get("author_id") is not None}

	train_groups = unique_groups(train_rows)
	validation_groups = unique_groups(validation_rows)
	final_test_groups = unique_groups(final_test_rows)
	train_authors = unique_authors(train_rows)
	validation_authors = unique_authors(validation_rows)
	final_test_authors = unique_authors(final_test_rows)
	if train_groups & validation_groups or train_groups & final_test_groups or validation_groups & final_test_groups:
		raise ValueError("Duplicate post groups overlap across forecasting partitions")
	if train_authors & validation_authors or train_authors & final_test_authors or validation_authors & final_test_authors:
		raise ValueError("Author identities overlap across forecasting partitions")

	return {
		"train_post_ids": [row["post_id"] for row in train_rows],
		"validation_post_ids": [row["post_id"] for row in validation_rows],
		"final_test_post_ids": [row["post_id"] for row in final_test_rows],
		"final_test_locked": True,
	}


def regression_benchmark(actual: Sequence[float], candidate_predictions: Mapping[str, Sequence[float]]) -> Dict[str, Any]:
	"""Compare mean and candidate predictions using MAE/RMSE and rank correlation."""
	actual_array = np.asarray(actual, dtype=float)
	if actual_array.ndim != 1 or len(actual_array) == 0:
		raise ValueError("Actual targets must be a nonempty one-dimensional sequence")

	def metrics(predicted_values: Sequence[float]) -> Dict[str, Any]:
		predicted_array = np.asarray(predicted_values, dtype=float)
		if predicted_array.shape != actual_array.shape:
			raise ValueError("Candidate predictions must match actual target shape")
		mae = float(np.mean(np.abs(actual_array - predicted_array)))
		rmse = float(np.sqrt(np.mean((actual_array - predicted_array) ** 2)))
		actual_ranks = np.asarray([rank for rank in np.argsort(np.argsort(actual_array))], dtype=float)
		predicted_ranks = np.asarray([rank for rank in np.argsort(np.argsort(predicted_array))], dtype=float)
		spearman = None
		if len(actual_array) > 1 and np.std(actual_ranks) > 0 and np.std(predicted_ranks) > 0:
			spearman = float(np.corrcoef(actual_ranks, predicted_ranks)[0, 1])
		return {"mae": mae, "rmse": rmse, "spearman": spearman}

	baseline = np.full(actual_array.shape, float(np.mean(actual_array)), dtype=float)
	results = {"mean_baseline": metrics(baseline)}
	for name, predictions in candidate_predictions.items():
		results[name] = metrics(predictions)
	return results
