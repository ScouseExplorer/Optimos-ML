import unittest

from intelligence.evaluation.readiness import PerformancePredictionEngine
from intelligence.performance.forecasting_foundation import (
	TARGET_METRIC,
	build_forecasting_dataset,
	calculate_impression_engagement_rate,
	chronological_author_split,
	regression_benchmark,
	validate_forecasting_records,
)


class ForecastingFoundationTests(unittest.TestCase):
	def test_impression_target_requires_complete_positive_context(self):
		complete = {"engagement": {"likes": 8, "comments": 2, "shares": 1}, "impressions": 100}
		self.assertEqual(calculate_impression_engagement_rate(complete), 11.0)
		self.assertIsNone(calculate_impression_engagement_rate({"engagement": {"likes": 8, "comments": 2, "shares": 1}, "impressions": 0}))
		self.assertIsNone(calculate_impression_engagement_rate({"engagement": {"likes": 8, "comments": None, "shares": 1}, "impressions": 100}))

	def test_current_raw_count_records_are_not_ready(self):
		records = [{
			"post_id": str(index),
			"published_at": "2026-11-19T10:00:00Z",
			"engagement": {"likes": index, "comments": 1, "shares": 0},
		} for index in range(10)]
		validation = validate_forecasting_records(records)
		self.assertEqual(validation["target_metric"], TARGET_METRIC)
		self.assertFalse(validation["ready"])
		self.assertGreater(validation["missing"]["prediction_at"], 0)
		self.assertGreater(validation["missing"]["impressions"], 0)

	def test_forecasting_dataset_is_built_but_training_is_disabled(self):
		records = [{
			"post_id": str(index),
			"published_at": "2026-11-19T10:00:00Z",
			"prediction_at": "2026-11-19T11:00:00Z",
			"outcome_at": "2026-11-26T10:00:00Z",
			"engagement": {"likes": index + 1, "comments": 1, "shares": 1},
			"impressions": 100,
			"duplicate_group_id": f"group-{index}",
		} for index in range(12)]
		dataset = build_forecasting_dataset(records, prediction_point_hours=1, outcome_window_hours=168)
		self.assertEqual(dataset["target_metric"], TARGET_METRIC)
		self.assertFalse(dataset["training_enabled"])
		self.assertEqual(len(dataset["records"]), 12)

	def test_duplicate_groups_do_not_cross_split_partitions(self):
		records = [{
			"post_id": str(index),
			"published_at": f"2026-11-{1 + index:02d}T10:00:00Z",
			"author_id": f"author-{index}",
			"duplicate_group_id": f"group-{index}",
			"target": float(index),
		} for index in range(20)]
		dataset = {"records": records}
		split = chronological_author_split(dataset)
		train = set(split["train_post_ids"])
		validation = set(split["validation_post_ids"])
		final_test = set(split["final_test_post_ids"])
		self.assertTrue(train.isdisjoint(validation))
		self.assertTrue(train.isdisjoint(final_test))
		self.assertTrue(validation.isdisjoint(final_test))
		self.assertTrue(split["final_test_locked"])

	def test_duplicate_group_overlap_is_rejected(self):
		records = [{
			"post_id": str(index),
			"published_at": f"2026-11-{1 + index:02d}T10:00:00Z",
			"author_id": f"author-{index}",
			"duplicate_group_id": "shared-group" if index in (13, 14, 17) else f"group-{index}",
		} for index in range(20)]
		with self.assertRaises(ValueError):
			chronological_author_split({"records": records})

	def test_author_overlap_is_rejected(self):
		records = [{
			"post_id": str(index),
			"published_at": f"2026-11-{1 + index:02d}T10:00:00Z",
			"author_id": "shared-author" if index in (13, 14, 17) else f"author-{index}",
			"duplicate_group_id": f"group-{index}",
		} for index in range(20)]
		with self.assertRaises(ValueError):
			chronological_author_split({"records": records})

	def test_mean_baseline_and_candidates_report_mae_rmse_and_rank(self):
		actual = [10.0, 20.0, 30.0, 40.0]
		results = regression_benchmark(actual, {
			"linear": [11.0, 19.0, 31.0, 39.0],
			"ridge": [12.0, 18.0, 32.0, 38.0],
		})
		self.assertIn("mean_baseline", results)
		self.assertEqual(results["linear"]["mae"], 1.0)
		self.assertGreater(results["linear"]["rmse"], 0.0)
		self.assertGreaterEqual(results["linear"]["spearman"], 0.99)

	def test_readiness_requires_compatible_model_and_locked_validation(self):
		current = PerformancePredictionEngine({
			"engagement_target": "observed likes + comments + shares",
			"metrics_capture_time_missing": 2949,
		}).readiness()
		self.assertEqual(current["status"], "not_ready")
		ready_metadata = {
			"label_context_fields": ["published_at", "metrics_captured_at", "audience_size", "impressions"],
			"target_metric": TARGET_METRIC,
			"prediction_model_available": True,
			"model_schema_compatible": True,
			"post_level_split_verified": True,
			"locked_final_test": True,
			"validation_status": "passed",
		}
		ready = PerformancePredictionEngine(ready_metadata).readiness()
		self.assertEqual(ready["status"], "ready")
		self.assertEqual(ready["missing_requirements"], [])

	def test_normal_retraining_is_not_a_forecasting_permission(self):
		from train_models import train_and_save_models
		with self.assertRaisesRegex(RuntimeError, "Forecasting retraining is disabled"):
			train_and_save_models()

	def test_invalid_split_windows_are_rejected(self):
		with self.assertRaises(ValueError):
			build_forecasting_dataset([], prediction_point_hours=-1, outcome_window_hours=168)
		with self.assertRaises(ValueError):
			build_forecasting_dataset([], prediction_point_hours=168, outcome_window_hours=168)


if __name__ == "__main__":
	unittest.main()
