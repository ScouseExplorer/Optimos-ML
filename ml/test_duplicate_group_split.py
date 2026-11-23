import unittest
from unittest.mock import patch

import pandas as pd

from ml.regression_models import LinkedInNLPRegressionModels


class DuplicateGroupSplitTests(unittest.TestCase):
	def test_chronological_partitions_are_author_and_duplicate_disjoint(self):
		count = 100
		frame = pd.DataFrame({
			"published_at": pd.date_range("2026-01-01", periods=count, freq="h").astype(str),
			"author_id": [f"author-{index // 2}" for index in range(count)],
			"author_name": [None] * count,
			"duplicate_group_id": [f"content-{index // 2}" for index in range(count)],
		})

		partitions = LinkedInNLPRegressionModels._chronological_author_split_indices(frame)
		partition_frames = [frame.iloc[indices] for indices in partitions]

		for left_index, left in enumerate(partition_frames):
			for right in partition_frames[left_index + 1:]:
				self.assertTrue(set(left["author_id"]).isdisjoint(right["author_id"]))
				self.assertTrue(set(left["duplicate_group_id"]).isdisjoint(right["duplicate_group_id"]))
		self.assertLess(partition_frames[0]["published_at"].max(), partition_frames[1]["published_at"].min())
		self.assertLess(partition_frames[1]["published_at"].max(), partition_frames[2]["published_at"].min())

	def test_model_selection_compares_baseline_and_excludes_metadata_features(self):
		model = LinkedInNLPRegressionModels()
		count = 100
		texts = [f"post {index}" for index in range(count)]
		features = [{"word_count": float(index), "style": float(index % 5)} for index in range(count)]
		groups = [f"group-{index}" for index in range(count)]
		authors = [f"author-{index}" for index in range(count)]
		dates = pd.date_range("2026-01-01", periods=count, freq="h").astype(str).tolist()
		targets = [float((index % 7) ** 2) for index in range(count)]
		comments = [index % 2 for index in range(count)]

		with patch.object(model.nlp_extractor, "extract_all_features", side_effect=features), \
				patch.object(model, "_analyze_feature_importance"):
			training_df = model.prepare_training_data_with_nlp(
				texts, targets, comments, groups, authors, authors, dates
			)
			model.train_models(training_df)

		self.assertNotIn("author_id", model.feature_names)
		self.assertNotIn("published_at", model.feature_names)
		self.assertIn("baseline_mean", model.validation_results)
		self.assertIn("baseline", model.regression_models)
		self.assertIn("mae", model.final_test_results["baseline_mean"])
		self.assertEqual(
			model.training_sample_count + model.validation_sample_count + model.evaluation_sample_count,
			count,
		)


if __name__ == "__main__":
	unittest.main()