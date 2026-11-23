import unittest
from types import ModuleType
from unittest.mock import patch

from intelligence.performance.engagement_intelligence import analyze_personal_performance


def classify(content):
	return {"topic": content, "format": "text", "hook": "direct claim"}


class PersonalPerformanceTests(unittest.TestCase):
	def test_matches_shuffled_metrics_and_groups_observed_rates(self):
		posts = [{"post_id": "a", "content": "AI"}, {"post_id": "b", "content": "Cloud"}]
		metrics = [
			{"post_id": "b", "likes": 8, "comments": 1, "shares": 1, "impressions": 100},
			{"post_id": "a", "likes": 2, "comments": 1, "reposts": 1, "impressions": 20},
			{"post_id": "unknown", "likes": 100},
		]
		result = analyze_personal_performance(posts, metrics, classifier=classify)
		self.assertEqual(result["observations"][0]["engagement_rate_by_impressions"], 0.2)
		self.assertEqual(result["groups"]["topic"]["Cloud"]["engagement_rate_by_impressions"], 0.1)
		self.assertEqual(result["groups"]["hook"]["direct claim"]["engagement_rate_by_impressions_sample_size"], 2)
		self.assertEqual(result["unmatched_performance_sample_size"], 1)
		self.assertEqual(result["evidence_status"], "descriptive_only")

	def test_missing_and_invalid_metrics_do_not_become_zero(self):
		posts = [{"post_id": str(index), "content": "AI"} for index in range(4)]
		metrics = [
			{"post_id": "0", "likes": 0, "comments": 0, "reposts": 0, "impressions": 10},
			{"post_id": "1", "likes": 2, "comments": 1, "impressions": 10},
			{"post_id": "2", "likes": 2, "comments": 1, "reposts": 1, "impressions": 0},
			{"post_id": "3", "likes": float("nan"), "comments": 1, "reposts": 1, "followers": 10},
		]
		result = analyze_personal_performance(posts, metrics, classifier=classify)
		self.assertEqual(result["engagement_rate_by_impressions"], 0)
		self.assertEqual(result["engagement_rate_by_impressions_sample_size"], 1)
		self.assertIsNone(result["observations"][1]["metrics"]["reposts"])
		self.assertIsNone(result["observations"][2]["engagement_rate_by_impressions"])
		self.assertIsNone(result["observations"][3]["engagement_rate_by_followers"])

	def test_distributions_author_type_length_and_source_metadata(self):
		posts = [
			{"post_id": "a", "content": "Short AI post", "author_type": "profile", "published_at": "2026-11-19T10:00:00Z", "source": "public_apify"},
			{"post_id": "b", "content": "A much longer post " * 80, "author_type": "company", "published_at": "2026-11-20T10:00:00Z", "source": "public_apify"},
		]
		metrics = [
			{"post_id": "a", "likes": 1, "comments": 0, "shares": 0, "impressions": 10, "captured_at": "2026-11-20T10:00:00Z", "source": "public_apify"},
			{"post_id": "b", "likes": 101, "comments": 9, "shares": 5, "impressions": 1000, "captured_at": "2026-11-21T10:00:00Z", "source": "public_apify"},
		]
		result = analyze_personal_performance(posts, metrics, classifier=classify)
		self.assertEqual(result["metric_distributions"]["likes"]["median"], 51)
		self.assertEqual(result["metric_distributions"]["likes"]["p90"], 91)
		self.assertEqual(result["groups"]["author_type"]["profile"]["sample_size"], 1)
		self.assertEqual(result["groups"]["author_type"]["company"]["sample_size"], 1)
		self.assertEqual(result["groups"]["content_length"]["short_under_100"]["sample_size"], 1)
		self.assertEqual(result["groups"]["content_length"]["long_over_300"]["sample_size"], 1)
		self.assertEqual(result["sources"], ["public_apify"])
		self.assertEqual(result["date_range"]["published_at_min"], "2026-11-19T10:00:00Z")
		self.assertEqual(result["missing_data"]["followers"]["missing_count"], 2)
		self.assertEqual(result["metric_distributions"]["engagement_total"]["median"], 58)
		self.assertEqual(result["observation_analysis"]["status"], "not_available")
		self.assertEqual(result["observation_analysis"]["captured_observation_count"], 2)

	def test_no_metrics_is_not_performance_evidence(self):
		result = analyze_personal_performance([{"post_id": "a", "content": "AI"}], classifier=classify)
		self.assertEqual(result["evidence_status"], "no_observed_rates")
		self.assertIsNone(result["engagement_rate_by_impressions"])
		self.assertEqual(result["groups"]["format"]["text"]["sample_size"], 1)

	def test_missing_or_duplicate_ids_are_rejected(self):
		with self.assertRaises(ValueError):
			analyze_personal_performance([{"content": "AI"}], classifier=classify)
		with self.assertRaises(ValueError):
			analyze_personal_performance([], [{"post_id": "a"}, {"post_id": "a"}], classifier=classify)

	def test_default_classifier_and_follower_rate(self):
		api_module = ModuleType("api.app")
		api_module.extract_nlp_features = lambda text: (None, {"word_count": len(text.split())})
		api_module._classify_topic = lambda features, text: {"topic": "AI"}
		api_module._classify_format = lambda features, text: {"format": "text"}
		api_module._classify_hook = lambda text: {"hook_type": "question"}
		with patch.dict("sys.modules", {"api.app": api_module}):
			result = analyze_personal_performance(
				[{"post_id": "a", "content": "AI?"}],
				[{"post_id": "a", "likes": 2, "comments": 1, "shares": 1, "follower_count": 20}],
			)
		self.assertEqual(result["groups"]["topic"]["AI"]["engagement_rate_by_followers"], 0.2)
		self.assertEqual(result["groups"]["hook"]["question"]["evidence_status"], "descriptive_only")
		self.assertIsNone(result["engagement_rate_by_impressions"])


if __name__ == "__main__":
	unittest.main()
