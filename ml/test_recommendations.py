import unittest
from unittest.mock import patch

from intelligence.personal.recommendations import recommend_strategy


def classify(content):
    return {"topic": content, "format": content, "hook": content}


class RecommendationTests(unittest.TestCase):
    def test_cold_start_uses_profile_and_asks_for_missing_facts(self):
        result = recommend_strategy(
            profile={"expertise": ["Cloud"], "experiences": ["Migrated a service"]},
            goal="help peers", historical_posts=["A previous post"],
        )
        self.assertEqual(result["topic"], "Cloud")
        self.assertEqual(result["evidence_level"], "profile_based")
        self.assertEqual(result["evidence_provenance"]["global"]["status"], "unavailable")
        self.assertEqual(result["evidence_to_use"][0]["text"], "Migrated a service")
        self.assertTrue(any("outcome" in question for question in result["questions"]))
        self.assertIsNone(result["expected_audience"])
        self.assertNotIn("confidence", result)
        self.assertEqual(result, recommend_strategy(
            profile={"expertise": ["Cloud"], "experiences": ["Migrated a service"]},
            goal="help peers", historical_posts=["A previous post"],
        ))

    def test_small_and_unmatched_samples_do_not_drive_strategy(self):
        posts = [{"post_id": str(index), "content": "question"} for index in range(4)]
        metrics = [{"post_id": str(index), "likes": 9, "comments": 1, "reposts": 0,
                    "impressions": 100} for index in range(4)]
        metrics.append({"post_id": "other", "likes": 1000})
        result = recommend_strategy({"expertise": ["Cloud"]}, historical_posts=posts,
                                    historical_performance=list(reversed(metrics)))
        self.assertEqual(result["format"], "educational post")
        self.assertEqual(result["evidence_level"], "profile_based")
        self.assertEqual(result["evidence_provenance"]["personal"]["engagement_rate_by_impressions_sample_size"], 4)
        self.assertEqual(result["evidence_provenance"]["personal"]["unmatched_performance_sample_size"], 1)

    def test_matched_comparable_groups_are_descriptive(self):
        posts = [{"post_id": str(index), "content": "story" if index < 5 else "list"}
                 for index in range(10)]
        metrics = [{"post_id": str(index), "likes": 9 if index < 5 else 1,
                    "comments": 0, "reposts": 0, "impressions": 100} for index in range(10)]
        with patch("intelligence.performance.engagement_intelligence._classify_post", side_effect=classify):
            result = recommend_strategy({"expertise": ["Cloud"]}, historical_posts=posts,
                                        historical_performance=list(reversed(metrics)))
        self.assertEqual(result["topic"], "Cloud")
        self.assertEqual(result["format"], "story")
        self.assertEqual(result["evidence_level"], "descriptive_personal")
        comparison = result["evidence_provenance"]["personal"]["selected_comparisons"]["format"]
        self.assertEqual(comparison["sample_size"], 5)
        self.assertEqual(comparison["comparison"]["sample_size"], 5)
        self.assertEqual(comparison["metric"], "engagement_rate_by_impressions")

    def test_global_evidence_requires_provenance(self):
        missing = recommend_strategy(global_evidence=[{"topic": "AI"}])
        self.assertIsNone(missing["topic"])
        self.assertEqual(missing["evidence_provenance"]["global"]["status"], "insufficient_provenance")
        supplied = recommend_strategy(global_evidence=[{"topic": "AI", "source": "Study", "sample_size": 20}])
        self.assertEqual(supplied["topic"], "AI")
        self.assertEqual(supplied["evidence_level"], "supplied_global")
        self.assertEqual(supplied["evidence_provenance"]["global"]["records"][0]["sample_size"], 20)
        self.assertEqual(supplied["evidence_to_use"][0]["provenance"], "Study")

    def test_experience_and_role_only_cold_starts(self):
        experience = recommend_strategy({"experiences": ["Migrated a service"]}, goal="help peers")
        self.assertEqual(experience["topic"], "Migrated a service")
        self.assertIn("help peers", experience["angle"])
        self.assertEqual(experience["evidence_provenance"]["topic_source"], "profile_experience")
        role = recommend_strategy({"role": "Engineer"})
        self.assertEqual(role["topic"], "Engineer")
        self.assertEqual(role["evidence_level"], "profile_based")

    def test_many_matched_but_incomplete_metrics_do_not_invoke_classifiers(self):
        posts = [{"post_id": str(index), "content": "post"} for index in range(10)]
        metrics = [{"post_id": str(index), "likes": 1} for index in range(10)]
        with patch("intelligence.performance.engagement_intelligence._classify_post", side_effect=AssertionError("unexpected classifier")):
            result = recommend_strategy({"expertise": ["Cloud"]}, historical_posts=posts,
                                        historical_performance=metrics)
        self.assertEqual(result["evidence_level"], "profile_based")
        self.assertEqual(result["evidence_provenance"]["personal"]["matched_metrics_sample_size"], 10)
        self.assertEqual(result["evidence_provenance"]["personal"]["engagement_rate_by_impressions_sample_size"], 0)

    def test_different_denominators_are_not_compared(self):
        posts = [{"post_id": str(index), "content": "story" if index < 5 else "list"}
                 for index in range(10)]
        metrics = [{"post_id": str(index), "likes": 1, "comments": 0, "reposts": 0,
                    "impressions" if index < 5 else "followers": 100} for index in range(10)]
        with patch("intelligence.performance.engagement_intelligence._classify_post", side_effect=AssertionError("unexpected classifier")):
            result = recommend_strategy({"expertise": ["Cloud"]}, historical_posts=posts,
                                        historical_performance=metrics)
        self.assertEqual(result["evidence_provenance"]["personal"]["selected_comparisons"], {})
        self.assertEqual(result["evidence_level"], "profile_based")

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            recommend_strategy(historical_posts=[{"post_id": "a", "content": "First"}],
                               historical_performance=[{"post_id": "a"}, {"post_id": "a"}])


if __name__ == "__main__":
    unittest.main()