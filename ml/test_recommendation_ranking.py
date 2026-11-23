import unittest

from intelligence.personal.ranking import rank_recommendations


class RecommendationRankingTests(unittest.TestCase):
	def test_relevant_draft_defects_are_ranked_with_evidence(self):
		assessment = {
			"dimension_scores": {"hook": 0.55, "specificity": 0.40, "structure": 0.30},
			"dimension_explanations": {},
			"hook": {"type": "direct_claim"},
			"structure": {"has_clear_paragraph_breaks": False},
			"interaction": {"has_question": False, "has_call_to_action": False},
			"specificity": {"numbered_detail_count": 0},
			"topic_relevance": {"status": "heuristic", "lexical_coverage": 0.0},
		}
		result = rank_recommendations(assessment, {})
		categories = [item["category"] for item in result["recommendations"]]
		self.assertIn("Improve the opening", categories)
		self.assertIn("Increase specificity", categories)
		self.assertIn("Shorten or reorganise a post", categories)
		self.assertIn("Improve the call to action", categories)
		self.assertIn("Change the topic emphasis", categories)
		self.assertTrue(all(item["rank"] == index for index, item in enumerate(result["recommendations"], start=1)))
		self.assertTrue(all(item["recommendation_id"].startswith("rec-") for item in result["recommendations"]))
		self.assertTrue(all("success" not in str(item).lower() or "no success" in str(item).lower() for item in result["recommendations"]))

	def test_personal_and_observed_evidence_add_optional_actions(self):
		assessment = {
			"dimension_scores": {"hook": 0.90, "specificity": 0.80, "structure": 0.80},
			"dimension_explanations": {},
			"hook": {"type": "question"},
			"structure": {"has_clear_paragraph_breaks": True},
			"interaction": {"has_question": True, "has_call_to_action": True},
			"specificity": {"numbered_detail_count": 2},
			"topic_relevance": {"status": "not_assessed", "lexical_coverage": None},
		}
		component_results = {
			"personal_baselines": {"result": {"evidence_status": "sufficient_personal_history", "observed_post_count": 5}},
			"engagement_intelligence": {"result": {"evidence_status": "descriptive_only"}},
		}
		result = rank_recommendations(assessment, component_results)
		categories = [item["category"] for item in result["recommendations"]]
		self.assertIn("Use a personal baseline", categories)
		self.assertIn("Test a different content format", categories)
		for item in result["recommendations"]:
			self.assertIn("qualification", item)
			self.assertIn("supporting_evidence", item)
			self.assertIn("limitations", item)

	def test_sparse_personal_history_adds_no_personal_action(self):
		assessment = {
			"dimension_scores": {"hook": 0.90, "specificity": 0.80, "structure": 0.80},
			"dimension_explanations": {},
			"hook": {"type": "question"},
			"structure": {"has_clear_paragraph_breaks": True},
			"interaction": {"has_question": True, "has_call_to_action": True},
			"specificity": {"numbered_detail_count": 2},
			"topic_relevance": {"status": "not_assessed", "lexical_coverage": None},
		}
		component_results = {
			"personal_baselines": {"result": {"evidence_status": "insufficient_personal_history", "observed_post_count": 2}},
			"engagement_intelligence": {"result": {"evidence_status": "no_observed_rates"}},
		}
		result = rank_recommendations(assessment, component_results)
		categories = [item["category"] for item in result["recommendations"]]
		self.assertNotIn("Use a personal baseline", categories)
		self.assertNotIn("Test a different content format", categories)


if __name__ == "__main__":
	unittest.main()
