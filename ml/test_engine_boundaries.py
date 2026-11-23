import unittest

from intelligence.content.content_effectiveness import ContentEffectivenessEngine
from intelligence.performance.engagement_intelligence import EngagementIntelligenceEngine
from intelligence.performance.performance_prediction import PerformancePredictionEngine
from intelligence.evaluation.rubric_evaluation import evaluate_rubric


class FakeFeatureExtractor:
	def extract_all_features(self, text):
		return {
			"word_count": len(text.split()),
			"sentence_count": 2,
			"paragraph_count": 2,
			"bullet_point_count": 0,
			"cta_score": 1,
			"has_question": 1,
			"flesch_reading_ease": 72,
			"unique_word_ratio": 0.8,
		}


def classify(content):
	return {"topic": "topic", "format": "text", "hook": "question"}


class EngineBoundaryTests(unittest.TestCase):
	def test_content_effectiveness_reports_text_only_signals(self):
		result = ContentEffectivenessEngine(FakeFeatureExtractor()).evaluate(
			"What worked in AI?\n\nHere is what I learned.", "AI"
		)
		self.assertEqual(result["engine"], "content_effectiveness")
		self.assertEqual(result["hook"]["type"], "question")
		self.assertEqual(result["topic_relevance"]["lexical_coverage"], 1.0)
		self.assertNotIn("engagement_score", result)

	def test_hook_types_are_distinguished_and_explained(self):
		examples = {
			"What happened?\n\nDetails.": "question",
			"Revenue rose 35% this quarter.": "specific_claim",
			"When I led the migration, I learned three lessons.": "personal_experience",
			"Everyone says hiring is easy. That is wrong.": "contrarian_claim",
			"I'm excited to share a new release.": "generic_introduction",
			"We changed the deployment process.": "direct_claim",
		}
		engine = ContentEffectivenessEngine(FakeFeatureExtractor())
		for text, expected in examples.items():
			with self.subTest(hook_type=expected):
				result = engine.evaluate(text)
				self.assertEqual(result["hook"]["type"], expected)
				self.assertEqual(result["rubric_version"], "content-rubric-1.0")
				self.assertIn("hook", result["dimension_explanations"])

	def test_rubric_human_agreement_is_reproducible(self):
		algorithm_scores = {
			"p1": {"hook": 88.0},
			"p2": {"hook": 62.0},
		}
		human_scores = {
			"p1": [88.0, 92.0],
			"p2": [58.0, 62.0],
		}
		result = evaluate_rubric(algorithm_scores, human_scores, "hook")
		self.assertEqual(result["posts_evaluated"], 2)
		self.assertEqual(result["algorithm_human_mae"], 2.0)
		self.assertEqual(result["score_stability"], 0.96)

	def test_engagement_intelligence_marks_outcomes_descriptive(self):
		result = EngagementIntelligenceEngine().analyze(
			[{"post_id": "p1", "content": "A post"}],
			[{"post_id": "p1", "likes": 4, "comments": 1, "shares": 0}],
			classifier=classify,
		)
		self.assertEqual(result["engine"], "engagement_intelligence")
		self.assertEqual(result["interpretation"], "descriptive_context_only")
		self.assertIsNone(result["engagement_rate_by_impressions"])

	def test_current_raw_count_artifact_is_not_ready_for_prediction(self):
		result = PerformancePredictionEngine({
			"engagement_target": "observed likes + comments + shares",
			"metrics_capture_time_missing": 2949,
		}).readiness()
		self.assertEqual(result["status"], "not_ready")
		self.assertIn("impressions", result["missing_requirements"])
		self.assertIn("trained_performance_model", result["missing_requirements"])
		self.assertIsNone(result["prediction"])


if __name__ == "__main__":
	unittest.main()