import unittest

from intelligence.content.content_effectiveness import ContentEffectivenessEngine
from intelligence.orchestrator.engine import IntelligenceOrchestrator
from intelligence.orchestrator.registry import ComponentRegistry, build_default_registry
from intelligence.orchestrator.schemas import AnalysisRequest, ComponentOutput
from intelligence.performance.engagement_intelligence import EngagementIntelligenceEngine
from intelligence.performance.performance_prediction import PerformancePredictionEngine

CONTRACT_KEYS = {"component", "version", "status", "result", "evidence", "limitations", "execution_ms", "error"}


class FakeExtractor:
	def extract_all_features(self, text):
		return {"word_count": len(text.split()), "sentence_count": 2, "paragraph_count": 2,
				"cta_score": 1, "has_question": 1, "flesch_reading_ease": 70, "unique_word_ratio": 0.8}


def classify(content):
	return {"topic": "topic", "format": "text", "hook": "question"}


def make_orchestrator(content_engine="default"):
	engine = ContentEffectivenessEngine(FakeExtractor()) if content_engine == "default" else content_engine
	registry = build_default_registry(
		content_engine=engine,
		engagement_engine=EngagementIntelligenceEngine(),
		prediction_engine=PerformancePredictionEngine({}),
		classifier=classify,
	)
	return IntelligenceOrchestrator(registry)


class OrchestratorTests(unittest.TestCase):
	def test_post_only_runs_content_and_prediction_readiness(self):
		response = make_orchestrator().analyze(AnalysisRequest(content="What worked?\n\nHere is the lesson."))
		self.assertEqual(set(response["components"]), {"content_effectiveness", "recommendation_ranking", "performance_prediction"})
		skipped = {item["component"]: item["reason"] for item in response["skipped"]}
		self.assertEqual(skipped["engagement_intelligence"], "required_inputs_not_supplied")
		self.assertEqual(skipped["recommendations"], "required_inputs_not_supplied")
		for component in response["components"].values():
			self.assertEqual(set(component), CONTRACT_KEYS)
			self.assertGreaterEqual(component["execution_ms"], 0)

	def test_prediction_is_unavailable_with_stated_limitations(self):
		response = make_orchestrator().analyze(AnalysisRequest(content="A post."))
		prediction = response["components"]["performance_prediction"]
		self.assertEqual(prediction["status"], "unavailable")
		self.assertIsNone(prediction["result"]["prediction"])
		self.assertIn("performance_prediction", response["unavailable"])
		self.assertTrue(any("Forecast unavailable" in item for item in response["limitations"]))

	def test_history_and_profile_activate_engagement_and_recommendations(self):
		response = make_orchestrator().analyze(AnalysisRequest(
			profile={"expertise": ["Cloud"]},
			historical_posts=[{"post_id": "a", "content": "A post"}],
			historical_performance=[{"post_id": "a", "likes": 2, "comments": 1, "shares": 0, "impressions": 20}],
		))
		self.assertEqual(set(response["components"]), {"engagement_intelligence", "recommendations"})
		engagement = response["components"]["engagement_intelligence"]
		self.assertEqual(engagement["result"]["interpretation"], "descriptive_context_only")
		self.assertEqual(engagement["result"]["engagement_rate_by_impressions"], 0.15)
		self.assertEqual(response["components"]["recommendations"]["result"]["topic"], "Cloud")
		self.assertTrue(any(item["component"] == "engagement_intelligence" for item in response["evidence"]))

	def test_component_filter_limits_execution(self):
		response = make_orchestrator().analyze(
			AnalysisRequest(content="A post.", components=["content_effectiveness"])
		)
		self.assertEqual(list(response["components"]), ["content_effectiveness"])
		reasons = {item["component"]: item["reason"] for item in response["skipped"]}
		self.assertEqual(reasons["performance_prediction"], "not_requested")

	def test_one_failing_component_does_not_fail_the_rest(self):
		class Broken:
			def evaluate(self, content, topic):
				raise RuntimeError("secret draft text should not leak")

		response = make_orchestrator(content_engine=Broken()).analyze(AnalysisRequest(content="A post."))
		self.assertEqual(response["status"], "partial")
		self.assertEqual(response["failed"], ["content_effectiveness"])
		self.assertEqual(response["components"]["recommendation_ranking"]["status"], "unavailable")
		failed = response["components"]["content_effectiveness"]
		self.assertEqual(failed["status"], "error")
		self.assertEqual(failed["error"]["type"], "RuntimeError")
		self.assertNotIn("secret", str(failed))
		self.assertIn("performance_prediction", response["components"])

	def test_uninitialized_content_engine_is_unavailable_not_error(self):
		response = make_orchestrator(content_engine=None).analyze(AnalysisRequest(content="A post."))
		self.assertEqual(response["components"]["content_effectiveness"]["status"], "unavailable")
		self.assertEqual(response["failed"], [])

	def test_invalid_input_is_rejected_before_execution(self):
		orchestrator = make_orchestrator()
		with self.assertRaises(ValueError):
			orchestrator.analyze(AnalysisRequest(content="   "))
		with self.assertRaises(ValueError):
			orchestrator.analyze(AnalysisRequest(content="A post.", components=["nonexistent"]))

	def test_request_with_no_applicable_components(self):
		response = make_orchestrator().analyze(AnalysisRequest())
		self.assertEqual(response["status"], "no_applicable_components")
		self.assertEqual(response["components"], {})

	def test_all_components_failing_reports_failed(self):
		registry = ComponentRegistry()
		def boom(request):
			raise ValueError("bad")
		registry.register("only", "1.0.0", boom, lambda r: True)
		response = IntelligenceOrchestrator(registry).analyze(AnalysisRequest())
		self.assertEqual(response["status"], "failed")

	def test_registry_rejects_duplicate_names(self):
		registry = ComponentRegistry()
		registry.register("a", "1", lambda r: ComponentOutput(result={}), lambda r: True)
		with self.assertRaises(ValueError):
			registry.register("a", "1", lambda r: ComponentOutput(result={}), lambda r: True)


if __name__ == "__main__":
	unittest.main()
