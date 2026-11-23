import unittest

from intelligence.generative.explanation import SYSTEM_PROMPT, build_refinement_prompt, compare_assessments
from intelligence.generative.llm_client import GenerativeProviderError, OpenAITextProvider
from intelligence.generative.refinement import RefinementService


class FakeCompletions:
	def __init__(self, text="A clearer post.", error=False):
		self.text = text
		self.error = error

	def create(self, **kwargs):
		if self.error:
			raise RuntimeError("provider failure")
		class Message:
			content = self.text
		class Choice:
			message = Message()
		class Response:
			choices = [Choice()]
		return Response()


class FakeClient:
	def __init__(self, text="A clearer post.", error=False):
		self.chat = type("Chat", (), {"completions": FakeCompletions(text, error)})()


class GenerativeTests(unittest.TestCase):
	def test_provider_returns_trimmed_completion(self):
		provider = OpenAITextProvider(FakeClient("  A clearer post.  "))
		self.assertEqual(provider.generate("system", "user"), "A clearer post.")

	def test_provider_converts_provider_failure(self):
		provider = OpenAITextProvider(FakeClient(error=True))
		with self.assertRaises(GenerativeProviderError):
			provider.generate("system", "user")

	def test_prompt_does_not_request_performance_evidence(self):
		prompt = build_refinement_prompt(
			"Original post",
			"professional",
			{"content_effectiveness_score": 50, "score_type": "heuristic_content_rubric_not_performance_prediction", "strengths": [], "weaknesses": ["No direct question"]},
			["Add a question"],
		)
		self.assertIn("Do not invent", SYSTEM_PROMPT)
		self.assertIn("Do not add numbers", prompt)
		self.assertIn("not an observed performance outcome", prompt)
		self.assertNotIn("viral", prompt.lower())

	def test_variant_count_is_bounded(self):
		service = RefinementService(OpenAITextProvider(FakeClient("A clearer post.")))
		assessment = {"content_effectiveness_score": 50, "score_type": "heuristic", "strengths": [], "weaknesses": []}
		variants = service.refine_variants("Original", "professional", assessment, [], 99)
		self.assertEqual(len(variants), 3)

	def test_draft_comparison_reports_rubric_deltas_only(self):
		original = {"content_effectiveness_score": 50, "dimension_scores": {"hook": 0.5, "readability": 0.7, "structure": 0.4}}
		refined = {"content_effectiveness_score": 65, "dimension_scores": {"hook": 0.8, "readability": 0.8, "structure": 0.6}}
		comparison = compare_assessments(original, refined)
		self.assertEqual(comparison["hook"], 0.3)
		self.assertEqual(comparison["overall_score_change"], 15.0)
		self.assertIn("not_performance_prediction", comparison["score_type"])


if __name__ == "__main__":
	unittest.main()
