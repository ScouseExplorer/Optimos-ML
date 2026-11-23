"""Variant orchestration for evidence-bounded post refinement."""

from typing import Any, Mapping, Sequence

from intelligence.generative.explanation import SYSTEM_PROMPT, build_refinement_prompt
from intelligence.generative.llm_client import OpenAITextProvider


class RefinementService:
	def __init__(self, provider: OpenAITextProvider):
		self.provider = provider

	def refine_variants(
		self,
		original: str,
		style: str,
		assessment: Mapping[str, Any],
		improvements: Sequence[str],
		variant_count: int,
	) -> list[str]:
		variant_count = max(1, min(int(variant_count), 3))
		prompt = build_refinement_prompt(original, style, assessment, list(improvements))
		variants = []
		for _ in range(variant_count):
			variants.append(self.provider.generate(SYSTEM_PROMPT, prompt, timeout_seconds=float(__import__("os").getenv("OPENAI_TIMEOUT_SECONDS", "30"))))
		return variants
