"""Small provider abstraction for evidence-bounded text generation."""

import os
from typing import List, Mapping, Optional


class GenerativeProviderError(RuntimeError):
	pass


class OpenAITextProvider:
	def __init__(self, client: object):
		self.client = client

	def generate(self, system_prompt: str, user_prompt: str, *, timeout_seconds: float = 30.0) -> str:
		model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
		max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "500"))
		if max_tokens < 100:
			max_tokens = 100
		if max_tokens > 800:
			max_tokens = 800
		try:
			response = self.client.chat.completions.create(
				model=model,
				messages=[
					{"role": "system", "content": system_prompt},
					{"role": "user", "content": user_prompt},
				],
				max_completion_tokens=max_tokens,
				timeout=timeout_seconds,
			)
		except Exception as exc:
			raise GenerativeProviderError(type(exc).__name__) from exc
		try:
			text = response.choices[0].message.content
		except (AttributeError, IndexError, KeyError) as exc:
			raise GenerativeProviderError("missing_completion") from exc
		if not isinstance(text, str) or not text.strip():
			raise GenerativeProviderError("empty_completion")
		return text.strip()
