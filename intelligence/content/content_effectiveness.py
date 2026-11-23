"""Heuristic analysis of content characteristics, separate from outcomes."""

import re
from typing import Any, Dict, Optional, Protocol


class FeatureExtractor(Protocol):
	def extract_all_features(self, text: str) -> Dict[str, float]: ...


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
	return max(lower, min(upper, value))


class ContentEffectivenessEngine:
	"""Inspect hooks, readability, structure, relevance, and interaction cues.

	Scores are rubric signals about the text itself, not forecasts of reach or
	engagement. Topic relevance is a lexical heuristic when a target is supplied.
	"""

	def __init__(self, feature_extractor: FeatureExtractor):
		self.feature_extractor = feature_extractor

	def evaluate(self, content: str, intended_topic: Optional[str] = None) -> Dict[str, Any]:
		if not isinstance(content, str) or not content.strip():
			raise ValueError("content must be a nonempty string")

		features = self.feature_extractor.extract_all_features(content)
		lines = [line.strip() for line in content.splitlines() if line.strip()]
		hook_text = lines[0] if lines else content.strip()
		word_count = int(features.get("word_count", len(content.split())))
		bullet_count = int(features.get("bullet_point_count", 0))
		paragraph_count = int(features.get("paragraph_count", len([p for p in content.split("\n\n") if p.strip()])))
		cta_score = float(features.get("cta_score", 0))
		flesch = float(features.get("flesch_reading_ease", 0))

		lower_hook = hook_text.casefold()
		if "?" in hook_text:
			hook_type = "question"
		elif re.search(r"\d+%|\d+x|\d+", hook_text):
			hook_type = "specific_claim"
		elif any(phrase in lower_hook for phrase in ("i learned", "i failed", "my experience", "when i ")):
			hook_type = "personal_experience"
		elif any(phrase in lower_hook for phrase in ("unpopular opinion", "contrary", "everyone says", "the truth is", "wrong")):
			hook_type = "contrarian_claim"
		elif any(phrase in lower_hook for phrase in ("in today's world", "as we all know", "i'm excited to share", "excited to announce")):
			hook_type = "generic_introduction"
		else:
			hook_type = "direct_claim"

		if intended_topic and intended_topic.strip():
			topic_tokens = set(re.findall(r"\w+", intended_topic.casefold()))
			content_tokens = set(re.findall(r"\w+", content.casefold()))
			topic_relevance = {
				"status": "heuristic",
				"intended_topic": intended_topic.strip(),
				"lexical_coverage": round(len(topic_tokens & content_tokens) / len(topic_tokens), 3) if topic_tokens else None,
			}
		else:
			topic_relevance = {
				"status": "not_assessed",
				"intended_topic": None,
				"lexical_coverage": None,
			}

		readability = {
			"flesch_reading_ease": round(flesch, 2),
			"assessment": "easier" if flesch >= 60 else "more_complex",
		}
		structure = {
			"word_count": word_count,
			"sentence_count": int(features.get("sentence_count", 0)),
			"paragraph_count": paragraph_count,
			"bullet_count": bullet_count,
			"has_clear_paragraph_breaks": paragraph_count > 1,
		}
		interaction = {
			"has_question": bool(features.get("has_question", 0)),
			"has_call_to_action": cta_score > 0,
			"cta_signal_count": cta_score,
		}
		cliches = (
			"game changer", "unlock your potential", "in today's world", "thought leader",
			"leverage", "synergy", "best practices",
		)
		genericity_signal = _clamp(
			0.2 * (word_count < 80)
			+ 0.2 * (float(features.get("unique_word_ratio", 0)) < 0.55)
			+ 0.2 * (float(features.get("personal_pronoun_ratio", 0)) < 0.01)
			+ 0.1 * (len(re.findall(r"\d", content)) / max(len(content), 1) < 0.003)
			+ min(0.3, sum(term in content.casefold() for term in cliches) * 0.08)
		)
		specificity = {
			"unique_word_ratio": round(float(features.get("unique_word_ratio", 0)), 3),
			"numbered_detail_count": len(re.findall(r"\b\d+(?:\.\d+)?%?\b", content)),
			"genericity_signal": round(genericity_signal, 3),
		}
		sentence_count = int(features.get("sentence_count", 0))
		avg_sentence_length = float(features.get("avg_sentence_length", 0))
		if 8 <= avg_sentence_length <= 24:
			readability_score = 0.9
		elif avg_sentence_length > 0:
			readability_score = _clamp(1.0 - abs(avg_sentence_length - 16) / 40.0)
		else:
			readability_score = _clamp(1.0 - abs(flesch - 65.0) / 100.0)

		hook_score = {
			"question": 0.9,
			"specific_claim": 0.85,
			"personal_experience": 0.8,
			"contrarian_claim": 0.75,
			"direct_claim": 0.55,
			"generic_introduction": 0.35,
		}[hook_type]
		structure_score = _clamp(
			0.35 * (paragraph_count > 1)
			+ 0.35 * (bullet_count > 0)
			+ 0.30 * (sentence_count > 2)
		)
		interaction_score = 0.5 * interaction["has_question"] + 0.5 * interaction["has_call_to_action"]
		specificity_score = _clamp(
			0.6 * specificity["unique_word_ratio"]
			+ 0.4 * min(specificity["numbered_detail_count"] / 3.0, 1.0)
		)
		dimension_scores = {
			"hook": round(hook_score, 3),
			"readability": round(readability_score, 3),
			"structure": round(structure_score, 3),
			"interaction": round(interaction_score, 3),
			"specificity": round(specificity_score, 3),
		}
		if topic_relevance["lexical_coverage"] is not None:
			dimension_scores["topic_relevance"] = topic_relevance["lexical_coverage"]
		content_score = round(100 * sum(dimension_scores.values()) / len(dimension_scores), 1)

		dimension_explanations = {
			"hook": {
				"positive": "The opening is identified as a question, specific claim, or personal experience." if hook_score >= 0.8 else None,
				"weakness": "The opening is classified as a direct claim and could be more specific or audience-focused." if hook_score < 0.8 else None,
			},
			"readability": {
				"positive": "Average sentence length is in a generally readable range." if readability_score >= 0.8 else None,
				"weakness": "Sentence length or readability ease is outside the current target range." if readability_score < 0.8 else None,
			},
			"structure": {
				"positive": "The post uses paragraphs, formatting or enough sentence count to expose structure." if structure_score >= 0.65 else None,
				"weakness": "The post has limited paragraph, list or sentence structure." if structure_score < 0.65 else None,
			},
			"interaction": {
				"positive": "The post includes a question or call to action." if interaction_score >= 0.5 else None,
				"weakness": "The post does not combine an explicit question with clear response language." if interaction_score < 0.5 else None,
			},
			"specificity": {
				"positive": "The wording is varied or includes a specific number." if specificity_score >= 0.65 else None,
				"weakness": "The post may need more varied wording or concrete evidence." if specificity_score < 0.65 else None,
			},
		}
		if topic_relevance["lexical_coverage"] is not None:
			topic_coverage = topic_relevance["lexical_coverage"]
			dimension_explanations["topic_relevance"] = {
				"positive": "The stated topic has direct lexical overlap with the post." if topic_coverage >= 0.5 else None,
				"weakness": "The post has limited lexical overlap with the stated topic; semantic relevance is not confirmed." if topic_coverage < 0.5 else None,
			}

		strengths = []
		weaknesses = []
		if interaction["has_question"]:
			strengths.append("Includes a question for the audience")
		else:
			weaknesses.append("No direct question")
		if interaction["has_call_to_action"]:
			strengths.append("Includes call-to-action language")
		else:
			weaknesses.append("No clear call to action")
		if bullet_count or paragraph_count > 1:
			strengths.append("Uses visible structure")
		else:
			weaknesses.append("Consider adding paragraph breaks or a list")
		if word_count < 60:
			weaknesses.append("Very short; check whether enough context is present")
		elif word_count > 400:
			weaknesses.append("Long post; consider tightening the structure")
		if intended_topic and topic_relevance["lexical_coverage"] == 0:
			weaknesses.append("No direct lexical overlap with the stated topic; verify the connection manually")

		return {
			"engine": "content_effectiveness",
			"rubric_version": "content-rubric-1.0",
			"evidence_type": "text_characteristics",
			"hook": {"text": hook_text[:180], "type": hook_type},
			"readability": readability,
			"structure": structure,
			"topic_relevance": topic_relevance,
			"interaction": interaction,
			"specificity": specificity,
			"dimension_scores": dimension_scores,
			"dimension_explanations": dimension_explanations,
			"content_effectiveness_score": content_score,
			"score_type": "heuristic_content_rubric_not_performance_prediction",
			"strengths": strengths,
			"weaknesses": weaknesses,
			"features": features,
		}