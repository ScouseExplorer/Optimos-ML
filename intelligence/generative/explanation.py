"""Evidence-bounded prompt assembly and alternative-draft comparison."""

from typing import Any, Dict, Mapping


SYSTEM_PROMPT = (
	"You improve LinkedIn drafts using only the supplied content and analytical evidence. "
	"Do not invent engagement metrics, user statistics, historical performance, predictions, "
	"confidence scores, or claims of success. Preserve factual claims and user experience."
)


def build_refinement_prompt(original: str, style: str, assessment: Mapping[str, Any], improvements: list[str]) -> str:
	strengths = assessment.get("strengths", [])
	weaknesses = assessment.get("weaknesses", [])
	limitations = [
		"The rubric is heuristic, not an observed performance outcome.",
		"Performance prediction is unavailable with the current evidence.",
	]
	return (
		"ORIGINAL POST:\n" + original + "\n\n"
		+ "STRUCTURED ANALYSIS:\n"
		+ f"- Rubric score: {assessment['content_effectiveness_score']} ({assessment['score_type']})\n"
		+ f"- Strengths: {', '.join(strengths) if strengths else 'None identified'}\n"
		+ f"- Weaknesses: {', '.join(weaknesses) if weaknesses else 'None identified'}\n"
		+ f"- Improvements: {', '.join(improvements) if improvements else 'No additional changes identified'}\n"
		+ f"- Limitations: {'; '.join(limitations)}\n\n"
		+ f"Rewrite for a {style} tone. Keep the author's stated experience and facts. "
		+ "Do not add numbers, outcomes, testimonials, credentials, or experiences that are not in the original. "
		+ "Improve clarity and structure only. Return only the revised post text."
	)


def compare_assessments(original: Mapping[str, Any], refined: Mapping[str, Any]) -> Dict[str, Any]:
	original_dimensions = original.get("dimension_scores", {})
	refined_dimensions = refined.get("dimension_scores", {})
	differences = {
		name: round(float(refined_dimensions.get(name, 0)) - float(original_dimensions.get(name, 0)), 3)
		for name in sorted(set(original_dimensions) | set(refined_dimensions))
	}
	return {
		"hook": differences.get("hook", 0.0),
			"readability": differences.get("readability", 0.0),
			"structure": differences.get("structure", 0.0),
			"specificity": differences.get("specificity", 0.0),
			"interaction": differences.get("interaction", 0.0),
			"topic_relevance": differences.get("topic_relevance", 0.0),
			"overall_score_change": round(
				float(refined.get("content_effectiveness_score", 0))
				- float(original.get("content_effectiveness_score", 0)),
				1,
			),
			"score_type": "heuristic_content_rubric_comparison_not_performance_prediction",
			"limitations": [
				"Differences are rubric changes, not evidence of future performance.",
			],
		}
