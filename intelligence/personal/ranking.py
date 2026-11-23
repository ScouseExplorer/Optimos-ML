"""Rank evidence-backed recommendation actions for the current draft."""

from hashlib import sha256
from typing import Any, Dict, List, Mapping, Optional


def _recommendation_id(category: str, action: str) -> str:
	seed = f"{category}|{action}".encode("utf-8")
	return "rec-" + sha256(seed).hexdigest()[:16]


def _evidence_strength(component_results: Mapping[str, Any]) -> Dict[str, Any]:
	personal = component_results.get("personal_baselines", {}).get("result", {})
	personal_status = personal.get("evidence_status")
	personal_reliable = personal_status == "sufficient_personal_history"
	engagement = component_results.get("engagement_intelligence", {}).get("result", {})
	engagement_status = engagement.get("evidence_status")
	engagement_reliable = engagement_status == "descriptive_only"
	return {
		"personal_reliable": personal_reliable,
		"engagement_reliable": engagement_reliable,
	}


def rank_recommendations(
	content_assessment: Mapping[str, Any],
	component_results: Mapping[str, Any],
	strategy: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
	"""Return ranked draft actions with evidence and limitations, never success probabilities."""
	assessment = content_assessment or {}
	dimension_scores = assessment.get("dimension_scores", {})
	explanations = assessment.get("dimension_explanations", {})
	hook = assessment.get("hook", {})
	structure = assessment.get("structure", {})
	interaction = assessment.get("interaction", {})
	specificity = assessment.get("specificity", {})
	topic_relevance = assessment.get("topic_relevance", {})
	evidence = _evidence_strength(component_results)

	recommendations: List[Dict[str, Any]] = []

	if not interaction.get("has_question"):
		recommendations.append({
			"category": "Improve the opening",
			"action": "Add one direct question or a more specific opening claim.",
			"reason": "The current draft does not contain a detected question.",
			"supporting_evidence": ["content.interaction.has_question=false"],
			"qualification": "heuristic",
			"limitations": ["A question is an interaction cue, not proof of higher engagement."],
			"rank_score": 0.92,
		})

	if float(dimension_scores.get("hook", 0.0)) < 0.70:
		recommendations.append({
			"category": "Improve the opening",
			"action": f"Rewrite the first line as a {hook.get('type', 'direct claim')} with more specificity.",
			"reason": "The hook dimension is below the current rubric threshold.",
			"supporting_evidence": [f"content.dimension_scores.hook={dimension_scores.get('hook')}"],
			"qualification": "heuristic",
			"limitations": ["Hook quality is rubric-based, not outcome-validated."],
			"rank_score": 0.88,
		})

	if not structure.get("has_clear_paragraph_breaks"):
		recommendations.append({
			"category": "Shorten or reorganise a post",
			"action": "Separate the idea into shorter paragraphs or add a concise list.",
			"reason": "The draft lacks clear paragraph breaks.",
			"supporting_evidence": ["content.structure.has_clear_paragraph_breaks=false"],
			"qualification": "heuristic",
			"limitations": ["Readable structure depends on context and audience."],
			"rank_score": 0.82,
		})

	if float(dimension_scores.get("specificity", 0.0)) < 0.65:
		recommendations.append({
			"category": "Increase specificity",
			"action": "Replace one general claim with a concrete example, number, or constraint.",
			"reason": "The specificity dimension is below the current rubric threshold.",
			"supporting_evidence": [f"content.dimension_scores.specificity={dimension_scores.get('specificity')}"],
			"qualification": "heuristic",
			"limitations": ["Specificity is assessed statistically and should be reviewed for authenticity."],
			"rank_score": 0.86,
		})

	if not interaction.get("has_call_to_action"):
		recommendations.append({
			"category": "Improve the call to action",
			"action": "End with one explicit, relevant question or response request.",
			"reason": "No call-to-action language was detected.",
			"supporting_evidence": ["content.interaction.has_call_to_action=false"],
			"qualification": "heuristic",
			"limitations": ["An explicit CTA is not evidence that a response will occur."],
			"rank_score": 0.84,
		})

	if topic_relevance.get("status") == "heuristic" and topic_relevance.get("lexical_coverage") == 0:
		recommendations.append({
			"category": "Change the topic emphasis",
			"action": "Tie the draft more explicitly to the supplied topic or change the stated topic.",
			"reason": "Lexical overlap with the stated topic is zero.",
			"supporting_evidence": ["content.topic_relevance.lexical_coverage=0"],
			"qualification": "heuristic",
			"limitations": ["Lexical overlap is not semantic relevance; verify it manually."],
			"rank_score": 0.80,
		})

	if evidence["personal_reliable"]:
		personal = component_results["personal_baselines"]["result"]
		recommendations.append({
			"category": "Use a personal baseline",
			"action": "Compare this draft's format and topic with the account's qualified personal groups.",
			"reason": "Matched personal history is sufficient for account-level descriptive comparison.",
			"supporting_evidence": [{
				"type": "personal_baseline",
				"observed_post_count": personal["observed_post_count"],
			}],
			"qualification": "descriptive_personal",
			"limitations": ["Associations are not causal effects and public observations are excluded."],
			"rank_score": 0.75,
		})

	if evidence["engagement_reliable"]:
		recommendations.append({
			"category": "Test a different content format",
			"action": "Test one alternate format with explicit measurement rather than changing everything at once.",
			"reason": "Matched observed engagement rates exist but do not prove future outcomes.",
			"supporting_evidence": ["engagement_intelligence.evidence_status=descriptive_only"],
			"qualification": "descriptive_observed",
			"limitations": ["Observed engagement is descriptive context, not a success forecast."],
			"rank_score": 0.72,
		})

	rank_score_map = {id(item): item["rank_score"] for item in recommendations}
	recommendations.sort(key=lambda item: (-item["rank_score"], item["category"], item["action"]))
	for rank, item in enumerate(recommendations, start=1):
		item["rank"] = rank
		item["recommendation_id"] = _recommendation_id(item["category"], item["action"])

	return {
		"engine": "recommendation_ranking",
		"version": "1.0.0",
		"recommendation_count": len(recommendations),
		"recommendations": recommendations,
		"limitations": [
			"Rankings express rule relevance, not probability of success.",
			"No recommendation claims a causal performance effect.",
		],
		"strategy": strategy or {},
	}
