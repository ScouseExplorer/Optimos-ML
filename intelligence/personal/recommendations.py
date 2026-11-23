"""Deterministic, evidence-aware content strategy recommendations.

Profile fields may include ``expertise``, ``experiences``, ``role``,
``industry``, and ``target_audience``. Expertise and experiences can be a
string or a list of strings. Historical posts can be strings or mappings with
``post_id`` and ``content``; only ID-bearing posts can be matched to the
separate performance records. See ``personal_intelligence`` for metric fields.

Global evidence is a sequence of mappings with a nonempty ``source``, positive
integer ``sample_size``, and optional ``topic`` and ``summary``. A source and
sample size document provenance, not the quality or representativeness of a
study. Personal comparisons require at least five valid rates in each of two
groups for the same denominator. This threshold is a conservative eligibility
rule, not a statistical confidence guarantee.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from intelligence.performance.engagement_intelligence import analyze_personal_performance


MIN_GROUP_SAMPLE = 5


def _text(value: Any) -> Optional[str]:
	return value.strip() if isinstance(value, str) and value.strip() else None


def _texts(value: Any) -> List[str]:
	values = value if isinstance(value, (list, tuple)) else [value]
	return [text for item in values if (text := _text(item)) is not None]


def _unclassified(content: str) -> Dict[str, str]:
	return {"topic": "unclassified", "format": "unclassified", "hook": "unclassified"}


def _personal_choice(groups: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
	for metric in ("engagement_rate_by_impressions", "engagement_rate_by_followers"):
		eligible = [
			(name, group) for name, group in groups.items()
			if group[f"{metric}_sample_size"] >= MIN_GROUP_SAMPLE and group[metric] is not None
		]
		if len(eligible) < 2:
			continue
		eligible.sort(key=lambda item: (-item[1][metric], item[0]))
		best, comparison = eligible[0], eligible[1]
		if best[1][metric] == comparison[1][metric]:
			continue
		return {
			"name": best[0],
			"metric": metric,
			"mean_rate": best[1][metric],
			"sample_size": best[1][f"{metric}_sample_size"],
			"comparison": {
				"name": comparison[0],
				"mean_rate": comparison[1][metric],
				"sample_size": comparison[1][f"{metric}_sample_size"],
			},
		}
	return None


def recommend_strategy(
	profile: Optional[Mapping[str, Any]] = None,
	goal: Optional[str] = None,
	historical_posts: Optional[Sequence[Union[str, Mapping[str, Any]]]] = None,
	historical_performance: Optional[Sequence[Mapping[str, Any]]] = None,
	global_evidence: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
	"""Return a strategy and its provenance without inventing user facts.

	Bare historical strings are usable as content references, but never as
	performance evidence. Global evidence is supplied context, not a trend or
	a verified performance prediction.
	"""
	profile = profile or {}
	goal_text = _text(goal)
	expertise = _texts(profile.get("expertise"))
	experiences = _texts(profile.get("experiences")) or _texts(profile.get("experience"))
	audience = _text(profile.get("target_audience")) or _text(profile.get("audience"))
	posts = historical_posts or []
	identified_posts = [post for post in posts if isinstance(post, Mapping)]
	history_references = []
	for post in posts:
		content = _text(post if isinstance(post, str) else post.get("content"))
		if content:
			history_references.append({
				"source": "historical_content",
				"post_id": post.get("post_id") if isinstance(post, Mapping) else None,
				"content": content,
			})

	performance = historical_performance or []
	personal = analyze_personal_performance(identified_posts, performance, classifier=_unclassified)
	has_comparable_samples = max(
		personal["engagement_rate_by_impressions_sample_size"],
		personal["engagement_rate_by_followers_sample_size"],
	) >= 2 * MIN_GROUP_SAMPLE
	if has_comparable_samples:
		personal = analyze_personal_performance(identified_posts, performance)

	usable_global = [
		record for record in (global_evidence or [])
		if _text(record.get("source"))
		and isinstance(record.get("sample_size"), int)
		and not isinstance(record.get("sample_size"), bool)
		and record["sample_size"] > 0
	]
	global_status = "supplied" if usable_global else (
		"insufficient_provenance" if global_evidence else "unavailable"
	)

	choices = {}
	if has_comparable_samples:
		for dimension in ("topic", "format", "hook"):
			choice = _personal_choice(personal["groups"][dimension])
			if choice:
				choices[dimension] = choice

	topic = expertise[0] if expertise else None
	topic_source = "profile_expertise" if topic else None
	if not topic and experiences:
		topic, topic_source = experiences[0], "profile_experience"
	if not topic:
		topic = _text(profile.get("industry")) or _text(profile.get("role"))
		if topic:
			topic_source = "profile"
	if not topic and "topic" in choices:
		topic, topic_source = choices["topic"]["name"], "personal_outcomes"
	if not topic:
		for record in usable_global:
			topic = _text(record.get("topic"))
			if topic:
				topic_source = "global_evidence"
				break

	used_choices = {dimension: choice for dimension, choice in choices.items()
					if dimension != "topic" or topic_source == "personal_outcomes"}
	post_format = used_choices.get("format", {}).get("name") or (
		"personal story" if experiences else "educational post"
	)
	hook_strategy = used_choices.get("hook", {}).get("name") or (
		"personal experience" if experiences else "direct claim"
	)
	evidence_to_use = [{"source": "profile_experience", "text": item} for item in experiences]
	evidence_to_use.extend(history_references)
	evidence_to_use.extend({
		"source": "global_evidence", "provenance": record["source"],
		"sample_size": record["sample_size"], "topic": _text(record.get("topic")),
		"summary": _text(record.get("summary")),
	} for record in usable_global)
	questions = []
	if not topic:
		questions.append("What topic or area of expertise should this post focus on?")
	if not experiences:
		questions.append("What relevant experience can you share?")
	questions.append("What specific outcome or evidence can you substantiate?")
	if not audience:
		questions.append("Who is the intended audience?")
	if not goal_text:
		questions.append("What is the goal of the post?")
	if global_status == "insufficient_provenance":
		questions.append("What are the source and sample size for the global evidence?")

	reasons = []
	if topic_source == "profile_expertise":
		reasons.append("The topic comes from the supplied expertise.")
	elif topic_source == "profile_experience":
		reasons.append("The topic comes from a supplied personal experience.")
	elif topic_source == "profile":
		reasons.append("The topic comes from the supplied industry or role.")
	elif topic_source == "personal_outcomes":
		reasons.append("The topic reflects a descriptive comparison of matched personal outcomes.")
	elif topic_source == "global_evidence":
		reasons.append("The topic comes from explicitly supplied global evidence.")
	if experiences:
		reasons.append("The strategy uses a supplied personal experience.")
	if goal_text:
		reasons.append(f"The supplied goal is: {goal_text}")
	if used_choices:
		reasons.append("Selected personal patterns are descriptive, not proof of causation or future results.")
	if global_status == "unavailable":
		reasons.append("Global evidence was not supplied.")

	evidence_level = "descriptive_personal" if used_choices else (
		"supplied_global" if topic_source == "global_evidence" else
		"profile_based" if expertise or experiences or goal_text or profile else "insufficient_information"
	)
	return {
		"topic": topic,
		"angle": (("Share a relevant personal experience and its lesson" if experiences else
			  "Explain a useful idea related to the topic") +
			  (f" in support of the goal: {goal_text}" if goal_text else "")) if topic else None,
		"format": post_format,
		"hook_strategy": hook_strategy,
		"outline": ["Open with the chosen hook", "Introduce the relevant context",
					"Describe the experience or idea", "Share a substantiated outcome or lesson",
					"Close in a way that supports the goal"],
		"evidence_to_use": evidence_to_use,
		"expected_audience": audience,
		"reasons": reasons,
		"things_to_avoid": ["Invented experiences or outcomes", "Unsupported claims",
							"Treating descriptive patterns as guaranteed results"],
		"questions": questions,
		"evidence_level": evidence_level,
		"evidence_provenance": {
			"topic_source": topic_source,
			"personal": {
				"matched_metrics_sample_size": personal["matched_metrics_sample_size"],
				"engagement_rate_by_impressions_sample_size": personal["engagement_rate_by_impressions_sample_size"],
				"engagement_rate_by_followers_sample_size": personal["engagement_rate_by_followers_sample_size"],
				"unmatched_performance_sample_size": personal["unmatched_performance_sample_size"],
				"minimum_group_sample_size": MIN_GROUP_SAMPLE,
				"selected_comparisons": used_choices,
			},
			"global": {"status": global_status, "records": [
				{"source": record["source"], "sample_size": record["sample_size"],
				 "topic": _text(record.get("topic")), "summary": _text(record.get("summary"))}
				for record in usable_global
			]},
		},
	}
