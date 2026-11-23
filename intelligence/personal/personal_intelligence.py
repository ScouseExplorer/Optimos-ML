"""Statistical baselines from explicitly supplied user-specific observations."""

from datetime import datetime
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


MIN_PERSONAL_BASELINE_SAMPLES = 5


def _number(value: Any) -> Optional[float]:
	if value is None or isinstance(value, bool):
		return None
	if not isinstance(value, (int, float)):
		return None
	number = float(value)
	return number if number >= 0 else None


def _distribution(values: Iterable[Optional[float]]) -> Dict[str, Any]:
	valid = [float(value) for value in values if value is not None]
	if not valid:
		return {"count": 0, "mean": None, "median": None}
	ordered = sorted(valid)
	position = (len(ordered) - 1) * 0.75
	lower = int(position)
	upper = min(lower + 1, len(ordered) - 1)
	fraction = position - lower
	p75 = ordered[lower] * (1 - fraction) + ordered[upper] * fraction
	return {"count": len(valid), "mean": round(mean(valid), 6), "median": round(median(valid), 6), "p75": round(p75, 6)}


def _published_hour(post: Mapping[str, Any], performance: Mapping[str, Any]) -> Optional[str]:
	value = post.get("published_at") or performance.get("published_at")
	if not isinstance(value, str) or not value:
		return None
	try:
		parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
		return f"{parsed.hour:02d}:00-{parsed.hour:02d}:59"
	except ValueError:
		return None


def _content_length(content: str) -> str:
	count = len(content.split())
	if count < 100:
		return "short_under_100"
	if count <= 300:
		return "medium_100_300"
	return "long_over_300"


def analyze_personal_baselines(
	account_id: str,
	historical_posts: Sequence[Mapping[str, Any]],
	performance: Sequence[Mapping[str, Any]],
	*,
	classify: Optional[Any] = None,
) -> Dict[str, Any]:
	"""Calculate account-owned baselines only from ID-matched personal observations."""
	if not isinstance(account_id, str) or not account_id.strip():
		raise ValueError("account_id must be a nonempty application account identifier")

	posts_by_id: Dict[str, Mapping[str, Any]] = {}
	for post in historical_posts:
		record_account_id = post.get("account_id")
		if record_account_id is not None and record_account_id != account_id:
			raise ValueError("Personal post belongs to a different account")
		post_id = post.get("post_id")
		if not isinstance(post_id, str) or not post_id.strip():
			raise ValueError("Each personal post must have a nonempty post_id")
		if post_id in posts_by_id:
			raise ValueError(f"Duplicate personal post_id: {post_id}")
		posts_by_id[post_id] = post

	performance_by_id: Dict[str, Mapping[str, Any]] = {}
	for record in performance:
		record_account_id = record.get("account_id")
		if record_account_id is not None and record_account_id != account_id:
			raise ValueError("Personal performance record belongs to a different account")
		post_id = record.get("post_id")
		if not isinstance(post_id, str) or not post_id.strip():
			raise ValueError("Each personal performance record must have a nonempty post_id")
		if post_id in performance_by_id:
			raise ValueError(f"Duplicate personal performance post_id: {post_id}")
		performance_by_id[post_id] = record

	observations = []
	for post_id, post in posts_by_id.items():
		metrics = performance_by_id.get(post_id)
		if metrics is None:
			continue
		content = post.get("content")
		if not isinstance(content, str) or not content.strip():
			raise ValueError(f"Personal post {post_id} must have nonempty content")
		likes = _number(metrics.get("likes"))
		comments = _number(metrics.get("comments"))
		shares = _number(metrics.get("reposts", metrics.get("shares")))
		impressions = _number(metrics.get("impressions"))
		engagement_total = sum(value for value in (likes, comments, shares) if value is not None)
		if likes is None or comments is None or shares is None:
			engagement_total = None
		engagement_rate = None
		if engagement_total is not None and impressions is not None and impressions > 0:
			engagement_rate = engagement_total / impressions * 100

		labels = classify(content) if classify else {"topic": "unclassified", "format": "unclassified", "hook": "unclassified"}
		observations.append({
			"post_id": post_id,
			"likes": likes,
			"comments": comments,
			"shares": shares,
			"engagement_total": engagement_total,
			"engagement_rate": engagement_rate,
			"topic": labels["topic"],
			"format": labels["format"],
			"hook": labels["hook"],
			"posting_hour": _published_hour(post, metrics),
			"content_length": _content_length(content),
			"source": "personal_import",
		})

	if len(observations) < MIN_PERSONAL_BASELINE_SAMPLES:
		return {
			"evidence_status": "insufficient_personal_history",
			"account_id": account_id,
			"observed_post_count": len(observations),
			"minimum_observation_count": MIN_PERSONAL_BASELINE_SAMPLES,
			"baselines": {},
			"group_comparisons": {},
			"limitations": [
				"No personal baselines were ranked until at least five matched observations exist.",
				"Public observations are not personal evidence.",
			],
		}

	metric_names = ("likes", "comments", "shares", "engagement_total", "engagement_rate")
	baselines = {name: _distribution([item[name] for item in observations]) for name in metric_names}

	def group_distributions(dimension: str) -> Dict[str, Any]:
		groups: Dict[str, List[Dict[str, Any]]] = {}
		for item in observations:
			key = str(item[dimension])
			groups.setdefault(key, []).append(item)
		result = {}
		for key, items in groups.items():
			eligible = len(items) >= MIN_PERSONAL_BASELINE_SAMPLES
			result[key] = {
				"sample_size": len(items),
				"eligible_for_comparison": eligible,
				"likes": _distribution([item["likes"] for item in items]),
				"comments": _distribution([item["comments"] for item in items]),
				"shares": _distribution([item["shares"] for item in items]),
				"engagement_total": _distribution([item["engagement_total"] for item in items]),
				"engagement_rate": _distribution([item["engagement_rate"] for item in items]),
			}
		return result

	return {
		"evidence_status": "sufficient_personal_history",
		"account_id": account_id,
		"observed_post_count": len(observations),
		"minimum_observation_count": MIN_PERSONAL_BASELINE_SAMPLES,
		"baselines": baselines,
		"group_comparisons": {
			"topic": group_distributions("topic"),
			"format": group_distributions("format"),
			"hook": group_distributions("hook"),
			"posting_hour": group_distributions("posting_hour"),
			"content_length": group_distributions("content_length"),
		},
		"limitations": [
			"Simple baselines describe associations, not causal effects.",
			"Public observations are not included.",
			"Small groups are qualified rather than ranked.",
		],
	}
