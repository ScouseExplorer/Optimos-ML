"""Descriptive analysis of observed post performance.

Posts are mappings with a unique, nonempty string ``post_id`` and ``content``.
Performance records use the same ``post_id`` and may contain ``likes``,
``comments``, ``reposts`` (or ``shares``), ``impressions``, and ``followers``
(or ``follower_count``). Rates are fractions, not percentages. A rate is
available only when all three engagement counts and its positive denominator
are present. Unmatched records are counted but never paired by position.
"""

from collections import defaultdict
from math import isfinite
from numbers import Real
from statistics import mean, median
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence


def _classify_post(content: str) -> Dict[str, str]:
	from api.app import _classify_format, _classify_hook, _classify_topic, extract_nlp_features

	_, features = extract_nlp_features(content)
	return {
		"topic": _classify_topic(features, content)["topic"],
		"format": _classify_format(features, content)["format"],
		"hook": _classify_hook(content)["hook_type"],
	}


def _number(record: Mapping[str, Any], *names: str) -> Optional[float]:
	for name in names:
		value = record.get(name)
		if value is not None:
			if isinstance(value, bool) or not isinstance(value, Real):
				return None
			number = float(value)
			return number if isfinite(number) and number >= 0 else None
	return None


def _percentile(values: List[float], percentile: float) -> Optional[float]:
	if not values:
		return None
	ordered = sorted(values)
	position = (len(ordered) - 1) * percentile
	lower = int(position)
	upper = min(lower + 1, len(ordered) - 1)
	fraction = position - lower
	return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _distribution(values: Sequence[Optional[float]]) -> Dict[str, Any]:
	valid = [float(value) for value in values if value is not None]
	if not valid:
		return {
			"count": 0,
			"mean": None,
			"median": None,
			"p25": None,
			"p75": None,
			"p90": None,
			"min": None,
			"max": None,
		}
	return {
		"count": len(valid),
		"mean": round(mean(valid), 6),
		"median": round(median(valid), 6),
		"p25": round(_percentile(valid, 0.25), 6),
		"p75": round(_percentile(valid, 0.75), 6),
		"p90": round(_percentile(valid, 0.90), 6),
		"min": round(min(valid), 6),
		"max": round(max(valid), 6),
	}


def _content_length_bucket(content: str) -> str:
	word_count = len(content.split())
	if word_count < 100:
		return "short_under_100"
	if word_count <= 300:
		return "medium_100_300"
	return "long_over_300"


def _index_records(records: Sequence[Mapping[str, Any]], label: str) -> Dict[str, Mapping[str, Any]]:
	indexed = {}
	for record in records:
		post_id = record.get("post_id")
		if not isinstance(post_id, str) or not post_id.strip():
			raise ValueError(f"Each {label} must have a nonempty string post_id")
		if post_id in indexed:
			raise ValueError(f"Duplicate {label} post_id: {post_id}")
		indexed[post_id] = record
	return indexed


def analyze_personal_performance(
	historical_posts: Sequence[Mapping[str, Any]],
	performance: Optional[Sequence[Mapping[str, Any]]] = None,
	*,
	classifier: Optional[Callable[[str], Mapping[str, str]]] = None,
) -> Dict[str, Any]:
	"""Summarize matched observed outcomes without ranking or inferred performance.

	``classifier`` can supply topic, format, and hook labels; by default the
	existing API classifiers and NLP feature extractor are used. The default
	requires the API's NLP extractor to be initialized.
	"""
	posts = _index_records(historical_posts, "post")
	metrics_by_id = _index_records(performance or [], "performance record")
	classify = classifier or _classify_post
	groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
		"topic": defaultdict(list),
		"format": defaultdict(list),
		"hook": defaultdict(list),
		"author_type": defaultdict(list),
		"content_length": defaultdict(list),
	}
	observations = []

	for post_id, post in posts.items():
		content = post.get("content")
		if not isinstance(content, str) or not content.strip():
			raise ValueError(f"Post {post_id} must have nonempty content")
		labels = classify(content)
		metrics = metrics_by_id.get(post_id, {})
		counts = {
			"likes": _number(metrics, "likes"),
			"comments": _number(metrics, "comments"),
			"reposts": _number(metrics, "reposts", "shares"),
		}
		denominators = {
			"impressions": _number(metrics, "impressions"),
			"followers": _number(metrics, "followers", "follower_count"),
		}
		total = sum(counts.values()) if all(value is not None for value in counts.values()) else None
		rates = {
			f"engagement_rate_by_{name}": total / value if total is not None and value is not None and value > 0 else None
			for name, value in denominators.items()
		}
		author_type = post.get("author_type")
		if not isinstance(author_type, str) or not author_type.strip():
			author_type = metrics.get("author_type")
		if not isinstance(author_type, str) or not author_type.strip():
			author_type = "unknown"
		content_length = _content_length_bucket(content)
		published_at = post.get("published_at") or metrics.get("published_at")
		captured_at = metrics.get("captured_at") or metrics.get("metrics_captured_at")
		snapshot_id = metrics.get("snapshot_id") or post.get("snapshot_id")
		source = metrics.get("source") or post.get("source") or "unspecified"
		observation = {
			"post_id": post_id,
			"metrics_matched": post_id in metrics_by_id,
			"metrics": {**counts, **denominators},
			"engagement_total": total,
			"author_type": author_type,
			"content_length": content_length,
			"published_at": published_at,
			"captured_at": captured_at,
			"snapshot_id": snapshot_id,
			"source": source,
			**rates,
		}
		observations.append(observation)
		for dimension in ("topic", "format", "hook"):
			groups[dimension][labels[dimension]].append(observation)
		groups["author_type"][author_type].append(observation)
		groups["content_length"][content_length].append(observation)

	def summarize(items: List[Dict[str, Any]]) -> Dict[str, Any]:
		summary: Dict[str, Any] = {
			"sample_size": len(items),
			"matched_metrics_sample_size": sum(item["metrics_matched"] for item in items),
		}
		metric_distributions = {
			"likes": _distribution([item["metrics"]["likes"] for item in items]),
			"comments": _distribution([item["metrics"]["comments"] for item in items]),
			"shares": _distribution([item["metrics"]["reposts"] for item in items]),
			"engagement_total": _distribution([item["engagement_total"] for item in items]),
		}
		summary["metric_distributions"] = metric_distributions
		for rate in ("engagement_rate_by_impressions", "engagement_rate_by_followers"):
			values = [item[rate] for item in items if item[rate] is not None]
			summary[rate] = sum(values) / len(values) if values else None
			summary[f"{rate}_sample_size"] = len(values)
			summary[f"{rate}_distribution"] = _distribution(values)
		rate_sample_size = max(summary["engagement_rate_by_impressions_sample_size"],
							   summary["engagement_rate_by_followers_sample_size"])
		summary["evidence_status"] = "no_observed_rates" if rate_sample_size == 0 else "descriptive_only"

		published_dates = [item["published_at"] for item in items if item["published_at"]]
		captured_dates = [item["captured_at"] for item in items if item["captured_at"]]
		summary["date_range"] = {
			"published_at_min": min(published_dates) if published_dates else None,
			"published_at_max": max(published_dates) if published_dates else None,
			"captured_at_min": min(captured_dates) if captured_dates else None,
			"captured_at_max": max(captured_dates) if captured_dates else None,
		}
		sources = sorted({str(item["source"]) for item in items if item["source"]})
		summary["sources"] = sources
		if items:
			missing_data = {}
			fields = ("likes", "comments", "reposts", "impressions", "followers")
			for field in fields:
				missing = sum(item["metrics"][field] is None for item in items)
				missing_data[field] = {
					"missing_count": missing,
					"missing_proportion": round(missing / len(items), 6),
				}
			summary["missing_data"] = missing_data
		else:
			summary["missing_data"] = {}
		return summary

	overall = summarize(observations)
	captured_count = sum(observation["captured_at"] is not None for observation in observations)
	observation_analysis = {
		"status": "not_available",
		"captured_observation_count": captured_count,
		"missing_capture_count": len(observations) - captured_count,
		"limitations": [
			"Repeated metric snapshots are required to calculate engagement growth.",
			"Reliable captured_at values are required to calculate performance at a specific post age.",
		],
	}
	return {
		**overall,
		"observation_analysis": observation_analysis,
		"unmatched_performance_sample_size": len(metrics_by_id.keys() - posts.keys()),
		"observations": observations,
		"groups": {
			dimension: {label: summarize(items) for label, items in labels.items()}
			for dimension, labels in groups.items()
		},
	}


class EngagementIntelligenceEngine:
	"""Summarize matched observed outcomes as descriptive evidence."""

	def analyze(
		self,
		historical_posts: Sequence[Mapping[str, Any]],
		performance: Optional[Sequence[Mapping[str, Any]]] = None,
		*,
		classifier: Optional[Callable[[str], Mapping[str, str]]] = None,
	) -> Dict[str, Any]:
		result = analyze_personal_performance(historical_posts, performance, classifier=classifier)
		result["engine"] = "engagement_intelligence"
		result["interpretation"] = "descriptive_context_only"
		return result
