"""Validate and canonicalize LinkedIn post records before model training."""

import hashlib
import json
import math
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


SCHEMA_VERSION = "1.0"
NEAR_DUPLICATE_SIMILARITY_THRESHOLD = 0.92
_NEAR_DUPLICATE_MIN_LENGTH_RATIO = 0.8
_METRIC_NAMES = ("likes", "comments", "shares")
_CONTEXT_NAMES = ("impressions", "audience_size")


def _normalize_text(text: str) -> str:
	return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _parse_timestamp(value: Any, field: str, *, required: bool) -> Optional[str]:
	if value is None or value == "":
		if required:
			raise ValueError(f"{field} is required")
		return None
	if not isinstance(value, str):
		raise ValueError(f"{field} must be an ISO-8601 string")
	try:
		parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
	except ValueError as exc:
		raise ValueError(f"{field} must be a valid ISO-8601 timestamp") from exc
	if parsed.tzinfo is None:
		raise ValueError(f"{field} must include a timezone")
	return parsed.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _metric_count(engagement: Any, name: str) -> Optional[int]:
	if engagement is None:
		return None
	if not isinstance(engagement, dict):
		raise ValueError("engagement must be an object or null")
	value = engagement.get(name)
	if value is None:
		return None
	if isinstance(value, bool) or not isinstance(value, (int, float)):
		raise ValueError(f"engagement.{name} must be a nonnegative integer or null")
	if not math.isfinite(value) or value < 0 or int(value) != value:
		raise ValueError(f"engagement.{name} must be a nonnegative integer or null")
	return int(value)


def _context_number(record: Mapping[str, Any], engagement: Any, name: str) -> Optional[float]:
	value = record.get(name)
	if value is None and isinstance(engagement, dict):
		value = engagement.get(name)
	if value is None:
		return None
	if isinstance(value, bool) or not isinstance(value, (int, float)):
		raise ValueError(f"{name} must be a nonnegative number or null")
	value = float(value)
	if not math.isfinite(value) or value < 0:
		raise ValueError(f"{name} must be a nonnegative number or null")
	return value


def _canonicalize_record(record: Any, source_index: int) -> Dict[str, Any]:
	if not isinstance(record, dict):
		raise ValueError("post record must be an object")
	post_id = record.get("post_id")
	if not isinstance(post_id, str) or not post_id.strip():
		raise ValueError("post_id must be a nonempty string")
	text = record.get("text")
	if not isinstance(text, str) or not text.strip():
		raise ValueError("text must be a nonempty string")

	author = record.get("author")
	if author is not None and not isinstance(author, dict):
		raise ValueError("author must be an object or null")
	search_query = record.get("search_topic")
	if search_query is not None and not isinstance(search_query, str):
		raise ValueError("search_topic must be a string or null")

	engagement = record.get("engagement")
	context = {_context_name: _context_number(record, engagement, _context_name) for _context_name in _CONTEXT_NAMES}
	observation_window = record.get("observation_window_hours")
	if observation_window is not None:
		if isinstance(observation_window, bool) or not isinstance(observation_window, (int, float)):
			raise ValueError("observation_window_hours must be a nonnegative number or null")
		observation_window = float(observation_window)
		if not math.isfinite(observation_window) or observation_window < 0:
			raise ValueError("observation_window_hours must be a nonnegative number or null")
	quality_status = record.get("quality_status")
	if quality_status is not None and not isinstance(quality_status, str):
		raise ValueError("quality_status must be a string or null")

	return {
		"post_id": post_id.strip(),
		"text": text,
		"published_at": _parse_timestamp(record.get("published_at"), "published_at", required=True),
		"metrics_captured_at": _parse_timestamp(
			record.get("metrics_captured_at") or record.get("captured_at"), "metrics_captured_at", required=False
		),
		"prediction_at": _parse_timestamp(record.get("prediction_at"), "prediction_at", required=False),
		"outcome_at": _parse_timestamp(record.get("outcome_at"), "outcome_at", required=False),
		"observation_window_hours": observation_window,
		"quality_status": quality_status,
		"engagement": {name: _metric_count(engagement, name) for name in _METRIC_NAMES},
		**context,
		"author_id": author.get("id") if author else None,
		"author_name": author.get("name") if author else None,
		"author_type": author.get("type") if author else None,
		"search_query": search_query,
		"source_index": source_index,
	}


def assign_duplicate_groups(
	records: List[Dict[str, Any]],
	threshold: float = NEAR_DUPLICATE_SIMILARITY_THRESHOLD,
) -> Dict[str, int]:
	"""Add stable content-group IDs so copies can be kept in the same split.

	Exact matches use Unicode-normalized, case-folded, whitespace-collapsed text.
	Near matches use character n-gram cosine similarity and a length-ratio guard.
	"""
	if not 0 < threshold <= 1:
		raise ValueError("threshold must be in the range (0, 1]")
	if not records:
		return {"exact_duplicate_groups": 0, "near_duplicate_groups": 0}

	normalized = [_normalize_text(record["text"]) for record in records]
	parents = list(range(len(records)))

	def find(index: int) -> int:
		while parents[index] != index:
			parents[index] = parents[parents[index]]
			index = parents[index]
		return index

	def union(left: int, right: int) -> None:
		left_root, right_root = find(left), find(right)
		if left_root != right_root:
			parents[max(left_root, right_root)] = min(left_root, right_root)

	first_by_text: Dict[str, int] = {}
	text_counts: Dict[str, int] = {}
	exact_groups = 0
	for index, text in enumerate(normalized):
		if text in first_by_text:
			union(index, first_by_text[text])
			if text_counts[text] == 1:
				exact_groups += 1
			text_counts[text] += 1
		else:
			first_by_text[text] = index
			text_counts[text] = 1

	unique_indices = list(first_by_text.values())
	if len(unique_indices) > 1 and threshold < 1:
		unique_texts = [normalized[index] for index in unique_indices]
		vectors = TfidfVectorizer(
			analyzer="char_wb",
			ngram_range=(3, 5),
			dtype=np.float32,
			max_features=250_000,
		).fit_transform(unique_texts)
		neighbors = NearestNeighbors(
			radius=1 - threshold,
			metric="cosine",
			algorithm="brute",
			n_jobs=1,
		).fit(vectors)
		distances, candidates = neighbors.radius_neighbors(vectors, return_distance=True)
		for local_index, (row_distances, row_candidates) in enumerate(zip(distances, candidates)):
			for distance, candidate in zip(row_distances, row_candidates):
				candidate = int(candidate)
				if candidate <= local_index:
					continue
				left_text, right_text = unique_texts[local_index], unique_texts[candidate]
				length_ratio = min(len(left_text), len(right_text)) / max(len(left_text), len(right_text))
				if length_ratio >= _NEAR_DUPLICATE_MIN_LENGTH_RATIO and 1 - float(distance) >= threshold:
					union(unique_indices[local_index], unique_indices[candidate])

	groups: Dict[int, List[int]] = {}
	for index in range(len(records)):
		groups.setdefault(find(index), []).append(index)

	near_groups = 0
	for indices in groups.values():
		texts = {normalized[index] for index in indices}
		if len(indices) > 1 and len(texts) == 1:
			kind = "exact_duplicate"
		elif len(indices) > 1:
			kind = "near_duplicate"
			near_groups += 1
		else:
			kind = "unique"
		group_seed = min(texts)
		group_id = "content-" + hashlib.sha256(group_seed.encode("utf-8")).hexdigest()[:16]
		for index in indices:
			records[index]["duplicate_group_id"] = group_id
			records[index]["duplicate_kind"] = kind

	return {"exact_duplicate_groups": exact_groups, "near_duplicate_groups": near_groups}


def prepare_dataset(
	input_path: str,
	output_path: str,
	*,
	rejected_path: Optional[str] = None,
	near_duplicate_threshold: float = NEAR_DUPLICATE_SIMILARITY_THRESHOLD,
) -> Dict[str, Any]:
	"""Write a canonical dataset and a separate report for rejected records."""
	source_path = Path(input_path)
	with source_path.open("r", encoding="utf-8") as source_file:
		posts = json.load(source_file)
	if not isinstance(posts, list):
		raise ValueError("Expected the source JSON to contain a list of posts")

	records: List[Dict[str, Any]] = []
	rejected: List[Dict[str, Any]] = []
	seen_ids = set()
	for source_index, post in enumerate(posts):
		try:
			canonical = _canonicalize_record(post, source_index)
			if canonical["post_id"] in seen_ids:
				raise ValueError(f"duplicate post_id: {canonical['post_id']}")
			seen_ids.add(canonical["post_id"])
			records.append(canonical)
		except ValueError as exc:
			rejected.append({
				"source_index": source_index,
				"post_id": post.get("post_id") if isinstance(post, dict) else None,
				"reason": str(exc),
			})

	duplicate_summary = assign_duplicate_groups(records, near_duplicate_threshold)
	missing_metric_values = sum(
		record["engagement"][name] is None for record in records for name in _METRIC_NAMES
	)
	prepared_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
	canonical_data = {
		"schema_version": SCHEMA_VERSION,
		"prepared_at": prepared_at,
		"source_file": source_path.name,
		"source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
		"near_duplicate_similarity_threshold": near_duplicate_threshold,
		"records": records,
		"summary": {
			"source_records": len(posts),
			"canonical_records": len(records),
			"rejected_records": len(rejected),
			"records_missing_metrics_capture_time": sum(
				record["metrics_captured_at"] is None for record in records
			),
			"missing_metric_values": missing_metric_values,
			**duplicate_summary,
		},
	}

	canonical_path = Path(output_path)
	canonical_path.parent.mkdir(parents=True, exist_ok=True)
	canonical_path.write_text(json.dumps(canonical_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	if rejected_path is None:
		rejected_path = str(canonical_path.with_name(f"{canonical_path.stem}_rejected{canonical_path.suffix}"))
	rejected_file = Path(rejected_path)
	rejected_file.parent.mkdir(parents=True, exist_ok=True)
	rejected_file.write_text(json.dumps(rejected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	return canonical_data