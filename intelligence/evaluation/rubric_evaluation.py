"""Deterministic agreement metrics for a versioned content rubric."""

from collections import defaultdict
from statistics import mean
from typing import Dict, Iterable, Mapping, Sequence


def evaluate_rubric(
	algorithm_scores: Mapping[str, Mapping[str, float]],
	human_scores: Mapping[str, Sequence[float]],
	dimension: str,
) -> Dict[str, float | int | None]:
	"""Compare per-post algorithm scores with human ratings in the same 0-100 scale."""
	common_ids = sorted(set(algorithm_scores) & set(human_scores))
	if not common_ids:
		return {
			"posts_evaluated": 0,
			"algorithm_human_mae": None,
			"mean_human_score": None,
			"score_stability": None,
		}

	algorithm_values = []
	human_mean_values = []
	human_variances = []
	for post_id in common_ids:
		algorithm_dimensions = algorithm_scores[post_id]
		algorithm_value = algorithm_dimensions.get(dimension)
		if algorithm_value is None:
			continue
		ratings = [rating for rating in human_scores[post_id] if isinstance(rating, (int, float))]
		if not ratings:
			continue
		human_mean = mean(ratings)
		algorithm_values.append(float(algorithm_value))
		human_mean_values.append(human_mean)
		human_variances.append(max(ratings) - min(ratings) if len(ratings) > 1 else 0.0)

	if not algorithm_values:
		return {
			"posts_evaluated": 0,
			"algorithm_human_mae": None,
			"mean_human_score": None,
			"score_stability": None,
		}

	mae = mean(abs(algorithm_value - human_value) for algorithm_value, human_value in zip(algorithm_values, human_mean_values))
	return {
		"posts_evaluated": len(algorithm_values),
		"algorithm_human_mae": round(mae, 3),
		"mean_human_score": round(mean(human_mean_values), 3),
		"score_stability": round(max(0.0, 1.0 - mean(human_variances) / 100.0), 3),
	}
