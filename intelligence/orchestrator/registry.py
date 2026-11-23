"""Component registry and adapters over the existing intelligence engines."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional

from intelligence.orchestrator.schemas import (
	STATUS_SUCCESS,
	STATUS_UNAVAILABLE,
	AnalysisRequest,
	ComponentOutput,
)
from intelligence.personal.recommendations import recommend_strategy
from intelligence.personal.personal_intelligence import analyze_personal_baselines
from intelligence.personal.ranking import rank_recommendations

Handler = Callable[[AnalysisRequest], ComponentOutput]
Applicability = Callable[[AnalysisRequest], bool]


@dataclass(frozen=True)
class RegisteredComponent:
	name: str
	version: str
	handler: Handler
	applies: Applicability


class ComponentRegistry:
	def __init__(self) -> None:
		self._components: Dict[str, RegisteredComponent] = {}

	def register(self, name: str, version: str, handler: Handler, applies: Applicability) -> None:
		if name in self._components:
			raise ValueError(f"Component already registered: {name}")
		self._components[name] = RegisteredComponent(name, version, handler, applies)

	def get(self, name: str) -> RegisteredComponent:
		return self._components[name]

	def names(self) -> List[str]:
		return list(self._components)

	def all(self) -> List[RegisteredComponent]:
		return list(self._components.values())


def _has_identified_posts(request: AnalysisRequest) -> bool:
	return any(isinstance(post, Mapping) for post in request.historical_posts)


def build_default_registry(
	*,
	content_engine: Optional[Any],
	engagement_engine: Any,
	prediction_engine: Any,
	classifier: Optional[Callable[[str], Mapping[str, str]]] = None,
) -> ComponentRegistry:
	registry = ComponentRegistry()

	def content(request: AnalysisRequest) -> ComponentOutput:
		if content_engine is None:
			return ComponentOutput(
				result={},
				limitations=["The content effectiveness engine is not initialized."],
				status=STATUS_UNAVAILABLE,
			)
		assessment = content_engine.evaluate(request.content, request.intended_topic)
		return ComponentOutput(
			result=assessment,
			evidence=[{"type": "text_characteristics", "source": "submitted_content"}],
			limitations=[
				"The content score is a heuristic rubric, not a performance prediction.",
				"Topic relevance is lexical overlap, not semantic understanding.",
			],
		)

	def engagement(request: AnalysisRequest) -> ComponentOutput:
		identified = [post for post in request.historical_posts if isinstance(post, Mapping)]
		result = engagement_engine.analyze(
			identified, list(request.historical_performance), classifier=classifier
		)
		limitations = ["Observed engagement is descriptive context, not a forecast."]
		if result["evidence_status"] == "no_observed_rates":
			limitations.append("No complete observed engagement rates were supplied.")
		return ComponentOutput(
			result=result,
			evidence=[{
				"type": "observed_engagement",
				"sample_size": result["sample_size"],
				"matched_metrics_sample_size": result["matched_metrics_sample_size"],
			}],
			limitations=limitations,
		)

	def prediction(request: AnalysisRequest) -> ComponentOutput:
		readiness = prediction_engine.readiness()
		ready = readiness["status"] == "ready"
		return ComponentOutput(
			result=readiness,
			limitations=[] if ready else [
				f"Forecast unavailable; missing requirement: {item}"
				for item in readiness["missing_requirements"]
			],
			status=STATUS_SUCCESS if ready else STATUS_UNAVAILABLE,
		)

	def personal_baselines(request: AnalysisRequest) -> ComponentOutput:
		if not request.account_id:
			return ComponentOutput(
				result={"evidence_status": "account_id_required", "account_id": None},
				limitations=["An application account ID is required for personal baselines."],
				status=STATUS_UNAVAILABLE,
			)
		result = analyze_personal_baselines(
			account_id=request.account_id,
			historical_posts=list(request.historical_posts),
			performance=list(request.historical_performance),
			classify=classifier,
		)
		limitations = [
			"Personal baselines describe associations, not causal effects.",
			"Public observations are not included.",
		]
		if result["evidence_status"] == "insufficient_personal_history":
			limitations.append("Five matched personal observations are required before baselines are ranked.")
		return ComponentOutput(result=result, evidence=[{
			"type": "personal_baseline",
			"account_id": request.account_id,
			"observed_post_count": result["observed_post_count"],
		}], limitations=limitations)

	def ranked_recommendations(request: AnalysisRequest) -> ComponentOutput:
		content = {}
		if content_engine is not None and request.content:
			try:
				content = content_engine.evaluate(request.content, request.intended_topic)
			except Exception as exc:
				return ComponentOutput(
					result={"recommendation_count": 0, "recommendations": []},
					limitations=["Ranking unavailable because content analysis failed."],
					status=STATUS_UNAVAILABLE,
				)

		component_results: Dict[str, Any] = {}
		if request.account_id:
			baseline_result = analyze_personal_baselines(
				account_id=request.account_id,
				historical_posts=list(request.historical_posts),
				performance=list(request.historical_performance),
				classify=classifier,
			)
			component_results["personal_baselines"] = {"result": baseline_result}
		if _has_identified_posts(request):
			observed_result = engagement_engine.analyze(
				[post for post in request.historical_posts if isinstance(post, Mapping)],
				list(request.historical_performance),
				classifier=classifier,
			)
			component_results["engagement_intelligence"] = {"result": observed_result}

		strategy = recommend_strategy(
			profile=request.profile,
			goal=request.goal,
			historical_posts=list(request.historical_posts),
			historical_performance=list(request.historical_performance),
			global_evidence=list(request.global_evidence),
		)
		result = rank_recommendations(content, component_results, strategy)
		return ComponentOutput(
			result=result,
			evidence=[{"type": "recommendation_ranking", "recommendation_count": result["recommendation_count"]}],
			limitations=result["limitations"],
		)

	def recommendations(request: AnalysisRequest) -> ComponentOutput:
		result = recommend_strategy(
			profile=request.profile,
			goal=request.goal,
			historical_posts=list(request.historical_posts),
			historical_performance=list(request.historical_performance),
			global_evidence=list(request.global_evidence),
		)
		return ComponentOutput(
			result=result,
			evidence=[{"type": "recommendation_provenance", **result["evidence_provenance"]}],
			limitations=["Recommendations are descriptive and are not proof of future results."],
		)

	registry.register("content_effectiveness", "1.0.0", content, lambda r: r.content is not None)
	registry.register("engagement_intelligence", "1.0.0", engagement, _has_identified_posts)
	registry.register(
		"recommendation_ranking",
		"1.0.0",
		ranked_recommendations,
		lambda r: r.content is not None or r.account_id is not None,
	)
	registry.register("personal_baselines", "1.0.0", personal_baselines, lambda r: r.account_id is not None)
	registry.register("performance_prediction", "0.1.0", prediction, lambda r: r.content is not None)
	registry.register(
		"recommendations",
		"1.0.0",
		recommendations,
		lambda r: bool(r.profile or r.goal or r.historical_posts or r.global_evidence),
	)
	return registry
