"""Validates a request, runs the relevant components and combines their results."""

import logging
from time import perf_counter
from typing import Any, Dict, List, Tuple

from intelligence.orchestrator.registry import ComponentRegistry, RegisteredComponent
from intelligence.orchestrator.result_builder import build_response, component_result, error_result
from intelligence.orchestrator.schemas import AnalysisRequest, ComponentResult

logger = logging.getLogger("linkedin_ml_service")


class IntelligenceOrchestrator:
	def __init__(self, registry: ComponentRegistry) -> None:
		self.registry = registry

	def analyze(self, request: AnalysisRequest) -> Dict[str, Any]:
		self._validate(request)
		selected, skipped = self._select(request)
		return build_response([self._run(component, request) for component in selected], skipped)

	def _validate(self, request: AnalysisRequest) -> None:
		if request.content is not None and (not isinstance(request.content, str) or not request.content.strip()):
			raise ValueError("content must be a nonempty string when supplied")
		if request.components is not None:
			unknown = [name for name in request.components if name not in self.registry.names()]
			if unknown:
				raise ValueError(f"Unknown components: {', '.join(sorted(unknown))}")

	def _select(self, request: AnalysisRequest) -> Tuple[List[RegisteredComponent], List[Dict[str, str]]]:
		requested = set(request.components) if request.components is not None else None
		selected: List[RegisteredComponent] = []
		skipped: List[Dict[str, str]] = []
		for component in self.registry.all():
			if requested is not None and component.name not in requested:
				skipped.append({"component": component.name, "reason": "not_requested"})
			elif not component.applies(request):
				skipped.append({"component": component.name, "reason": "required_inputs_not_supplied"})
			else:
				selected.append(component)
		return selected, skipped

	def _run(self, component: RegisteredComponent, request: AnalysisRequest) -> ComponentResult:
		start = perf_counter()
		try:
			output = component.handler(request)
		except Exception as exc:
			elapsed = round((perf_counter() - start) * 1000, 2)
			logger.warning("component_failed component=%s error_type=%s", component.name, type(exc).__name__)
			return error_result(component.name, component.version, exc, elapsed)
		return component_result(component.name, component.version, output, round((perf_counter() - start) * 1000, 2))
