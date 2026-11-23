"""Builds component results and combines them into one orchestrated response."""

from typing import Any, Dict, List, Sequence

from intelligence.orchestrator.schemas import (
	STATUS_ERROR,
	STATUS_UNAVAILABLE,
	ComponentOutput,
	ComponentResult,
)


def component_result(name: str, version: str, output: ComponentOutput, execution_ms: float) -> ComponentResult:
	return ComponentResult(
		component=name,
		version=version,
		status=output.status,
		result=output.result,
		evidence=output.evidence,
		limitations=output.limitations,
		execution_ms=execution_ms,
	)


def error_result(name: str, version: str, exc: Exception, execution_ms: float) -> ComponentResult:
	# Only the exception type is exposed; messages can echo submitted content.
	return ComponentResult(
		component=name,
		version=version,
		status=STATUS_ERROR,
		result={},
		evidence=[],
		limitations=["This component failed, so its evidence is missing from the response."],
		execution_ms=execution_ms,
		error={"type": type(exc).__name__, "message": "Component execution failed"},
	)


def build_response(results: Sequence[ComponentResult], skipped: Sequence[Dict[str, str]]) -> Dict[str, Any]:
	errors = [item for item in results if item.status == STATUS_ERROR]
	if not results:
		status = "no_applicable_components"
	elif len(errors) == len(results):
		status = "failed"
	elif errors:
		status = "partial"
	else:
		status = "success"

	evidence: List[Dict[str, Any]] = []
	limitations: List[str] = []
	for item in results:
		evidence.extend({"component": item.component, **entry} for entry in item.evidence)
		for limitation in item.limitations:
			if limitation not in limitations:
				limitations.append(limitation)

	return {
		"status": status,
		"components": {item.component: item.to_dict() for item in results},
		"unavailable": [item.component for item in results if item.status == STATUS_UNAVAILABLE],
		"failed": [item.component for item in errors],
		"skipped": list(skipped),
		"evidence": evidence,
		"limitations": limitations,
		"total_execution_ms": round(sum(item.execution_ms for item in results), 2),
	}
