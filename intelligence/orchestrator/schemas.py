"""Shared request and result contracts for orchestrated components."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

STATUS_SUCCESS = "success"
STATUS_UNAVAILABLE = "unavailable"
STATUS_ERROR = "error"


@dataclass(frozen=True)
class AnalysisRequest:
	account_id: Optional[str] = None
	content: Optional[str] = None
	intended_topic: Optional[str] = None
	profile: Optional[Mapping[str, Any]] = None
	goal: Optional[str] = None
	historical_posts: Sequence[Any] = ()
	historical_performance: Sequence[Mapping[str, Any]] = ()
	global_evidence: Sequence[Mapping[str, Any]] = ()
	components: Optional[Sequence[str]] = None


@dataclass
class ComponentOutput:
	"""What a component handler returns; the orchestrator adds timing and identity."""

	result: Dict[str, Any]
	evidence: List[Dict[str, Any]] = field(default_factory=list)
	limitations: List[str] = field(default_factory=list)
	status: str = STATUS_SUCCESS


@dataclass
class ComponentResult:
	component: str
	version: str
	status: str
	result: Dict[str, Any]
	evidence: List[Dict[str, Any]]
	limitations: List[str]
	execution_ms: float
	error: Optional[Dict[str, str]] = None

	def to_dict(self) -> Dict[str, Any]:
		return {
			"component": self.component,
			"version": self.version,
			"status": self.status,
			"result": self.result,
			"evidence": self.evidence,
			"limitations": self.limitations,
			"execution_ms": self.execution_ms,
			"error": self.error,
		}
