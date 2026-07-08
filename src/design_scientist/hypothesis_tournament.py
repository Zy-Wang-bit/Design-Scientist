from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FakeHypothesisAgent:
    """Deterministic agent for tests and offline tournament fixtures."""

    name: str
    output: Mapping[str, Any] | Iterable[Mapping[str, Any]]

    def propose(
        self,
        mechanism_library: Iterable[Mapping[str, Any]],
        gap_matrix: Iterable[Mapping[str, Any]],
        failure_memory: Iterable[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        del mechanism_library, gap_matrix, failure_memory
        if isinstance(self.output, Mapping):
            return [dict(self.output)]
        return [dict(proposal) for proposal in self.output]


def run_hypothesis_tournament(
    mechanism_library: Iterable[Mapping[str, Any]],
    gap_matrix: Iterable[Mapping[str, Any]],
    failure_memory: Iterable[Mapping[str, Any]],
    agents: Iterable[Any],
) -> dict[str, Any]:
    library = list(mechanism_library)
    gaps = list(gap_matrix)
    failures = list(failure_memory)
    baseline_names = _baseline_names(library)
    known_failure_signatures = {
        _normalize(failure.get("failure_signature"))
        for failure in failures
        if _has_text(failure.get("failure_signature"))
    }

    candidates: list[dict[str, Any]] = []
    selected_for_implementation: list[str] = []

    for agent in agents:
        for proposal in _agent_proposals(agent, library, gaps, failures):
            mechanism_id = _mechanism_id(proposal)
            reasons = _rejection_reasons(
                proposal=proposal,
                mechanism_id=mechanism_id,
                baseline_names=baseline_names,
                known_failure_signatures=known_failure_signatures,
            )
            status = "rejected" if reasons else "accepted"

            candidate = {
                "mechanism_id": mechanism_id,
                "status": status,
                "reasons": reasons,
                "proposal": proposal,
                "skeptic_notes": _list_field(proposal.get("skeptic_notes")),
                "novelty_notes": _list_field(proposal.get("novelty_notes")),
                "engineering_notes": _list_field(proposal.get("engineering_notes")),
            }
            candidates.append(candidate)
            if status == "accepted" and mechanism_id:
                selected_for_implementation.append(mechanism_id)

    return {
        "candidates": candidates,
        "selected_for_implementation": selected_for_implementation,
    }


def _agent_proposals(
    agent: Any,
    mechanism_library: list[Mapping[str, Any]],
    gap_matrix: list[Mapping[str, Any]],
    failure_memory: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if hasattr(agent, "propose"):
        output = agent.propose(mechanism_library, gap_matrix, failure_memory)
    elif callable(agent):
        output = agent(mechanism_library, gap_matrix, failure_memory)
    else:
        output = getattr(agent, "output")

    if isinstance(output, Mapping):
        return [dict(output)]
    return [dict(proposal) for proposal in output]


def _rejection_reasons(
    proposal: Mapping[str, Any],
    mechanism_id: str,
    baseline_names: set[str],
    known_failure_signatures: set[str],
) -> list[str]:
    reasons: list[str] = []

    if not _has_text(proposal.get("literature_gap")):
        reasons.append("missing_literature_gap")

    if not _has_components(proposal.get("components")):
        reasons.append("missing_components")

    mechanism_names = {
        _normalize(mechanism_id),
        _normalize(proposal.get("mechanism_name")),
    }
    if mechanism_names & baseline_names:
        reasons.append("copies_baseline_name")

    failure_signature = _normalize(proposal.get("failure_signature"))
    if failure_signature and failure_signature in known_failure_signatures:
        reasons.append("known_failure_signature")

    return reasons


def _baseline_names(mechanism_library: Iterable[Mapping[str, Any]]) -> set[str]:
    names: set[str] = set()
    for mechanism in mechanism_library:
        is_baseline = bool(
            mechanism.get("is_baseline")
            or mechanism.get("baseline")
            or _normalize(mechanism.get("kind")) == "baseline"
            or _normalize(mechanism.get("type")) == "baseline"
            or _normalize(mechanism.get("role")) == "baseline"
        )
        if not is_baseline:
            continue
        for key in ("mechanism_id", "mechanism_name", "name", "id"):
            name = _normalize(mechanism.get(key))
            if name:
                names.add(name)
    return names


def _mechanism_id(proposal: Mapping[str, Any]) -> str:
    for key in ("mechanism_id", "mechanism_name", "name", "id"):
        value = proposal.get(key)
        if _has_text(value):
            return str(value).strip()
    return ""


def _has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _has_components(value: Any) -> bool:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
        return False
    return any(_has_component(component) for component in value)


def _has_component(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return bool(value)
    return True


def _list_field(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _normalize(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower()
