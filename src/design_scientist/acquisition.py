"""Acquisition policy scoring."""

from __future__ import annotations

from typing import Any

from design_scientist.schemas import AcquisitionPolicy, Candidate


FORBIDDEN_MODULE_PAIRS = frozenset({("KD31N", "SY92F")})

POLICIES = {
    "mechanism_aware": {
        "performance": 1.0,
        "decision_value": 0.7,
        "contrast_value": 0.45,
        "lattice_repair": 0.35,
        "coverage_qc": 0.25,
        "risk": -0.7,
    },
    "fixed_mix": {
        "performance": 0.8,
        "decision_value": 0.4,
        "contrast_value": 0.3,
        "lattice_repair": 0.3,
        "coverage_qc": 0.2,
        "risk": -0.5,
    },
    "pure_lattice_repair": {
        "performance": 0.25,
        "decision_value": 0.2,
        "contrast_value": 0.5,
        "lattice_repair": 1.0,
        "coverage_qc": 0.1,
        "risk": -0.4,
    },
}


def has_forbidden_module_pair(modules: list[str] | tuple[str, ...]) -> bool:
    module_set = {str(module) for module in modules}
    return any(set(pair).issubset(module_set) for pair in FORBIDDEN_MODULE_PAIRS)


def candidate_is_selectable(candidate: Candidate) -> bool:
    return (
        candidate.feasibility_status != "infeasible"
        and not has_forbidden_module_pair(candidate.modules)
    )


def default_policy(name: str = "mechanism_aware") -> AcquisitionPolicy:
    weights = POLICIES.get(name, POLICIES["mechanism_aware"])
    return AcquisitionPolicy(
        name=name,
        description="Performance-first, mechanism-aware dry-run acquisition policy.",
        weights=dict(weights),
        constraints={"forbid_unsupported_1e62_module_causality": True},
    )


def score_candidate(candidate: Candidate, policy: AcquisitionPolicy) -> float:
    score = 0.0
    for key, value in candidate.score_components.items():
        score += policy.weights.get(key, 0.0) * float(value)
    if candidate.risk_flags:
        score += policy.weights.get("risk", -0.5) * len(candidate.risk_flags)
    return round(score, 6)


def score_panel(panel: list[Candidate], policy: AcquisitionPolicy) -> float:
    return round(sum(score_candidate(candidate, policy) for candidate in panel), 6)


def metrics_for_candidate(candidate: Candidate, policy: AcquisitionPolicy) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "policy": policy.name,
        "score": score_candidate(candidate, policy),
        "score_components": candidate.score_components,
        "risk_flags": candidate.risk_flags,
    }
