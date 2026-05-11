"""Panel selection and baseline comparisons."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

from design_scientist.acquisition import default_policy, score_candidate, score_panel
from design_scientist.schemas import AcquisitionPolicy, Candidate


def select_panel(
    candidates: list[Candidate],
    policy: AcquisitionPolicy | None = None,
    budget: int = 24,
    *,
    return_diagnostics: bool = False,
) -> tuple[list[Candidate], float] | tuple[list[Candidate], float, dict[str, Any]]:
    policy = policy or default_policy()
    limit = max(0, budget)
    pool = _dedupe_candidates(candidates)
    ranked = sorted(
        pool,
        key=lambda candidate: (-score_candidate(candidate, policy), candidate.candidate_id),
    )
    panel = ranked[:limit]
    _enforce_panel_invariants(pool, panel, limit)
    score = score_panel(panel, policy)
    if return_diagnostics:
        return panel, score, panel_diagnostics(pool, panel, policy, budget=limit, score=score)
    return panel, score


def random_feasible(candidates: list[Candidate], budget: int) -> list[Candidate]:
    rng = random.Random(0)
    sample = sorted(_dedupe_candidates(candidates), key=lambda candidate: candidate.candidate_id)
    rng.shuffle(sample)
    return sample[: max(0, budget)]


def top_predicted_utility(candidates: list[Candidate], budget: int) -> list[Candidate]:
    return sorted(
        _dedupe_candidates(candidates),
        key=lambda c: (
            -(
                c.score_components.get("performance", 0.0)
                + c.score_components.get("decision_value", 0.0)
            ),
            c.candidate_id,
        ),
    )[: max(0, budget)]


def pure_lattice_repair(candidates: list[Candidate], budget: int) -> list[Candidate]:
    return sorted(
        _dedupe_candidates(candidates),
        key=lambda c: (-c.score_components.get("lattice_repair", 0.0), c.candidate_id),
    )[: max(0, budget)]


def fixed_mix(candidates: list[Candidate], budget: int) -> list[Candidate]:
    limit = max(0, budget)
    if limit == 0:
        return []
    categories = ["champion", "champion_contrast", "lattice_repair", "interaction_square", "control", "repeat"]
    selected: list[Candidate] = []
    selected_ids: set[str] = set()
    pool = sorted(_dedupe_candidates(candidates), key=lambda candidate: candidate.candidate_id)
    for category in categories:
        for candidate in [c for c in pool if c.category == category][: max(1, limit // len(categories))]:
            if candidate.candidate_id in selected_ids:
                continue
            selected.append(candidate)
            selected_ids.add(candidate.candidate_id)
            if len(selected) >= limit:
                return selected[:limit]
    for candidate in pool:
        if candidate.candidate_id not in selected_ids:
            selected.append(candidate)
            selected_ids.add(candidate.candidate_id)
        if len(selected) >= limit:
            break
    return selected


def panel_diagnostics(
    candidates: list[Candidate],
    panel: list[Candidate],
    policy: AcquisitionPolicy,
    *,
    budget: int,
    score: float | None = None,
) -> dict[str, Any]:
    pool = _dedupe_candidates(candidates)
    panel_ids = [candidate.candidate_id for candidate in panel]
    pool_ids = {candidate.candidate_id for candidate in pool}
    duplicate_ids = sorted(
        candidate_id
        for candidate_id, count in Counter(panel_ids).items()
        if count > 1
    )
    selected_set = set(panel_ids)
    baseline_budget = len(panel)
    baselines = {
        "top_predicted_utility": top_predicted_utility(pool, baseline_budget),
        "random_feasible": random_feasible(pool, baseline_budget),
        "pure_lattice_repair": pure_lattice_repair(pool, baseline_budget),
        "fixed_mix": fixed_mix(pool, baseline_budget),
    }
    return {
        "policy": policy.name,
        "budget": budget,
        "pool_count": len(pool),
        "selected_count": len(panel),
        "score": score_panel(panel, policy) if score is None else score,
        "subset_of_pool": selected_set <= pool_ids,
        "duplicate_candidate_ids": duplicate_ids,
        "overlap": {
            name: _overlap_summary(selected_set, {candidate.candidate_id for candidate in baseline})
            for name, baseline in baselines.items()
        },
        "coverage": {
            "category_counts": dict(sorted(Counter(candidate.category for candidate in panel).items())),
            "operator_counts": dict(sorted(Counter(candidate.operator for candidate in panel).items())),
            "background_counts": dict(
                sorted(Counter(str(candidate.background or "") for candidate in panel).items())
            ),
            "module_counts": dict(
                sorted(Counter(module for candidate in panel for module in candidate.modules).items())
            ),
            "required_measurements": sorted(
                {measurement for candidate in panel for measurement in candidate.required_measurements}
            ),
        },
        "risk": _risk_summary(panel),
    }


def compare_baselines(candidates: list[Candidate], budget: int = 24) -> list[str]:
    policy = default_policy("mechanism_aware")
    baselines = {
        "random_feasible": random_feasible(candidates, budget),
        "top_predicted_utility": top_predicted_utility(candidates, budget),
        "pure_lattice_repair": pure_lattice_repair(candidates, budget),
        "fixed_mix": fixed_mix(candidates, budget),
    }
    return [
        f"{name}: n={len(panel)} score={score_panel(panel, policy):.3f}"
        for name, panel in baselines.items()
    ]


def _dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    unique: dict[str, Candidate] = {}
    for candidate in candidates:
        unique.setdefault(candidate.candidate_id, candidate)
    return list(unique.values())


def _enforce_panel_invariants(
    pool: list[Candidate],
    panel: list[Candidate],
    budget: int,
) -> None:
    pool_ids = {candidate.candidate_id for candidate in pool}
    panel_ids = [candidate.candidate_id for candidate in panel]
    if len(panel) > budget:
        raise ValueError("Panel exceeds budget")
    if set(panel_ids) - pool_ids:
        raise ValueError("Panel contains candidates outside the pool")
    if len(panel_ids) != len(set(panel_ids)):
        raise ValueError("Panel contains duplicate candidates")


def _overlap_summary(selected_ids: set[str], baseline_ids: set[str]) -> dict[str, float | int]:
    overlap_count = len(selected_ids & baseline_ids)
    union_count = len(selected_ids | baseline_ids)
    return {
        "count": overlap_count,
        "fraction_of_panel": round(overlap_count / len(selected_ids), 6) if selected_ids else 0.0,
        "jaccard": round(overlap_count / union_count, 6) if union_count else 0.0,
    }


def _risk_summary(panel: list[Candidate]) -> dict[str, Any]:
    risk_flags = Counter(flag for candidate in panel for flag in candidate.risk_flags)
    risky_count = sum(1 for candidate in panel if candidate.risk_flags)
    return {
        "risky_candidate_count": risky_count,
        "risky_fraction": round(risky_count / len(panel), 6) if panel else 0.0,
        "risk_flag_counts": dict(sorted(risk_flags.items())),
    }
