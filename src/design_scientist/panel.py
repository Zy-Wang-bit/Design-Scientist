"""Panel selection and baseline comparisons."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

from design_scientist.acquisition import (
    candidate_is_selectable,
    default_policy,
    score_candidate,
    score_panel,
)
from design_scientist.schemas import AcquisitionPolicy, Candidate


def select_panel(
    candidates: list[Candidate],
    policy: AcquisitionPolicy | None = None,
    budget: int = 24,
    *,
    return_diagnostics: bool = False,
) -> tuple[list[Candidate], float] | tuple[list[Candidate], float, dict[str, Any]]:
    policy = policy or default_policy()
    cost_budget = _cost_budget(budget)
    pool = _selectable_candidates(candidates)
    ranked = sorted(
        pool,
        key=lambda candidate: (-score_candidate(candidate, policy), candidate.candidate_id),
    )
    panel = _take_within_budget(ranked, cost_budget)
    _enforce_panel_invariants(_dedupe_candidates(candidates), panel, cost_budget)
    score = score_panel(panel, policy)
    if return_diagnostics:
        return panel, score, panel_diagnostics(candidates, panel, policy, budget=cost_budget, score=score)
    return panel, score


def random_feasible(candidates: list[Candidate], budget: int) -> list[Candidate]:
    rng = random.Random(0)
    sample = sorted(_selectable_candidates(candidates), key=lambda candidate: candidate.candidate_id)
    rng.shuffle(sample)
    return _take_within_budget(sample, _cost_budget(budget))


def top_predicted_utility(candidates: list[Candidate], budget: int) -> list[Candidate]:
    ranked = sorted(
        _selectable_candidates(candidates),
        key=lambda c: (
            -(
                c.score_components.get("performance", 0.0)
                + c.score_components.get("decision_value", 0.0)
            ),
            c.candidate_id,
        ),
    )
    return _take_within_budget(ranked, _cost_budget(budget))


def pure_lattice_repair(candidates: list[Candidate], budget: int) -> list[Candidate]:
    ranked = sorted(
        _selectable_candidates(candidates),
        key=lambda c: (-c.score_components.get("lattice_repair", 0.0), c.candidate_id),
    )
    return _take_within_budget(ranked, _cost_budget(budget))


def fixed_mix(candidates: list[Candidate], budget: int) -> list[Candidate]:
    cost_budget = _cost_budget(budget)
    if cost_budget == 0:
        return []
    categories = ["champion", "champion_contrast", "lattice_repair", "interaction_square", "control", "repeat"]
    selected: list[Candidate] = []
    selected_ids: set[str] = set()
    selected_cost = 0.0
    pool = sorted(_selectable_candidates(candidates), key=lambda candidate: candidate.candidate_id)
    for category in categories:
        quota = max(1, int(cost_budget) // len(categories))
        for candidate in [c for c in pool if c.category == category][:quota]:
            if candidate.candidate_id in selected_ids:
                continue
            if selected_cost + _candidate_cost(candidate) > cost_budget:
                continue
            selected.append(candidate)
            selected_ids.add(candidate.candidate_id)
            selected_cost += _candidate_cost(candidate)
            if selected_cost >= cost_budget:
                return selected
    for candidate in pool:
        if candidate.candidate_id not in selected_ids:
            if selected_cost + _candidate_cost(candidate) > cost_budget:
                continue
            selected.append(candidate)
            selected_ids.add(candidate.candidate_id)
            selected_cost += _candidate_cost(candidate)
        if selected_cost >= cost_budget:
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
    selectable_pool = _selectable_candidates(candidates)
    panel_ids = [candidate.candidate_id for candidate in panel]
    pool_ids = {candidate.candidate_id for candidate in pool}
    duplicate_ids = sorted(
        candidate_id
        for candidate_id, count in Counter(panel_ids).items()
        if count > 1
    )
    selected_set = set(panel_ids)
    baselines = {
        "top_predicted_utility": top_predicted_utility(pool, int(budget)),
        "random_feasible": random_feasible(pool, int(budget)),
        "pure_lattice_repair": pure_lattice_repair(pool, int(budget)),
        "fixed_mix": fixed_mix(pool, int(budget)),
    }
    return {
        "policy": policy.name,
        "budget": budget,
        "cost_budget": budget,
        "total_cost": round(sum(_candidate_cost(candidate) for candidate in panel), 6),
        "pool_count": len(pool),
        "selectable_pool_count": len(selectable_pool),
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


def _selectable_candidates(candidates: list[Candidate]) -> list[Candidate]:
    return [
        candidate
        for candidate in _dedupe_candidates(candidates)
        if candidate_is_selectable(candidate)
    ]


def _cost_budget(budget: int | float) -> float:
    return max(0.0, float(budget))


def _candidate_cost(candidate: Candidate) -> float:
    try:
        return max(0.0, float(candidate.cost))
    except (TypeError, ValueError):
        return 1.0


def _take_within_budget(candidates: list[Candidate], cost_budget: float) -> list[Candidate]:
    panel: list[Candidate] = []
    total_cost = 0.0
    for candidate in candidates:
        cost = _candidate_cost(candidate)
        if total_cost + cost > cost_budget:
            continue
        panel.append(candidate)
        total_cost += cost
    return panel


def _enforce_panel_invariants(
    pool: list[Candidate],
    panel: list[Candidate],
    budget: float,
) -> None:
    pool_ids = {candidate.candidate_id for candidate in pool}
    panel_ids = [candidate.candidate_id for candidate in panel]
    if sum(_candidate_cost(candidate) for candidate in panel) > budget:
        raise ValueError("Panel exceeds cost budget")
    if set(panel_ids) - pool_ids:
        raise ValueError("Panel contains candidates outside the pool")
    if len(panel_ids) != len(set(panel_ids)):
        raise ValueError("Panel contains duplicate candidates")
    if any(not candidate_is_selectable(candidate) for candidate in panel):
        raise ValueError("Panel contains infeasible candidates")


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
