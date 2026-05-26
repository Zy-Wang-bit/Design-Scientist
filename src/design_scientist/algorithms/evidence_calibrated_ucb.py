"""Evidence-calibrated UCB for sparse wet-lab variant design.

The algorithm is intentionally generic. It consumes policy records with
variant identifiers, module lists, optional endpoint summaries, cost, and risk
metadata; it does not depend on a specific antibody, antigen, or benchmark
fixture.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from statistics import mean
from typing import Any


DEFAULT_BETA = 0.35
PAIR_WEIGHT = 0.50
MODULE_SHRINKAGE = 2.0
PAIR_SHRINKAGE = 3.0
TOP_NEIGHBOR_WEIGHT = 0.25
MOTIF_ENRICHMENT_WEIGHT = 0.10
BAD_NEIGHBOR_WEIGHT = 0.25
BATCH_REDUNDANCY_WEIGHT = 0.45


@dataclass(frozen=True)
class ObservedNeighborEvidence:
    """Observed variant summary used for generic neighborhood calibration."""

    record_id: str
    modules: tuple[str, ...]
    utility: float
    favorable: bool
    unfavorable: bool


@dataclass(frozen=True)
class EvidenceCalibratedState:
    """Small fitted additive-plus-pair model used by the acquisition rule."""

    observed_ids: frozenset[str]
    global_mean: float
    module_effects: dict[str, float]
    module_counts: dict[str, int]
    pair_effects: dict[tuple[str, str], float]
    pair_counts: dict[tuple[str, str], int]
    endpoint_names: tuple[str, ...]
    neighbor_records: tuple[ObservedNeighborEvidence, ...]


class EvidenceCalibratedLifecycle:
    """V3 lifecycle adapter for evidence-calibrated UCB."""

    name = "evidence_calibrated_ucb"
    claims = (
        "A shrinkage additive-plus-interaction state model with calibrated "
        "uncertainty, observed-neighborhood calibration, and batch architecture "
        "diversity should improve sparse constrained design decisions.",
    )
    architecture_clone = False
    claim_guardrail = True
    confounding_correction = True
    stress_test_worlds = ("sparse_early_round", "noisy_endpoint", "epistatic")

    def fit_state(self, context: Mapping[str, Any]) -> dict[str, Any]:
        observed = list(context.get("observed_records", []))
        candidates = list(context.get("candidate_records", []))
        return {
            **dict(context),
            "algorithm_state": _fit_state(observed, candidates),
            "round_index": int(context.get("round_index", 0) or 0),
        }

    def generate_candidates(self, state: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [dict(item) for item in state.get("candidate_records", [])]

    def score_candidates(
        self,
        state: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        observed = list(state.get("observed_records", []))
        return score_candidates(
            observed,
            candidates,
            round_index=int(state.get("round_index", 0) or 0),
            fitted_state=state.get("algorithm_state"),
        )

    def select_panel(
        self,
        state: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        budget: int,
        rng: random.Random,
    ) -> list[str]:
        del rng
        observed = list(state.get("observed_records", []))
        return select_batch(
            observed,
            candidates,
            budget,
            int(state.get("round_index", 0) or 0),
            random.Random(0),
            fitted_state=state.get("algorithm_state"),
        )

    def plan_ablations(self, state: Mapping[str, Any]) -> list[dict[str, Any]]:
        del state
        return [
            {"name": "none"},
            {
                "name": "no_uncertainty",
                "beta": 0.0,
                "removed_component": "calibrated_uncertainty",
            },
            {
                "name": "no_pair_interactions",
                "pair_weight": 0.0,
                "removed_component": "pair_interaction_state",
            },
            {
                "name": "no_neighbor_calibration",
                "removed_component": "top_neighbor_and_bad_neighbor_evidence",
            },
            {
                "name": "no_batch_architecture_diversity",
                "removed_component": "batch_redundancy_adjustment",
            },
        ]


def mechanism_lifecycle() -> EvidenceCalibratedLifecycle:
    """Return a V3-compatible lifecycle object."""

    return EvidenceCalibratedLifecycle()


def paired_ph_contrast_ucb(
    observed: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    budget: int,
    round_index: int,
    rng: random.Random,
) -> list[str]:
    """Alias for the paper-facing constrained UCB acquisition rule."""

    return select_batch(observed, candidates, budget, round_index, rng)


def select_batch(
    observed: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    budget: int,
    round_index: int,
    rng: random.Random,
    *,
    fitted_state: EvidenceCalibratedState | Any | None = None,
    beta: float = DEFAULT_BETA,
) -> list[str]:
    """Select candidates under a cost budget using calibrated UCB scores."""

    del rng
    if budget <= 0:
        return []
    scored = score_candidates(
        observed,
        candidates,
        round_index=round_index,
        beta=beta,
        fitted_state=fitted_state,
    )
    selected: list[str] = []
    selected_rows: list[dict[str, Any]] = []
    remaining = list(scored)
    spent = 0.0
    cost_budget = float(budget)
    while remaining:
        affordable = [
            row
            for row in remaining
            if spent + _positive_cost(row) <= cost_budget
        ]
        if not affordable:
            break
        row = max(
            affordable,
            key=lambda item: (
                _batch_adjusted_score(item, selected_rows),
                float(item["score"]),
                str(item["candidate_id"]),
            ),
        )
        cost = float(row.get("cost", 1.0) or 1.0)
        if cost <= 0:
            cost = 1.0
        selected.append(str(row["candidate_id"]))
        selected_rows.append(row)
        spent += cost
        remaining.remove(row)
        if spent >= cost_budget:
            break
    return selected


def score_candidates(
    observed: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *,
    round_index: int = 0,
    beta: float = DEFAULT_BETA,
    fitted_state: EvidenceCalibratedState | Any | None = None,
    pair_weight: float = PAIR_WEIGHT,
) -> list[dict[str, Any]]:
    """Score candidate records and expose interpretable score components."""

    state = fitted_state if isinstance(fitted_state, EvidenceCalibratedState) else _fit_state(observed, candidates)
    observed_ids = set(state.observed_ids)
    rows: list[dict[str, Any]] = []
    for raw_candidate in candidates:
        candidate = _normalize_record(raw_candidate)
        candidate_id = _record_id(candidate)
        if candidate_id in observed_ids or not _is_selectable(candidate):
            continue
        components = _score_components(
            candidate,
            state,
            round_index=round_index,
            beta=beta,
            pair_weight=pair_weight,
        )
        row = dict(candidate)
        row["candidate_id"] = candidate_id
        row["variant_id"] = candidate_id
        row["score"] = components["total_score"]
        row["score_components"] = components
        rows.append(row)
    return sorted(
        rows,
        key=lambda item: (float(item["score"]), str(item["candidate_id"])),
        reverse=True,
    )


def _fit_state(
    observed: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> EvidenceCalibratedState:
    observed_records = [_normalize_record(item) for item in observed]
    candidate_records = [_normalize_record(item) for item in candidates]
    utilities = [_record_utility(item) for item in observed_records if _record_utility(item) is not None]
    global_mean = mean(utilities) if utilities else 0.0

    module_values: dict[str, list[float]] = {}
    module_counts: dict[str, int] = {}
    for record in observed_records:
        utility = _record_utility(record)
        if utility is None:
            continue
        modules = _record_modules(record)
        if not modules:
            continue
        residual = utility - global_mean
        share = residual / max(len(modules), 1)
        for module in modules:
            module_values.setdefault(module, []).append(share)
            module_counts[module] = module_counts.get(module, 0) + 1

    module_effects = {
        module: _shrink(mean(values), len(values), MODULE_SHRINKAGE)
        for module, values in module_values.items()
    }

    pair_values: dict[tuple[str, str], list[float]] = {}
    pair_counts: dict[tuple[str, str], int] = {}
    for record in observed_records:
        utility = _record_utility(record)
        modules = _record_modules(record)
        if utility is None or len(modules) < 2:
            continue
        predicted_additive = global_mean + sum(module_effects.get(module, 0.0) for module in modules)
        residual = utility - predicted_additive
        pairs = [tuple(sorted(pair)) for pair in combinations(modules, 2)]
        share = residual / max(len(pairs), 1)
        for pair in pairs:
            pair_values.setdefault(pair, []).append(share)
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    pair_effects = {
        pair: _shrink(mean(values), len(values), PAIR_SHRINKAGE)
        for pair, values in pair_values.items()
    }
    endpoint_names = tuple(sorted(_endpoint_names([*observed_records, *candidate_records])))
    neighbor_records = _neighbor_records(observed_records, utilities)
    return EvidenceCalibratedState(
        observed_ids=frozenset(_record_id(item) for item in observed_records),
        global_mean=float(global_mean),
        module_effects=module_effects,
        module_counts=module_counts,
        pair_effects=pair_effects,
        pair_counts=pair_counts,
        endpoint_names=endpoint_names,
        neighbor_records=neighbor_records,
    )


def _score_components(
    candidate: Mapping[str, Any],
    state: EvidenceCalibratedState,
    *,
    round_index: int,
    beta: float,
    pair_weight: float,
) -> dict[str, float]:
    modules = _record_modules(candidate)
    predicted = state.global_mean + sum(state.module_effects.get(module, 0.0) for module in modules)
    for pair in combinations(modules, 2):
        predicted += pair_weight * state.pair_effects.get(tuple(sorted(pair)), 0.0)

    uncertainty = _candidate_uncertainty(candidate, state)
    neighbor_components = _neighbor_components(candidate, state)
    contrast_bonus = _contrast_completion_bonus(candidate, state)
    coverage_bonus = _coverage_bonus(candidate, state)
    retention_penalty = _retention_risk(candidate)
    guardrail_penalty = _guardrail_risk(candidate)
    cost_penalty = 0.035 * max(float(candidate.get("cost", 1.0) or 1.0) - 1.0, 0.0)
    leakage_penalty = _leakage_penalty(candidate)
    missing_penalty = _missing_evidence_penalty(candidate)
    round_decay = 1.0 / (1.0 + max(round_index, 0))
    uncertainty_bonus = beta * uncertainty * round_decay
    total = (
        predicted
        + uncertainty_bonus
        + neighbor_components["top_neighbor_bonus"]
        + neighbor_components["motif_enrichment_bonus"]
        + contrast_bonus
        + coverage_bonus
        - neighbor_components["bad_neighbor_penalty"]
        - retention_penalty
        - guardrail_penalty
        - cost_penalty
        - leakage_penalty
        - missing_penalty
    )
    return {
        "predicted_utility": _round(predicted),
        "uncertainty": _round(uncertainty),
        "uncertainty_bonus": _round(uncertainty_bonus),
        "top_neighbor_bonus": _round(neighbor_components["top_neighbor_bonus"]),
        "motif_enrichment_bonus": _round(neighbor_components["motif_enrichment_bonus"]),
        "contrast_bonus": _round(contrast_bonus),
        "coverage_bonus": _round(coverage_bonus),
        "bad_neighbor_penalty": _round(neighbor_components["bad_neighbor_penalty"]),
        "retention_risk_penalty": _round(retention_penalty),
        "guardrail_risk_penalty": _round(guardrail_penalty),
        "cost_penalty": _round(cost_penalty),
        "leakage_penalty": _round(leakage_penalty),
        "missing_evidence_penalty": _round(missing_penalty),
        "total_score": _round(total),
    }


def _neighbor_records(
    observed_records: Sequence[Mapping[str, Any]],
    utilities: Sequence[float],
) -> tuple[ObservedNeighborEvidence, ...]:
    if not utilities:
        return ()
    top_threshold = _quantile(utilities, 0.70)
    bottom_threshold = _quantile(utilities, 0.30)
    records: list[ObservedNeighborEvidence] = []
    for record in observed_records:
        utility = _record_utility(record)
        modules = _record_modules(record)
        if utility is None or not modules:
            continue
        feasible = _record_true_feasible(record)
        favorable = bool(feasible is True or (feasible is not False and utility >= top_threshold))
        unfavorable = bool(feasible is False or (feasible is not True and utility <= bottom_threshold))
        records.append(
            ObservedNeighborEvidence(
                record_id=_record_id(record),
                modules=modules,
                utility=float(utility),
                favorable=favorable,
                unfavorable=unfavorable,
            )
        )
    return tuple(records)


def _neighbor_components(
    candidate: Mapping[str, Any],
    state: EvidenceCalibratedState,
) -> dict[str, float]:
    modules = _record_modules(candidate)
    if not modules or not state.neighbor_records:
        return {
            "top_neighbor_bonus": 0.0,
            "motif_enrichment_bonus": 0.0,
            "bad_neighbor_penalty": 0.0,
        }

    neighbors = sorted(
        (
            (_containment_similarity(modules, record.modules), record)
            for record in state.neighbor_records
        ),
        key=lambda item: (item[0], item[1].utility, item[1].record_id),
        reverse=True,
    )
    nearest = [(similarity, record) for similarity, record in neighbors[:5] if similarity > 0.0]
    total_similarity = sum(similarity for similarity, _ in nearest)
    if total_similarity <= 0.0:
        top_neighbor_bonus = 0.0
        bad_neighbor_penalty = 0.0
    else:
        favorable_similarity = sum(
            similarity for similarity, record in nearest if record.favorable
        )
        unfavorable_similarity = sum(
            similarity for similarity, record in nearest if record.unfavorable
        )
        breadth = min(1.0, len(set(modules)) / 2.0)
        top_neighbor_bonus = (
            TOP_NEIGHBOR_WEIGHT
            * min(1.0, favorable_similarity / total_similarity)
            * breadth
        )
        bad_neighbor_penalty = BAD_NEIGHBOR_WEIGHT * min(1.0, unfavorable_similarity / total_similarity)

    return {
        "top_neighbor_bonus": top_neighbor_bonus,
        "motif_enrichment_bonus": MOTIF_ENRICHMENT_WEIGHT
        * _motif_enrichment_score(modules, state.neighbor_records)
        * min(1.0, len(set(modules)) / 2.0),
        "bad_neighbor_penalty": bad_neighbor_penalty,
    }


def _motif_enrichment_score(
    modules: Sequence[str],
    observed_records: Sequence[ObservedNeighborEvidence],
) -> float:
    favorable = [record for record in observed_records if record.favorable]
    unfavorable = [record for record in observed_records if record.unfavorable]
    if not modules or not favorable:
        return 0.0
    module_scores: list[float] = []
    for module in modules:
        favorable_rate = sum(module in record.modules for record in favorable) / len(favorable)
        unfavorable_rate = (
            sum(module in record.modules for record in unfavorable) / len(unfavorable)
            if unfavorable
            else 0.0
        )
        module_scores.append(max(favorable_rate - unfavorable_rate, 0.0))
    return min(1.0, mean(module_scores) if module_scores else 0.0)


def _batch_adjusted_score(
    candidate: Mapping[str, Any],
    selected_rows: Sequence[Mapping[str, Any]],
) -> float:
    score = float(candidate["score"])
    if not selected_rows:
        return score
    modules = _record_modules(candidate)
    redundancy = max(
        (_jaccard_similarity(modules, _record_modules(selected)) for selected in selected_rows),
        default=0.0,
    )
    return score - BATCH_REDUNDANCY_WEIGHT * redundancy


def _positive_cost(candidate: Mapping[str, Any]) -> float:
    cost = float(candidate.get("cost", 1.0) or 1.0)
    return cost if cost > 0.0 else 1.0


def _candidate_uncertainty(candidate: Mapping[str, Any], state: EvidenceCalibratedState) -> float:
    explicit = _to_float(candidate.get("uncertainty"))
    if explicit is not None:
        return max(0.0, min(explicit, 1.0))
    modules = _record_modules(candidate)
    if not modules:
        return 0.05
    module_uncertainty = [
        1.0 / (1.0 + state.module_counts.get(module, 0))
        for module in modules
    ]
    pair_uncertainty = [
        1.0 / (1.0 + state.pair_counts.get(tuple(sorted(pair)), 0))
        for pair in combinations(modules, 2)
    ]
    values = [*module_uncertainty, *pair_uncertainty]
    return mean(values) if values else 0.05


def _contrast_completion_bonus(candidate: Mapping[str, Any], state: EvidenceCalibratedState) -> float:
    modules = _record_modules(candidate)
    if len(modules) < 2:
        return 0.0
    missing_pairs = sum(
        1
        for pair in combinations(modules, 2)
        if tuple(sorted(pair)) not in state.pair_counts
    )
    return min(0.20, 0.05 * missing_pairs)


def _coverage_bonus(candidate: Mapping[str, Any], state: EvidenceCalibratedState) -> float:
    required = _as_list(candidate.get("required_measurements"))
    endpoints = _endpoint_names([candidate])
    if not required and len(_record_modules(candidate)) >= 2:
        return 0.04
    covered = len(set(required) & set(endpoints))
    if required:
        if covered:
            return min(0.12, 0.04 * covered)
        if not _as_list(candidate.get("missing_endpoints")):
            return min(0.08, 0.02 * len(required))
        return 0.0
    return 0.0 if not state.endpoint_names else 0.02


def _retention_risk(candidate: Mapping[str, Any]) -> float:
    flags = {str(flag).lower() for flag in _as_list(candidate.get("risk_flags"))}
    penalty = 0.0
    if any("retention" in flag or "neutral" in flag for flag in flags):
        penalty += 0.12
    values = _endpoint_mapping(candidate)
    for key, value in values.items():
        if ("retention" in key.lower() or "ph7" in key.lower() or "pH7" in key) and value < 0.20:
            penalty += 0.08
    return penalty


def _guardrail_risk(candidate: Mapping[str, Any]) -> float:
    flags = {str(flag).lower() for flag in _as_list(candidate.get("risk_flags"))}
    penalty = 0.0
    if any("qc" in flag or "guardrail" in flag or "expression" in flag for flag in flags):
        penalty += 0.10
    missing = _as_list(candidate.get("missing_endpoints"))
    penalty += min(0.12, 0.03 * len(missing))
    return penalty


def _leakage_penalty(candidate: Mapping[str, Any]) -> float:
    refs = candidate.get("source_refs")
    refs = refs if isinstance(refs, list) else []
    for ref in refs:
        text = str(ref).lower()
        if "holdout" in text or "future" in text or "masked" in text:
            return 0.25
    return 0.0


def _missing_evidence_penalty(candidate: Mapping[str, Any]) -> float:
    return min(0.18, 0.045 * len(_as_list(candidate.get("missing_endpoints"))))


def _normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(record)
    record_id = _record_id(out)
    out["candidate_id"] = record_id
    out["variant_id"] = record_id
    out["modules"] = tuple(_record_modules(out))
    return out


def _record_id(record: Mapping[str, Any]) -> str:
    for key in ("variant_id", "candidate_id", "id", "name"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    raise ValueError("Record is missing variant_id/candidate_id/id/name")


def _record_modules(record: Mapping[str, Any]) -> tuple[str, ...]:
    modules = record.get("modules")
    if modules is None:
        modules = record.get("mutation_tokens")
    if modules is None:
        modules = record.get("mutations")
    if isinstance(modules, str):
        parts = [part.strip() for part in modules.replace(",", ";").split(";")]
    elif isinstance(modules, Sequence):
        parts = [str(part).strip() for part in modules]
    else:
        parts = []
    return tuple(part for part in parts if part)


def _record_utility(record: Mapping[str, Any]) -> float | None:
    for key in ("observed_utility", "utility", "score"):
        value = _to_float(record.get(key))
        if value is not None:
            return value
    components = record.get("utility_components")
    if isinstance(components, Mapping):
        values = [_to_float(value) for value in components.values()]
        values = [value for value in values if value is not None]
        if values:
            return mean(values)
    endpoints = _endpoint_mapping(record)
    if endpoints:
        return mean(endpoints.values())
    return None


def _endpoint_mapping(record: Mapping[str, Any]) -> dict[str, float]:
    for key in ("endpoint_values", "observed_endpoints", "true_endpoints"):
        value = record.get(key)
        if isinstance(value, Mapping):
            out: dict[str, float] = {}
            for endpoint, raw in value.items():
                numeric = _to_float(raw)
                if numeric is not None:
                    out[str(endpoint)] = numeric
            if out:
                return out
    return {}


def _endpoint_names(records: Sequence[Mapping[str, Any]]) -> set[str]:
    names: set[str] = set()
    for record in records:
        names.update(_endpoint_mapping(record))
    return names


def _is_selectable(candidate: Mapping[str, Any]) -> bool:
    status = str(candidate.get("feasibility_status", "feasible")).lower()
    if status in {"infeasible", "blocked", "invalid"}:
        return False
    return True


def _record_true_feasible(record: Mapping[str, Any]) -> bool | None:
    for key in ("true_feasible", "observed_feasible", "feasible_observed"):
        value = record.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes"}:
                return True
            if normalized in {"false", "0", "no"}:
                return False
    return None


def _as_list(value: Any) -> list[Any]:
    if value in (None, "", [], {}, ()):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _shrink(value: float, count: int, strength: float) -> float:
    return float(value) * (float(count) / (float(count) + strength))


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = round(max(0.0, min(fraction, 1.0)) * (len(ordered) - 1))
    return ordered[int(index)]


def _containment_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / min(len(left_set), len(right_set))


def _jaccard_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _round(value: float) -> float:
    return round(float(value), 6)


__all__ = [
    "EvidenceCalibratedLifecycle",
    "EvidenceCalibratedState",
    "mechanism_lifecycle",
    "paired_ph_contrast_ucb",
    "score_candidates",
    "select_batch",
]
