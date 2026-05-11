"""Unified policy API for design panel selection."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from statistics import mean
from typing import Any

from design_scientist.acquisition import FORBIDDEN_MODULE_PAIRS, has_forbidden_module_pair


PolicyRecord = Mapping[str, Any]
PolicyRecordLike = Any
PolicyCallable = Callable[
    [Sequence[PolicyRecordLike], Sequence[PolicyRecordLike], int, int, random.Random],
    list[str],
]

DEFAULT_POLICY_NAMES = (
    "mechanism_aware",
    "random_feasible",
    "top_observed",
    "greedy_utility",
    "fixed_mix",
    "pure_uncertainty",
    "pure_lattice_repair",
)

DEFAULT_TARGET_BACKGROUND = "transfer_bg"
DEFAULT_MODULES = (
    "HD110H",
    "HG56H",
    "HN54H",
    "HV105H",
    "HA23K",
    "SY92F",
    "KD31N",
    "LT47Y",
)
MAX_MODULES = 3
MODULE_ROLES = {
    "HD110H": "acid",
    "HG56H": "neutral",
    "HN54H": "acid",
    "HV105H": "neutral",
    "HA23K": "surrogate_confounded",
    "SY92F": "liability",
    "KD31N": "liability",
    "LT47Y": "developability_repair",
}
PRIOR_INTERACTION_PAIRS = {
    ("HD110H", "HG56H"),
    ("HD110H", "HN54H"),
    ("HG56H", "HV105H"),
    ("HN54H", "HV105H"),
    ("HD110H", "LT47Y"),
}
RISK_MODULES = {"HA23K", "SY92F", "KD31N"}
FORBIDDEN_DESIGN_PAIRS = set(FORBIDDEN_MODULE_PAIRS)

POLICY_RECORD_ID_KEYS = ("variant_id", "candidate_id", "id", "name")
POLICY_RECORD_UTILITY_KEYS = ("observed_utility", "utility", "score")
POLICY_RECORD_OPTIONAL_KEYS = (
    "observed_endpoints",
    "true_endpoints",
    "true_utility",
    "true_feasible",
    "target_system",
    "category",
    "operator",
    "score_components",
    "risk_flags",
    "required_measurements",
    "feasibility_status",
    "feasibility_reasons",
    "cost",
    "design_context",
    "world_id",
)


@dataclass(frozen=True)
class PolicyModel:
    """State estimated from the currently observed records."""

    observed_ids: frozenset[str]
    control_utility_by_background: dict[str, float]
    global_control_utility: float
    module_effects: dict[str, float]
    module_counts: dict[str, int]
    module_backgrounds: dict[str, frozenset[str]]
    singleton_backgrounds: dict[tuple[str, str], int]
    pair_effects: dict[tuple[str, str], float]
    pair_counts: dict[tuple[str, str], int]
    best_observed_utility: float
    best_observed_ids: tuple[str, ...]
    confounding_correction: bool
    target_background: str


def to_policy_record(
    record: PolicyRecordLike,
    *,
    target_background: str | None = None,
    policy_ablation: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize a real or synthetic candidate-like object to the policy contract.

    Stable policy records use:
    - ``variant_id``: stable candidate/variant identifier.
    - ``background``: design background; parsed from ``variant_id`` when absent.
    - ``modules``: tuple of module identifiers.
    - ``observed_utility``: optional utility for observed records.

    Additional fields are preserved where present so callers can carry
    project-specific diagnostics without changing the policy API.
    """

    raw = dict(record) if isinstance(record, Mapping) else {}
    variant = _record_value(record, "variant") or raw.get("variant")
    source = variant if variant is not None else record

    variant_id = _first_record_value(record, POLICY_RECORD_ID_KEYS)
    if variant_id is None and variant is not None:
        variant_id = _first_record_value(variant, POLICY_RECORD_ID_KEYS)
    if variant_id is None:
        raise ValueError("Policy record is missing variant_id/candidate_id/id/name")
    variant_id = str(variant_id)

    background = _record_value(record, "background")
    if background is None and variant is not None:
        background = _record_value(variant, "background")
    if background is None:
        background = _parse_variant_id(variant_id)[0]

    modules = _record_value(record, "modules")
    if modules is None and variant is not None:
        modules = _record_value(variant, "modules")

    normalized: dict[str, Any] = dict(raw)
    normalized["variant_id"] = variant_id
    normalized["background"] = str(background) if background is not None else ""
    normalized["modules"] = _normalize_modules(modules, variant_id)

    utility = _first_record_value(record, POLICY_RECORD_UTILITY_KEYS)
    if utility is None and variant is not None:
        utility = _first_record_value(variant, POLICY_RECORD_UTILITY_KEYS)
    if utility is not None:
        normalized["observed_utility"] = float(utility)

    resolved_target = target_background
    if resolved_target is None:
        resolved_target = _record_value(record, "target_background")
    if resolved_target is None and variant is not None:
        resolved_target = _record_value(variant, "target_background")
    if resolved_target is None:
        design_context = _record_value(record, "design_context")
        if design_context is None and variant is not None:
            design_context = _record_value(variant, "design_context")
        if isinstance(design_context, Mapping):
            resolved_target = design_context.get("target_background")
    if resolved_target is None and _record_value(record, "target_system") is not None and background is not None:
        resolved_target = background
    if resolved_target is not None:
        normalized["target_background"] = str(resolved_target)

    resolved_ablation = policy_ablation
    if resolved_ablation is None:
        resolved_ablation = _record_value(record, "policy_ablation")
    if resolved_ablation is not None:
        normalized["policy_ablation"] = str(resolved_ablation)

    for key in POLICY_RECORD_OPTIONAL_KEYS:
        if key in normalized:
            continue
        value = _record_value(record, key)
        if value is None and variant is not None:
            value = _record_value(variant, key)
        if value is not None:
            normalized[key] = value

    if metadata:
        normalized.update(dict(metadata))
    return normalized


def to_policy_records(records: Sequence[PolicyRecordLike]) -> list[dict[str, Any]]:
    """Normalize a sequence of records to the stable policy contract."""

    return [to_policy_record(record) for record in records]


def select_batch(
    observed: Sequence[PolicyRecordLike],
    candidates: Sequence[PolicyRecordLike],
    budget: int,
    round_index: int,
    rng: random.Random,
) -> list[str]:
    """Default mechanism-aware policy using the shared selection API."""

    return mechanism_aware(observed, candidates, budget, round_index, rng)


def random_feasible(
    observed: Sequence[PolicyRecordLike],
    candidates: Sequence[PolicyRecordLike],
    budget: int,
    round_index: int,
    rng: random.Random,
) -> list[str]:
    """Select a fixed-seed random feasible batch."""

    del round_index
    pool = _candidate_pool(observed, candidates)
    pool = sorted(pool, key=_record_id)
    rng.shuffle(pool)
    return [_record_id(candidate) for candidate in _take_within_budget(pool, budget)]


def top_observed(
    observed: Sequence[PolicyRecordLike],
    candidates: Sequence[PolicyRecordLike],
    budget: int,
    round_index: int,
    rng: random.Random,
) -> list[str]:
    """Exploit candidates similar to the best observed records."""

    del rng
    model = _fit_model(observed, candidates, confounding_correction=False)
    ranked = _rank_policy(
        "top_observed",
        _candidate_pool(observed, candidates),
        model,
        round_index=round_index,
    )
    return [_record_id(candidate) for candidate in _take_within_budget(ranked, budget)]


def greedy_utility(
    observed: Sequence[PolicyRecordLike],
    candidates: Sequence[PolicyRecordLike],
    budget: int,
    round_index: int,
    rng: random.Random,
) -> list[str]:
    """Select candidates with the highest current additive utility estimate."""

    del rng
    model = _fit_model(observed, candidates, confounding_correction=False)
    ranked = _rank_policy(
        "greedy_utility",
        _candidate_pool(observed, candidates),
        model,
        round_index=round_index,
    )
    return [_record_id(candidate) for candidate in _take_within_budget(ranked, budget)]


def fixed_mix(
    observed: Sequence[PolicyRecordLike],
    candidates: Sequence[PolicyRecordLike],
    budget: int,
    round_index: int,
    rng: random.Random,
) -> list[str]:
    """Blend repair, uncertainty, and exploitation with fixed quotas."""

    del rng
    model = _fit_model(observed, candidates, confounding_correction=False)
    pool = _candidate_pool(observed, candidates)
    quotas = [
        ("pure_lattice_repair", max(1, int(round(budget * 0.35))), lambda record: len(_record_modules(record)) == 1),
        ("pure_lattice_repair", max(1, int(round(budget * 0.30))), lambda record: len(_record_modules(record)) == 2),
        ("pure_uncertainty", max(1, int(round(budget * 0.20))), lambda record: len(_record_modules(record)) <= 2),
        ("top_observed", budget, lambda record: len(_record_modules(record)) <= 2),
    ]
    selected: list[PolicyRecord] = []
    selected_ids: set[str] = set()
    selected_cost = 0.0
    cost_budget = _cost_budget(budget)
    for ranker, quota, predicate in quotas:
        ranker_pool = [
            candidate
            for candidate in pool
            if _record_id(candidate) not in selected_ids and predicate(candidate)
        ]
        ranked = _rank_policy(ranker, ranker_pool, model, round_index=round_index)
        added_for_quota = 0
        for candidate in ranked:
            if added_for_quota >= quota:
                break
            candidate_cost = _record_cost(candidate)
            if selected_cost + candidate_cost > cost_budget:
                continue
            selected.append(candidate)
            selected_ids.add(_record_id(candidate))
            selected_cost += candidate_cost
            added_for_quota += 1
            if selected_cost >= cost_budget:
                return [_record_id(record) for record in selected]

    if selected_cost < cost_budget:
        fallback_pool = [
            candidate
            for candidate in pool
            if _record_id(candidate) not in selected_ids and len(_record_modules(candidate)) <= 2
        ]
        ranked = _rank_policy("top_observed", fallback_pool, model, round_index=round_index)
        for candidate in ranked:
            candidate_cost = _record_cost(candidate)
            if selected_cost + candidate_cost > cost_budget:
                continue
            selected.append(candidate)
            selected_ids.add(_record_id(candidate))
            selected_cost += candidate_cost
            if selected_cost >= cost_budget:
                break
    return [_record_id(record) for record in selected]


def pure_uncertainty(
    observed: Sequence[PolicyRecordLike],
    candidates: Sequence[PolicyRecordLike],
    budget: int,
    round_index: int,
    rng: random.Random,
) -> list[str]:
    """Select the batch with highest model uncertainty."""

    del rng
    model = _fit_model(observed, candidates, confounding_correction=True)
    ranked = _rank_policy(
        "pure_uncertainty",
        _candidate_pool(observed, candidates),
        model,
        round_index=round_index,
    )
    return [_record_id(candidate) for candidate in _take_within_budget(ranked, budget)]


def pure_lattice_repair(
    observed: Sequence[PolicyRecordLike],
    candidates: Sequence[PolicyRecordLike],
    budget: int,
    round_index: int,
    rng: random.Random,
) -> list[str]:
    """Select candidates that complete missing singleton and interaction evidence."""

    del rng
    model = _fit_model(observed, candidates, confounding_correction=True)
    ranked = _rank_policy(
        "pure_lattice_repair",
        _candidate_pool(observed, candidates),
        model,
        round_index=round_index,
    )
    return [_record_id(candidate) for candidate in _take_within_budget(ranked, budget)]


def mechanism_aware(
    observed: Sequence[PolicyRecordLike],
    candidates: Sequence[PolicyRecordLike],
    budget: int,
    round_index: int,
    rng: random.Random,
) -> list[str]:
    """Balance predicted utility, uncertainty, lattice repair, priors, and risk."""

    del rng
    ablation = _policy_ablation(candidates)
    model = _fit_model(
        observed,
        candidates,
        confounding_correction=(ablation != "confounding_correction"),
    )
    ranked = _rank_policy(
        "mechanism_aware",
        _candidate_pool(observed, candidates),
        model,
        round_index=round_index,
        ablation=ablation,
    )
    return [_record_id(candidate) for candidate in _take_within_budget(ranked, budget)]


POLICY_REGISTRY: dict[str, PolicyCallable] = {
    "random_feasible": random_feasible,
    "top_observed": top_observed,
    "greedy_utility": greedy_utility,
    "fixed_mix": fixed_mix,
    "pure_uncertainty": pure_uncertainty,
    "pure_lattice_repair": pure_lattice_repair,
    "mechanism_aware": mechanism_aware,
}


def get_policy(name: str) -> PolicyCallable:
    """Return a registered policy callable by name."""

    try:
        return POLICY_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown policy: {name}") from exc


def _candidate_pool(
    observed: Sequence[PolicyRecordLike],
    candidates: Sequence[PolicyRecordLike],
) -> list[PolicyRecord]:
    observed = to_policy_records(observed)
    candidates = to_policy_records(candidates)
    observed_ids = {_record_id(record) for record in observed}
    unique: dict[str, PolicyRecord] = {}
    for candidate in candidates:
        candidate_id = _record_id(candidate)
        if candidate_id in observed_ids:
            continue
        if not _record_selectable(candidate):
            continue
        unique.setdefault(candidate_id, candidate)
    return list(unique.values())


def _fit_model(
    observed: Sequence[PolicyRecordLike],
    candidates: Sequence[PolicyRecordLike],
    *,
    confounding_correction: bool,
) -> PolicyModel:
    observed = to_policy_records(observed)
    candidates = to_policy_records(candidates)
    target_background = _target_background(observed, candidates)
    control_utility_by_background: dict[str, float] = {}
    for record in observed:
        if not _record_modules(record):
            control_utility_by_background[_record_background(record)] = _record_utility(record)

    global_control = mean(control_utility_by_background.values()) if control_utility_by_background else 0.50
    modules = _module_universe(observed, candidates)
    singleton_deltas: dict[str, list[float]] = {module: [] for module in modules}
    singleton_backgrounds: dict[tuple[str, str], int] = {}
    module_counts: dict[str, int] = {module: 0 for module in modules}
    module_backgrounds: dict[str, set[str]] = {module: set() for module in modules}

    for record in observed:
        record_modules = _record_modules(record)
        background = _record_background(record)
        for module in record_modules:
            module_counts[module] = module_counts.get(module, 0) + 1
            module_backgrounds.setdefault(module, set()).add(background)
        if len(record_modules) != 1:
            continue
        module = record_modules[0]
        baseline = _baseline_utility(background, control_utility_by_background, global_control)
        if not confounding_correction:
            baseline = global_control
        singleton_deltas.setdefault(module, []).append(_record_utility(record) - baseline)
        singleton_backgrounds[(module, background)] = singleton_backgrounds.get((module, background), 0) + 1

    module_effects = {
        module: mean(values) if values else 0.0
        for module, values in singleton_deltas.items()
    }

    pair_residuals: dict[tuple[str, str], list[float]] = {}
    pair_counts: dict[tuple[str, str], int] = {}
    for record in observed:
        record_modules = _record_modules(record)
        if len(record_modules) < 2:
            continue
        background = _record_background(record)
        baseline = _baseline_utility(background, control_utility_by_background, global_control)
        if not confounding_correction:
            baseline = global_control
        additive = baseline + sum(module_effects.get(module, 0.0) for module in record_modules)
        residual = _record_utility(record) - additive
        pairs = list(combinations(record_modules, 2))
        if not pairs:
            continue
        residual_per_pair = residual / len(pairs)
        for pair in pairs:
            ordered_pair = tuple(sorted(pair))
            pair_residuals.setdefault(ordered_pair, []).append(residual_per_pair)
            pair_counts[ordered_pair] = pair_counts.get(ordered_pair, 0) + 1

    pair_effects = {
        pair: mean(values) if values else 0.0
        for pair, values in pair_residuals.items()
    }
    best_observed = sorted(observed, key=_record_utility, reverse=True)[:8]
    return PolicyModel(
        observed_ids=frozenset(_record_id(record) for record in observed),
        control_utility_by_background=control_utility_by_background,
        global_control_utility=global_control,
        module_effects=module_effects,
        module_counts=module_counts,
        module_backgrounds={
            module: frozenset(backgrounds)
            for module, backgrounds in module_backgrounds.items()
        },
        singleton_backgrounds=singleton_backgrounds,
        pair_effects=pair_effects,
        pair_counts=pair_counts,
        best_observed_utility=_record_utility(best_observed[0]) if best_observed else 0.0,
        best_observed_ids=tuple(_record_id(record) for record in best_observed),
        confounding_correction=confounding_correction,
        target_background=target_background,
    )


def _rank_policy(
    method_name: str,
    candidates: Sequence[PolicyRecord],
    model: PolicyModel,
    *,
    round_index: int,
    ablation: str | None = None,
) -> list[PolicyRecord]:
    return sorted(
        candidates,
        key=lambda candidate: (
            _score_candidate(method_name, candidate, model, round_index=round_index, ablation=ablation),
            _record_id(candidate),
        ),
        reverse=True,
    )


def _score_candidate(
    method_name: str,
    candidate: PolicyRecord,
    model: PolicyModel,
    *,
    round_index: int,
    ablation: str | None,
) -> float:
    del round_index
    predicted = _predicted_utility(candidate, model)
    target = 1.0 if _record_background(candidate) == model.target_background else 0.12
    uncertainty = _uncertainty_score(candidate, model)
    lattice = _lattice_repair_score(candidate, model)
    interaction = _interaction_prior_score(candidate)
    risk = _risk_score(candidate, model)
    observed_neighbor = _top_observed_similarity(candidate, model)

    if method_name == "top_observed":
        return observed_neighbor + 0.04 * target - 0.03 * risk
    if method_name == "greedy_utility":
        return predicted + 0.08 * target - 0.08 * risk
    if method_name == "pure_uncertainty":
        return uncertainty + 0.03 * target
    if method_name == "pure_lattice_repair":
        return lattice + 0.04 * target - 0.02 * risk
    if method_name != "mechanism_aware":
        raise ValueError(f"Unknown policy: {method_name}")

    weights = {
        "predicted": 1.00,
        "target": 0.16,
        "uncertainty": 0.18,
        "lattice": 0.24,
        "interaction": 0.22,
        "risk": -0.24,
    }
    if ablation == "interaction_prior":
        weights["interaction"] = 0.0
    elif ablation == "uncertainty":
        weights["uncertainty"] = 0.0
    elif ablation == "lattice_repair":
        weights["lattice"] = 0.0
    elif ablation == "evidence_guardrail":
        weights["risk"] = -0.05
        weights["lattice"] = 0.08

    return (
        weights["predicted"] * predicted
        + weights["target"] * target
        + weights["uncertainty"] * uncertainty
        + weights["lattice"] * lattice
        + weights["interaction"] * interaction
        + weights["risk"] * risk
    )


def _baseline_utility(
    background: str,
    control_utility_by_background: dict[str, float],
    global_control: float,
) -> float:
    return control_utility_by_background.get(background, global_control)


def _predicted_utility(candidate: PolicyRecord, model: PolicyModel) -> float:
    modules = _record_modules(candidate)
    value = _baseline_utility(
        _record_background(candidate),
        model.control_utility_by_background,
        model.global_control_utility,
    )
    value += sum(model.module_effects.get(module, 0.0) for module in modules)
    for pair in combinations(modules, 2):
        value += model.pair_effects.get(tuple(sorted(pair)), 0.0)
    value -= 0.020 * max(0, len(modules) - 2)
    return value


def _uncertainty_score(candidate: PolicyRecord, model: PolicyModel) -> float:
    modules = _record_modules(candidate)
    if not modules:
        return 0.05 if _record_id(candidate) not in model.observed_ids else 0.0

    module_uncertainty = [
        1.0 / (1.0 + model.module_counts.get(module, 0))
        for module in modules
    ]
    pair_uncertainty = [
        1.0 / (1.0 + model.pair_counts.get(tuple(sorted(pair)), 0))
        for pair in combinations(modules, 2)
    ]
    missing_target_singletons = sum(
        1
        for module in modules
        if (module, model.target_background) not in model.singleton_backgrounds
    )
    value = 0.50 * mean(module_uncertainty)
    if pair_uncertainty:
        value += 0.35 * mean(pair_uncertainty)
    value += 0.15 * (missing_target_singletons / max(1, len(modules)))
    return min(1.0, value)


def _lattice_repair_score(candidate: PolicyRecord, model: PolicyModel) -> float:
    modules = _record_modules(candidate)
    if not modules:
        return 0.10 if _record_id(candidate) not in model.observed_ids else 0.0

    background = _record_background(candidate)
    score = 0.0
    if len(modules) == 1:
        module = modules[0]
        if (module, background) not in model.singleton_backgrounds:
            score += 0.80 if background == model.target_background else 0.35
        if len(model.module_backgrounds.get(module, frozenset())) == 1:
            score += 0.20
    else:
        for module in modules:
            if (module, background) not in model.singleton_backgrounds:
                score += 0.12
        for pair in combinations(modules, 2):
            ordered_pair = tuple(sorted(pair))
            if ordered_pair not in model.pair_counts:
                score += 0.25
            if ordered_pair in PRIOR_INTERACTION_PAIRS and background == model.target_background:
                score += 0.15
    return min(1.0, score)


def _interaction_prior_score(candidate: PolicyRecord) -> float:
    modules = _record_modules(candidate)
    if len(modules) < 2:
        return 0.0

    module_set = set(modules)
    roles = {MODULE_ROLES.get(module, "unknown") for module in modules}
    score = 0.0
    for pair in combinations(modules, 2):
        ordered_pair = tuple(sorted(pair))
        if ordered_pair in PRIOR_INTERACTION_PAIRS:
            score += 0.32
        if ordered_pair == ("HA23K", "KD31N"):
            score -= 0.40
    if {"acid", "neutral"}.issubset(roles):
        score += 0.20
    if "developability_repair" in roles and "HD110H" in module_set:
        score += 0.12
    return max(-0.75, min(1.0, score))


def _risk_score(candidate: PolicyRecord, model: PolicyModel) -> float:
    modules = _record_modules(candidate)
    background = _record_background(candidate)
    risk = 0.0
    for module in modules:
        if module in RISK_MODULES:
            risk += 0.20
        if background == model.target_background and (module, model.target_background) not in model.singleton_backgrounds:
            risk += 0.05
    for pair in combinations(modules, 2):
        ordered_pair = tuple(sorted(pair))
        if ordered_pair in FORBIDDEN_DESIGN_PAIRS:
            risk += 1.00
        if ordered_pair == ("HA23K", "KD31N"):
            risk += 0.45
    if len(modules) == MAX_MODULES:
        observed_pairs = sum(
            1
            for pair in combinations(modules, 2)
            if tuple(sorted(pair)) in model.pair_counts
        )
        if observed_pairs == 0:
            risk += 0.12
    return min(1.0, risk)


def _top_observed_similarity(candidate: PolicyRecord, model: PolicyModel) -> float:
    if not model.best_observed_ids:
        return 0.0
    best_score = 0.0
    candidate_modules = set(_record_modules(candidate))
    candidate_background = _record_background(candidate)
    for variant_id in model.best_observed_ids:
        background, modules = _parse_variant_id(variant_id)
        observed_modules = set(modules)
        union_size = len(candidate_modules | observed_modules)
        jaccard = 1.0 if union_size == 0 else len(candidate_modules & observed_modules) / union_size
        background_bonus = 0.15 if background == candidate_background else 0.0
        score = 0.35 + 0.65 * jaccard + background_bonus
        best_score = max(best_score, score * model.best_observed_utility)
    return best_score


def _record_id(record: PolicyRecord) -> str:
    for key in POLICY_RECORD_ID_KEYS:
        value = _record_value(record, key)
        if value is not None:
            return str(value)
    variant = _record_value(record, "variant")
    if variant is not None:
        for key in POLICY_RECORD_ID_KEYS:
            value = _record_value(variant, key)
            if value is not None:
                return str(value)
    raise ValueError("Policy record is missing variant_id/candidate_id/id/name")


def _record_background(record: PolicyRecord) -> str:
    background = _record_value(record, "background")
    variant = _record_value(record, "variant")
    if background is None and variant is not None:
        background = _record_value(variant, "background")
    if background is not None:
        return str(background)
    variant_id = _record_id(record)
    if "__" in variant_id:
        return variant_id.split("__", 1)[0]
    return ""


def _record_modules(record: PolicyRecord) -> tuple[str, ...]:
    modules = _record_value(record, "modules")
    variant = _record_value(record, "variant")
    if modules is None and variant is not None:
        modules = _record_value(variant, "modules")
    return _normalize_modules(modules, _record_id(record))


def _record_utility(record: PolicyRecord) -> float:
    for key in POLICY_RECORD_UTILITY_KEYS:
        value = _record_value(record, key)
        if value is not None:
            return float(value)
    return 0.0


def _target_background(
    observed: Sequence[PolicyRecord],
    candidates: Sequence[PolicyRecord],
) -> str:
    for record in (*observed, *candidates):
        target = _record_value(record, "target_background")
        if target:
            return str(target)
    backgrounds = {_record_background(record) for record in (*observed, *candidates)}
    if DEFAULT_TARGET_BACKGROUND in backgrounds:
        return DEFAULT_TARGET_BACKGROUND
    return DEFAULT_TARGET_BACKGROUND


def _module_universe(
    observed: Sequence[PolicyRecord],
    candidates: Sequence[PolicyRecord],
) -> tuple[str, ...]:
    modules: list[str] = list(DEFAULT_MODULES)
    seen = set(modules)
    for record in (*observed, *candidates):
        for module in _record_modules(record):
            if module not in seen:
                modules.append(module)
                seen.add(module)
    return tuple(modules)


def _design_feasible(modules: tuple[str, ...]) -> bool:
    return not has_forbidden_module_pair(modules)


def _parse_variant_id(variant_id: str) -> tuple[str, tuple[str, ...]]:
    if "__" not in variant_id:
        return "", ()
    background, suffix = variant_id.split("__", 1)
    modules = () if suffix == "WT" else tuple(suffix.split("_"))
    return background, modules


def _policy_ablation(candidates: Sequence[PolicyRecord]) -> str | None:
    for candidate in candidates:
        ablation = _record_value(candidate, "policy_ablation")
        if ablation:
            return str(ablation)
    return None


def _cost_budget(budget: int | float) -> float:
    return max(0.0, float(budget))


def _take_within_budget(records: Sequence[PolicyRecord], budget: int | float) -> list[PolicyRecord]:
    selected: list[PolicyRecord] = []
    total_cost = 0.0
    cost_budget = _cost_budget(budget)
    for record in records:
        cost = _record_cost(record)
        if total_cost + cost > cost_budget:
            continue
        selected.append(record)
        total_cost += cost
    return selected


def _record_cost(record: PolicyRecord) -> float:
    value = _record_value(record, "cost")
    if value is None:
        return 1.0
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 1.0


def _record_selectable(record: PolicyRecord) -> bool:
    return _record_value(record, "feasibility_status") != "infeasible" and _design_feasible(
        _record_modules(record)
    )


def _record_value(record: Any, key: str) -> Any:
    if isinstance(record, Mapping):
        return record.get(key)
    return getattr(record, key, None)


def _first_record_value(record: Any, keys: Sequence[str]) -> Any:
    for key in keys:
        value = _record_value(record, key)
        if value is not None:
            return value
    return None


def _normalize_modules(modules: Any, variant_id: str) -> tuple[str, ...]:
    if modules is None:
        return _parse_variant_id(variant_id)[1]
    if isinstance(modules, str):
        if modules in {"", "WT"}:
            return ()
        if "+" in modules:
            parts = modules.split("+")
        elif "," in modules:
            parts = modules.split(",")
        else:
            parts = [modules]
        return tuple(part.strip() for part in parts if part.strip())
    return tuple(str(module) for module in modules)


__all__ = [
    "DEFAULT_POLICY_NAMES",
    "POLICY_REGISTRY",
    "PolicyCallable",
    "PolicyRecord",
    "PolicyRecordLike",
    "fixed_mix",
    "get_policy",
    "greedy_utility",
    "mechanism_aware",
    "pure_lattice_repair",
    "pure_uncertainty",
    "random_feasible",
    "select_batch",
    "to_policy_record",
    "to_policy_records",
    "top_observed",
]
