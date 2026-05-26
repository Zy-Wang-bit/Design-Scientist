"""Mechanism-Calibrated Constrained Bayesian Design.

MCCBD is a generic constrained Bayesian design policy for sparse variant
optimization. It fits closed-form Bayesian ridge posteriors over module,
module-pair, and numeric endpoint features, then acquires candidates by
constrained expected improvement with explicit feasibility and information
value terms.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any


ABLATION_MODES = {
    "full",
    "no_constraint",
    "no_information_value",
    "no_batch_diversity",
}


@dataclass(frozen=True)
class MCCBDConfig:
    """Configuration for the MCCBD posterior and acquisition mechanism."""

    prior_variance: float = 1.0
    noise_variance: float = 0.05
    exploration_weight: float = 0.02
    constraint_weight: float = 1.0
    information_weight: float = 0.10
    diversity_weight: float = 0.08
    feasibility_threshold: float = 0.50
    ablation_mode: str = "full"


@dataclass(frozen=True)
class MCCBDNeighborEvidence:
    """Observed local-neighborhood evidence for empirical posterior calibration."""

    record_id: str
    modules: tuple[str, ...]
    utility: float
    feasible: bool | None


@dataclass(frozen=True)
class MCCBDState:
    """Fitted Bayesian utility and feasibility state for MCCBD."""

    config: MCCBDConfig
    feature_names: tuple[str, ...]
    posterior_mean: tuple[float, ...]
    posterior_covariance: tuple[tuple[float, ...], ...]
    feasibility_posterior_mean: tuple[float, ...]
    feasibility_posterior_covariance: tuple[tuple[float, ...], ...]
    features: dict[str, tuple[float, ...]]
    observed_ids: frozenset[str]
    best_observed_feasible_utility: float
    observed_count: int
    feasible_observed_count: int
    utility_training_ids: tuple[str, ...]
    feasibility_training_ids: tuple[str, ...]
    observed_modules: frozenset[str]
    neighbor_evidence: tuple[MCCBDNeighborEvidence, ...]


class MCCBDLifecycle:
    """V3 lifecycle adapter for MCCBD."""

    name = "mccbd"
    claims = (
        "A closed-form Bayesian feature posterior with explicit feasibility "
        "calibration, constrained expected improvement, information value, "
        "and batch diversity should improve constrained sparse design.",
    )
    architecture_clone = False
    claim_guardrail = True
    confounding_correction = True
    stress_test_worlds = ("sparse_early_round", "constraint_shift", "epistatic")

    def fit_state(self, context: Mapping[str, Any]) -> dict[str, Any]:
        observed = list(context.get("observed_records", []))
        candidates = list(context.get("candidate_records", []))
        return {
            **dict(context),
            "algorithm_state": fit_state(observed, candidates),
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
        observed = list(state.get("observed_records", []))
        return select_batch(
            observed,
            candidates,
            budget,
            int(state.get("round_index", 0) or 0),
            rng,
            fitted_state=state.get("algorithm_state"),
        )

    def plan_ablations(self, state: Mapping[str, Any]) -> list[dict[str, Any]]:
        del state
        return [
            {"name": "full", "ablation_mode": "full"},
            {
                "name": "no_constraint",
                "ablation_mode": "no_constraint",
                "removed_component": "feasibility_probability_and_threshold",
            },
            {
                "name": "no_information_value",
                "ablation_mode": "no_information_value",
                "removed_component": "posterior_information_gain_proxy",
            },
            {
                "name": "no_batch_diversity",
                "ablation_mode": "no_batch_diversity",
                "removed_component": "feature_space_batch_diversity",
            },
        ]


def mechanism_lifecycle() -> MCCBDLifecycle:
    """Return a V3-compatible lifecycle object."""

    return MCCBDLifecycle()


def fit_state(
    observed: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    config: MCCBDConfig | None = None,
) -> MCCBDState:
    """Fit closed-form Bayesian ridge utility and feasibility posteriors."""

    cfg = _normalize_config(config)
    observed_records = [_normalize_record(item) for item in observed]
    candidate_records = [_normalize_record(item) for item in candidates]
    all_records = [*observed_records, *candidate_records]
    feature_names = _feature_names(all_records)
    features = {
        _record_id(record): _feature_vector(record, feature_names)
        for record in all_records
    }

    utility_vectors: list[tuple[float, ...]] = []
    utility_targets: list[float] = []
    utility_training_ids: list[str] = []
    feasibility_vectors: list[tuple[float, ...]] = []
    feasibility_targets: list[float] = []
    feasibility_training_ids: list[str] = []
    observed_utilities: list[float] = []
    feasible_utilities: list[float] = []
    neighbor_evidence: list[MCCBDNeighborEvidence] = []
    has_feasibility_labels = False

    for record in observed_records:
        record_id = _record_id(record)
        vector = features[record_id]
        utility = _record_utility(record)
        feasibility = _observed_feasibility_label(record)
        if feasibility is not None:
            has_feasibility_labels = True
            feasibility_vectors.append(vector)
            feasibility_targets.append(1.0 if feasibility else 0.0)
            feasibility_training_ids.append(record_id)
        if utility is None:
            continue
        neighbor_evidence.append(
            MCCBDNeighborEvidence(
                record_id=record_id,
                modules=_record_modules(record),
                utility=float(utility),
                feasible=feasibility,
            )
        )
        observed_utilities.append(float(utility))
        utility_vectors.append(vector)
        utility_targets.append(float(utility))
        utility_training_ids.append(record_id)
        if feasibility is True:
            feasible_utilities.append(float(utility))

    posterior_mean, posterior_covariance = _fit_bayesian_ridge(
        utility_vectors,
        utility_targets,
        len(feature_names),
        cfg,
    )
    feasibility_mean, feasibility_covariance = _fit_bayesian_ridge(
        feasibility_vectors,
        feasibility_targets,
        len(feature_names),
        cfg,
    )
    if feasible_utilities:
        best_feasible = max(feasible_utilities)
    elif has_feasibility_labels:
        best_feasible = max(observed_utilities, default=0.0)
    else:
        best_feasible = max(observed_utilities, default=0.0)

    return MCCBDState(
        config=cfg,
        feature_names=feature_names,
        posterior_mean=posterior_mean,
        posterior_covariance=posterior_covariance,
        feasibility_posterior_mean=feasibility_mean,
        feasibility_posterior_covariance=feasibility_covariance,
        features=features,
        observed_ids=frozenset(_record_id(item) for item in observed_records),
        best_observed_feasible_utility=float(best_feasible),
        observed_count=len(observed_records),
        feasible_observed_count=len(feasible_utilities),
        utility_training_ids=tuple(utility_training_ids),
        feasibility_training_ids=tuple(feasibility_training_ids),
        observed_modules=frozenset(
            module for record in observed_records for module in _record_modules(record)
        ),
        neighbor_evidence=tuple(neighbor_evidence),
    )


def score_candidates(
    observed: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    round_index: int = 0,
    config: MCCBDConfig | None = None,
    fitted_state: MCCBDState | Any | None = None,
) -> list[dict[str, Any]]:
    """Score candidates with posterior constrained expected improvement."""

    state = (
        fitted_state
        if isinstance(fitted_state, MCCBDState)
        else fit_state(observed, candidates, config)
    )
    cfg = _normalize_config(config or state.config)
    rows: list[dict[str, Any]] = []
    for raw_candidate in candidates:
        candidate = _normalize_record(raw_candidate)
        candidate_id = _record_id(candidate)
        if candidate_id in state.observed_ids:
            continue
        vector = _feature_vector(candidate, state.feature_names)
        components = _candidate_score_components(
            candidate,
            vector,
            state,
            cfg,
            round_index=round_index,
        )
        row = dict(candidate)
        row["candidate_id"] = candidate_id
        row["variant_id"] = candidate_id
        row["posterior_mean"] = components["posterior_mean"]
        row["posterior_std"] = components["posterior_std"]
        row["constrained_expected_improvement"] = components[
            "constrained_expected_improvement"
        ]
        row["feasibility_probability"] = components["feasibility_probability"]
        row["information_gain_proxy"] = components["information_gain_proxy"]
        row["score"] = components["total_score"]
        row["score_components"] = components
        rows.append(row)
    return sorted(
        rows,
        key=lambda item: (float(item["score"]), str(item["candidate_id"])),
        reverse=True,
    )


def select_batch(
    observed: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    budget: int,
    round_index: int,
    rng: random.Random,
    config: MCCBDConfig | None = None,
    fitted_state: MCCBDState | Any | None = None,
) -> list[str]:
    """Select candidate identifiers under a cost budget."""

    del rng
    if budget <= 0:
        return []
    state = (
        fitted_state
        if isinstance(fitted_state, MCCBDState)
        else fit_state(observed, candidates, config)
    )
    cfg = _normalize_config(config or state.config)
    scored = score_candidates(
        observed,
        candidates,
        round_index=round_index,
        config=cfg,
        fitted_state=state,
    )
    remaining = [
        row for row in scored if _eligible_for_batch(row, cfg)
    ]
    selected: list[str] = []
    selected_rows: list[Mapping[str, Any]] = []
    spent = 0.0
    cost_budget = float(budget)
    current_covariance = _copy_matrix(state.posterior_covariance)

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
                _batch_adjusted_value(
                    item,
                    selected_rows,
                    current_covariance,
                    state,
                    cfg,
                ),
                float(item["score"]),
                -_positive_cost(item),
                str(item["candidate_id"]),
            ),
        )
        selected.append(str(row["candidate_id"]))
        selected_rows.append(row)
        spent += _positive_cost(row)
        remaining.remove(row)
        vector = _feature_vector(row, state.feature_names)
        current_covariance = _covariance_after_virtual_observation(
            current_covariance,
            vector,
            cfg.noise_variance,
        )
        if spent >= cost_budget:
            break
    return selected


def _candidate_score_components(
    candidate: Mapping[str, Any],
    vector: Sequence[float],
    state: MCCBDState,
    cfg: MCCBDConfig,
    *,
    round_index: int,
) -> dict[str, float]:
    raw_posterior_mean = _dot(vector, state.posterior_mean)
    parameter_variance = max(
        0.0,
        _quadratic_form(state.posterior_covariance, vector),
    )
    posterior_std = math.sqrt(max(cfg.noise_variance + parameter_variance, 0.0))
    posterior_mean = _neighbor_calibrated_mean(
        candidate,
        raw_posterior_mean,
        posterior_std,
        state,
        cfg,
    )
    expected_improvement = _expected_improvement(
        posterior_mean,
        posterior_std,
        state.best_observed_feasible_utility,
    )
    evidence_support = _evidence_support(candidate, state)
    feasibility_probability = _feasibility_probability(candidate, vector, state, cfg)
    if cfg.ablation_mode == "no_constraint":
        constrained_expected_improvement = expected_improvement
        constraint_penalty = 0.0
    else:
        constrained_expected_improvement = feasibility_probability * expected_improvement
        constraint_penalty = cfg.constraint_weight * max(
            0.0,
            cfg.feasibility_threshold - feasibility_probability,
        )
    information_gain = _information_gain_proxy(
        parameter_variance,
        cfg.noise_variance,
    )
    if cfg.ablation_mode == "no_information_value":
        information_value = 0.0
    else:
        information_value = cfg.information_weight * information_gain * (0.25 + 0.75 * evidence_support)
    round_discount = 1.0 / math.sqrt(1.0 + max(float(round_index), 0.0))
    exploration_value = cfg.exploration_weight * posterior_std * (0.25 + 0.75 * evidence_support) * round_discount
    total_score = (
        constrained_expected_improvement
        + exploration_value
        + information_value * round_discount
        - constraint_penalty
    )
    return {
        "posterior_mean": _round(posterior_mean),
        "raw_posterior_mean": _round(raw_posterior_mean),
        "posterior_std": _round(posterior_std),
        "posterior_parameter_variance": _round(parameter_variance),
        "expected_improvement": _round(expected_improvement),
        "constrained_expected_improvement": _round(constrained_expected_improvement),
        "feasibility_probability": _round(feasibility_probability),
        "feasibility_threshold": _round(cfg.feasibility_threshold),
        "constraint_penalty": _round(constraint_penalty),
        "information_gain_proxy": _round(information_gain),
        "information_value": _round(information_value * round_discount),
        "exploration_value": _round(exploration_value),
        "evidence_support": _round(evidence_support),
        "neighbor_support": _round(_neighbor_support(candidate, state)),
        "best_observed_feasible_utility": _round(
            state.best_observed_feasible_utility
        ),
        "total_score": _round(total_score),
    }


def _batch_adjusted_value(
    row: Mapping[str, Any],
    selected_rows: Sequence[Mapping[str, Any]],
    current_covariance: Sequence[Sequence[float]],
    state: MCCBDState,
    cfg: MCCBDConfig,
) -> float:
    vector = _feature_vector(row, state.feature_names)
    parameter_variance = max(0.0, _quadratic_form(current_covariance, vector))
    dynamic_information = _information_gain_proxy(
        parameter_variance,
        cfg.noise_variance,
    )
    components = row["score_components"]
    value = (
        float(components["constrained_expected_improvement"])
        + float(components["exploration_value"])
        - float(components["constraint_penalty"])
    )
    if cfg.ablation_mode != "no_information_value":
        value += cfg.information_weight * dynamic_information
    if selected_rows and cfg.ablation_mode != "no_batch_diversity":
        similarity = max(
            (
                _module_jaccard_similarity(row, selected)
                for selected in selected_rows
            ),
            default=0.0,
        )
        value += cfg.diversity_weight * (1.0 - 2.0 * similarity)
    return value


def _eligible_for_batch(row: Mapping[str, Any], cfg: MCCBDConfig) -> bool:
    if cfg.ablation_mode == "no_constraint":
        return True
    if _hard_infeasible(row) or _observed_feasibility_label(row) is False:
        return False
    return True


def _fit_bayesian_ridge(
    vectors: Sequence[Sequence[float]],
    targets: Sequence[float],
    dimension: int,
    cfg: MCCBDConfig,
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    if dimension <= 0:
        return (), ()
    numpy_result = _fit_bayesian_ridge_numpy(vectors, targets, dimension, cfg)
    if numpy_result is not None:
        return numpy_result
    prior_precision = 1.0 / cfg.prior_variance
    noise_precision = 1.0 / cfg.noise_variance
    precision = [
        [prior_precision if row == col else 0.0 for col in range(dimension)]
        for row in range(dimension)
    ]
    linear = [0.0 for _ in range(dimension)]
    for vector, target in zip(vectors, targets):
        for row_index, row_value in enumerate(vector):
            if row_value == 0.0:
                continue
            linear[row_index] += noise_precision * row_value * float(target)
            for col_index, col_value in enumerate(vector):
                if col_value == 0.0:
                    continue
                precision[row_index][col_index] += (
                    noise_precision * row_value * col_value
                )
    covariance = _invert_matrix(precision)
    mean = _matvec(covariance, linear)
    return tuple(mean), tuple(tuple(row) for row in covariance)


def _fit_bayesian_ridge_numpy(
    vectors: Sequence[Sequence[float]],
    targets: Sequence[float],
    dimension: int,
    cfg: MCCBDConfig,
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]] | None:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - pure Python fallback is tested implicitly.
        return None
    prior_precision = 1.0 / cfg.prior_variance
    noise_precision = 1.0 / cfg.noise_variance
    if vectors:
        matrix = np.asarray(vectors, dtype=float)
        target_vector = np.asarray(targets, dtype=float)
        precision = prior_precision * np.eye(dimension) + noise_precision * matrix.T @ matrix
        linear = noise_precision * matrix.T @ target_vector
    else:
        precision = prior_precision * np.eye(dimension)
        linear = np.zeros(dimension)
    try:
        covariance = np.linalg.inv(precision)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(precision)
    mean = covariance @ linear
    return (
        tuple(float(value) for value in mean.tolist()),
        tuple(tuple(float(value) for value in row) for row in covariance.tolist()),
    )


def _feature_names(records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    names = {"intercept"}
    for record in records:
        names.update(_record_feature_values(record))
    return tuple(["intercept", *sorted(name for name in names if name != "intercept")])


def _record_feature_values(record: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {"intercept": 1.0}
    modules = _record_modules(record)
    for module in modules:
        values[f"module:{module}"] = 1.0
    for left, right in combinations(modules, 2):
        values[f"pair:{left}|{right}"] = 1.0
    for endpoint, value in _numeric_endpoint_features(record).items():
        values[f"endpoint:{endpoint}"] = value
    for name, value in _numeric_feature_mapping(record).items():
        values[f"numeric:{name}"] = value
    return values


def _feature_vector(
    record: Mapping[str, Any],
    feature_names: Sequence[str],
) -> tuple[float, ...]:
    values = _record_feature_values(record)
    active_non_intercept = [
        name
        for name in feature_names
        if name != "intercept" and float(values.get(name, 0.0) or 0.0) != 0.0
    ]
    scale = 1.0 / math.sqrt(len(active_non_intercept)) if active_non_intercept else 1.0
    return tuple(
        float(values.get(name, 0.0))
        if name == "intercept"
        else float(values.get(name, 0.0)) * scale
        for name in feature_names
    )


def _normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(record)
    record_id = _record_id(out)
    out["candidate_id"] = record_id
    out["variant_id"] = record_id
    out["modules"] = _record_modules(out)
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
        modules = record.get("module_id")
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
    return tuple(sorted({part for part in parts if part}))


def _record_utility(record: Mapping[str, Any]) -> float | None:
    for key in ("observed_utility", "utility", "true_utility", "score"):
        value = _to_float(record.get(key))
        if value is not None:
            return value
    endpoint_values = record.get("endpoint_values")
    if isinstance(endpoint_values, Mapping):
        value = _to_float(endpoint_values.get("utility"))
        if value is not None:
            return value
    components = record.get("utility_components")
    if isinstance(components, Mapping):
        values = [_to_float(value) for value in components.values()]
        numeric_values = [value for value in values if value is not None]
        if numeric_values:
            return sum(numeric_values) / len(numeric_values)
    endpoints = _numeric_endpoint_features(record)
    if endpoints:
        return sum(endpoints.values()) / len(endpoints)
    return None


def _record_feasibility(record: Mapping[str, Any]) -> bool | None:
    for key in (
        "true_feasible",
        "observed_feasible",
        "feasible_observed",
        "feasible",
        "is_feasible",
    ):
        value = _parse_bool(record.get(key))
        if value is not None:
            return value
    status = record.get("feasibility_status")
    if isinstance(status, str):
        normalized = status.strip().lower()
        if normalized in {
            "feasible",
            "valid",
            "pass",
            "passed",
            "true",
            "yes",
            "1",
        }:
            return True
        if normalized in {
            "infeasible",
            "blocked",
            "invalid",
            "fail",
            "failed",
            "false",
            "no",
            "0",
        }:
            return False
    return None


def _observed_feasibility_label(record: Mapping[str, Any]) -> bool | None:
    for key in (
        "true_feasible",
        "observed_feasible",
        "feasible_observed",
        "feasible_label",
        "assay_feasible",
    ):
        value = _parse_bool(record.get(key))
        if value is not None:
            return value
    return None


def _hard_infeasible(record: Mapping[str, Any]) -> bool:
    status = record.get("feasibility_status")
    if not isinstance(status, str):
        return False
    normalized = status.strip().lower()
    return normalized in {
        "infeasible",
        "blocked",
        "invalid",
        "fail",
        "failed",
        "false",
        "no",
        "0",
    }


def _numeric_endpoint_features(record: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in ("endpoint_values", "observed_endpoints", "true_endpoints", "endpoints"):
        value = record.get(key)
        if not isinstance(value, Mapping):
            continue
        for endpoint, raw_value in value.items():
            endpoint_name = str(endpoint)
            if endpoint_name.lower() in {
                "utility",
                "observed_utility",
                "true_utility",
                "score",
            }:
                continue
            numeric = _to_float(raw_value)
            if numeric is not None:
                out[endpoint_name] = numeric
    return out


def _numeric_feature_mapping(record: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in ("numeric_features", "features"):
        value = record.get(key)
        if not isinstance(value, Mapping):
            continue
        for name, raw_value in value.items():
            numeric = _to_float(raw_value)
            if numeric is not None:
                out[str(name)] = numeric
    return out


def _feasibility_probability(
    candidate: Mapping[str, Any],
    vector: Sequence[float],
    state: MCCBDState,
    cfg: MCCBDConfig,
) -> float:
    if cfg.ablation_mode == "no_constraint":
        return 1.0
    explicit = _observed_feasibility_label(candidate)
    if _hard_infeasible(candidate) or explicit is False:
        return 0.0
    if not state.feasibility_training_ids:
        return 0.95 if explicit is True else 1.0
    linear_mean = _dot(vector, state.feasibility_posterior_mean)
    parameter_variance = max(
        0.0,
        _quadratic_form(state.feasibility_posterior_covariance, vector),
    )
    predictive_std = math.sqrt(max(cfg.noise_variance + parameter_variance, 1e-12))
    probability = _normal_cdf((linear_mean - 0.5) / predictive_std)
    if explicit is True:
        probability = max(probability, 0.95)
    elif explicit is None:
        probability *= 0.55 + 0.45 * _evidence_support(candidate, state)
        neighbor_probability, neighbor_weight = _neighbor_feasibility(candidate, state)
        if neighbor_weight > 0.0:
            blend = min(neighbor_weight / (neighbor_weight + 1.0), 0.75)
            probability = (1.0 - blend) * probability + blend * neighbor_probability
    return _clamp(probability, 0.0, 1.0)


def _evidence_support(candidate: Mapping[str, Any], state: MCCBDState) -> float:
    modules = _record_modules(candidate)
    if not modules:
        return 1.0
    if not state.observed_modules:
        return 0.5
    supported = sum(1 for module in modules if module in state.observed_modules)
    return supported / len(modules)


def _neighbor_calibrated_mean(
    candidate: Mapping[str, Any],
    posterior_mean: float,
    posterior_std: float,
    state: MCCBDState,
    cfg: MCCBDConfig,
) -> float:
    neighbor_mean, neighbor_weight = _neighbor_utility(candidate, state)
    if neighbor_weight <= 0.0:
        return posterior_mean
    posterior_precision = 1.0 / max(posterior_std * posterior_std, 1e-9)
    neighbor_precision = neighbor_weight / max(cfg.noise_variance, 1e-9)
    return (
        posterior_mean * posterior_precision + neighbor_mean * neighbor_precision
    ) / (posterior_precision + neighbor_precision)


def _neighbor_utility(candidate: Mapping[str, Any], state: MCCBDState) -> tuple[float, float]:
    weights: list[tuple[float, float]] = []
    for neighbor in state.neighbor_evidence:
        similarity = _module_similarity(_record_modules(candidate), neighbor.modules)
        if similarity <= 0.0:
            continue
        weights.append((similarity * similarity, neighbor.utility))
    if not weights:
        return 0.0, 0.0
    total_weight = sum(weight for weight, _value in weights)
    return (
        sum(weight * value for weight, value in weights) / total_weight,
        total_weight,
    )


def _neighbor_feasibility(candidate: Mapping[str, Any], state: MCCBDState) -> tuple[float, float]:
    weights: list[tuple[float, float]] = []
    for neighbor in state.neighbor_evidence:
        if neighbor.feasible is None:
            continue
        similarity = _module_similarity(_record_modules(candidate), neighbor.modules)
        if similarity <= 0.0:
            continue
        weights.append((similarity * similarity, 1.0 if neighbor.feasible else 0.0))
    if not weights:
        return 0.5, 0.0
    total_weight = sum(weight for weight, _value in weights)
    return (
        sum(weight * value for weight, value in weights) / total_weight,
        total_weight,
    )


def _neighbor_support(candidate: Mapping[str, Any], state: MCCBDState) -> float:
    if not state.neighbor_evidence:
        return 0.0
    return max(
        (
            _module_similarity(_record_modules(candidate), neighbor.modules)
            for neighbor in state.neighbor_evidence
        ),
        default=0.0,
    )


def _module_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _expected_improvement(mean: float, std: float, incumbent: float) -> float:
    if std <= 0.0:
        return max(mean - incumbent, 0.0)
    improvement = mean - incumbent
    z_value = improvement / std
    return max(
        0.0,
        improvement * _normal_cdf(z_value) + std * _normal_pdf(z_value),
    )


def _information_gain_proxy(parameter_variance: float, noise_variance: float) -> float:
    if noise_variance <= 0.0:
        return 0.0
    return 0.5 * math.log1p(max(parameter_variance, 0.0) / noise_variance)


def _covariance_after_virtual_observation(
    covariance: Sequence[Sequence[float]],
    vector: Sequence[float],
    noise_variance: float,
) -> list[list[float]]:
    sigma_x = _matvec(covariance, vector)
    denominator = noise_variance + _dot(vector, sigma_x)
    if denominator <= 0.0:
        return _copy_matrix(covariance)
    updated: list[list[float]] = []
    for row_index, row in enumerate(covariance):
        updated_row: list[float] = []
        for col_index, value in enumerate(row):
            updated_row.append(
                float(value)
                - (sigma_x[row_index] * sigma_x[col_index]) / denominator
            )
        updated.append(updated_row)
    return updated


def _invert_matrix(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    size = len(matrix)
    augmented = [
        [
            *[float(matrix[row][col]) for col in range(size)],
            *[1.0 if row == col else 0.0 for col in range(size)],
        ]
        for row in range(size)
    ]
    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            augmented[pivot][col] += 1e-9
        if pivot != col:
            augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        pivot_value = augmented[col][col]
        if abs(pivot_value) < 1e-15:
            raise ValueError("Bayesian ridge precision matrix is singular")
        augmented[col] = [value / pivot_value for value in augmented[col]]
        for row in range(size):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor == 0.0:
                continue
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[col])
            ]
    return [row[size:] for row in augmented]


def _matvec(
    matrix: Sequence[Sequence[float]],
    vector: Sequence[float],
) -> list[float]:
    return [
        sum(float(row[col]) * float(vector[col]) for col in range(len(vector)))
        for row in matrix
    ]


def _quadratic_form(
    matrix: Sequence[Sequence[float]],
    vector: Sequence[float],
) -> float:
    return _dot(vector, _matvec(matrix, vector))


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _copy_matrix(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def _module_jaccard_similarity(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> float:
    left_modules = set(_record_modules(left))
    right_modules = set(_record_modules(right))
    if not left_modules and not right_modules:
        return 0.0
    return len(left_modules & right_modules) / len(left_modules | right_modules)


def _positive_cost(candidate: Mapping[str, Any]) -> float:
    cost = _to_float(candidate.get("cost"))
    if cost is None or cost <= 0.0:
        return 1.0
    return cost


def _normalize_config(config: MCCBDConfig | None) -> MCCBDConfig:
    cfg = config or MCCBDConfig()
    if cfg.ablation_mode not in ABLATION_MODES:
        raise ValueError(
            "ablation_mode must be one of "
            + ", ".join(sorted(ABLATION_MODES))
        )
    if cfg.prior_variance <= 0.0:
        raise ValueError("prior_variance must be positive")
    if cfg.noise_variance <= 0.0:
        raise ValueError("noise_variance must be positive")
    if not 0.0 <= cfg.feasibility_threshold <= 1.0:
        raise ValueError("feasibility_threshold must be between 0 and 1")
    return cfg


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "feasible", "pass", "passed"}:
            return True
        if normalized in {"false", "0", "no", "n", "infeasible", "fail", "failed"}:
            return False
    return None


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _round(value: float) -> float:
    return round(float(value), 6)


__all__ = [
    "MCCBDConfig",
    "MCCBDLifecycle",
    "MCCBDState",
    "fit_state",
    "mechanism_lifecycle",
    "score_candidates",
    "select_batch",
]
