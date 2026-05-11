"""Deterministic synthetic multi-round wet-lab replay benchmark."""

from __future__ import annotations

import csv
import hashlib
import random
import re
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import yaml

from design_scientist import policies as policy_api


SYNTHETIC_SEED = 1729
DEFAULT_RUN_ID = "synthetic_replay_seed_1729"
TARGET_BACKGROUND = "transfer_bg"
BACKGROUND_ORDER = ("screening_bg", TARGET_BACKGROUND)
MODULES = (
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
DEFAULT_METHODS = tuple(policy_api.DEFAULT_POLICY_NAMES)
DEFAULT_WORLD_IDS = (
    "additive",
    "epistatic",
    "confounded_transfer",
    "noisy_endpoint",
    "sparse_early_round",
)

ENDPOINTS = ("neutral_binding", "acid_release", "expression", "specificity")
UTILITY_WEIGHTS = {
    "neutral_binding": 0.38,
    "acid_release": 0.32,
    "expression": 0.18,
    "specificity": 0.12,
}
FEASIBILITY_THRESHOLDS = {
    "neutral_binding": 0.62,
    "acid_release": 0.56,
    "expression": 0.46,
    "specificity": 0.50,
}
BACKGROUND_BASE = {
    "screening_bg": {
        "neutral_binding": 0.55,
        "acid_release": 0.36,
        "expression": 0.62,
        "specificity": 0.60,
    },
    TARGET_BACKGROUND: {
        "neutral_binding": 0.42,
        "acid_release": 0.42,
        "expression": 0.56,
        "specificity": 0.58,
    },
}
MODULE_EFFECTS = {
    "HD110H": {
        "neutral_binding": 0.16,
        "acid_release": 0.18,
        "expression": -0.02,
        "specificity": 0.02,
    },
    "HG56H": {
        "neutral_binding": 0.13,
        "acid_release": 0.12,
        "expression": 0.03,
        "specificity": 0.04,
    },
    "HN54H": {
        "neutral_binding": 0.05,
        "acid_release": 0.20,
        "expression": -0.01,
        "specificity": 0.02,
    },
    "HV105H": {
        "neutral_binding": 0.11,
        "acid_release": 0.05,
        "expression": 0.04,
        "specificity": 0.03,
    },
    "HA23K": {
        "neutral_binding": 0.03,
        "acid_release": 0.04,
        "expression": -0.12,
        "specificity": -0.09,
    },
    "SY92F": {
        "neutral_binding": 0.04,
        "acid_release": 0.08,
        "expression": 0.02,
        "specificity": -0.15,
    },
    "KD31N": {
        "neutral_binding": 0.18,
        "acid_release": -0.06,
        "expression": -0.19,
        "specificity": 0.00,
    },
    "LT47Y": {
        "neutral_binding": -0.02,
        "acid_release": 0.04,
        "expression": 0.18,
        "specificity": 0.08,
    },
}
PAIR_INTERACTIONS = {
    ("HD110H", "HG56H"): {
        "neutral_binding": 0.07,
        "acid_release": 0.08,
        "expression": 0.00,
        "specificity": 0.00,
    },
    ("HD110H", "HN54H"): {
        "neutral_binding": 0.01,
        "acid_release": 0.10,
        "expression": 0.00,
        "specificity": 0.03,
    },
    ("HG56H", "HV105H"): {
        "neutral_binding": 0.04,
        "acid_release": 0.00,
        "expression": 0.03,
        "specificity": 0.00,
    },
    ("HN54H", "HV105H"): {
        "neutral_binding": 0.00,
        "acid_release": 0.05,
        "expression": 0.00,
        "specificity": 0.02,
    },
    ("HD110H", "LT47Y"): {
        "neutral_binding": 0.00,
        "acid_release": 0.00,
        "expression": 0.08,
        "specificity": 0.06,
    },
    ("HA23K", "KD31N"): {
        "neutral_binding": 0.06,
        "acid_release": -0.04,
        "expression": -0.10,
        "specificity": -0.15,
    },
}
TRIPLE_INTERACTIONS = {
    ("HD110H", "HG56H", "HN54H"): {
        "neutral_binding": 0.03,
        "acid_release": 0.05,
        "expression": 0.00,
        "specificity": 0.00,
    },
    ("HD110H", "HG56H", "LT47Y"): {
        "neutral_binding": 0.00,
        "acid_release": 0.00,
        "expression": 0.05,
        "specificity": 0.04,
    },
}
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
FORBIDDEN_DESIGN_PAIRS = {("KD31N", "SY92F")}
NOISE_SCALE = {
    "neutral_binding": 0.035,
    "acid_release": 0.035,
    "expression": 0.025,
    "specificity": 0.025,
}


@dataclass(frozen=True)
class SyntheticVariant:
    """A designable synthetic variant in the replay lattice."""

    variant_id: str
    background: str
    modules: tuple[str, ...]


@dataclass(frozen=True)
class SyntheticObservation:
    """Truth and fixed noisy observation for one synthetic variant."""

    variant: SyntheticVariant
    true_endpoints: dict[str, float]
    observed_endpoints: dict[str, float]
    true_utility: float
    observed_utility: float
    true_feasible: bool


@dataclass(frozen=True)
class ReplayModel:
    """State estimated from currently revealed observations."""

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


@dataclass(frozen=True)
class WorldSpec:
    """Deterministic synthetic world transformation."""

    world_id: str
    pair_scale: float
    triple_scale: float
    noise_multiplier: float
    initial_mode: str = "standard"
    confounded_transfer: bool = False


@dataclass(frozen=True)
class MethodSpec:
    """Resolved benchmark method name plus policy callable."""

    name: str
    policy: policy_api.PolicyCallable


def run_synthetic_benchmark(
    project_dir: str | Path,
    run_id: str | None = None,
    rounds: int = 3,
    budget: int = 24,
    methods: Iterable[str | policy_api.PolicyCallable] | str | policy_api.PolicyCallable | None = None,
) -> dict[str, Any]:
    """Run a fixed-seed synthetic wet-lab replay and write benchmark artifacts.

    The replay uses a complete latent design lattice, reveals noisy endpoint
    observations only when a policy spends budget on a variant, and computes all
    metrics against the hidden truth. The default method set includes the
    mechanism-aware policy plus six simple baselines.
    """

    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if budget <= 0:
        raise ValueError("budget must be positive")

    method_specs = _normalize_methods(methods)
    method_names = [method.name for method in method_specs]
    safe_run_id = _safe_run_id(run_id)
    root = Path(project_dir).expanduser().resolve()
    _ensure_dir(root)

    world_contexts: list[dict[str, Any]] = []
    for world in _world_suite():
        observations_by_id = _build_observation_table(seed=SYNTHETIC_SEED, world=world)
        variants = [record.variant for record in observations_by_id.values()]
        initial_ids = _initial_observation_ids(observations_by_id, world=world)
        truth = _truth_summary(observations_by_id)
        world_contexts.append(
            {
                "world": world,
                "observations_by_id": observations_by_id,
                "variants": variants,
                "initial_ids": initial_ids,
                "truth": truth,
            }
        )

    config = _benchmark_config(
        rounds=rounds,
        budget=budget,
        methods=method_names,
        world_contexts=world_contexts,
    )
    config_path = root / "benchmarks" / "synthetic_replay" / "config.yaml"
    _write_yaml(config_path, config)

    benchmark_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    random_reference = MethodSpec("random_feasible", policy_api.random_feasible)
    fixed_mix_reference = MethodSpec("fixed_mix", policy_api.fixed_mix)
    mechanism_reference = MethodSpec("mechanism_aware", policy_api.mechanism_aware)

    for context in world_contexts:
        world = context["world"]
        observations_by_id = context["observations_by_id"]
        variants = context["variants"]
        initial_ids = context["initial_ids"]
        truth = context["truth"]

        world_results = [
            _run_replay(
                method,
                world=world,
                variants=variants,
                observations_by_id=observations_by_id,
                initial_ids=initial_ids,
                rounds=rounds,
                budget=budget,
                ablation=None,
            )
            for method in method_specs
        ]
        random_result = next(
            (result for result in world_results if result["method"] == "random_feasible"),
            None,
        )
        if random_result is None:
            random_result = _run_replay(
                random_reference,
                world=world,
                variants=variants,
                observations_by_id=observations_by_id,
                initial_ids=initial_ids,
                rounds=rounds,
                budget=budget,
                ablation=None,
            )
        fixed_mix_result = next(
            (result for result in world_results if result["method"] == "fixed_mix"),
            None,
        )
        if fixed_mix_result is None:
            fixed_mix_result = _run_replay(
                fixed_mix_reference,
                world=world,
                variants=variants,
                observations_by_id=observations_by_id,
                initial_ids=initial_ids,
                rounds=rounds,
                budget=budget,
                ablation=None,
            )
        mechanism_result = next(
            (result for result in world_results if result["method"] == "mechanism_aware"),
            None,
        )
        if mechanism_result is None:
            mechanism_result = _run_replay(
                mechanism_reference,
                world=world,
                variants=variants,
                observations_by_id=observations_by_id,
                initial_ids=initial_ids,
                rounds=rounds,
                budget=budget,
                ablation=None,
            )
        reference_selected_ids = {
            "mechanism_aware": set(mechanism_result["selected_ids"]),
            "random_feasible": set(random_result["selected_ids"]),
            "fixed_mix": set(fixed_mix_result["selected_ids"]),
        }

        for result in world_results:
            row = _result_row(result, truth, rounds, budget, len(initial_ids))
            row["novelty_score"] = _selection_novelty(
                result["selected_ids"],
                initial_ids,
                observations_by_id,
            )
            row["baseline_overlap"] = _baseline_overlap(
                result["selected_ids"],
                reference_selected_ids["random_feasible"],
            )
            row["selected_ids_digest"] = _selection_digest(result["selected_ids"])
            for baseline, selected_ids in reference_selected_ids.items():
                row[f"selection_overlap_{baseline}"] = _baseline_overlap(
                    result["selected_ids"],
                    selected_ids,
                )
            benchmark_rows.append(row)

        full_mechanism = _run_replay(
            mechanism_reference,
            world=world,
            variants=variants,
            observations_by_id=observations_by_id,
            initial_ids=initial_ids,
            rounds=rounds,
            budget=budget,
            ablation=None,
        )
        for removed_component in (
            "none",
            "interaction_prior",
            "uncertainty",
            "lattice_repair",
            "confounding_correction",
            "evidence_guardrail",
        ):
            ablation = None if removed_component == "none" else removed_component
            result = full_mechanism if ablation is None else _run_replay(
                mechanism_reference,
                world=world,
                variants=variants,
                observations_by_id=observations_by_id,
                initial_ids=initial_ids,
                rounds=rounds,
                budget=budget,
                ablation=ablation,
            )
            row = _result_row(result, truth, rounds, budget, len(initial_ids))
            row["method"] = "mechanism_aware"
            row["removed_component"] = removed_component
            row["delta_from_full_best_feasible_utility"] = _round_metric(
                row["best_feasible_utility"] - full_mechanism["metrics"]["best_feasible_utility"]
            )
            row["delta_from_full_regret_proxy"] = _round_metric(
                row["regret_proxy"] - full_mechanism["metrics"]["regret_proxy"]
            )
            row["novelty_score"] = _selection_novelty(
                result["selected_ids"],
                initial_ids,
                observations_by_id,
            )
            row["baseline_overlap"] = _baseline_overlap(
                result["selected_ids"],
                reference_selected_ids["random_feasible"],
            )
            row["selected_ids_digest"] = _selection_digest(result["selected_ids"])
            for baseline, selected_ids in reference_selected_ids.items():
                row[f"selection_overlap_{baseline}"] = _baseline_overlap(
                    result["selected_ids"],
                    selected_ids,
                )
            ablation_rows.append(row)

    run_dir = root / "runs" / safe_run_id
    benchmark_path = run_dir / "benchmark_results.csv"
    ablation_path = run_dir / "ablation_results.csv"
    summary = _benchmark_summary(benchmark_rows)
    summary_path = run_dir / "benchmark_summary.csv"
    _write_csv(benchmark_path, benchmark_rows)
    _write_csv(ablation_path, ablation_rows)
    _write_csv(summary_path, summary["method_rankings"])

    return {
        "run_id": safe_run_id,
        "config_path": str(config_path),
        "benchmark_results_path": str(benchmark_path),
        "ablation_results_path": str(ablation_path),
        "summary_results_path": str(summary_path),
        "benchmark_results": benchmark_rows,
        "ablation_results": ablation_rows,
        "summary": summary,
        "config": config,
    }


def _normalize_methods(
    methods: Iterable[str | policy_api.PolicyCallable] | str | policy_api.PolicyCallable | None,
) -> list[MethodSpec]:
    if methods is None:
        raw_methods = list(DEFAULT_METHODS)
    elif isinstance(methods, str) or callable(methods):
        raw_methods = [methods]
    else:
        raw_methods = list(methods)

    normalized: list[MethodSpec] = []
    seen: set[str] = set()
    for method in raw_methods:
        if isinstance(method, str):
            if method not in policy_api.POLICY_REGISTRY:
                raise ValueError(f"Unknown synthetic replay method: {method}")
            method_name = method
            policy = policy_api.get_policy(method)
        elif callable(method):
            method_name = _callable_method_name(method)
            policy = method
        else:
            raise TypeError("methods must be method names or policy callables")

        if method_name not in seen:
            normalized.append(MethodSpec(method_name, policy))
            seen.add(method_name)
    if not normalized:
        raise ValueError("At least one method is required")
    return normalized


def _callable_method_name(policy: Callable[..., Any]) -> str:
    name = getattr(policy, "policy_name", None) or getattr(policy, "__name__", None)
    if not name:
        name = policy.__class__.__name__
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("._-")
    return safe or "policy_callable"


def _safe_run_id(run_id: str | None) -> str:
    if run_id is None:
        return DEFAULT_RUN_ID
    if not run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a non-empty relative name")
    if "/" in run_id or "\\" in run_id or ".." in Path(run_id).parts:
        raise ValueError("run_id must not contain path separators or '..'")
    return run_id


def _benchmark_config(
    *,
    rounds: int,
    budget: int,
    methods: list[str],
    world_contexts: list[dict[str, Any]],
) -> dict[str, Any]:
    first_context = world_contexts[0]
    initial_ids = first_context["initial_ids"]
    candidate_count = len(first_context["variants"])
    truth_by_world = {
        context["world"].world_id: context["truth"]
        for context in world_contexts
    }
    initial_ids_by_world = {
        context["world"].world_id: context["initial_ids"]
        for context in world_contexts
    }
    return {
        "benchmark": "synthetic_multi_round_wet_lab_replay",
        "seed": SYNTHETIC_SEED,
        "rounds": rounds,
        "budget_per_round": budget,
        "methods": methods,
        "worlds": [context["world"].world_id for context in world_contexts],
        "default_method": "mechanism_aware",
        "candidate_count": candidate_count,
        "initial_observation_count": len(initial_ids),
        "initial_observation_ids": initial_ids,
        "initial_observation_ids_by_world": initial_ids_by_world,
        "target_background": TARGET_BACKGROUND,
        "backgrounds": list(BACKGROUND_ORDER),
        "modules": list(MODULES),
        "max_modules": MAX_MODULES,
        "endpoints": list(ENDPOINTS),
        "utility_weights": dict(UTILITY_WEIGHTS),
        "feasibility_thresholds": dict(FEASIBILITY_THRESHOLDS),
        "latent_features": {
            "module_effects": MODULE_EFFECTS,
            "pair_interactions": {
                "+".join(pair): effects for pair, effects in PAIR_INTERACTIONS.items()
            },
            "triple_interactions": {
                "+".join(triple): effects for triple, effects in TRIPLE_INTERACTIONS.items()
            },
            "risk_modules": sorted(RISK_MODULES),
            "forbidden_design_pairs": ["+".join(pair) for pair in sorted(FORBIDDEN_DESIGN_PAIRS)],
            "confounding": (
                "Initial singleton observations are enriched in screening_bg, while final "
                "feasible utility is scored on transfer_bg variants."
            ),
        },
        "truth_summary": truth_by_world,
    }


def _world_suite() -> tuple[WorldSpec, ...]:
    return (
        WorldSpec(
            world_id="additive",
            pair_scale=0.0,
            triple_scale=0.0,
            noise_multiplier=0.80,
        ),
        WorldSpec(
            world_id="epistatic",
            pair_scale=1.20,
            triple_scale=1.15,
            noise_multiplier=1.00,
        ),
        WorldSpec(
            world_id="confounded_transfer",
            pair_scale=1.00,
            triple_scale=1.00,
            noise_multiplier=1.00,
            initial_mode="confounded",
            confounded_transfer=True,
        ),
        WorldSpec(
            world_id="noisy_endpoint",
            pair_scale=1.00,
            triple_scale=1.00,
            noise_multiplier=2.25,
        ),
        WorldSpec(
            world_id="sparse_early_round",
            pair_scale=1.10,
            triple_scale=1.10,
            noise_multiplier=1.00,
            initial_mode="sparse",
        ),
    )


def _default_world() -> WorldSpec:
    return _world_suite()[1]


def _build_observation_table(
    seed: int,
    world: WorldSpec | None = None,
) -> dict[str, SyntheticObservation]:
    world = world or _default_world()
    table: dict[str, SyntheticObservation] = {}
    for background in BACKGROUND_ORDER:
        for module_count in range(MAX_MODULES + 1):
            for modules in combinations(MODULES, module_count):
                if not _design_feasible(modules):
                    continue
                variant = SyntheticVariant(
                    variant_id=_variant_id(background, modules),
                    background=background,
                    modules=tuple(modules),
                )
                true_endpoints = _true_endpoints(variant, world=world)
                observed_endpoints = _observed_endpoints(variant, true_endpoints, seed, world=world)
                true_utility = _utility(true_endpoints, len(modules))
                observed_utility = _utility(observed_endpoints, len(modules))
                table[variant.variant_id] = SyntheticObservation(
                    variant=variant,
                    true_endpoints=true_endpoints,
                    observed_endpoints=observed_endpoints,
                    true_utility=true_utility,
                    observed_utility=observed_utility,
                    true_feasible=_true_feasible(variant, true_endpoints),
                )
    return table


def _variant_id(background: str, modules: tuple[str, ...] | Iterable[str]) -> str:
    ordered = tuple(modules)
    suffix = "WT" if not ordered else "_".join(ordered)
    return f"{background}__{suffix}"


def _design_feasible(modules: tuple[str, ...]) -> bool:
    module_set = set(modules)
    return not any(set(pair).issubset(module_set) for pair in FORBIDDEN_DESIGN_PAIRS)


def _true_endpoints(
    variant: SyntheticVariant,
    world: WorldSpec | None = None,
) -> dict[str, float]:
    world = world or _default_world()
    endpoints = dict(BACKGROUND_BASE[variant.background])
    module_set = set(variant.modules)

    for module in variant.modules:
        for endpoint, effect in MODULE_EFFECTS[module].items():
            endpoints[endpoint] += effect
        if world.confounded_transfer and module == "HA23K":
            if variant.background == "screening_bg":
                endpoints["neutral_binding"] += 0.12
                endpoints["acid_release"] += 0.08
                endpoints["expression"] += 0.06
                endpoints["specificity"] += 0.12
            else:
                endpoints["expression"] -= 0.08
                endpoints["specificity"] -= 0.06

    for pair, effects in PAIR_INTERACTIONS.items():
        if set(pair).issubset(module_set):
            for endpoint, effect in effects.items():
                endpoints[endpoint] += world.pair_scale * effect

    for triple, effects in TRIPLE_INTERACTIONS.items():
        if set(triple).issubset(module_set):
            for endpoint, effect in effects.items():
                endpoints[endpoint] += world.triple_scale * effect

    return {endpoint: _round_metric(_clamp(value)) for endpoint, value in endpoints.items()}


def _observed_endpoints(
    variant: SyntheticVariant,
    true_endpoints: dict[str, float],
    seed: int,
    world: WorldSpec | None = None,
) -> dict[str, float]:
    world = world or _default_world()
    observed: dict[str, float] = {}
    for endpoint, true_value in true_endpoints.items():
        noise = _stable_noise(
            seed,
            world.world_id,
            variant.variant_id,
            endpoint,
            scale=NOISE_SCALE[endpoint] * world.noise_multiplier,
        )
        observed[endpoint] = _round_metric(_clamp(true_value + noise))
    return observed


def _utility(endpoints: dict[str, float], module_count: int) -> float:
    value = sum(UTILITY_WEIGHTS[endpoint] * endpoints[endpoint] for endpoint in ENDPOINTS)
    value -= 0.035 * max(0, module_count - 2)
    return _round_metric(value)


def _true_feasible(variant: SyntheticVariant, endpoints: dict[str, float]) -> bool:
    if variant.background != TARGET_BACKGROUND:
        return False
    return all(endpoints[endpoint] >= threshold for endpoint, threshold in FEASIBILITY_THRESHOLDS.items())


def _initial_observation_ids(
    observations_by_id: dict[str, SyntheticObservation],
    world: WorldSpec | None = None,
) -> list[str]:
    world = world or _default_world()
    if world.initial_mode == "sparse":
        requested = [
            _variant_id("screening_bg", ()),
            _variant_id(TARGET_BACKGROUND, ()),
            _variant_id("screening_bg", ("HD110H",)),
            _variant_id("screening_bg", ("HG56H",)),
            _variant_id(TARGET_BACKGROUND, ("HD110H",)),
        ]
    elif world.initial_mode == "confounded":
        requested = [
            _variant_id("screening_bg", ()),
            _variant_id(TARGET_BACKGROUND, ()),
            *[_variant_id("screening_bg", (module,)) for module in MODULES],
            _variant_id(TARGET_BACKGROUND, ("HD110H",)),
            _variant_id(TARGET_BACKGROUND, ("HG56H",)),
            _variant_id("screening_bg", ("HA23K", "KD31N")),
        ]
    else:
        requested = [
            _variant_id("screening_bg", ()),
            _variant_id(TARGET_BACKGROUND, ()),
            *[_variant_id("screening_bg", (module,)) for module in MODULES],
            _variant_id(TARGET_BACKGROUND, ("HD110H",)),
            _variant_id(TARGET_BACKGROUND, ("HG56H",)),
            _variant_id(TARGET_BACKGROUND, ("HA23K",)),
            _variant_id(TARGET_BACKGROUND, ("LT47Y",)),
            _variant_id("screening_bg", ("HA23K", "KD31N")),
        ]
    return [variant_id for variant_id in requested if variant_id in observations_by_id]


def _run_replay(
    method: MethodSpec,
    *,
    world: WorldSpec,
    variants: list[SyntheticVariant],
    observations_by_id: dict[str, SyntheticObservation],
    initial_ids: list[str],
    rounds: int,
    budget: int,
    ablation: str | None,
) -> dict[str, Any]:
    method_name = method.name
    observed_ids = set(initial_ids)
    observations = [observations_by_id[variant_id] for variant_id in initial_ids]
    round_best: list[float] = []
    selected_ids: list[str] = []
    selected_count = 0

    for round_index in range(1, rounds + 1):
        selected = _select_round(
            method,
            world=world,
            round_index=round_index,
            budget=budget,
            variants=variants,
            observations=observations,
            observed_ids=observed_ids,
            ablation=ablation,
        )
        for variant in selected:
            observed_ids.add(variant.variant_id)
            observations.append(observations_by_id[variant.variant_id])
            selected_ids.append(variant.variant_id)
        selected_count += len(selected)
        round_best.append(_best_feasible_utility(observations))

    metrics = _metrics(
        observations=observations,
        selected_count=selected_count,
        observations_by_id=observations_by_id,
        round_best=round_best,
        claim_guardrail=(method_name == "mechanism_aware" and ablation != "evidence_guardrail"),
        confounding_correction=(method_name == "mechanism_aware" and ablation != "confounding_correction"),
    )
    return {
        "world_id": world.world_id,
        "method": method_name,
        "ablation": ablation,
        "selected_count": selected_count,
        "total_observations": len(observations),
        "selected_ids": tuple(selected_ids),
        "metrics": metrics,
    }


def _select_round(
    method: MethodSpec,
    *,
    world: WorldSpec,
    round_index: int,
    budget: int,
    variants: list[SyntheticVariant],
    observations: list[SyntheticObservation],
    observed_ids: set[str],
    ablation: str | None,
) -> list[SyntheticVariant]:
    available = [variant for variant in variants if variant.variant_id not in observed_ids]
    if not available:
        return []

    observed_records = [
        _policy_observation_record(observation, world=world)
        for observation in observations
    ]
    candidate_records = [
        _policy_candidate_record(variant, world=world, ablation=ablation)
        for variant in available
    ]
    rng = random.Random(
        _stable_seed(SYNTHETIC_SEED, "policy", world.world_id, method.name, round_index, ablation or "full")
    )
    selected_ids = method.policy(
        observed_records,
        candidate_records,
        budget,
        round_index,
        rng,
    )
    return _resolve_selected_variants(selected_ids, available, budget)


def _policy_observation_record(
    observation: SyntheticObservation,
    *,
    world: WorldSpec,
) -> dict[str, Any]:
    return policy_api.to_policy_record(
        observation,
        target_background=TARGET_BACKGROUND,
        metadata={"world_id": world.world_id},
    )


def _policy_candidate_record(
    variant: SyntheticVariant,
    *,
    world: WorldSpec,
    ablation: str | None,
) -> dict[str, Any]:
    return policy_api.to_policy_record(
        variant,
        target_background=TARGET_BACKGROUND,
        policy_ablation=ablation,
        metadata={"world_id": world.world_id},
    )


def _resolve_selected_variants(
    selected_ids: Iterable[Any],
    available: list[SyntheticVariant],
    budget: int,
) -> list[SyntheticVariant]:
    available_by_id = {variant.variant_id: variant for variant in available}
    selected: list[SyntheticVariant] = []
    seen: set[str] = set()
    for raw_selection in selected_ids:
        variant_id = _selected_variant_id(raw_selection)
        if variant_id in seen:
            continue
        if variant_id not in available_by_id:
            raise ValueError(f"Policy selected unavailable candidate: {variant_id}")
        selected.append(available_by_id[variant_id])
        seen.add(variant_id)
        if len(selected) >= budget:
            break
    return selected


def _selected_variant_id(selection: Any) -> str:
    if isinstance(selection, dict):
        for key in ("variant_id", "candidate_id", "id", "name"):
            value = selection.get(key)
            if value:
                return str(value)
    for attr in ("variant_id", "candidate_id", "id", "name"):
        value = getattr(selection, attr, None)
        if value:
            return str(value)
    return str(selection)


def _select_fixed_mix(
    round_index: int,
    budget: int,
    available: list[SyntheticVariant],
    observations: list[SyntheticObservation],
) -> list[SyntheticVariant]:
    model = _fit_model(observations, confounding_correction=False)
    quotas = [
        ("pure_lattice_repair", max(1, int(round(budget * 0.35))), lambda variant: len(variant.modules) == 1),
        ("pure_lattice_repair", max(1, int(round(budget * 0.30))), lambda variant: len(variant.modules) == 2),
        ("pure_uncertainty", max(1, int(round(budget * 0.20))), lambda variant: len(variant.modules) <= 2),
        ("top_observed", budget, lambda variant: len(variant.modules) <= 2),
    ]
    selected: list[SyntheticVariant] = []
    selected_ids: set[str] = set()
    for ranker, quota, predicate in quotas:
        pool = [
            variant
            for variant in available
            if variant.variant_id not in selected_ids and predicate(variant)
        ]
        ranked = _rank_variants(ranker, pool, model, round_index=round_index, ablation=None)
        for variant in ranked[:quota]:
            if len(selected) >= budget:
                return selected
            selected.append(variant)
            selected_ids.add(variant.variant_id)
    if len(selected) < budget:
        pool = [
            variant
            for variant in available
            if variant.variant_id not in selected_ids and len(variant.modules) <= 2
        ]
        ranked = _rank_variants("top_observed", pool, model, round_index=round_index, ablation=None)
        selected.extend(ranked[: budget - len(selected)])
    return selected[:budget]


def _rank_variants(
    method_name: str,
    variants: list[SyntheticVariant],
    model: ReplayModel,
    *,
    round_index: int,
    ablation: str | None,
) -> list[SyntheticVariant]:
    return sorted(
        variants,
        key=lambda variant: (
            _score_variant(method_name, variant, model, round_index=round_index, ablation=ablation),
            variant.variant_id,
        ),
        reverse=True,
    )


def _score_variant(
    method_name: str,
    variant: SyntheticVariant,
    model: ReplayModel,
    *,
    round_index: int,
    ablation: str | None,
) -> float:
    if method_name == "random_feasible":
        return _stable_unit(SYNTHETIC_SEED, "random_feasible", variant.variant_id)

    predicted = _predicted_utility(variant, model)
    target = 1.0 if variant.background == TARGET_BACKGROUND else 0.12
    uncertainty = _uncertainty_score(variant, model)
    lattice = _lattice_repair_score(variant, model)
    interaction = _interaction_prior_score(variant)
    risk = _risk_score(variant, model)
    observed_neighbor = _top_observed_similarity(variant, model)

    if method_name == "top_observed":
        return observed_neighbor + 0.04 * target - 0.03 * risk
    if method_name == "greedy_utility":
        return predicted + 0.08 * target - 0.08 * risk
    if method_name == "pure_uncertainty":
        return uncertainty + 0.03 * target
    if method_name == "pure_lattice_repair":
        return lattice + 0.04 * target - 0.02 * risk
    if method_name != "mechanism_aware":
        raise ValueError(f"Unknown synthetic replay method: {method_name}")

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


def _fit_model(
    observations: list[SyntheticObservation],
    *,
    confounding_correction: bool,
) -> ReplayModel:
    control_utility_by_background: dict[str, float] = {}
    for observation in observations:
        if not observation.variant.modules:
            control_utility_by_background[observation.variant.background] = observation.observed_utility

    if control_utility_by_background:
        global_control = mean(control_utility_by_background.values())
    else:
        global_control = 0.50

    singleton_deltas: dict[str, list[float]] = {module: [] for module in MODULES}
    singleton_backgrounds: dict[tuple[str, str], int] = {}
    module_counts: dict[str, int] = {module: 0 for module in MODULES}
    module_backgrounds: dict[str, set[str]] = {module: set() for module in MODULES}

    for observation in observations:
        modules = observation.variant.modules
        for module in modules:
            module_counts[module] += 1
            module_backgrounds[module].add(observation.variant.background)
        if len(modules) != 1:
            continue
        module = modules[0]
        baseline = _baseline_utility(observation.variant.background, control_utility_by_background, global_control)
        if not confounding_correction:
            baseline = global_control
        singleton_deltas[module].append(observation.observed_utility - baseline)
        singleton_backgrounds[(module, observation.variant.background)] = (
            singleton_backgrounds.get((module, observation.variant.background), 0) + 1
        )

    module_effects = {
        module: mean(values) if values else 0.0
        for module, values in singleton_deltas.items()
    }

    pair_residuals: dict[tuple[str, str], list[float]] = {}
    pair_counts: dict[tuple[str, str], int] = {}
    for observation in observations:
        modules = observation.variant.modules
        if len(modules) < 2:
            continue
        baseline = _baseline_utility(observation.variant.background, control_utility_by_background, global_control)
        if not confounding_correction:
            baseline = global_control
        additive = baseline + sum(module_effects[module] for module in modules)
        residual = observation.observed_utility - additive
        pairs = list(combinations(modules, 2))
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
    best_observed = sorted(observations, key=lambda obs: obs.observed_utility, reverse=True)[:8]
    return ReplayModel(
        observed_ids=frozenset(observation.variant.variant_id for observation in observations),
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
        best_observed_utility=best_observed[0].observed_utility if best_observed else 0.0,
        best_observed_ids=tuple(observation.variant.variant_id for observation in best_observed),
        confounding_correction=confounding_correction,
    )


def _baseline_utility(
    background: str,
    control_utility_by_background: dict[str, float],
    global_control: float,
) -> float:
    return control_utility_by_background.get(background, global_control)


def _predicted_utility(variant: SyntheticVariant, model: ReplayModel) -> float:
    value = _baseline_utility(
        variant.background,
        model.control_utility_by_background,
        model.global_control_utility,
    )
    value += sum(model.module_effects.get(module, 0.0) for module in variant.modules)
    for pair in combinations(variant.modules, 2):
        value += model.pair_effects.get(tuple(sorted(pair)), 0.0)
    value -= 0.020 * max(0, len(variant.modules) - 2)
    return value


def _uncertainty_score(variant: SyntheticVariant, model: ReplayModel) -> float:
    if not variant.modules:
        return 0.05 if variant.variant_id not in model.observed_ids else 0.0

    module_uncertainty = [
        1.0 / (1.0 + model.module_counts.get(module, 0))
        for module in variant.modules
    ]
    pair_uncertainty = [
        1.0 / (1.0 + model.pair_counts.get(tuple(sorted(pair)), 0))
        for pair in combinations(variant.modules, 2)
    ]
    missing_target_singletons = sum(
        1 for module in variant.modules
        if (module, TARGET_BACKGROUND) not in model.singleton_backgrounds
    )
    value = 0.50 * mean(module_uncertainty)
    if pair_uncertainty:
        value += 0.35 * mean(pair_uncertainty)
    value += 0.15 * (missing_target_singletons / max(1, len(variant.modules)))
    return min(1.0, value)


def _lattice_repair_score(variant: SyntheticVariant, model: ReplayModel) -> float:
    if not variant.modules:
        return 0.10 if variant.variant_id not in model.observed_ids else 0.0

    score = 0.0
    if len(variant.modules) == 1:
        module = variant.modules[0]
        if (module, variant.background) not in model.singleton_backgrounds:
            score += 0.80 if variant.background == TARGET_BACKGROUND else 0.35
        if len(model.module_backgrounds.get(module, frozenset())) == 1:
            score += 0.20
    else:
        for module in variant.modules:
            if (module, variant.background) not in model.singleton_backgrounds:
                score += 0.12
        for pair in combinations(variant.modules, 2):
            ordered_pair = tuple(sorted(pair))
            if ordered_pair not in model.pair_counts:
                score += 0.25
            if ordered_pair in PRIOR_INTERACTION_PAIRS and variant.background == TARGET_BACKGROUND:
                score += 0.15
    return min(1.0, score)


def _interaction_prior_score(variant: SyntheticVariant) -> float:
    if len(variant.modules) < 2:
        return 0.0

    module_set = set(variant.modules)
    roles = {MODULE_ROLES[module] for module in variant.modules}
    score = 0.0
    for pair in combinations(variant.modules, 2):
        ordered_pair = tuple(sorted(pair))
        if ordered_pair in PRIOR_INTERACTION_PAIRS:
            score += 0.32
        if ordered_pair in PAIR_INTERACTIONS and any(module in RISK_MODULES for module in ordered_pair):
            score -= 0.40
    if {"acid", "neutral"}.issubset(roles):
        score += 0.20
    if "developability_repair" in roles and "HD110H" in module_set:
        score += 0.12
    return max(-0.75, min(1.0, score))


def _risk_score(variant: SyntheticVariant, model: ReplayModel) -> float:
    risk = 0.0
    for module in variant.modules:
        if module in RISK_MODULES:
            risk += 0.20
        if variant.background == TARGET_BACKGROUND and (module, TARGET_BACKGROUND) not in model.singleton_backgrounds:
            risk += 0.05
    for pair in combinations(variant.modules, 2):
        ordered_pair = tuple(sorted(pair))
        if ordered_pair in FORBIDDEN_DESIGN_PAIRS:
            risk += 1.00
        if ordered_pair == ("HA23K", "KD31N"):
            risk += 0.45
    if len(variant.modules) == MAX_MODULES:
        observed_pairs = sum(
            1 for pair in combinations(variant.modules, 2)
            if tuple(sorted(pair)) in model.pair_counts
        )
        if observed_pairs == 0:
            risk += 0.12
    return min(1.0, risk)


def _top_observed_similarity(variant: SyntheticVariant, model: ReplayModel) -> float:
    if not model.best_observed_ids:
        return 0.0
    best_score = 0.0
    variant_modules = set(variant.modules)
    for variant_id in model.best_observed_ids:
        background, modules = _parse_variant_id(variant_id)
        observed_modules = set(modules)
        union_size = len(variant_modules | observed_modules)
        jaccard = 1.0 if union_size == 0 else len(variant_modules & observed_modules) / union_size
        background_bonus = 0.15 if background == variant.background else 0.0
        score = 0.35 + 0.65 * jaccard + background_bonus
        best_score = max(best_score, score * model.best_observed_utility)
    return best_score


def _parse_variant_id(variant_id: str) -> tuple[str, tuple[str, ...]]:
    background, suffix = variant_id.split("__", 1)
    modules = () if suffix == "WT" else tuple(suffix.split("_"))
    return background, modules


def _metrics(
    *,
    observations: list[SyntheticObservation],
    selected_count: int,
    observations_by_id: dict[str, SyntheticObservation],
    round_best: list[float],
    claim_guardrail: bool,
    confounding_correction: bool,
) -> dict[str, float]:
    truth = _truth_summary(observations_by_id)
    best_feasible = _best_feasible_utility(observations)
    global_best = float(truth["global_best_feasible_utility"])
    regret = max(0.0, global_best - best_feasible)
    elite_ids = _elite_feasible_ids(observations_by_id)
    observed_ids = {observation.variant.variant_id for observation in observations}
    hit_rate = len(observed_ids & elite_ids) / len(elite_ids)
    evidence_coverage = _evidence_coverage(observations)
    false_claim_rate = _false_claim_rate(
        observations,
        observations_by_id,
        claim_guardrail=claim_guardrail,
        confounding_correction=confounding_correction,
    )
    if round_best and global_best > 0:
        round_efficiency = mean(round_best) / global_best
    else:
        round_efficiency = 0.0

    return {
        "selected_count": float(selected_count),
        "best_feasible_utility": _round_metric(best_feasible),
        "hit_rate": _round_metric(hit_rate),
        "regret_proxy": _round_metric(regret),
        "false_claim_rate": _round_metric(false_claim_rate),
        "evidence_coverage": _round_metric(evidence_coverage),
        "round_efficiency": _round_metric(round_efficiency),
    }


def _best_feasible_utility(observations: list[SyntheticObservation]) -> float:
    feasible = [observation.true_utility for observation in observations if observation.true_feasible]
    return max(feasible) if feasible else 0.0


def _truth_summary(observations_by_id: dict[str, SyntheticObservation]) -> dict[str, Any]:
    feasible = [
        observation
        for observation in observations_by_id.values()
        if observation.true_feasible
    ]
    best = max(feasible, key=lambda observation: observation.true_utility)
    return {
        "global_best_feasible_variant_id": best.variant.variant_id,
        "global_best_feasible_utility": best.true_utility,
        "feasible_variant_count": len(feasible),
        "elite_feasible_count": len(_elite_feasible_ids(observations_by_id)),
    }


def _elite_feasible_ids(observations_by_id: dict[str, SyntheticObservation]) -> set[str]:
    feasible = sorted(
        [
            observation
            for observation in observations_by_id.values()
            if observation.true_feasible
        ],
        key=lambda observation: observation.true_utility,
        reverse=True,
    )
    elite_count = max(1, min(12, int(round(len(feasible) * 0.10))))
    return {observation.variant.variant_id for observation in feasible[:elite_count]}


def _evidence_coverage(observations: list[SyntheticObservation]) -> float:
    observed_atoms: set[str] = set()
    for observation in observations:
        observed_atoms.update(_evidence_atoms(observation.variant))

    target_atoms: set[str] = set()
    for background in BACKGROUND_ORDER:
        target_atoms.add(f"control:{background}")
        for module in MODULES:
            target_atoms.add(f"single:{background}:{module}")
    for pair in PRIOR_INTERACTION_PAIRS:
        target_atoms.add(f"pair:{TARGET_BACKGROUND}:{pair[0]}:{pair[1]}")

    return len(observed_atoms & target_atoms) / len(target_atoms)


def _evidence_atoms(variant: SyntheticVariant) -> set[str]:
    atoms = set()
    if not variant.modules:
        atoms.add(f"control:{variant.background}")
    if len(variant.modules) == 1:
        atoms.add(f"single:{variant.background}:{variant.modules[0]}")
    for pair in combinations(variant.modules, 2):
        ordered = tuple(sorted(pair))
        atoms.add(f"pair:{variant.background}:{ordered[0]}:{ordered[1]}")
    return atoms


def _false_claim_rate(
    observations: list[SyntheticObservation],
    observations_by_id: dict[str, SyntheticObservation],
    *,
    claim_guardrail: bool,
    confounding_correction: bool,
) -> float:
    model = _fit_model(observations, confounding_correction=confounding_correction)
    true_marginals = _true_module_marginals(observations_by_id)
    claims: list[str] = []
    for module, estimate in model.module_effects.items():
        if estimate <= 0.030:
            continue
        if claim_guardrail and len(model.module_backgrounds.get(module, frozenset())) < 2:
            continue
        claims.append(module)
    if not claims:
        return 0.0
    false_claims = [
        module
        for module in claims
        if true_marginals.get(module, 0.0) <= 0.0
    ]
    return len(false_claims) / len(claims)


def _true_module_marginals(
    observations_by_id: dict[str, SyntheticObservation],
) -> dict[str, float]:
    marginals: dict[str, list[float]] = {module: [] for module in MODULES}
    by_key = {
        (observation.variant.background, observation.variant.modules): observation
        for observation in observations_by_id.values()
    }
    for observation in observations_by_id.values():
        variant = observation.variant
        if variant.background != TARGET_BACKGROUND:
            continue
        module_set = set(variant.modules)
        if len(module_set) >= MAX_MODULES:
            continue
        for candidate_module in MODULES:
            if candidate_module in module_set:
                continue
            expanded_set = module_set | {candidate_module}
            expanded = tuple(module for module in MODULES if module in expanded_set)
            if not _design_feasible(expanded):
                continue
            expanded_observation = by_key.get((variant.background, expanded))
            if expanded_observation is None:
                continue
            marginals[candidate_module].append(expanded_observation.true_utility - observation.true_utility)
    return {
        module: mean(values) if values else 0.0
        for module, values in marginals.items()
    }


def _result_row(
    result: dict[str, Any],
    truth: dict[str, Any],
    rounds: int,
    budget: int,
    initial_observations: int,
) -> dict[str, Any]:
    metrics = result["metrics"]
    return {
        "world_id": result["world_id"],
        "method": result["method"],
        "rounds": rounds,
        "budget_per_round": budget,
        "initial_observations": initial_observations,
        "selected_count": int(result["selected_count"]),
        "total_observations": int(result["total_observations"]),
        "global_best_feasible_utility": truth["global_best_feasible_utility"],
        "best_feasible_utility": metrics["best_feasible_utility"],
        "hit_rate": metrics["hit_rate"],
        "regret_proxy": metrics["regret_proxy"],
        "false_claim_rate": metrics["false_claim_rate"],
        "evidence_coverage": metrics["evidence_coverage"],
        "round_efficiency": metrics["round_efficiency"],
    }


def _selection_novelty(
    selected_ids: Iterable[str],
    initial_ids: list[str],
    observations_by_id: dict[str, SyntheticObservation],
) -> float:
    selected = [observations_by_id[variant_id].variant for variant_id in selected_ids]
    initial = [observations_by_id[variant_id].variant for variant_id in initial_ids]
    if not selected or not initial:
        return 0.0

    novelty_values: list[float] = []
    for variant in selected:
        variant_modules = set(variant.modules)
        best_similarity = 0.0
        for initial_variant in initial:
            initial_modules = set(initial_variant.modules)
            union_size = len(variant_modules | initial_modules)
            module_similarity = 1.0 if union_size == 0 else len(variant_modules & initial_modules) / union_size
            background_similarity = 1.0 if variant.background == initial_variant.background else 0.0
            similarity = 0.80 * module_similarity + 0.20 * background_similarity
            best_similarity = max(best_similarity, similarity)
        novelty_values.append(1.0 - best_similarity)
    return _round_metric(mean(novelty_values))


def _baseline_overlap(
    selected_ids: Iterable[str],
    baseline_ids: set[str],
) -> float:
    selected = set(selected_ids)
    if not selected:
        return 0.0
    return _round_metric(len(selected & baseline_ids) / len(selected))


def _selection_digest(selected_ids: Iterable[str]) -> str:
    text = "\n".join(sorted(str(variant_id) for variant_id in selected_ids))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _benchmark_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    by_world_method: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        method = str(row["method"])
        world_id = str(row["world_id"])
        by_method.setdefault(method, []).append(row)
        by_world_method[(world_id, method)] = row

    rankings: list[dict[str, Any]] = []
    for method, method_rows in by_method.items():
        worlds = {str(row["world_id"]) for row in method_rows}
        majority_signal = _majority_signal(method_rows, by_world_method)
        rankings.append(
            {
                "method": method,
                "world_count": len(worlds),
                "mean_best_feasible_utility": _round_metric(
                    mean(float(row["best_feasible_utility"]) for row in method_rows)
                ),
                "mean_regret_proxy": _round_metric(
                    mean(float(row["regret_proxy"]) for row in method_rows)
                ),
                "mean_hit_rate": _round_metric(
                    mean(float(row["hit_rate"]) for row in method_rows)
                ),
                "mean_novelty_score": _round_metric(
                    mean(float(row["novelty_score"]) for row in method_rows)
                ),
                "mean_baseline_overlap": _round_metric(
                    mean(float(row["baseline_overlap"]) for row in method_rows)
                ),
                **majority_signal,
            }
        )

    rankings.sort(
        key=lambda row: (
            float(row["mean_best_feasible_utility"]),
            -float(row["mean_regret_proxy"]),
            float(row["mean_hit_rate"]),
        ),
        reverse=True,
    )
    for rank, row in enumerate(rankings, start=1):
        row["rank"] = rank
    return {"method_rankings": rankings}


def _majority_signal(
    method_rows: list[dict[str, Any]],
    by_world_method: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    world_count = len({str(row["world_id"]) for row in method_rows})
    signal: dict[str, Any] = {}
    for baseline in ("random_feasible", "fixed_mix"):
        wins = 0
        comparable = 0
        for row in method_rows:
            baseline_row = by_world_method.get((str(row["world_id"]), baseline))
            if baseline_row is None:
                continue
            selected_best = _optional_float(row.get("best_feasible_utility"))
            baseline_best = _optional_float(baseline_row.get("best_feasible_utility"))
            if selected_best is None or baseline_best is None:
                continue
            comparable += 1
            if selected_best > baseline_best:
                wins += 1
        denominator = comparable or world_count
        signal[f"beats_{baseline}_worlds"] = wins
        signal[f"beats_{baseline}_majority"] = "true" if denominator > 0 and wins > denominator / 2 else "false"
    return signal


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stable_seed(seed: int, *parts: Any) -> int:
    payload = "|".join(str(part) for part in (seed, *parts))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _stable_noise(seed: int, *parts: Any, scale: float) -> float:
    first = _stable_unit(seed, *parts, "a")
    second = _stable_unit(seed, *parts, "b")
    return (first + second - 1.0) * scale


def _stable_unit(seed: int, *parts: Any) -> float:
    payload = "|".join(str(part) for part in (seed, *parts))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def _round_metric(value: float) -> float:
    return round(float(value), 6)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=False)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _ensure_dir(path.parent)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
