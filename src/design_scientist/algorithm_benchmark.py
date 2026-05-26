"""Generic MCCBD algorithm benchmark over controlled synthetic mechanisms.

The benchmark is deliberately independent of any one biological project. It
constructs small mechanism worlds with known truth, exposes only generic
candidate records to selectors, and reports deterministic multi-round active
design metrics for MCCBD, ablations, and simple baselines.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import math
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_RUN_ID = "mccbd_algorithm"
DEFAULT_SEEDS = range(10)
DEFAULT_ROUNDS = 3
DEFAULT_BUDGET = 6
DEFAULT_WORLDS = (
    "additive",
    "epistatic",
    "constrained_risky_decoy",
    "sparse_observation",
    "batch_redundancy",
)
DEFAULT_MECHANISMS = (
    "mccbd",
    "mccbd_no_constraint",
    "mccbd_no_information_value",
    "mccbd_no_batch_diversity",
    "evidence_calibrated_ucb",
    "random_feasible",
    "fixed_mix",
    "greedy_observed",
)
MCCBD_ABLATIONS = (
    ("full", "mccbd"),
    ("mccbd_no_constraint", "mccbd_no_constraint"),
    ("mccbd_no_information_value", "mccbd_no_information_value"),
    ("mccbd_no_batch_diversity", "mccbd_no_batch_diversity"),
)
MCCBD_ABLATION_MODES = {
    "mccbd": "full",
    "mccbd_no_constraint": "no_constraint",
    "mccbd_no_information_value": "no_information_value",
    "mccbd_no_batch_diversity": "no_batch_diversity",
}
BASELINE_MECHANISMS = ("random_feasible", "fixed_mix")
SELECTABLE_PREFERRED = ("mccbd", "evidence_calibrated_ucb")

MODULES = (
    "stabilizer_A",
    "binder_B",
    "release_C",
    "specificity_D",
    "developability_E",
    "scout_F",
    "liability_R",
    "decoy_X",
)
MAX_MODULES = 3
ENDPOINT_WEIGHTS = {
    "activity": 0.42,
    "selectivity": 0.24,
    "developability": 0.22,
    "robustness": 0.12,
}
FEASIBILITY_THRESHOLDS = {
    "activity": 0.34,
    "selectivity": 0.42,
    "developability": 0.42,
    "robustness": 0.38,
}

BENCHMARK_COLUMNS = (
    "status",
    "error",
    "seed",
    "world_id",
    "mechanism",
    "rounds",
    "budget",
    "initial_observation_count",
    "candidate_count",
    "selected_count",
    "selected_ids",
    "selected_feasible_ids",
    "selected_infeasible_ids",
    "best_feasible_utility",
    "hit_rate",
    "regret_proxy",
    "false_claim_rate",
    "evidence_coverage",
    "selected_diversity",
    "candidate_record_example",
)
SUMMARY_COLUMNS = (
    "mechanism",
    "world_id",
    "replicate_count",
    "mean_best_feasible_utility",
    "mean_hit_rate",
    "mean_regret_proxy",
    "mean_false_claim_rate",
    "mean_evidence_coverage",
    "mean_selected_diversity",
    "majority_win_vs_random_feasible",
    "majority_win_vs_fixed_mix",
    "selected_eligible",
)
ABLATION_COLUMNS = (
    "status",
    "error",
    "seed",
    "world_id",
    "mechanism",
    "ablation",
    "rounds",
    "budget",
    "selected_count",
    "selected_ids",
    "best_feasible_utility",
    "hit_rate",
    "regret_proxy",
    "false_claim_rate",
    "evidence_coverage",
    "selected_diversity",
    "delta_from_full_best_feasible_utility",
)
CANDIDATE_VISIBLE_FIELDS = (
    "variant_id",
    "modules",
    "cost",
    "mechanism_world",
    "mechanism_provenance",
    "prior_utility",
    "predicted_feasible",
    "constraint_margin",
    "uncertainty",
    "information_value",
    "risk_flags",
    "mechanism_tags",
    "architecture_id",
    "numeric_features",
    "required_measurements",
    "missing_endpoints",
    "feasibility_status",
    "source_refs",
)
CANDIDATE_VISIBLE_PROVENANCE_FIELDS = ("world_id", "seed", "description")
CANDIDATE_FORBIDDEN_KEYS = frozenset(
    {
        "endpoint_values",
        "true_utility",
        "true_feasible",
        "observed_utility",
        "observed_score",
        "observed_feasible",
        "utility",
        "score",
        "true_endpoints",
        "observed_endpoints",
        "truth_terms",
        "constraint_terms",
    }
)


@dataclass(frozen=True)
class WorldSpec:
    """Synthetic mechanism world specification."""

    world_id: str
    description: str
    initial_singletons: tuple[str, ...]
    forbidden_pairs: tuple[tuple[str, str], ...]
    risky_modules: tuple[str, ...]
    hidden_modules: tuple[str, ...] = ()


def run_mccbd_benchmark(
    project_dir: str | Path,
    run_id: str = DEFAULT_RUN_ID,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    rounds: int = DEFAULT_ROUNDS,
    budget: int = DEFAULT_BUDGET,
    mechanisms: Iterable[str] | str | None = None,
    worlds: Iterable[str] | str | None = None,
) -> dict[str, Any]:
    """Run a deterministic generic benchmark comparing MCCBD to baselines."""

    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if budget <= 0:
        raise ValueError("budget must be positive")

    safe_run_id = _safe_run_id(run_id)
    root = Path(project_dir).expanduser().resolve()
    run_dir = root / "runs" / safe_run_id
    seed_values = [int(seed) for seed in seeds]
    if not seed_values:
        raise ValueError("seeds must contain at least one seed")
    mechanism_names = _normalize_mechanisms(mechanisms)
    world_specs = _select_worlds(worlds)

    result_cache: dict[tuple[str, int, str], dict[str, Any]] = {}
    benchmark_rows: list[dict[str, Any]] = []
    for world in world_specs:
        for seed in seed_values:
            records = generate_synthetic_mechanism_world(world.world_id, seed)
            for mechanism in mechanism_names:
                row = _run_cached_simulation(
                    result_cache,
                    records=records,
                    world=world,
                    seed=seed,
                    mechanism=mechanism,
                    rounds=rounds,
                    budget=budget,
                )
                benchmark_rows.append(_project_row(row, BENCHMARK_COLUMNS))

    ablation_rows = _ablation_rows(
        result_cache,
        world_specs=world_specs,
        seeds=seed_values,
        rounds=rounds,
        budget=budget,
        include_mccbd="mccbd" in mechanism_names,
    )
    summary_rows = _summary_rows(benchmark_rows)
    selected_mechanism, selected_passes_gate = _select_winner(summary_rows, benchmark_rows)

    config = {
        "benchmark": "generic_mccbd_algorithm_benchmark",
        "schema_version": 1,
        "run_id": safe_run_id,
        "seeds": seed_values,
        "rounds": rounds,
        "budget": budget,
        "worlds": [world.world_id for world in world_specs],
        "mechanisms": mechanism_names,
        "required_candidate_fields": [
            "variant_id",
            "modules",
            "cost",
            "prior_utility",
            "predicted_feasible",
            "feasibility_status",
            "uncertainty",
            "information_value",
            "risk_flags",
            "mechanism_tags",
            "mechanism_world",
            "mechanism_provenance",
        ],
        "metrics": [
            "best_feasible_utility",
            "hit_rate",
            "regret_proxy",
            "false_claim_rate",
            "evidence_coverage",
            "selected_diversity",
        ],
        "winner_gate": {
            "baselines": list(BASELINE_MECHANISMS),
            "rule": (
                "selected mechanism must beat random_feasible and fixed_mix on "
                "a majority of worlds, counting equal best feasible utility "
                "with lower false_claim_rate as a win, and have "
                "false_claim_rate no worse than the worse of those baselines"
            ),
        },
        "selected_mechanism": selected_mechanism,
        "selected_passes_gate": selected_passes_gate,
    }

    benchmark_path = run_dir / "mccbd_benchmark_results.csv"
    summary_path = run_dir / "mccbd_benchmark_summary.csv"
    ablation_path = run_dir / "mccbd_ablation_results.csv"
    config_path = run_dir / "mccbd_benchmark_config.json"
    _write_csv(benchmark_path, benchmark_rows, BENCHMARK_COLUMNS)
    _write_csv(summary_path, summary_rows, SUMMARY_COLUMNS)
    _write_csv(ablation_path, ablation_rows, ABLATION_COLUMNS)
    _write_json(config_path, config)

    artifact_paths = {
        "benchmark_results": str(benchmark_path),
        "benchmark_summary": str(summary_path),
        "ablation_results": str(ablation_path),
        "benchmark_config": str(config_path),
    }
    return {
        "status": "completed",
        "run_id": safe_run_id,
        "artifact_paths": artifact_paths,
        "benchmark_results_path": str(benchmark_path),
        "summary_results_path": str(summary_path),
        "ablation_results_path": str(ablation_path),
        "config_path": str(config_path),
        "mccbd_benchmark_results_path": str(benchmark_path),
        "mccbd_benchmark_summary_path": str(summary_path),
        "mccbd_ablation_results_path": str(ablation_path),
        "mccbd_benchmark_config_path": str(config_path),
        "selected_mechanism": selected_mechanism,
        "selected_passes_gate": selected_passes_gate,
        "summary": {"mechanism_rankings": summary_rows},
    }


def generate_synthetic_mechanism_world(world_id: str, seed: int) -> list[dict[str, Any]]:
    """Generate candidate records for a controlled generic mechanism world."""

    world = _world_by_id(world_id)
    records: list[dict[str, Any]] = []
    for size in range(0, MAX_MODULES + 1):
        for modules in combinations(MODULES, size):
            endpoints, truth_terms = _endpoint_truth(world, modules)
            true_feasible, constraint_terms = _true_feasibility(world, modules, endpoints)
            true_utility = _utility(endpoints)
            prior_utility, observed_utility = _observed_surrogates(
                world,
                modules,
                true_utility,
                true_feasible,
                seed,
            )
            predicted_feasible, constraint_margin = _predicted_feasibility(
                world,
                modules,
                endpoints,
                seed,
            )
            risk_flags = _risk_flags(world, modules)
            record = {
                "variant_id": _variant_id(world.world_id, modules),
                "modules": list(modules),
                "endpoint_values": _rounded_mapping(endpoints),
                "true_utility": _round_metric(true_utility),
                "true_feasible": bool(true_feasible),
                "cost": _round_metric(_candidate_cost(modules)),
                "mechanism_world": world.world_id,
                "mechanism_provenance": {
                    "world_id": world.world_id,
                    "seed": int(seed),
                    "description": world.description,
                    "truth_terms": truth_terms,
                    "constraint_terms": constraint_terms,
                    "forbidden_pairs": [list(pair) for pair in world.forbidden_pairs],
                    "risky_modules": list(world.risky_modules),
                    "hidden_modules": list(world.hidden_modules),
                },
                "observed_utility": _round_metric(observed_utility),
                "prior_utility": _round_metric(prior_utility),
                "observed_score": _round_metric(observed_utility),
                "predicted_feasible": bool(predicted_feasible),
                "constraint_margin": _round_metric(constraint_margin),
                "uncertainty": _round_metric(_candidate_uncertainty(world, modules)),
                "information_value": _round_metric(_static_information_value(world, modules)),
                "risk_flags": risk_flags,
                "mechanism_tags": _mechanism_tags(world, modules),
                "architecture_id": _architecture_id(modules),
                "required_measurements": list(ENDPOINT_WEIGHTS),
                "missing_endpoints": [],
                "feasibility_status": "feasible" if predicted_feasible else "infeasible",
                "source_refs": [
                    {
                        "kind": "synthetic_mechanism_world",
                        "world_id": world.world_id,
                        "seed": int(seed),
                    }
                ],
            }
            records.append(record)
    return sorted(records, key=lambda item: str(item["variant_id"]))


def _run_cached_simulation(
    cache: dict[tuple[str, int, str], dict[str, Any]],
    *,
    records: Sequence[Mapping[str, Any]],
    world: WorldSpec,
    seed: int,
    mechanism: str,
    rounds: int,
    budget: int,
) -> dict[str, Any]:
    key = (world.world_id, int(seed), mechanism)
    if key not in cache:
        cache[key] = _run_simulation(
            records=records,
            world=world,
            seed=seed,
            mechanism=mechanism,
            rounds=rounds,
            budget=budget,
        )
    return cache[key]


def _run_simulation(
    *,
    records: Sequence[Mapping[str, Any]],
    world: WorldSpec,
    seed: int,
    mechanism: str,
    rounds: int,
    budget: int,
) -> dict[str, Any]:
    records_by_id = {str(record["variant_id"]): dict(record) for record in records}
    initial_ids = _initial_observation_ids(world, records_by_id)
    observed_ids = set(initial_ids)
    selected_ids: list[str] = []
    evidence_coverages: list[float] = []
    status = "completed"
    error = ""

    try:
        for round_index in range(rounds):
            observed_records = [
                _observed_record(records_by_id[variant_id])
                for variant_id in sorted(observed_ids)
            ]
            candidate_records = [
                _candidate_record(records_by_id[variant_id])
                for variant_id in sorted(records_by_id)
                if variant_id not in observed_ids
            ]
            if not candidate_records:
                break
            rng = random.Random(
                _stable_seed(seed, world.world_id, mechanism, "round", round_index)
            )
            raw_selected = _select_for_mechanism(
                mechanism,
                observed_records,
                candidate_records,
                budget,
                round_index,
                rng,
            )
            selected = _resolve_selected(raw_selected, candidate_records, budget)
            if not selected:
                break
            observed_before = set(observed_ids)
            for variant_id in selected:
                if variant_id in observed_ids:
                    continue
                coverage = _evidence_coverage_for_candidate(
                    records_by_id[variant_id],
                    [records_by_id[item] for item in observed_before],
                )
                evidence_coverages.append(coverage)
                selected_ids.append(variant_id)
                observed_ids.add(variant_id)
    except Exception as exc:  # pragma: no cover - retained as CSV diagnostic path
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"

    metrics = _metrics(records_by_id, selected_ids, evidence_coverages)
    selected_records = [records_by_id[variant_id] for variant_id in selected_ids]
    selected_feasible = [
        str(record["variant_id"]) for record in selected_records if bool(record["true_feasible"])
    ]
    selected_infeasible = [
        str(record["variant_id"]) for record in selected_records if not bool(record["true_feasible"])
    ]
    example = _candidate_record(next(iter(records_by_id.values())))
    return {
        "status": status,
        "error": error,
        "seed": int(seed),
        "world_id": world.world_id,
        "mechanism": mechanism,
        "rounds": int(rounds),
        "budget": int(budget),
        "initial_observation_count": len(initial_ids),
        "candidate_count": len(records_by_id),
        "selected_count": len(selected_ids),
        "selected_ids": selected_ids,
        "selected_feasible_ids": selected_feasible,
        "selected_infeasible_ids": selected_infeasible,
        "candidate_record_example": example,
        **metrics,
    }


def _select_for_mechanism(
    mechanism: str,
    observed: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    budget: int,
    round_index: int,
    rng: random.Random,
) -> list[str]:
    if mechanism in MCCBD_ABLATION_MODES:
        ablation_mode = MCCBD_ABLATION_MODES[mechanism]
        external = _external_mccbd_selection(
            observed,
            candidates,
            budget,
            round_index,
            rng,
            ablation_mode,
        )
        resolved = _resolve_selected(external, candidates, budget) if external is not None else []
        if external is not None:
            return resolved
        return _fallback_mccbd_select(
            observed,
            candidates,
            budget,
            use_constraint=ablation_mode != "no_constraint",
            use_information_value=ablation_mode != "no_information_value",
            use_batch_diversity=ablation_mode != "no_batch_diversity",
        )
    if mechanism == "evidence_calibrated_ucb":
        return _evidence_calibrated_ucb_select(observed, candidates, budget, round_index, rng)
    if mechanism == "random_feasible":
        return _random_feasible_select(candidates, budget, rng)
    if mechanism == "fixed_mix":
        return _fixed_mix_select(candidates, budget)
    if mechanism == "greedy_observed":
        return _rank_select(candidates, budget, lambda item, _selected: _observed_score(item))
    raise ValueError(f"Unknown benchmark mechanism: {mechanism}")


def _external_mccbd_selection(
    observed: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    budget: int,
    round_index: int,
    rng: random.Random,
    ablation_mode: str,
) -> list[Any] | None:
    try:
        module = importlib.import_module("design_scientist.algorithms.mccbd")
    except ImportError:
        return None

    select_batch = getattr(module, "select_batch", None)
    config_cls = getattr(module, "MCCBDConfig", None)
    if not callable(select_batch) or config_cls is None:
        raise AttributeError("design_scientist.algorithms.mccbd.select_batch is unavailable")
    return list(
        select_batch(
            observed,
            candidates,
            budget,
            round_index,
            rng,
            config=config_cls(ablation_mode=ablation_mode),
        )
    )


def _fallback_mccbd_select(
    observed: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    budget: int,
    *,
    use_constraint: bool,
    use_information_value: bool,
    use_batch_diversity: bool,
) -> list[str]:
    model = _fit_observed_model(observed)

    def score(candidate: Mapping[str, Any], selected: Sequence[Mapping[str, Any]]) -> float:
        predicted = _model_prediction(model, candidate)
        prior = _to_float(candidate.get("prior_utility"), 0.0)
        total = 0.58 * predicted + 0.42 * prior
        if use_information_value:
            total += 0.20 * _to_float(candidate.get("uncertainty"), 0.0)
            total += 0.24 * _dynamic_information_value(candidate, observed)
        if use_constraint:
            if not _predicted_feasible(candidate):
                total -= 1.20
            total += 0.18 * _to_float(candidate.get("constraint_margin"), 0.0)
            total -= 0.08 * len(_as_list(candidate.get("risk_flags")))
        total -= 0.035 * max(_cost(candidate) - 1.0, 0.0)
        if use_batch_diversity and selected:
            total -= 0.42 * max(
                (_jaccard(candidate, item) for item in selected),
                default=0.0,
            )
            total += 0.08 * _architecture_novelty(candidate, selected)
        return total

    return _rank_select(candidates, budget, score)


def _evidence_calibrated_ucb_select(
    observed: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    budget: int,
    round_index: int,
    rng: random.Random,
) -> list[str]:
    try:
        from design_scientist.algorithms import evidence_calibrated_ucb

        return list(
            evidence_calibrated_ucb.select_batch(
                observed,
                candidates,
                budget,
                round_index,
                rng,
            )
        )
    except Exception:
        return _fallback_mccbd_select(
            observed,
            candidates,
            budget,
            use_constraint=True,
            use_information_value=True,
            use_batch_diversity=False,
        )


def _random_feasible_select(
    candidates: Sequence[Mapping[str, Any]],
    budget: int,
    rng: random.Random,
) -> list[str]:
    feasible = [
        candidate
        for candidate in candidates
        if _predicted_feasible(candidate) and str(candidate.get("feasibility_status")) != "blocked"
    ]
    rng.shuffle(feasible)
    return _resolve_selected(feasible, feasible, budget)


def _fixed_mix_select(candidates: Sequence[Mapping[str, Any]], budget: int) -> list[str]:
    ranked_by_prior = sorted(
        candidates,
        key=lambda item: (_observed_score(item), -len(_modules(item)), str(item["variant_id"])),
        reverse=True,
    )
    ranked_by_uncertainty = sorted(
        candidates,
        key=lambda item: (
            _to_float(item.get("uncertainty"), 0.0),
            -_cost(item),
            str(item["variant_id"]),
        ),
        reverse=True,
    )
    ranked_by_cost = sorted(
        candidates,
        key=lambda item: (_cost(item), -_observed_score(item), str(item["variant_id"])),
    )
    schedule = [ranked_by_prior, ranked_by_cost, ranked_by_uncertainty, ranked_by_prior]
    ordered: list[Mapping[str, Any]] = []
    for bucket in schedule:
        for candidate in bucket:
            if str(candidate["variant_id"]) not in {str(item["variant_id"]) for item in ordered}:
                ordered.append(candidate)
                break
    ordered.extend(
        candidate
        for candidate in ranked_by_prior
        if str(candidate["variant_id"]) not in {str(item["variant_id"]) for item in ordered}
    )
    return _resolve_selected(ordered, candidates, budget)


def _rank_select(
    candidates: Sequence[Mapping[str, Any]],
    budget: int,
    score_fn: Any,
) -> list[str]:
    selected_rows: list[Mapping[str, Any]] = []
    selected_ids: list[str] = []
    spent = 0.0
    remaining = [candidate for candidate in candidates if _cost(candidate) <= budget]
    while remaining:
        affordable = [candidate for candidate in remaining if spent + _cost(candidate) <= budget]
        if not affordable:
            break
        row = max(
            affordable,
            key=lambda item: (
                float(score_fn(item, selected_rows)),
                -_cost(item),
                str(item["variant_id"]),
            ),
        )
        selected_rows.append(row)
        selected_ids.append(str(row["variant_id"]))
        spent += _cost(row)
        remaining.remove(row)
        if spent >= budget:
            break
    return selected_ids


def _resolve_selected(
    selected: Sequence[Any] | None,
    candidates: Sequence[Mapping[str, Any]],
    budget: int,
) -> list[str]:
    if selected is None:
        return []
    by_id = {
        str(candidate.get("variant_id") or candidate.get("candidate_id")): candidate
        for candidate in candidates
    }
    resolved: list[str] = []
    spent = 0.0
    for item in selected:
        if isinstance(item, Mapping):
            variant_id = str(item.get("variant_id") or item.get("candidate_id") or "")
        else:
            variant_id = str(item)
        if not variant_id or variant_id in resolved or variant_id not in by_id:
            continue
        cost = _cost(by_id[variant_id])
        if spent + cost > budget:
            continue
        resolved.append(variant_id)
        spent += cost
        if spent >= budget:
            break
    return resolved


def _metrics(
    records_by_id: Mapping[str, Mapping[str, Any]],
    selected_ids: Sequence[str],
    evidence_coverages: Sequence[float],
) -> dict[str, float]:
    selected = [records_by_id[variant_id] for variant_id in selected_ids if variant_id in records_by_id]
    feasible_selected = [record for record in selected if bool(record["true_feasible"])]
    feasible_all = [record for record in records_by_id.values() if bool(record["true_feasible"])]
    best_possible = max((_to_float(record.get("true_utility"), 0.0) for record in feasible_all), default=0.0)
    best_feasible = max(
        (_to_float(record.get("true_utility"), 0.0) for record in feasible_selected),
        default=0.0,
    )
    hit_threshold = _quantile(
        [_to_float(record.get("true_utility"), 0.0) for record in feasible_all],
        0.80,
    )
    selected_count = len(selected)
    hit_count = sum(
        bool(record["true_feasible"]) and _to_float(record.get("true_utility"), 0.0) >= hit_threshold
        for record in selected
    )
    false_count = sum(not bool(record["true_feasible"]) for record in selected)
    return {
        "best_feasible_utility": _round_metric(best_feasible),
        "hit_rate": _round_metric(hit_count / selected_count if selected_count else 0.0),
        "regret_proxy": _round_metric(max(best_possible - best_feasible, 0.0)),
        "false_claim_rate": _round_metric(false_count / selected_count if selected_count else 0.0),
        "evidence_coverage": _round_metric(mean(evidence_coverages) if evidence_coverages else 0.0),
        "selected_diversity": _round_metric(_selected_diversity(selected)),
    }


def _summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    mechanisms = sorted({str(row["mechanism"]) for row in rows})
    worlds = sorted({str(row["world_id"]) for row in rows})
    for row in rows:
        grouped.setdefault((str(row["mechanism"]), str(row["world_id"])), []).append(row)
        grouped.setdefault((str(row["mechanism"]), "overall"), []).append(row)

    summary: list[dict[str, Any]] = []
    for mechanism in mechanisms:
        for world_id in [*worlds, "overall"]:
            mechanism_rows = grouped.get((mechanism, world_id), [])
            if not mechanism_rows:
                continue
            row = _aggregate_summary_row(mechanism, world_id, mechanism_rows)
            row["majority_win_vs_random_feasible"] = _bool_text(
                _majority_win(rows, mechanism, "random_feasible")
                if world_id == "overall"
                else _world_win(rows, mechanism, "random_feasible", world_id)
            )
            row["majority_win_vs_fixed_mix"] = _bool_text(
                _majority_win(rows, mechanism, "fixed_mix")
                if world_id == "overall"
                else _world_win(rows, mechanism, "fixed_mix", world_id)
            )
            row["selected_eligible"] = _bool_text(
                _passes_gate(rows, mechanism) if world_id == "overall" else False
            )
            summary.append(row)

    return sorted(summary, key=lambda item: (str(item["mechanism"]), str(item["world_id"])))


def _aggregate_summary_row(
    mechanism: str,
    world_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "mechanism": mechanism,
        "world_id": world_id,
        "replicate_count": len(rows),
        "mean_best_feasible_utility": _mean_metric(rows, "best_feasible_utility"),
        "mean_hit_rate": _mean_metric(rows, "hit_rate"),
        "mean_regret_proxy": _mean_metric(rows, "regret_proxy"),
        "mean_false_claim_rate": _mean_metric(rows, "false_claim_rate"),
        "mean_evidence_coverage": _mean_metric(rows, "evidence_coverage"),
        "mean_selected_diversity": _mean_metric(rows, "selected_diversity"),
    }


def _select_winner(
    summary_rows: Sequence[Mapping[str, Any]],
    benchmark_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, bool]:
    overall = [row for row in summary_rows if row.get("world_id") == "overall"]
    by_mechanism = {str(row["mechanism"]): row for row in overall}
    if "mccbd" in by_mechanism and _passes_gate(benchmark_rows, "mccbd"):
        return "mccbd", True

    preferred_candidates = [
        mechanism for mechanism in SELECTABLE_PREFERRED if mechanism in by_mechanism
    ]
    other_candidates = [
        mechanism
        for mechanism in by_mechanism
        if mechanism not in set(BASELINE_MECHANISMS)
        and not mechanism.startswith("mccbd_no_")
        and mechanism not in preferred_candidates
    ]
    ranked = sorted(
        [*preferred_candidates, *other_candidates, *by_mechanism],
        key=lambda name: (
            _passes_gate(benchmark_rows, name),
            _to_float(by_mechanism[name].get("mean_best_feasible_utility"), 0.0),
            _to_float(by_mechanism[name].get("mean_hit_rate"), 0.0),
            -_to_float(by_mechanism[name].get("mean_false_claim_rate"), 0.0),
            -_mechanism_priority(name),
        ),
        reverse=True,
    )
    selected = ranked[0] if ranked else ""
    return selected, _passes_gate(benchmark_rows, selected) if selected else False


def _passes_gate(rows: Sequence[Mapping[str, Any]], mechanism: str) -> bool:
    if mechanism in {"", *BASELINE_MECHANISMS} or mechanism.startswith("mccbd_no_"):
        return False
    if not _majority_win(rows, mechanism, "random_feasible"):
        return False
    if not _majority_win(rows, mechanism, "fixed_mix"):
        return False
    selected_false = _overall_mean(rows, mechanism, "false_claim_rate")
    random_false = _overall_mean(rows, "random_feasible", "false_claim_rate")
    fixed_false = _overall_mean(rows, "fixed_mix", "false_claim_rate")
    if any(value is None for value in (selected_false, random_false, fixed_false)):
        return False
    return float(selected_false) <= max(float(random_false), float(fixed_false)) + 1e-12


def _majority_win(
    rows: Sequence[Mapping[str, Any]],
    mechanism: str,
    baseline: str,
) -> bool:
    worlds = sorted({str(row["world_id"]) for row in rows})
    comparable = 0
    wins = 0
    for world_id in worlds:
        if _world_mean(rows, mechanism, world_id, "best_feasible_utility") is None:
            continue
        if _world_mean(rows, baseline, world_id, "best_feasible_utility") is None:
            continue
        comparable += 1
        if _world_win(rows, mechanism, baseline, world_id):
            wins += 1
    return comparable > 0 and wins > comparable / 2.0


def _world_win(
    rows: Sequence[Mapping[str, Any]],
    mechanism: str,
    baseline: str,
    world_id: str,
) -> bool:
    selected = _world_mean(rows, mechanism, world_id, "best_feasible_utility")
    base = _world_mean(rows, baseline, world_id, "best_feasible_utility")
    if selected is None or base is None:
        return False
    if selected > base + 1e-12:
        return True
    if abs(selected - base) > 1e-12:
        return False
    selected_false = _world_mean(rows, mechanism, world_id, "false_claim_rate")
    base_false = _world_mean(rows, baseline, world_id, "false_claim_rate")
    return bool(
        selected_false is not None
        and base_false is not None
        and selected_false < base_false - 1e-12
    )


def _ablation_rows(
    cache: dict[tuple[str, int, str], dict[str, Any]],
    *,
    world_specs: Sequence[WorldSpec],
    seeds: Sequence[int],
    rounds: int,
    budget: int,
    include_mccbd: bool,
) -> list[dict[str, Any]]:
    if not include_mccbd:
        return []

    rows: list[dict[str, Any]] = []
    for world in world_specs:
        for seed in seeds:
            records = generate_synthetic_mechanism_world(world.world_id, seed)
            full = _run_cached_simulation(
                cache,
                records=records,
                world=world,
                seed=seed,
                mechanism="mccbd",
                rounds=rounds,
                budget=budget,
            )
            full_best = _to_float(full.get("best_feasible_utility"), 0.0)
            for ablation, mechanism in MCCBD_ABLATIONS:
                result = _run_cached_simulation(
                    cache,
                    records=records,
                    world=world,
                    seed=seed,
                    mechanism=mechanism,
                    rounds=rounds,
                    budget=budget,
                )
                row = {
                    **result,
                    "mechanism": "mccbd",
                    "ablation": ablation,
                    "delta_from_full_best_feasible_utility": _round_metric(
                        full_best - _to_float(result.get("best_feasible_utility"), 0.0)
                    ),
                }
                rows.append(_project_row(row, ABLATION_COLUMNS))
    return rows


def _endpoint_truth(
    world: WorldSpec,
    modules: Sequence[str],
) -> tuple[dict[str, float], dict[str, Any]]:
    endpoints = {
        "activity": 0.31,
        "selectivity": 0.54,
        "developability": 0.57,
        "robustness": 0.51,
    }
    effects = _module_effects(world.world_id)
    interactions = _interaction_effects(world.world_id)
    terms: dict[str, Any] = {"module_effects": {}, "interaction_effects": {}}

    for module in modules:
        module_effect = effects.get(module, {})
        terms["module_effects"][module] = module_effect
        for endpoint, delta in module_effect.items():
            endpoints[endpoint] += delta

    for pair in combinations(sorted(modules), 2):
        pair_key = tuple(pair)
        pair_effect = interactions.get(pair_key, {})
        if pair_effect:
            terms["interaction_effects"]["+".join(pair_key)] = pair_effect
        for endpoint, delta in pair_effect.items():
            endpoints[endpoint] += delta

    for triple in combinations(sorted(modules), 3):
        triple_effect = _triple_interaction_effect(world.world_id, triple)
        if triple_effect:
            terms.setdefault("triple_interaction_effects", {})["+".join(triple)] = triple_effect
        for endpoint, delta in triple_effect.items():
            endpoints[endpoint] += delta

    return {key: _clip(value) for key, value in endpoints.items()}, terms


def _module_effects(world_id: str) -> dict[str, dict[str, float]]:
    base = {
        "stabilizer_A": {"activity": 0.10, "developability": 0.08, "robustness": 0.04},
        "binder_B": {"activity": 0.13, "selectivity": 0.05},
        "release_C": {"activity": 0.08, "robustness": 0.10},
        "specificity_D": {"selectivity": 0.13, "activity": 0.04},
        "developability_E": {"developability": 0.15, "robustness": 0.04},
        "scout_F": {"activity": 0.03, "robustness": 0.08},
        "liability_R": {"activity": 0.16, "selectivity": -0.18, "developability": -0.15},
        "decoy_X": {
            "activity": 0.30,
            "selectivity": -0.28,
            "developability": -0.24,
            "robustness": -0.12,
        },
    }
    if world_id == "epistatic":
        for item in base.values():
            for endpoint in list(item):
                item[endpoint] *= 0.62
    elif world_id == "constrained_risky_decoy":
        base["decoy_X"] = {
            "activity": 0.42,
            "selectivity": -0.34,
            "developability": -0.31,
            "robustness": -0.16,
        }
        base["liability_R"] = {
            "activity": 0.22,
            "selectivity": -0.24,
            "developability": -0.20,
        }
        base["stabilizer_A"]["developability"] = 0.13
        base["specificity_D"]["selectivity"] = 0.18
    elif world_id == "sparse_observation":
        base["developability_E"] = {"developability": 0.08, "robustness": 0.03}
        base["scout_F"] = {"activity": 0.02, "robustness": 0.05}
        base["binder_B"] = {"activity": 0.10, "selectivity": 0.04}
    elif world_id == "batch_redundancy":
        base["stabilizer_A"] = {"activity": 0.12, "developability": 0.08}
        base["binder_B"] = {"activity": 0.11, "selectivity": 0.05}
        base["release_C"] = {"activity": 0.10, "robustness": 0.08}
        base["specificity_D"] = {"activity": 0.09, "selectivity": 0.09}
        base["developability_E"] = {"developability": 0.16, "robustness": 0.04}
        base["scout_F"] = {"selectivity": 0.10, "robustness": 0.07}
    return base


def _interaction_effects(world_id: str) -> dict[tuple[str, str], dict[str, float]]:
    if world_id == "additive":
        return {
            ("binder_B", "release_C"): {"activity": 0.03},
            ("specificity_D", "stabilizer_A"): {"developability": 0.03},
        }
    if world_id == "epistatic":
        return {
            ("binder_B", "stabilizer_A"): {"activity": 0.22, "selectivity": 0.07},
            ("release_C", "specificity_D"): {"activity": 0.12, "robustness": 0.16},
            ("developability_E", "scout_F"): {"developability": 0.12, "robustness": 0.10},
            ("decoy_X", "liability_R"): {"activity": 0.18, "selectivity": -0.12},
        }
    if world_id == "constrained_risky_decoy":
        return {
            ("binder_B", "specificity_D"): {"activity": 0.12, "selectivity": 0.10},
            ("developability_E", "stabilizer_A"): {"developability": 0.12},
            ("decoy_X", "liability_R"): {
                "activity": 0.22,
                "selectivity": -0.15,
                "developability": -0.13,
            },
        }
    if world_id == "sparse_observation":
        return {
            ("developability_E", "scout_F"): {
                "activity": 0.26,
                "developability": 0.14,
                "robustness": 0.12,
            },
            ("binder_B", "release_C"): {"activity": 0.07, "robustness": 0.04},
            ("specificity_D", "stabilizer_A"): {"selectivity": 0.06},
        }
    if world_id == "batch_redundancy":
        return {
            ("binder_B", "stabilizer_A"): {"activity": 0.12, "selectivity": 0.04},
            ("release_C", "stabilizer_A"): {"activity": 0.11, "robustness": 0.05},
            ("specificity_D", "stabilizer_A"): {"activity": 0.10, "selectivity": 0.05},
            ("developability_E", "scout_F"): {"developability": 0.13, "robustness": 0.09},
        }
    return {}


def _triple_interaction_effect(
    world_id: str,
    modules: Sequence[str],
) -> dict[str, float]:
    key = tuple(sorted(modules))
    if world_id == "epistatic" and key == (
        "binder_B",
        "release_C",
        "stabilizer_A",
    ):
        return {"activity": 0.08, "robustness": 0.04}
    if world_id == "sparse_observation" and key == (
        "developability_E",
        "scout_F",
        "specificity_D",
    ):
        return {"selectivity": 0.08, "robustness": 0.05}
    if world_id == "batch_redundancy" and key == (
        "developability_E",
        "scout_F",
        "specificity_D",
    ):
        return {"selectivity": 0.08, "developability": 0.06}
    return {}


def _true_feasibility(
    world: WorldSpec,
    modules: Sequence[str],
    endpoints: Mapping[str, float],
) -> tuple[bool, dict[str, Any]]:
    module_set = set(modules)
    threshold_margins = {
        endpoint: float(endpoints[endpoint]) - threshold
        for endpoint, threshold in FEASIBILITY_THRESHOLDS.items()
    }
    forbidden_hits = [
        pair for pair in world.forbidden_pairs if set(pair).issubset(module_set)
    ]
    risky_hits = sorted(module_set & set(world.risky_modules))
    feasible = all(value >= 0.0 for value in threshold_margins.values()) and not forbidden_hits
    return feasible, {
        "threshold_margins": _rounded_mapping(threshold_margins),
        "forbidden_pair_hits": [list(pair) for pair in forbidden_hits],
        "risky_module_hits": risky_hits,
    }


def _observed_surrogates(
    world: WorldSpec,
    modules: Sequence[str],
    true_utility: float,
    true_feasible: bool,
    seed: int,
) -> tuple[float, float]:
    module_set = set(modules)
    bias = 0.0
    if "decoy_X" in module_set:
        bias += 0.24 if world.world_id == "constrained_risky_decoy" else 0.10
    if "liability_R" in module_set:
        bias += 0.08
    if world.world_id == "sparse_observation" and module_set & set(world.hidden_modules):
        bias -= 0.06
    if world.world_id == "batch_redundancy" and "stabilizer_A" in module_set:
        bias += 0.05
    if not true_feasible and world.world_id != "constrained_risky_decoy":
        bias -= 0.02
    noise = _noise(seed, world.world_id, modules, "observed") * 0.035
    prior_noise = _noise(seed, world.world_id, modules, "prior") * 0.020
    observed = _clip(true_utility + bias + noise, lower=0.0, upper=1.3)
    prior = _clip(true_utility + 0.60 * bias + prior_noise, lower=0.0, upper=1.2)
    return prior, observed


def _predicted_feasibility(
    world: WorldSpec,
    modules: Sequence[str],
    endpoints: Mapping[str, float],
    seed: int,
) -> tuple[bool, float]:
    del seed
    margins = [float(endpoints[key]) - value for key, value in FEASIBILITY_THRESHOLDS.items()]
    margin = min(margins) if margins else 0.0
    module_set = set(modules)
    forbidden = any(set(pair).issubset(module_set) for pair in world.forbidden_pairs)
    predicted = margin >= -0.02 and not forbidden
    if "decoy_X" in module_set or "liability_R" in module_set:
        predicted = False
        margin -= 0.12
    return predicted, max(min(margin + 0.30, 0.60) / 0.60, 0.0)


def _risk_flags(world: WorldSpec, modules: Sequence[str]) -> list[str]:
    flags: list[str] = []
    module_set = set(modules)
    if "decoy_X" in module_set:
        flags.extend(["constraint_decoy", "specificity_liability"])
    if "liability_R" in module_set:
        flags.extend(["developability_liability", "constraint_guardrail"])
    for pair in world.forbidden_pairs:
        if set(pair).issubset(module_set):
            flags.append("forbidden_pair")
    return sorted(set(flags))


def _mechanism_tags(world: WorldSpec, modules: Sequence[str]) -> list[str]:
    tags = [world.world_id]
    module_set = set(modules)
    if not modules:
        tags.append("wild_type_control")
    if len(modules) >= 2:
        tags.append("combination")
    if module_set & set(world.hidden_modules):
        tags.append("sparse_evidence")
    if module_set & set(world.risky_modules):
        tags.append("constraint_risk")
    if len(modules) == MAX_MODULES:
        tags.append("higher_order")
    return tags


def _candidate_uncertainty(world: WorldSpec, modules: Sequence[str]) -> float:
    if not modules:
        return 0.04
    module_set = set(modules)
    value = 0.08 + 0.07 * len(modules) + 0.04 * max(len(modules) - 1, 0)
    if module_set & set(world.hidden_modules):
        value += 0.22
    if world.world_id == "epistatic" and len(modules) >= 2:
        value += 0.08
    if world.world_id == "batch_redundancy":
        value += 0.03
    return min(value, 0.95)


def _static_information_value(world: WorldSpec, modules: Sequence[str]) -> float:
    if not modules:
        return 0.0
    module_set = set(modules)
    value = 0.10 * len(modules)
    if len(modules) >= 2:
        value += 0.18
    if module_set & set(world.hidden_modules):
        value += 0.30
    if world.world_id == "epistatic" and len(modules) >= 2:
        value += 0.15
    if world.world_id == "batch_redundancy":
        value += 0.08 * len(_architecture_groups(modules))
    return min(value, 1.0)


def _dynamic_information_value(
    candidate: Mapping[str, Any],
    observed: Sequence[Mapping[str, Any]],
) -> float:
    modules = _modules(candidate)
    observed_modules = set().union(*(_modules(record) for record in observed)) if observed else set()
    observed_pairs = {
        tuple(sorted(pair))
        for record in observed
        for pair in combinations(_modules(record), 2)
    }
    candidate_pairs = {tuple(sorted(pair)) for pair in combinations(modules, 2)}
    unseen_modules = sum(module not in observed_modules for module in modules)
    unseen_pairs = sum(pair not in observed_pairs for pair in candidate_pairs)
    static_value = _to_float(candidate.get("information_value"), 0.0)
    denominator = max(len(modules) + len(candidate_pairs), 1)
    novelty = (unseen_modules + unseen_pairs) / denominator
    return min(1.0, 0.55 * static_value + 0.45 * novelty)


def _candidate_cost(modules: Sequence[str]) -> float:
    cost = 1.0 + 0.16 * max(len(modules) - 1, 0)
    if "decoy_X" in modules or "liability_R" in modules:
        cost += 0.08
    return cost


def _utility(endpoints: Mapping[str, float]) -> float:
    return sum(float(endpoints[key]) * weight for key, weight in ENDPOINT_WEIGHTS.items())


def _initial_observation_ids(
    world: WorldSpec,
    records_by_id: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    module_sets = {(): True}
    for module in world.initial_singletons:
        module_sets[(module,)] = True
    if world.world_id in {"epistatic", "batch_redundancy"}:
        module_sets[("binder_B", "stabilizer_A")] = True
    if world.world_id == "constrained_risky_decoy":
        module_sets[("decoy_X",)] = True
        module_sets[("liability_R",)] = True
    ids: list[str] = []
    for modules in module_sets:
        variant_id = _variant_id(world.world_id, modules)
        if variant_id in records_by_id:
            ids.append(variant_id)
    return sorted(ids)


def _observed_record(record: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(record)
    out["candidate_id"] = str(out["variant_id"])
    out["observed_utility"] = _to_float(out.get("observed_utility"), 0.0)
    out["utility"] = out["observed_utility"]
    out["observed_feasible"] = bool(out.get("predicted_feasible", out.get("true_feasible")))
    out["numeric_features"] = _selector_numeric_features(out)
    return out


def _candidate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    out = {
        key: _copy_visible_value(record[key])
        for key in CANDIDATE_VISIBLE_FIELDS
        if key in record and key != "mechanism_provenance"
    }
    provenance = _candidate_visible_provenance(record.get("mechanism_provenance"))
    if provenance:
        out["mechanism_provenance"] = provenance
    out["numeric_features"] = _selector_numeric_features(record)
    out["candidate_id"] = str(record["variant_id"])
    _assert_candidate_record_blind(out)
    return out


def _candidate_visible_provenance(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        key: _copy_visible_value(value[key])
        for key in CANDIDATE_VISIBLE_PROVENANCE_FIELDS
        if key in value
    }


def _copy_visible_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _copy_visible_value(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_copy_visible_value(item) for item in value]
    if isinstance(value, tuple):
        return [_copy_visible_value(item) for item in value]
    return value


def _selector_numeric_features(record: Mapping[str, Any]) -> dict[str, float]:
    features: dict[str, float] = {}
    existing = record.get("numeric_features")
    if isinstance(existing, Mapping):
        for key, value in existing.items():
            numeric = _to_float(value, None)
            if numeric is not None:
                features[str(key)] = numeric
    for key in ("prior_utility", "uncertainty", "information_value", "constraint_margin"):
        if key not in record:
            continue
        numeric = _to_float(record.get(key), None)
        if numeric is not None:
            features[key] = numeric
    if "predicted_feasible" in record:
        features["predicted_feasible"] = 1.0 if _predicted_feasible(record) else 0.0
    features["risk_flag_count"] = float(len(_as_list(record.get("risk_flags"))))
    return features


def _assert_candidate_record_blind(record: Mapping[str, Any]) -> None:
    leaked = sorted(_candidate_leakage_paths(record))
    if leaked:
        raise AssertionError(
            "candidate-visible record leaks scoring labels: " + ", ".join(leaked)
        )


def _candidate_leakage_paths(value: Any, path: str = "") -> set[str]:
    if isinstance(value, Mapping):
        leaked: set[str] = set()
        for key, nested in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}" if path else key_text
            if key_text in CANDIDATE_FORBIDDEN_KEYS:
                leaked.add(child_path)
            leaked.update(_candidate_leakage_paths(nested, child_path))
        return leaked
    if isinstance(value, list):
        leaked = set()
        for index, nested in enumerate(value):
            leaked.update(_candidate_leakage_paths(nested, f"{path}[{index}]"))
        return leaked
    return set()


def _fit_observed_model(observed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    utilities = [_to_float(record.get("observed_utility"), 0.0) for record in observed]
    global_mean = mean(utilities) if utilities else 0.0
    module_values: dict[str, list[float]] = {}
    pair_values: dict[tuple[str, str], list[float]] = {}
    for record in observed:
        modules = _modules(record)
        if not modules:
            continue
        utility = _to_float(record.get("observed_utility"), global_mean)
        residual = utility - global_mean
        for module in modules:
            module_values.setdefault(module, []).append(residual / max(len(modules), 1))
        if len(modules) >= 2:
            additive = global_mean + sum(
                mean(module_values.get(module, [0.0])) for module in modules
            )
            pair_residual = utility - additive
            pairs = [tuple(sorted(pair)) for pair in combinations(modules, 2)]
            for pair in pairs:
                pair_values.setdefault(pair, []).append(pair_residual / max(len(pairs), 1))
    return {
        "global_mean": global_mean,
        "module_effects": {
            module: _shrink(mean(values), len(values), 2.0)
            for module, values in module_values.items()
        },
        "pair_effects": {
            pair: _shrink(mean(values), len(values), 3.0)
            for pair, values in pair_values.items()
        },
    }


def _model_prediction(model: Mapping[str, Any], candidate: Mapping[str, Any]) -> float:
    modules = _modules(candidate)
    value = _to_float(model.get("global_mean"), 0.0)
    module_effects = model.get("module_effects", {})
    pair_effects = model.get("pair_effects", {})
    if isinstance(module_effects, Mapping):
        value += sum(_to_float(module_effects.get(module), 0.0) for module in modules)
    if isinstance(pair_effects, Mapping):
        for pair in combinations(modules, 2):
            value += 0.65 * _to_float(pair_effects.get(tuple(sorted(pair))), 0.0)
    return value


def _evidence_coverage_for_candidate(
    candidate: Mapping[str, Any],
    observed: Sequence[Mapping[str, Any]],
) -> float:
    modules = _modules(candidate)
    if not modules:
        return 1.0
    observed_modules = set().union(*(_modules(record) for record in observed)) if observed else set()
    observed_pairs = {
        tuple(sorted(pair))
        for record in observed
        for pair in combinations(_modules(record), 2)
    }
    candidate_pairs = {tuple(sorted(pair)) for pair in combinations(modules, 2)}
    module_coverage = sum(module in observed_modules for module in modules) / len(modules)
    if not candidate_pairs:
        pair_coverage = 1.0
    else:
        pair_coverage = sum(pair in observed_pairs for pair in candidate_pairs) / len(candidate_pairs)
    return 0.60 * module_coverage + 0.40 * pair_coverage


def _selected_diversity(selected: Sequence[Mapping[str, Any]]) -> float:
    if len(selected) < 2:
        return 0.0
    values = [
        1.0 - _jaccard(left, right)
        for left, right in combinations(selected, 2)
    ]
    return mean(values) if values else 0.0


def _jaccard(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    left_modules = set(_modules(left))
    right_modules = set(_modules(right))
    union = left_modules | right_modules
    if not union:
        return 1.0
    return len(left_modules & right_modules) / len(union)


def _architecture_novelty(
    candidate: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
) -> float:
    candidate_groups = _architecture_groups(_modules(candidate))
    selected_groups = set().union(*(_architecture_groups(_modules(row)) for row in selected))
    if not candidate_groups:
        return 0.0
    return len(candidate_groups - selected_groups) / len(candidate_groups)


def _architecture_groups(modules: Sequence[str]) -> set[str]:
    groups: set[str] = set()
    for module in modules:
        if module in {"stabilizer_A", "binder_B", "release_C", "specificity_D"}:
            groups.add("performance_core")
        elif module in {"developability_E", "scout_F"}:
            groups.add("developability_repair")
        else:
            groups.add("liability_or_decoy")
    return groups


def _world_suite() -> tuple[WorldSpec, ...]:
    return (
        WorldSpec(
            world_id="additive",
            description="Independent module effects dominate and constraints are mild.",
            initial_singletons=(
                "stabilizer_A",
                "binder_B",
                "release_C",
                "specificity_D",
                "developability_E",
                "scout_F",
            ),
            forbidden_pairs=(("decoy_X", "liability_R"),),
            risky_modules=("decoy_X", "liability_R"),
        ),
        WorldSpec(
            world_id="epistatic",
            description="Pair and triple interactions dominate singleton effects.",
            initial_singletons=(
                "stabilizer_A",
                "binder_B",
                "release_C",
                "specificity_D",
                "developability_E",
                "scout_F",
            ),
            forbidden_pairs=(("decoy_X", "liability_R"),),
            risky_modules=("decoy_X", "liability_R"),
        ),
        WorldSpec(
            world_id="constrained_risky_decoy",
            description="High apparent activity decoys violate feasibility constraints.",
            initial_singletons=(
                "stabilizer_A",
                "binder_B",
                "release_C",
                "specificity_D",
                "developability_E",
            ),
            forbidden_pairs=(("decoy_X", "liability_R"), ("decoy_X", "scout_F")),
            risky_modules=("decoy_X", "liability_R"),
        ),
        WorldSpec(
            world_id="sparse_observation",
            description="Early observations miss a high-value interaction subspace.",
            initial_singletons=("stabilizer_A", "binder_B", "release_C"),
            forbidden_pairs=(("decoy_X", "liability_R"),),
            risky_modules=("decoy_X", "liability_R"),
            hidden_modules=("developability_E", "scout_F"),
        ),
        WorldSpec(
            world_id="batch_redundancy",
            description="Many high-scoring candidates occupy redundant architectures.",
            initial_singletons=(
                "stabilizer_A",
                "binder_B",
                "release_C",
                "specificity_D",
                "developability_E",
                "scout_F",
            ),
            forbidden_pairs=(("decoy_X", "liability_R"),),
            risky_modules=("decoy_X", "liability_R"),
        ),
    )


def _select_worlds(worlds: Iterable[str] | str | None) -> list[WorldSpec]:
    if worlds is None:
        world_ids = list(DEFAULT_WORLDS)
    elif isinstance(worlds, str):
        world_ids = [worlds]
    else:
        world_ids = [str(world) for world in worlds]
    return [_world_by_id(world_id) for world_id in world_ids]


def _world_by_id(world_id: str) -> WorldSpec:
    by_id = {world.world_id: world for world in _world_suite()}
    try:
        return by_id[str(world_id)]
    except KeyError as exc:
        raise ValueError(f"Unknown MCCBD benchmark world: {world_id}") from exc


def _normalize_mechanisms(mechanisms: Iterable[str] | str | None) -> list[str]:
    if mechanisms is None:
        names = list(DEFAULT_MECHANISMS)
    elif isinstance(mechanisms, str):
        names = [mechanisms]
    else:
        names = [str(mechanism) for mechanism in mechanisms]
    valid = set(DEFAULT_MECHANISMS)
    unknown = [name for name in names if name not in valid]
    if unknown:
        raise ValueError(f"Unknown MCCBD benchmark mechanisms: {', '.join(unknown)}")
    return _dedupe(names)


def _project_row(row: Mapping[str, Any], columns: Sequence[str]) -> dict[str, Any]:
    return {column: _csv_value(row.get(column, "")) for column in columns}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column, "")) for column in columns})


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_dumps(payload) + "\n", encoding="utf-8")


def _csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return _bool_text(value)
    if isinstance(value, (dict, list, tuple)):
        return _json_dumps(value)
    if isinstance(value, float):
        return _round_metric(value)
    return value


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _mean_metric(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return _round_metric(mean(_to_float(row.get(field), 0.0) for row in rows) if rows else 0.0)


def _world_mean(
    rows: Sequence[Mapping[str, Any]],
    mechanism: str,
    world_id: str,
    field: str,
) -> float | None:
    values = [
        _to_float(row.get(field), 0.0)
        for row in rows
        if row.get("mechanism") == mechanism and row.get("world_id") == world_id
    ]
    return mean(values) if values else None


def _overall_mean(
    rows: Sequence[Mapping[str, Any]],
    mechanism: str,
    field: str,
) -> float | None:
    values = [
        _to_float(row.get(field), 0.0)
        for row in rows
        if row.get("mechanism") == mechanism
    ]
    return mean(values) if values else None


def _quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(max(math.ceil(len(ordered) * fraction) - 1, 0), len(ordered) - 1)
    return ordered[index]


def _observed_score(candidate: Mapping[str, Any]) -> float:
    return _to_float(
        candidate.get("observed_score", candidate.get("prior_utility")),
        0.0,
    )


def _predicted_feasible(candidate: Mapping[str, Any]) -> bool:
    value = candidate.get("predicted_feasible", True)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "feasible"}
    return bool(value)


def _modules(record: Mapping[str, Any]) -> tuple[str, ...]:
    value = record.get("modules", ())
    if isinstance(value, str):
        return tuple(part.strip() for part in value.replace(",", ";").split(";") if part.strip())
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return ()


def _cost(record: Mapping[str, Any]) -> float:
    value = _to_float(record.get("cost"), 1.0)
    return value if value > 0.0 else 1.0


def _as_list(value: Any) -> list[Any]:
    if value in (None, "", (), [], {}):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _to_float(value: Any, default: float | None = None) -> float:
    try:
        if value in (None, ""):
            if default is None:
                raise ValueError
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        if default is None:
            raise
        return float(default)


def _shrink(value: float, count: int, shrinkage: float) -> float:
    return float(value) * (float(count) / (float(count) + float(shrinkage)))


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(max(float(value), lower), upper)


def _rounded_mapping(values: Mapping[str, float]) -> dict[str, float]:
    return {str(key): _round_metric(float(value)) for key, value in values.items()}


def _round_metric(value: float) -> float:
    return round(float(value), 6)


def _bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def _variant_id(world_id: str, modules: Sequence[str]) -> str:
    if not modules:
        suffix = "WT"
    else:
        suffix = "_".join(_slug(module) for module in modules)
    return f"{world_id}__{suffix}"


def _architecture_id(modules: Sequence[str]) -> str:
    if not modules:
        return "control"
    return "+".join(sorted(_architecture_groups(modules))) or "unassigned"


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", value)


def _safe_run_id(run_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_id)).strip("._")
    return safe or DEFAULT_RUN_ID


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _noise(seed: int, world_id: str, modules: Sequence[str], channel: str) -> float:
    rng = random.Random(_stable_seed(seed, world_id, ",".join(modules), channel))
    return rng.uniform(-1.0, 1.0)


def _dedupe(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _mechanism_priority(mechanism: str) -> int:
    order = {
        "mccbd": 0,
        "evidence_calibrated_ucb": 1,
        "mccbd_no_constraint": 2,
        "mccbd_no_information_value": 3,
        "mccbd_no_batch_diversity": 4,
        "greedy_observed": 5,
        "fixed_mix": 6,
        "random_feasible": 7,
    }
    return order.get(mechanism, 99)
