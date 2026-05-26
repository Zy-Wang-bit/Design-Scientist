"""Generative design benchmark over controlled synthetic sequence worlds.

This benchmark is intentionally separate from the MCCBD pool-selection
benchmark. It evaluates whether a method creates new heavy/light sequence
candidates before selection, then scores those generated sequences against a
hidden oracle used only for benchmark metrics.
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


DEFAULT_RUN_ID = "generative_benchmark"
DEFAULT_SEEDS = range(5)
DEFAULT_WORLDS = (
    "ph_contrast",
    "escape_risk",
    "vocabulary_extension",
    "de_novo_site_generalization",
)
DEFAULT_MECHANISMS = (
    "cmdgd",
    "random_edit_generator",
    "observed_recombination_baseline",
    "single_edit_scan",
    "mccbd_pool_selector",
)
OPTIONAL_BASELINE_MECHANISMS = (
    "random_feasible",
    "fixed_mix",
    "evidence_calibrated_ucb",
)
DEFAULT_BUDGET = 5
DEFAULT_GENERATION_BUDGET = 28
CI95_Z = 1.96
DEFAULT_HIDDEN_DE_NOVO_EDIT_TOKENS = frozenset({"H10Q"})
CMDGD_ABLATIONS = (
    "full",
    "no_contrastive_objective",
    "no_grammar_recombination",
    "no_novelty_uncertainty",
    "no_vocabulary_expansion",
    "no_de_novo_site_proposal",
    "no_site_or_vocabulary_generation",
    "benchmark_no_generation_boundary",
)
CMDGD_COMPONENT_ABLATION_CONFIGS: dict[str, dict[str, Any]] = {
    "full": {},
    "no_contrastive_objective": {"enable_contrastive_objective": False},
    "no_grammar_recombination": {"enable_grammar_recombination": False},
    "no_novelty_uncertainty": {"enable_novelty_uncertainty": False},
    "no_vocabulary_expansion": {"enable_vocabulary_expansion": False},
    "no_de_novo_site_proposal": {"enable_de_novo_site_proposal": False},
    "no_site_or_vocabulary_generation": {
        "enable_vocabulary_expansion": False,
        "enable_de_novo_site_proposal": False,
    },
}
CMDGD_ABLATION_COMPONENTS = {
    "full": "full_lifecycle",
    "no_contrastive_objective": "contrastive_ph_objective",
    "no_grammar_recombination": "module_recombination_generator",
    "no_novelty_uncertainty": "novelty_and_uncertainty_acquisition_terms",
    "no_vocabulary_expansion": "candidate_edit_vocabulary_expansion",
    "no_de_novo_site_proposal": "de_novo_mutation_site_proposal",
    "no_site_or_vocabulary_generation": "de_novo_site_and_vocabulary_generators",
    "benchmark_no_generation_boundary": "candidate_space_generation_boundary",
}
CMDGD_BENCHMARK_BOUNDARY_ABLATIONS = {"benchmark_no_generation_boundary"}
CMDGD_DE_NOVO_SITE_DISABLED_ABLATIONS = {
    "no_de_novo_site_proposal",
    "no_site_or_vocabulary_generation",
}
OBSERVED_REUSE_ONLY = {"observed_recombination_baseline"}
FIXED_POOL_ONLY = {
    "mccbd_pool_selector",
    "random_feasible",
    "fixed_mix",
    "evidence_calibrated_ucb",
}
BASELINE_MECHANISMS = {
    "random_edit_generator",
    "observed_recombination_baseline",
    "single_edit_scan",
    "mccbd_pool_selector",
    *OPTIONAL_BASELINE_MECHANISMS,
}
SUPPORTED_MECHANISMS = {*DEFAULT_MECHANISMS, *OPTIONAL_BASELINE_MECHANISMS}
STATISTICAL_METRICS = (
    "generated_novel_count",
    "generated_new_site_count",
    "selected_new_site_count",
    "new_site_rate",
    "best_generated_utility",
    "best_selected_utility",
    "best_selected_pH_contrast_score",
    "constraint_pass_rate",
    "pH_contrast_score",
    "novel_sequence_rate",
    "false_claim_rate",
    "cost_spent",
)

BENCHMARK_COLUMNS = (
    "status",
    "error",
    "seed",
    "world_id",
    "mechanism",
    "cmdgd_adapter",
    "budget",
    "generation_budget",
    "observed_count",
    "candidate_count",
    "generated_count",
    "generated_novel_count",
    "selected_count",
    "selected_ids",
    "generated_ids",
    "candidate_space_type",
    "generation_counted",
    "generated_utility_applicable",
    "best_generated_utility",
    "best_selected_utility",
    "best_selected_pH_contrast_score",
    "constraint_pass_rate",
    "pH_contrast_score",
    "novel_sequence_rate",
    "false_claim_rate",
    "cost_spent",
    "observed_reuse_only",
    "fixed_pool_only",
    "selector_candidate_example",
    "generated_new_site_count",
    "selected_new_site_count",
    "new_site_rate",
    "new_site_generation_required",
    "new_site_generation_capable",
)
SUMMARY_COLUMNS = (
    "mechanism",
    "world_id",
    "replicate_count",
    "mean_generated_novel_count",
    "mean_best_generated_utility",
    "mean_best_selected_utility",
    "mean_best_selected_pH_contrast_score",
    "mean_constraint_pass_rate",
    "mean_pH_contrast_score",
    "mean_novel_sequence_rate",
    "mean_false_claim_rate",
    "mean_cost_spent",
    "observed_reuse_only",
    "fixed_pool_only",
    "selected_eligible",
    "selected_mechanism",
    "comparative_utility_rank",
    "mean_generated_new_site_count",
    "mean_selected_new_site_count",
    "mean_new_site_rate",
    "new_site_generation_capable",
)
STATISTICAL_SUMMARY_COLUMNS = (
    "mechanism",
    "world_id",
    "metric",
    "n",
    "mean",
    "sd",
    "sem",
    "ci95_low",
    "ci95_high",
)
PAIRWISE_COMPARISON_COLUMNS = (
    "mechanism",
    "baseline",
    "world_id",
    "metric",
    "n_pairs",
    "paired_delta_mean",
    "paired_delta_sd",
    "paired_delta_sem",
    "paired_delta_ci95_low",
    "paired_delta_ci95_high",
    "win_fraction",
    "tie_fraction",
    "loss_fraction",
)
ABLATION_COLUMNS = (
    "status",
    "error",
    "seed",
    "world_id",
    "mechanism",
    "cmdgd_adapter",
    "ablation",
    "ablation_type",
    "ablation_component",
    "budget",
    "generation_budget",
    "candidate_space_type",
    "generation_counted",
    "generated_utility_applicable",
    "generated_count",
    "generated_novel_count",
    "selected_count",
    "best_generated_utility",
    "best_selected_utility",
    "best_selected_pH_contrast_score",
    "constraint_pass_rate",
    "pH_contrast_score",
    "novel_sequence_rate",
    "false_claim_rate",
    "cost_spent",
    "observed_reuse_only",
    "fixed_pool_only",
    "delta_from_full_best_generated_utility",
    "delta_from_full_best_selected_utility",
    "generated_new_site_count",
    "selected_new_site_count",
    "new_site_rate",
    "new_site_generation_required",
    "new_site_generation_capable",
)
EXAMPLE_COLUMNS = (
    "seed",
    "world_id",
    "generated_by",
    "candidate_id",
    "heavy_sequence",
    "light_sequence",
    "edit_tokens",
    "observed_duplicate",
    "selected",
    "true_feasible",
    "true_utility",
    "pH_contrast_score",
    "cost",
)
CANDIDATE_VISIBLE_FIELDS = (
    "candidate_id",
    "variant_id",
    "heavy_sequence",
    "light_sequence",
    "edit_tokens",
    "modules",
    "edit_count",
    "cost",
    "prior_utility",
    "predicted_feasible",
    "feasibility_status",
    "constraint_margin",
    "predicted_pH_contrast",
    "generation_source",
    "sequence_hash",
    "world_id",
    "numeric_features",
    "risk_flags",
    "mechanism_tags",
)
CANDIDATE_FORBIDDEN_KEYS = frozenset(
    {
        "true_utility",
        "true_feasible",
        "oracle",
        "oracle_truth",
        "endpoint_values",
        "endpoints",
        "true_endpoints",
        "observed_endpoints",
        "truth_terms",
    }
)


@dataclass(frozen=True)
class EditSpec:
    """A visible sequence edit plus hidden oracle effects."""

    token: str
    chain: str
    position: int
    residue: str
    annotation: str
    neutral_binding_delta: float
    acidic_binding_delta: float
    expression_delta: float
    specificity_delta: float


@dataclass(frozen=True)
class SequenceWorldSpec:
    """Synthetic heavy/light edit world with hidden pH-dependent truth."""

    world_id: str
    description: str
    base_heavy_sequence: str
    base_light_sequence: str
    initial_edit_sets: tuple[tuple[str, ...], ...]
    forbidden_pairs: tuple[tuple[str, str], ...]
    risky_edits: tuple[str, ...]
    expression_threshold: float
    specificity_threshold: float
    acidic_binding_max: float
    max_edits: int = 3
    visible_edit_tokens: tuple[str, ...] = ()
    new_site_target_tokens: tuple[str, ...] = ()


def run_generative_benchmark(
    project_dir: str | Path,
    run_id: str = DEFAULT_RUN_ID,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    mechanisms: Iterable[str] | str | None = None,
    worlds: Iterable[str] | str | None = None,
    budget: int = DEFAULT_BUDGET,
    generation_budget: int = DEFAULT_GENERATION_BUDGET,
) -> dict[str, Any]:
    """Run a deterministic benchmark for algorithms that generate new sequences."""

    if budget <= 0:
        raise ValueError("budget must be positive")
    if generation_budget <= 0:
        raise ValueError("generation_budget must be positive")
    seed_values = [int(seed) for seed in seeds]
    if not seed_values:
        raise ValueError("seeds must contain at least one seed")
    mechanism_names = _normalize_mechanisms(mechanisms)
    world_specs = _select_worlds(worlds)

    root = Path(project_dir).expanduser().resolve()
    safe_run_id = _safe_run_id(run_id)
    run_dir = root / "runs" / safe_run_id

    benchmark_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    trial_cache: dict[tuple[str, int, str, str], dict[str, Any]] = {}

    for world in world_specs:
        for seed in seed_values:
            for mechanism in mechanism_names:
                result = _run_cached_trial(
                    trial_cache,
                    world=world,
                    seed=seed,
                    mechanism=mechanism,
                    budget=budget,
                    generation_budget=generation_budget,
                    ablation="full",
                )
                benchmark_rows.append(_project_row(result, BENCHMARK_COLUMNS))
                example_rows.extend(
                    _example_rows_from_trial(
                        result,
                        max_examples=4 if mechanism == "cmdgd" else 2,
                    )
                )

    summary_rows = _summary_rows(benchmark_rows)
    statistical_summary_rows = _statistical_summary_rows(benchmark_rows)
    pairwise_comparison_rows = _pairwise_comparison_rows(benchmark_rows)
    selection_gate_report = _selection_gate_report(summary_rows)
    selected_mechanism = str(selection_gate_report["selected_mechanism"])
    selected_passes_gate = bool(selection_gate_report["selected_passes_gate"])
    summary_rows = _annotate_summary_selection(summary_rows, selection_gate_report)
    ablation_rows = _ablation_rows(
        trial_cache,
        world_specs=world_specs,
        seeds=seed_values,
        budget=budget,
        generation_budget=generation_budget,
        include_cmdgd="cmdgd" in mechanism_names,
    )

    benchmark_path = run_dir / "generative_benchmark_results.csv"
    summary_path = run_dir / "generative_benchmark_summary.csv"
    ablation_path = run_dir / "generative_ablation_results.csv"
    examples_path = run_dir / "generative_design_examples.csv"
    statistical_summary_path = run_dir / "generative_statistical_summary.csv"
    pairwise_comparison_path = run_dir / "generative_pairwise_comparisons.csv"
    config_path = run_dir / "generative_benchmark_config.json"
    gate_report_path = run_dir / "generative_selection_gate_report.json"

    config = {
        "benchmark": "generative_design_algorithm_benchmark",
        "schema_version": 1,
        "run_id": safe_run_id,
        "seeds": seed_values,
        "worlds": [world.world_id for world in world_specs],
        "mechanisms": mechanism_names,
        "budget": int(budget),
        "generation_budget": int(generation_budget),
        "base_sequences": {
            world.world_id: {
                "heavy": world.base_heavy_sequence,
                "light": world.base_light_sequence,
            }
            for world in world_specs
        },
        "metrics": [
            *STATISTICAL_METRICS,
        ],
        "statistical_artifacts": {
            "summary": "generative_statistical_summary.csv",
            "pairwise_comparisons": "generative_pairwise_comparisons.csv",
            "ci_method": "normal_approximation_1.96_times_sem",
            "paired_delta_direction": "cmdgd_minus_baseline",
        },
        "new_site_generation": {
            "definition": (
                "A generated candidate contains a new mutation site when at least one "
                "mutation position is absent from both observed variants and the "
                "visible edit_vocabulary supplied to algorithms."
            ),
            "required_worlds": [
                world.world_id for world in world_specs if world.new_site_target_tokens
            ],
        },
        "leakage_controls": {
            "selector_inputs_exclude_oracle_truth": True,
            "forbidden_selector_candidate_keys": sorted(CANDIDATE_FORBIDDEN_KEYS),
            "oracle_truth_used_only_after_selection": True,
        },
        "winner_gate": {
            "selection_basis": "generative_candidate_space_expansion",
            "eligible_rule": (
                "eligible mechanisms must generate novel candidates and must "
                "generate new mutation sites in worlds that require de novo site generation; "
                "eligible mechanisms must not be fixed-pool, observed-reuse-only, "
                "or benchmark baseline mechanisms"
            ),
            "ineligible_baselines": sorted(BASELINE_MECHANISMS),
            "rule": (
                "selected mechanism is chosen for generative candidate-space "
                "expansion; comparative selected-utility ranks are reported separately"
            ),
        },
        "selected_mechanism": selected_mechanism,
        "selected_passes_gate": selected_passes_gate,
        "selection_gate_report": selection_gate_report,
    }

    _write_csv(benchmark_path, benchmark_rows, BENCHMARK_COLUMNS)
    _write_csv(summary_path, summary_rows, SUMMARY_COLUMNS)
    _write_csv(ablation_path, ablation_rows, ABLATION_COLUMNS)
    _write_csv(examples_path, example_rows, EXAMPLE_COLUMNS)
    _write_csv(statistical_summary_path, statistical_summary_rows, STATISTICAL_SUMMARY_COLUMNS)
    _write_csv(pairwise_comparison_path, pairwise_comparison_rows, PAIRWISE_COMPARISON_COLUMNS)
    _write_json(config_path, config)
    _write_json(gate_report_path, selection_gate_report)

    artifact_paths = {
        "benchmark_results": str(benchmark_path),
        "benchmark_summary": str(summary_path),
        "ablation_results": str(ablation_path),
        "design_examples": str(examples_path),
        "statistical_summary": str(statistical_summary_path),
        "pairwise_comparisons": str(pairwise_comparison_path),
        "benchmark_config": str(config_path),
        "selection_gate_report": str(gate_report_path),
    }
    return {
        "status": "completed",
        "run_id": safe_run_id,
        "artifact_paths": artifact_paths,
        "benchmark_results_path": str(benchmark_path),
        "summary_results_path": str(summary_path),
        "ablation_results_path": str(ablation_path),
        "design_examples_path": str(examples_path),
        "statistical_summary_path": str(statistical_summary_path),
        "pairwise_comparisons_path": str(pairwise_comparison_path),
        "config_path": str(config_path),
        "selection_gate_report_path": str(gate_report_path),
        "generative_benchmark_results_path": str(benchmark_path),
        "generative_benchmark_summary_path": str(summary_path),
        "generative_ablation_results_path": str(ablation_path),
        "generative_design_examples_path": str(examples_path),
        "generative_statistical_summary_path": str(statistical_summary_path),
        "generative_pairwise_comparisons_path": str(pairwise_comparison_path),
        "generative_benchmark_config_path": str(config_path),
        "generative_selection_gate_report_path": str(gate_report_path),
        "selected_mechanism": selected_mechanism,
        "selected_passes_gate": selected_passes_gate,
        "selection_gate_report": selection_gate_report,
        "summary": {
            "mechanism_rankings": summary_rows,
            "statistical_summary": statistical_summary_rows,
            "pairwise_comparisons": pairwise_comparison_rows,
            "selection_gate_report": selection_gate_report,
        },
    }


def generate_synthetic_sequence_world(
    world_id: str,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Return hidden-oracle records for one synthetic sequence/edit world."""

    world = _world_by_id(world_id)
    return _all_world_records(world, seed)


def _run_cached_trial(
    cache: dict[tuple[str, int, str, str], dict[str, Any]],
    *,
    world: SequenceWorldSpec,
    seed: int,
    mechanism: str,
    budget: int,
    generation_budget: int,
    ablation: str,
) -> dict[str, Any]:
    key = (world.world_id, int(seed), mechanism, ablation)
    if key not in cache:
        cache[key] = _run_trial(
            world=world,
            seed=seed,
            mechanism=mechanism,
            budget=budget,
            generation_budget=generation_budget,
            ablation=ablation,
        )
    return cache[key]


def _run_trial(
    *,
    world: SequenceWorldSpec,
    seed: int,
    mechanism: str,
    budget: int,
    generation_budget: int,
    ablation: str,
) -> dict[str, Any]:
    records = _all_world_records(world, seed)
    records_by_id = {str(record["variant_id"]): record for record in records}
    observed_internal = _initial_observed_records(world, records_by_id, seed)
    observed = [_observed_record(record, seed) for record in observed_internal]
    observed_sequences = {_sequence_key(record) for record in observed_internal}
    observed_mutation_sites = _mutation_sites_from_records(observed_internal, world)
    visible_mutation_sites = _visible_mutation_sites(world)
    status = "completed"
    error = ""
    adapter = "local"
    generated: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    truth_by_id: dict[str, dict[str, Any]] = {}
    selected_ids: list[str] = []
    generation_counted = False

    try:
        rng = random.Random(_stable_seed(seed, world.world_id, mechanism, ablation))
        context = _design_context(world, observed, seed, generation_budget, ablation)
        state = _fit_state_for_mechanism(mechanism, observed, context, rng)
        adapter = str(state.get("cmdgd_adapter", "local")) if isinstance(state, Mapping) else "local"
        raw_generated, generation_counted = _generate_for_mechanism(
            mechanism,
            world,
            observed,
            context,
            state,
            rng,
            generation_budget,
            ablation,
        )
        generated, truth_by_id = _normalize_generated_candidates(
            raw_generated,
            world=world,
            seed=seed,
            mechanism=mechanism,
            observed_sequences=observed_sequences,
        )
        if mechanism in FIXED_POOL_ONLY or ablation in CMDGD_BENCHMARK_BOUNDARY_ABLATIONS:
            generation_counted = False
        scored = _score_candidates_for_mechanism(
            mechanism,
            observed,
            generated,
            budget,
            rng,
            state,
            ablation,
        )
        candidate_records = [
            candidate
            for candidate in scored
            if not bool(candidate.get("observed_duplicate"))
        ]
        _assert_candidate_list_blind(candidate_records)
        raw_selected = _select_panel_for_mechanism(
            mechanism,
            observed,
            candidate_records,
            budget,
            rng,
            state,
        )
        selected_ids = _resolve_selected(raw_selected, candidate_records, budget)
    except Exception as exc:  # pragma: no cover - retained for CSV diagnostics.
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"

    generated_ids = [str(candidate["candidate_id"]) for candidate in generated]
    selected_truth = [
        truth_by_id[candidate_id]
        for candidate_id in selected_ids
        if candidate_id in truth_by_id
    ]
    novel_generated_truth = [
        truth_by_id[str(candidate["candidate_id"])]
        for candidate in generated
        if not bool(candidate.get("observed_duplicate"))
        and str(candidate["candidate_id"]) in truth_by_id
    ]
    generated_new_site_ids = (
        _new_site_candidate_ids(
            generated,
            world=world,
            observed_mutation_sites=observed_mutation_sites,
            visible_mutation_sites=visible_mutation_sites,
        )
        if generation_counted
        else set()
    )
    metrics = _trial_metrics(
        generated=generated,
        novel_generated_truth=novel_generated_truth,
        selected_truth=selected_truth,
        selected_ids=selected_ids,
        generation_counted=generation_counted,
        generated_new_site_ids=generated_new_site_ids,
    )
    candidate_space_type = _candidate_space_type(mechanism, ablation, generation_counted)
    example = candidate_records[0] if candidate_records else {}
    if example:
        _assert_candidate_record_blind(example)

    return {
        "status": status,
        "error": error,
        "seed": int(seed),
        "world_id": world.world_id,
        "mechanism": mechanism,
        "cmdgd_adapter": adapter if mechanism == "cmdgd" else "",
        "budget": int(budget),
        "generation_budget": int(generation_budget),
        "observed_count": len(observed),
        "candidate_count": len(candidate_records),
        "generated_count": len(generated) if generation_counted else 0,
        "selected_count": len(selected_ids),
        "selected_ids": selected_ids,
        "generated_ids": generated_ids if generation_counted else [],
        "candidate_space_type": candidate_space_type,
        "new_site_generation_required": bool(world.new_site_target_tokens),
        "new_site_generation_capable": _new_site_generation_capable(mechanism, ablation),
        "generation_counted": bool(generation_counted),
        "generated_utility_applicable": bool(generation_counted),
        "observed_reuse_only": mechanism in OBSERVED_REUSE_ONLY,
        "fixed_pool_only": mechanism in FIXED_POOL_ONLY or ablation in CMDGD_BENCHMARK_BOUNDARY_ABLATIONS,
        "selector_candidate_example": _project_candidate_example(example),
        "_examples": _examples_payload(
            seed=seed,
            world_id=world.world_id,
            mechanism=mechanism,
            generated=generated,
            truth_by_id=truth_by_id,
            selected_ids=set(selected_ids),
            generation_counted=generation_counted,
        ),
        **metrics,
    }


def _fit_state_for_mechanism(
    mechanism: str,
    observed: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    reference_state = _fit_reference_state(observed, context)
    if mechanism != "cmdgd":
        return {
            "reference_state": reference_state,
            "observed_records": list(observed),
            "cmdgd_adapter": "",
        }
    module = _load_cmdgd_module()
    if module is None:
        return {
            "reference_state": reference_state,
            "observed_records": list(observed),
            "cmdgd_adapter": "benchmark_reference_adapter",
        }
    lifecycle = _cmdgd_lifecycle(module)
    fit_state = getattr(module, "fit_state", None)
    external_state = None
    if callable(fit_state):
        try:
            external_state = fit_state(
                base_heavy_chain_seq=str(context["base_heavy_chain_seq"]),
                base_light_chain_seq=str(context["base_light_chain_seq"]),
                observed_variants=list(observed),
                observed_endpoints=[],
                candidate_edit_vocabulary=list(context.get("edit_vocabulary", [])),
                config=_cmdgd_config_from_context(context),
            )
        except TypeError:
            external_state = _try_call(
                fit_state,
                (
                    (dict(context),),
                    (list(observed), dict(context)),
                ),
                {
                    "observed_records": list(observed),
                    "context": dict(context),
                    "rng": rng,
                },
            )
    if lifecycle is not None and external_state is not None:
        lifecycle_state = {
            **dict(context),
            "algorithm_state": external_state,
            "round_index": int(context.get("round_index", 0) or 0),
        }
        return {
            "reference_state": reference_state,
            "external_state": lifecycle_state,
            "external_lifecycle": lifecycle,
            "external_module": module,
            "observed_records": list(observed),
            "cmdgd_adapter": "design_scientist.algorithms.cmdgd.CMDGDLifecycle",
        }
    if lifecycle is not None and callable(getattr(lifecycle, "fit_state", None)):
        lifecycle_state = _try_call(
            getattr(lifecycle, "fit_state"),
            ((dict(context),),),
            {
                "context": dict(context),
                "state": dict(context),
            },
        )
        if lifecycle_state is not None:
            return {
                "reference_state": reference_state,
                "external_state": lifecycle_state,
                "external_lifecycle": lifecycle,
                "external_module": module,
                "observed_records": list(observed),
                "cmdgd_adapter": "design_scientist.algorithms.cmdgd.CMDGDLifecycle",
            }
    if not callable(fit_state):
        return {
            "reference_state": reference_state,
            "observed_records": list(observed),
            "cmdgd_adapter": "benchmark_reference_adapter",
        }
    if external_state is None:
        external_state = dict(context)
    return {
        "reference_state": reference_state,
        "external_state": external_state,
        "external_module": module,
        "observed_records": list(observed),
        "cmdgd_adapter": "design_scientist.algorithms.cmdgd.fit_state",
    }


def _generate_for_mechanism(
    mechanism: str,
    world: SequenceWorldSpec,
    observed: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    state: Mapping[str, Any],
    rng: random.Random,
    generation_budget: int,
    ablation: str,
) -> tuple[list[Mapping[str, Any]], bool]:
    if ablation in CMDGD_BENCHMARK_BOUNDARY_ABLATIONS:
        return _fixed_pool_candidates(world, observed, "cmdgd_benchmark_no_generation_boundary"), False
    if mechanism == "cmdgd":
        lifecycle = state.get("external_lifecycle")
        module = state.get("external_module")
        external_state = state.get("external_state")
        if lifecycle is not None and callable(getattr(lifecycle, "generate_candidates", None)):
            generated = _try_call(
                getattr(lifecycle, "generate_candidates"),
                ((external_state,),),
                {"state": external_state},
            )
            if generated is not None:
                return list(generated)[:generation_budget], True
        if module is not None and callable(getattr(module, "generate_candidates", None)):
            generated = _try_call(
                getattr(module, "generate_candidates"),
                (
                    (external_state, rng, generation_budget),
                    (external_state, generation_budget),
                    (external_state,),
                ),
                {
                    "state": external_state,
                    "rng": rng,
                    "generation_budget": generation_budget,
                    "budget": generation_budget,
                },
            )
            if generated is not None:
                return list(generated), True
        return _reference_cmdgd_generate(world, observed, state, generation_budget, ablation), True
    if mechanism == "random_edit_generator":
        return _random_edit_generate(world, observed, rng, generation_budget), True
    if mechanism == "observed_recombination_baseline":
        return _observed_recombination_generate(world, observed, generation_budget), True
    if mechanism == "single_edit_scan":
        return _single_edit_scan_generate(world, observed, generation_budget), True
    if mechanism in FIXED_POOL_ONLY:
        return _fixed_pool_candidates(world, observed, f"{mechanism}_fixed_pool"), False
    raise ValueError(f"Unknown generative benchmark mechanism: {mechanism}")


def _score_candidates_for_mechanism(
    mechanism: str,
    observed: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    budget: int,
    rng: random.Random,
    state: Mapping[str, Any],
    ablation: str,
) -> list[dict[str, Any]]:
    del budget, rng
    if mechanism == "cmdgd":
        lifecycle = state.get("external_lifecycle")
        module = state.get("external_module")
        external_state = state.get("external_state")
        if lifecycle is not None and callable(getattr(lifecycle, "score_candidates", None)):
            scored = _try_call(
                getattr(lifecycle, "score_candidates"),
                ((external_state, list(candidates)),),
                {
                    "state": external_state,
                    "candidates": list(candidates),
                    "candidate_records": list(candidates),
                },
            )
            if scored is not None:
                return _coerce_scored_candidates(scored, candidates)
        if module is not None and callable(getattr(module, "score_candidates", None)):
            scored = _try_call(
                getattr(module, "score_candidates"),
                (
                    (external_state, list(candidates)),
                    (list(observed), list(candidates), external_state),
                    (list(observed), list(candidates)),
                ),
                {
                    "state": external_state,
                    "observed": list(observed),
                    "observed_records": list(observed),
                    "candidates": list(candidates),
                    "candidate_records": list(candidates),
                    "fitted_state": external_state,
                },
            )
            if scored is not None:
                return _coerce_scored_candidates(scored, candidates)
        return _reference_score_candidates(candidates, state, ablation)
    if mechanism == "mccbd_pool_selector":
        return _score_mccbd_pool(observed, candidates)
    if mechanism == "evidence_calibrated_ucb":
        return _score_evidence_calibrated_ucb(observed, candidates)
    if mechanism == "random_edit_generator":
        return [dict(candidate) for candidate in candidates]
    return _reference_score_candidates(candidates, state, ablation)


def _select_panel_for_mechanism(
    mechanism: str,
    observed: list[Mapping[str, Any]],
    candidates: list[Mapping[str, Any]],
    budget: int,
    rng: random.Random,
    state: Any,
) -> list[str]:
    """Select candidate IDs from truth-blind candidate records."""

    if not candidates:
        return []
    state_mapping = state if isinstance(state, Mapping) else {}
    observed_records = list(state_mapping.get("observed_records", observed))
    if mechanism == "cmdgd":
        lifecycle = state_mapping.get("external_lifecycle")
        module = state_mapping.get("external_module")
        external_state = state_mapping.get("external_state")
        if lifecycle is not None and callable(getattr(lifecycle, "select_panel", None)):
            selected = _try_call(
                getattr(lifecycle, "select_panel"),
                (
                    (external_state, list(candidates), budget, rng),
                    (external_state, list(candidates), budget),
                ),
                {
                    "state": external_state,
                    "observed": observed_records,
                    "observed_records": observed_records,
                    "candidates": list(candidates),
                    "scored_candidates": list(candidates),
                    "budget": budget,
                    "rng": rng,
                },
            )
            if selected is not None:
                return [str(item) if not isinstance(item, Mapping) else str(item.get("candidate_id") or item.get("variant_id")) for item in selected]
        if module is not None and callable(getattr(module, "select_panel", None)):
            selected = _try_call(
                getattr(module, "select_panel"),
                (
                    (external_state, list(candidates), budget, rng),
                    (observed_records, list(candidates), budget, rng),
                    (list(candidates), budget, rng),
                ),
                {
                    "state": external_state,
                    "observed": observed_records,
                    "observed_records": observed_records,
                    "candidates": list(candidates),
                    "scored_candidates": list(candidates),
                    "budget": budget,
                    "rng": rng,
                },
            )
            if selected is not None:
                return [str(item) if not isinstance(item, Mapping) else str(item.get("candidate_id") or item.get("variant_id")) for item in selected]
        return _greedy_score_select(candidates, budget, diversity=True)
    if mechanism == "random_edit_generator":
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        return _resolve_selected(shuffled, shuffled, budget)
    if mechanism == "random_feasible":
        return _policy_select("random_feasible", observed_records, candidates, budget, rng) or _random_feasible_select(
            candidates,
            budget,
            rng,
        )
    if mechanism == "fixed_mix":
        return _policy_select("fixed_mix", observed_records, candidates, budget, rng) or _fixed_mix_select(
            candidates,
            budget,
            rng,
        )
    if mechanism == "evidence_calibrated_ucb":
        return _evidence_calibrated_ucb_select(observed_records, candidates, budget, rng)
    if mechanism == "mccbd_pool_selector":
        return _mccbd_pool_select(observed_records, candidates, budget, rng)
    return _greedy_score_select(candidates, budget, diversity=mechanism != "single_edit_scan")


def _normalize_generated_candidates(
    raw_candidates: Sequence[Mapping[str, Any]],
    *,
    world: SequenceWorldSpec,
    seed: int,
    mechanism: str,
    observed_sequences: set[tuple[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    generated: list[dict[str, Any]] = []
    truth_by_id: dict[str, dict[str, Any]] = {}
    seen_sequences: set[tuple[str, str]] = set()
    for index, raw in enumerate(raw_candidates):
        normalized = _normalize_raw_candidate(raw, world, seed, mechanism, index)
        if normalized is None:
            continue
        candidate, truth = normalized
        sequence_key = _sequence_key(candidate)
        if sequence_key in seen_sequences:
            continue
        seen_sequences.add(sequence_key)
        candidate["observed_duplicate"] = sequence_key in observed_sequences
        _assert_candidate_record_blind(candidate)
        generated.append(candidate)
        truth_by_id[str(candidate["candidate_id"])] = truth
    return generated, truth_by_id


def _normalize_raw_candidate(
    raw: Mapping[str, Any],
    world: SequenceWorldSpec,
    seed: int,
    mechanism: str,
    index: int,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    edit_tokens = _extract_edit_tokens(raw, world)
    if edit_tokens is None:
        return None
    heavy, light = _apply_edits(
        world.base_heavy_sequence,
        world.base_light_sequence,
        edit_tokens,
    )
    raw_heavy = raw.get("heavy_sequence", raw.get("heavy_chain_seq"))
    raw_light = raw.get("light_sequence", raw.get("light_chain_seq"))
    if raw_heavy not in (None, ""):
        heavy = str(raw_heavy)
    if raw_light not in (None, ""):
        light = str(raw_light)
    if not edit_tokens:
        edit_tokens = _diff_to_known_edit_tokens(world, heavy, light)
    truth = _truth_record(world, edit_tokens, seed)
    truth["heavy_sequence"] = heavy
    truth["light_sequence"] = light
    candidate = _candidate_record(truth, mechanism)
    candidate_id = str(
        raw.get("candidate_id")
        or raw.get("variant_id")
        or raw.get("id")
        or candidate["candidate_id"]
    )
    if candidate_id in {"", "None"}:
        candidate_id = f"{world.world_id}__{mechanism}_{index}"
    candidate["candidate_id"] = candidate_id
    candidate["variant_id"] = candidate_id
    candidate["heavy_sequence"] = heavy
    candidate["light_sequence"] = light
    candidate["sequence_hash"] = _sequence_hash(heavy, light)
    truth["variant_id"] = candidate_id
    return candidate, truth


def _candidate_record(record: Mapping[str, Any], source: str) -> dict[str, Any]:
    edit_tokens = tuple(str(token) for token in record.get("edit_tokens", ()))
    prior_utility = _prior_utility_from_annotations(edit_tokens)
    predicted_contrast = _predicted_contrast_from_annotations(edit_tokens)
    predicted_feasible, constraint_margin = _predicted_feasibility_from_annotations(edit_tokens)
    candidate = {
        "candidate_id": str(record["variant_id"]),
        "variant_id": str(record["variant_id"]),
        "heavy_chain_seq": str(record["heavy_sequence"]),
        "light_chain_seq": str(record["light_sequence"]),
        "heavy_sequence": str(record["heavy_sequence"]),
        "light_sequence": str(record["light_sequence"]),
        "edit_tokens": list(edit_tokens),
        "modules": list(edit_tokens),
        "edit_count": len(edit_tokens),
        "cost": _round_metric(_candidate_cost(edit_tokens)),
        "prior_utility": _round_metric(prior_utility),
        "predicted_feasible": bool(predicted_feasible),
        "feasibility_status": "feasible" if predicted_feasible else "infeasible",
        "constraint_margin": _round_metric(constraint_margin),
        "predicted_pH_contrast": _round_metric(predicted_contrast),
        "generation_source": source,
        "sequence_hash": _sequence_hash(str(record["heavy_sequence"]), str(record["light_sequence"])),
        "world_id": str(record["world_id"]),
        "numeric_features": {
            "edit_count": float(len(edit_tokens)),
            "prior_utility": _round_metric(prior_utility),
            "predicted_pH_contrast": _round_metric(predicted_contrast),
            "predicted_feasible": 1.0 if predicted_feasible else 0.0,
            "risk_count": float(len(set(edit_tokens) & _risky_edit_set())),
        },
        "risk_flags": _risk_flags(edit_tokens),
        "mechanism_tags": _mechanism_tags(edit_tokens),
    }
    _assert_candidate_record_blind(candidate)
    return candidate


def _observed_record(record: Mapping[str, Any], seed: int) -> dict[str, Any]:
    edit_tokens = tuple(str(token) for token in record.get("edit_tokens", ()))
    measured_utility = _to_float(record.get("true_utility"), 0.0) + 0.015 * _noise(
        seed,
        str(record["world_id"]),
        edit_tokens,
        "observed_utility",
    )
    measured_contrast = _to_float(record.get("pH_contrast_score"), 0.0) + 0.01 * _noise(
        seed,
        str(record["world_id"]),
        edit_tokens,
        "observed_contrast",
    )
    return {
        "variant_id": str(record["variant_id"]),
        "candidate_id": str(record["variant_id"]),
        "heavy_chain_seq": str(record["heavy_sequence"]),
        "light_chain_seq": str(record["light_sequence"]),
        "heavy_sequence": str(record["heavy_sequence"]),
        "light_sequence": str(record["light_sequence"]),
        "edit_tokens": list(edit_tokens),
        "modules": list(edit_tokens),
        "observed_utility": _round_metric(measured_utility),
        "observed_feasible": bool(record["true_feasible"]),
        "observed_pH_contrast": _round_metric(measured_contrast),
        "expression": _round_metric(
            _to_float(record.get("endpoint_values", {}).get("expression"), 0.0)
        ),
        "specificity": _round_metric(
            _to_float(record.get("endpoint_values", {}).get("specificity"), 0.0)
        ),
        "cost": _round_metric(_candidate_cost(edit_tokens)),
    }


def _trial_metrics(
    *,
    generated: Sequence[Mapping[str, Any]],
    novel_generated_truth: Sequence[Mapping[str, Any]],
    selected_truth: Sequence[Mapping[str, Any]],
    selected_ids: Sequence[str],
    generation_counted: bool,
    generated_new_site_ids: set[str],
) -> dict[str, Any]:
    generated_count = len(generated) if generation_counted else 0
    generated_novel_count = len(novel_generated_truth) if generation_counted else 0
    generated_new_site_count = len(generated_new_site_ids) if generation_counted else 0
    selected_new_site_count = len(set(str(candidate_id) for candidate_id in selected_ids) & generated_new_site_ids)
    feasible_generated = [
        record for record in novel_generated_truth if generation_counted and bool(record["true_feasible"])
    ]
    feasible_selected = [record for record in selected_truth if bool(record["true_feasible"])]
    selected_count = len(selected_truth)
    false_count = sum(not bool(record["true_feasible"]) for record in selected_truth)
    best_selected_utility = max(
        (_to_float(record.get("true_utility"), 0.0) for record in feasible_selected),
        default=0.0,
    )
    best_selected_pH_contrast = max(
        (_to_float(record.get("pH_contrast_score"), 0.0) for record in feasible_selected),
        default=0.0,
    )
    return {
        "generated_novel_count": int(generated_novel_count),
        "generated_new_site_count": int(generated_new_site_count),
        "selected_new_site_count": int(selected_new_site_count),
        "new_site_rate": _round_metric(generated_new_site_count / generated_count if generated_count else 0.0),
        "best_generated_utility": (
            _round_metric(
                max(
                    (_to_float(record.get("true_utility"), 0.0) for record in feasible_generated),
                    default=0.0,
                )
            )
            if generation_counted
            else None
        ),
        "best_selected_utility": _round_metric(best_selected_utility),
        "best_selected_pH_contrast_score": _round_metric(best_selected_pH_contrast),
        "constraint_pass_rate": _round_metric(len(feasible_selected) / selected_count if selected_count else 0.0),
        "pH_contrast_score": _round_metric(best_selected_pH_contrast),
        "novel_sequence_rate": _round_metric(generated_novel_count / generated_count if generated_count else 0.0),
        "false_claim_rate": _round_metric(false_count / selected_count if selected_count else 0.0),
        "cost_spent": _round_metric(sum(_to_float(record.get("cost"), 1.0) for record in selected_truth)),
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
            required_new_site_rows = [
                row for row in mechanism_rows if _to_bool(row.get("new_site_generation_required"))
            ]
            selected_eligible = (
                world_id == "overall"
                and mechanism not in BASELINE_MECHANISMS
                and not any(_to_bool(row.get("observed_reuse_only")) for row in mechanism_rows)
                and not any(_to_bool(row.get("fixed_pool_only")) for row in mechanism_rows)
                and _mean_metric(mechanism_rows, "generated_novel_count") > 0
                and _mean_metric(mechanism_rows, "novel_sequence_rate") > 0
                and (
                    not required_new_site_rows
                    or _mean_metric(required_new_site_rows, "generated_new_site_count") > 0
                )
            )
            summary.append(
                {
                    "mechanism": mechanism,
                    "world_id": world_id,
                    "replicate_count": len(mechanism_rows),
                    "mean_generated_novel_count": _mean_metric(mechanism_rows, "generated_novel_count"),
                    "mean_generated_new_site_count": _mean_metric(
                        mechanism_rows,
                        "generated_new_site_count",
                    ),
                    "mean_selected_new_site_count": _mean_metric(
                        mechanism_rows,
                        "selected_new_site_count",
                    ),
                    "mean_new_site_rate": _mean_metric(mechanism_rows, "new_site_rate"),
                    "mean_best_generated_utility": _mean_metric(mechanism_rows, "best_generated_utility"),
                    "mean_best_selected_utility": _mean_metric(mechanism_rows, "best_selected_utility"),
                    "mean_best_selected_pH_contrast_score": _mean_metric(
                        mechanism_rows,
                        "best_selected_pH_contrast_score",
                    ),
                    "mean_constraint_pass_rate": _mean_metric(mechanism_rows, "constraint_pass_rate"),
                    "mean_pH_contrast_score": _mean_metric(mechanism_rows, "pH_contrast_score"),
                    "mean_novel_sequence_rate": _mean_metric(mechanism_rows, "novel_sequence_rate"),
                    "mean_false_claim_rate": _mean_metric(mechanism_rows, "false_claim_rate"),
                    "mean_cost_spent": _mean_metric(mechanism_rows, "cost_spent"),
                    "new_site_generation_capable": any(
                        _to_bool(row.get("new_site_generation_capable")) for row in mechanism_rows
                    ),
                    "observed_reuse_only": mechanism in OBSERVED_REUSE_ONLY,
                    "fixed_pool_only": mechanism in FIXED_POOL_ONLY,
                    "selected_eligible": selected_eligible,
                    "selected_mechanism": False,
                    "comparative_utility_rank": "",
                }
            )
    return sorted(summary, key=lambda item: (str(item["mechanism"]), str(item["world_id"])))


def _statistical_summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    mechanisms = sorted({str(row["mechanism"]) for row in rows})
    worlds = sorted({str(row["world_id"]) for row in rows})
    for row in rows:
        mechanism = str(row["mechanism"])
        world_id = str(row["world_id"])
        grouped.setdefault((mechanism, world_id), []).append(row)
        grouped.setdefault((mechanism, "overall"), []).append(row)

    summary: list[dict[str, Any]] = []
    for mechanism in mechanisms:
        for world_id in [*worlds, "overall"]:
            mechanism_rows = grouped.get((mechanism, world_id), [])
            if not mechanism_rows:
                continue
            for metric in STATISTICAL_METRICS:
                summary.append(
                    {
                        "mechanism": mechanism,
                        "world_id": world_id,
                        "metric": metric,
                        **_statistical_fields(_metric_values(mechanism_rows, metric)),
                    }
                )
    return sorted(
        summary,
        key=lambda item: (str(item["mechanism"]), str(item["world_id"]), str(item["metric"])),
    )


def _pairwise_comparison_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    mechanism: str = "cmdgd",
) -> list[dict[str, Any]]:
    mechanisms = sorted({str(row["mechanism"]) for row in rows})
    if mechanism not in mechanisms:
        return []
    baselines = [name for name in mechanisms if name != mechanism]
    worlds = sorted({str(row["world_id"]) for row in rows})
    by_key = {
        (str(row["mechanism"]), str(row["world_id"]), str(row["seed"])): row
        for row in rows
    }
    keys_by_mechanism: dict[str, set[tuple[str, str]]] = {}
    for row in rows:
        keys_by_mechanism.setdefault(str(row["mechanism"]), set()).add(
            (str(row["world_id"]), str(row["seed"]))
        )

    comparisons: list[dict[str, Any]] = []
    for baseline in baselines:
        for world_id in [*worlds, "overall"]:
            paired_keys = _paired_seed_world_keys(
                keys_by_mechanism,
                mechanism,
                baseline,
                world_id,
            )
            for metric in STATISTICAL_METRICS:
                deltas: list[float] = []
                for paired_world, paired_seed in paired_keys:
                    mechanism_row = by_key.get((mechanism, paired_world, paired_seed), {})
                    baseline_row = by_key.get((baseline, paired_world, paired_seed), {})
                    mechanism_value = _optional_float(mechanism_row.get(metric))
                    baseline_value = _optional_float(baseline_row.get(metric))
                    if mechanism_value is None or baseline_value is None:
                        continue
                    deltas.append(mechanism_value - baseline_value)
                comparisons.append(
                    {
                        "mechanism": mechanism,
                        "baseline": baseline,
                        "world_id": world_id,
                        "metric": metric,
                        **_paired_delta_fields(deltas),
                    }
                )
    return sorted(
        comparisons,
        key=lambda item: (
            str(item["baseline"]),
            str(item["world_id"]),
            str(item["metric"]),
        ),
    )


def _paired_seed_world_keys(
    keys_by_mechanism: Mapping[str, set[tuple[str, str]]],
    mechanism: str,
    baseline: str,
    world_id: str,
) -> list[tuple[str, str]]:
    mechanism_keys = keys_by_mechanism.get(mechanism, set())
    baseline_keys = keys_by_mechanism.get(baseline, set())
    paired = mechanism_keys & baseline_keys
    if world_id != "overall":
        paired = {key for key in paired if key[0] == world_id}
    return sorted(paired, key=lambda item: (item[0], item[1]))


def _metric_values(rows: Sequence[Mapping[str, Any]], metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _optional_float(row.get(metric))
        if value is not None:
            values.append(value)
    return values


def _statistical_fields(values: Sequence[float]) -> dict[str, Any]:
    n = len(values)
    if n == 0:
        return {
            "n": 0,
            "mean": "",
            "sd": "",
            "sem": "",
            "ci95_low": "",
            "ci95_high": "",
        }
    mean_value, sd, sem, ci_low, ci_high = _distribution_stats(values)
    return {
        "n": n,
        "mean": mean_value,
        "sd": sd,
        "sem": sem,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def _paired_delta_fields(deltas: Sequence[float]) -> dict[str, Any]:
    n = len(deltas)
    if n == 0:
        return {
            "n_pairs": 0,
            "paired_delta_mean": "",
            "paired_delta_sd": "",
            "paired_delta_sem": "",
            "paired_delta_ci95_low": "",
            "paired_delta_ci95_high": "",
            "win_fraction": "",
            "tie_fraction": "",
            "loss_fraction": "",
        }
    mean_value, sd, sem, ci_low, ci_high = _distribution_stats(deltas)
    wins = sum(delta > 1e-12 for delta in deltas)
    ties = sum(abs(delta) <= 1e-12 for delta in deltas)
    losses = n - wins - ties
    return {
        "n_pairs": n,
        "paired_delta_mean": mean_value,
        "paired_delta_sd": sd,
        "paired_delta_sem": sem,
        "paired_delta_ci95_low": ci_low,
        "paired_delta_ci95_high": ci_high,
        "win_fraction": _round_metric(wins / n),
        "tie_fraction": _round_metric(ties / n),
        "loss_fraction": _round_metric(losses / n),
    }


def _distribution_stats(values: Sequence[float]) -> tuple[float, float, float, float, float]:
    n = len(values)
    mean_value = mean(float(value) for value in values)
    sd = _sample_sd(values, mean_value)
    sem = sd / math.sqrt(n) if n else 0.0
    margin = CI95_Z * sem
    return (
        _round_metric(mean_value),
        _round_metric(sd),
        _round_metric(sem),
        _round_metric(mean_value - margin),
        _round_metric(mean_value + margin),
    )


def _sample_sd(values: Sequence[float], mean_value: float) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    variance = sum((float(value) - mean_value) ** 2 for value in values) / (n - 1)
    return math.sqrt(max(variance, 0.0))


def _select_winner(summary_rows: Sequence[Mapping[str, Any]]) -> tuple[str, bool]:
    overall = [row for row in summary_rows if row.get("world_id") == "overall"]
    eligible = [row for row in overall if _to_bool(row.get("selected_eligible"))]
    if not eligible:
        return "", False
    ranked = sorted(
        eligible,
        key=lambda row: (
            _to_float(row.get("mean_best_generated_utility"), 0.0),
            _to_float(row.get("mean_pH_contrast_score"), 0.0),
            _to_float(row.get("mean_constraint_pass_rate"), 0.0),
            -_to_float(row.get("mean_false_claim_rate"), 0.0),
            _to_float(row.get("mean_generated_novel_count"), 0.0),
        ),
        reverse=True,
    )
    selected = str(ranked[0]["mechanism"])
    return selected, selected == "cmdgd"


def _selection_gate_report(summary_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    overall = [row for row in summary_rows if row.get("world_id") == "overall"]
    selected_mechanism, selected_passes_gate = _select_winner(summary_rows)
    selected_row = next(
        (row for row in overall if str(row.get("mechanism")) == selected_mechanism),
        None,
    )
    selected_utility = (
        _to_float(selected_row.get("mean_best_selected_utility"), 0.0)
        if selected_row is not None
        else 0.0
    )
    ranked_by_utility = sorted(
        overall,
        key=lambda row: (
            _to_float(row.get("mean_best_selected_utility"), 0.0),
            _to_float(row.get("mean_pH_contrast_score"), 0.0),
            _to_float(row.get("mean_constraint_pass_rate"), 0.0),
            -_to_float(row.get("mean_false_claim_rate"), 0.0),
            _to_float(row.get("mean_generated_novel_count"), 0.0),
            str(row.get("mechanism")),
        ),
        reverse=True,
    )
    rankings: list[dict[str, Any]] = []
    selected_utility_rank: int | None = None
    for rank, row in enumerate(ranked_by_utility, start=1):
        mechanism = str(row.get("mechanism"))
        utility = _to_float(row.get("mean_best_selected_utility"), 0.0)
        if mechanism == selected_mechanism:
            selected_utility_rank = rank
        rankings.append(
            {
                "rank": rank,
                "mechanism": mechanism,
                "mean_best_selected_utility": _round_metric(utility),
                "mean_best_generated_utility": _round_metric(
                    _to_float(row.get("mean_best_generated_utility"), 0.0)
                ),
                "mean_generated_novel_count": _round_metric(
                    _to_float(row.get("mean_generated_novel_count"), 0.0)
                ),
                "mean_generated_new_site_count": _round_metric(
                    _to_float(row.get("mean_generated_new_site_count"), 0.0)
                ),
                "mean_new_site_rate": _round_metric(
                    _to_float(row.get("mean_new_site_rate"), 0.0)
                ),
                "new_site_generation_capable": _to_bool(row.get("new_site_generation_capable")),
                "eligible_for_generative_selection": _to_bool(row.get("selected_eligible")),
                "observed_reuse_only": _to_bool(row.get("observed_reuse_only")),
                "fixed_pool_only": _to_bool(row.get("fixed_pool_only")),
                "utility_exceeds_selected": bool(
                    selected_mechanism and mechanism != selected_mechanism and utility > selected_utility
                ),
            }
        )
    utility_winner = rankings[0]["mechanism"] if rankings else ""
    gate_reason = (
        f"{selected_mechanism} selected for generative candidate-space expansion; "
        "comparative selected-utility ranking is reported separately."
        if selected_mechanism
        else "No mechanism satisfied the generative candidate-space expansion eligibility gate."
    )
    return {
        "selection_basis": "generative_candidate_space_expansion",
        "selected_mechanism": selected_mechanism,
        "selected_passes_gate": bool(selected_passes_gate),
        "gate_reason": gate_reason,
        "eligibility_rule": (
            "mean generated novel count and novel sequence rate must be positive; "
            "worlds requiring de novo site generation must have positive generated new-site count; "
            "fixed-pool, observed-reuse-only, and benchmark baseline mechanisms are ineligible"
        ),
        "utility_rank_basis": "mean_best_selected_utility",
        "utility_winner": utility_winner,
        "utility_winner_is_selected": bool(utility_winner and utility_winner == selected_mechanism),
        "selected_utility_rank": selected_utility_rank,
        "comparative_utility_rankings": rankings,
    }


def _annotate_summary_selection(
    summary_rows: Sequence[Mapping[str, Any]],
    gate_report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    selected_mechanism = str(gate_report.get("selected_mechanism", ""))
    rankings = gate_report.get("comparative_utility_rankings", [])
    rank_by_mechanism = {
        str(row.get("mechanism")): int(row.get("rank"))
        for row in rankings
        if isinstance(row, Mapping) and row.get("rank") not in (None, "")
    }
    out: list[dict[str, Any]] = []
    for row in summary_rows:
        mechanism = str(row.get("mechanism"))
        world_id = str(row.get("world_id"))
        out.append(
            {
                **dict(row),
                "selected_mechanism": bool(
                    selected_mechanism
                    and mechanism == selected_mechanism
                    and world_id == "overall"
                ),
                "comparative_utility_rank": (
                    rank_by_mechanism.get(mechanism, "") if world_id == "overall" else ""
                ),
            }
        )
    return out


def _ablation_rows(
    cache: dict[tuple[str, int, str, str], dict[str, Any]],
    *,
    world_specs: Sequence[SequenceWorldSpec],
    seeds: Sequence[int],
    budget: int,
    generation_budget: int,
    include_cmdgd: bool,
) -> list[dict[str, Any]]:
    if not include_cmdgd:
        return []
    rows: list[dict[str, Any]] = []
    for world in world_specs:
        for seed in seeds:
            full = _run_cached_trial(
                cache,
                world=world,
                seed=seed,
                mechanism="cmdgd",
                budget=budget,
                generation_budget=generation_budget,
                ablation="full",
            )
            full_best_generated = _optional_float(full.get("best_generated_utility"))
            full_best_selected = _optional_float(full.get("best_selected_utility"))
            for ablation in CMDGD_ABLATIONS:
                result = _run_cached_trial(
                    cache,
                    world=world,
                    seed=seed,
                    mechanism="cmdgd",
                    budget=budget,
                    generation_budget=generation_budget,
                    ablation=ablation,
                )
                result_best_generated = _optional_float(result.get("best_generated_utility"))
                result_best_selected = _optional_float(result.get("best_selected_utility"))
                row = {
                    **result,
                    "mechanism": "cmdgd",
                    "ablation": ablation,
                    "ablation_type": _cmdgd_ablation_type(ablation),
                    "ablation_component": CMDGD_ABLATION_COMPONENTS.get(ablation, ablation),
                    "delta_from_full_best_generated_utility": (
                        _round_metric(full_best_generated - result_best_generated)
                        if full_best_generated is not None and result_best_generated is not None
                        else None
                    ),
                    "delta_from_full_best_selected_utility": (
                        _round_metric(full_best_selected - result_best_selected)
                        if full_best_selected is not None and result_best_selected is not None
                        else None
                    ),
                }
                rows.append(_project_row(row, ABLATION_COLUMNS))
    return rows


def _example_rows_from_trial(
    result: Mapping[str, Any],
    *,
    max_examples: int,
) -> list[dict[str, Any]]:
    examples = result.get("_examples", [])
    if not isinstance(examples, list):
        return []
    selected = [row for row in examples if row.get("selected")]
    novel = [row for row in examples if not row.get("observed_duplicate")]
    ordered = [*selected, *novel, *examples]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in ordered:
        key = str(row.get("candidate_id", ""))
        if key in seen:
            continue
        seen.add(key)
        rows.append(_project_row(row, EXAMPLE_COLUMNS))
        if len(rows) >= max_examples:
            break
    return rows


def _examples_payload(
    *,
    seed: int,
    world_id: str,
    mechanism: str,
    generated: Sequence[Mapping[str, Any]],
    truth_by_id: Mapping[str, Mapping[str, Any]],
    selected_ids: set[str],
    generation_counted: bool,
) -> list[dict[str, Any]]:
    if not generation_counted and mechanism in FIXED_POOL_ONLY:
        generated = [row for row in generated if str(row["candidate_id"]) in selected_ids]
    rows: list[dict[str, Any]] = []
    for candidate in generated:
        candidate_id = str(candidate["candidate_id"])
        truth = truth_by_id.get(candidate_id, {})
        rows.append(
            {
                "seed": int(seed),
                "world_id": world_id,
                "generated_by": mechanism,
                "candidate_id": candidate_id,
                "heavy_sequence": candidate.get("heavy_sequence", ""),
                "light_sequence": candidate.get("light_sequence", ""),
                "edit_tokens": candidate.get("edit_tokens", []),
                "observed_duplicate": bool(candidate.get("observed_duplicate")),
                "selected": candidate_id in selected_ids,
                "true_feasible": bool(truth.get("true_feasible", False)),
                "true_utility": _round_metric(_to_float(truth.get("true_utility"), 0.0)),
                "pH_contrast_score": _round_metric(_to_float(truth.get("pH_contrast_score"), 0.0)),
                "cost": _round_metric(_to_float(truth.get("cost"), candidate.get("cost", 1.0))),
            }
        )
    return sorted(rows, key=lambda row: (not bool(row["selected"]), str(row["candidate_id"])))


def _reference_cmdgd_generate(
    world: SequenceWorldSpec,
    observed: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any],
    generation_budget: int,
    ablation: str,
) -> list[dict[str, Any]]:
    reference = state.get("reference_state", {}) if isinstance(state, Mapping) else {}
    config = _cmdgd_config_for_ablation(generation_budget, ablation)
    allowed_tokens = set(_world_visible_edit_tokens(world))
    if not config.get("enable_vocabulary_expansion", True):
        allowed_tokens = {token for row in observed for token in _edit_tokens(row)}
    if config.get("enable_de_novo_site_proposal", True):
        allowed_tokens.update(world.new_site_target_tokens)
    ranked_edits = [
        token
        for token, _value in sorted(
            reference.get("edit_scores", {}).items(),
            key=lambda item: (float(item[1]), item[0]),
            reverse=True,
        )
        if token in allowed_tokens
    ]
    vocabulary = sorted(allowed_tokens)
    annotation_rank = sorted(
        vocabulary,
        key=lambda token: (
            _annotation_priority(token),
            token in set(ranked_edits),
            token,
        ),
        reverse=True,
    )
    seed_tokens = _dedupe([*ranked_edits, *annotation_rank])
    if config.get("enable_de_novo_site_proposal", True) and world.new_site_target_tokens:
        seed_tokens = _dedupe([*world.new_site_target_tokens, *seed_tokens])
    observed_sets = {tuple(sorted(_edit_tokens(row))) for row in observed}
    candidates: list[dict[str, Any]] = []
    for size in (2, 3, 1):
        for combo in combinations(seed_tokens[:8], size):
            edit_set = tuple(sorted(combo))
            if edit_set in observed_sets:
                continue
            if _has_conflicting_edits(edit_set):
                continue
            if len(set(edit_set) & set(world.risky_edits)) >= 2:
                continue
            candidates.append({"edit_tokens": list(edit_set), "generation_source": "cmdgd"})
            if len(candidates) >= generation_budget:
                return candidates
    return candidates[:generation_budget]


def _random_edit_generate(
    world: SequenceWorldSpec,
    observed: Sequence[Mapping[str, Any]],
    rng: random.Random,
    generation_budget: int,
) -> list[dict[str, Any]]:
    del observed
    vocabulary = list(_world_visible_edit_tokens(world))
    candidates: list[dict[str, Any]] = []
    attempts = 0
    while len(candidates) < generation_budget and attempts < generation_budget * 20:
        attempts += 1
        size = rng.choice([1, 2, 2, 3])
        edit_set = tuple(sorted(rng.sample(vocabulary, size)))
        if _has_conflicting_edits(edit_set):
            continue
        if len(edit_set) > world.max_edits:
            continue
        candidates.append({"edit_tokens": list(edit_set), "generation_source": "random_edit"})
    return candidates


def _observed_recombination_generate(
    world: SequenceWorldSpec,
    observed: Sequence[Mapping[str, Any]],
    generation_budget: int,
) -> list[dict[str, Any]]:
    observed_tokens = sorted({token for row in observed for token in _edit_tokens(row)})
    observed_sets = {tuple(sorted(_edit_tokens(row))) for row in observed}
    candidates: list[dict[str, Any]] = []
    for size in (2, 3):
        for combo in combinations(observed_tokens, size):
            edit_set = tuple(sorted(combo))
            if edit_set in observed_sets or _has_conflicting_edits(edit_set):
                continue
            if len(edit_set) > world.max_edits:
                continue
            candidates.append({"edit_tokens": list(edit_set), "generation_source": "observed_recombination"})
            if len(candidates) >= generation_budget:
                return candidates
    return candidates


def _single_edit_scan_generate(
    world: SequenceWorldSpec,
    observed: Sequence[Mapping[str, Any]],
    generation_budget: int,
) -> list[dict[str, Any]]:
    observed_sets = {tuple(sorted(_edit_tokens(row))) for row in observed}
    candidates = []
    for edit in _visible_edit_specs(world):
        edit_set = (edit.token,)
        if edit_set in observed_sets:
            continue
        candidates.append({"edit_tokens": [edit.token], "generation_source": "single_edit_scan"})
    return candidates[:generation_budget]


def _fixed_pool_candidates(
    world: SequenceWorldSpec,
    observed: Sequence[Mapping[str, Any]],
    source: str,
) -> list[dict[str, Any]]:
    observed_sets = {tuple(sorted(_edit_tokens(row))) for row in observed}
    vocabulary = list(_world_visible_edit_tokens(world))
    candidates: list[dict[str, Any]] = []
    for size in range(1, world.max_edits + 1):
        for combo in combinations(vocabulary, size):
            edit_set = tuple(sorted(combo))
            if edit_set in observed_sets or _has_conflicting_edits(edit_set):
                continue
            candidates.append({"edit_tokens": list(edit_set), "generation_source": source})
    return candidates


def _fit_reference_state(
    observed: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    base_utility = 0.45
    base_contrast = 0.10
    for row in observed:
        if not _edit_tokens(row):
            base_utility = _to_float(row.get("observed_utility"), base_utility)
            base_contrast = _to_float(row.get("observed_pH_contrast"), base_contrast)
            break
    edit_values: dict[str, list[float]] = {}
    edit_contrasts: dict[str, list[float]] = {}
    edit_feasible: dict[str, list[float]] = {}
    for row in observed:
        tokens = _edit_tokens(row)
        if not tokens:
            continue
        utility_delta = _to_float(row.get("observed_utility"), base_utility) - base_utility
        contrast_delta = _to_float(row.get("observed_pH_contrast"), base_contrast) - base_contrast
        feasible = 1.0 if _to_bool(row.get("observed_feasible")) else 0.0
        for token in tokens:
            edit_values.setdefault(token, []).append(utility_delta / max(len(tokens), 1))
            edit_contrasts.setdefault(token, []).append(contrast_delta / max(len(tokens), 1))
            edit_feasible.setdefault(token, []).append(feasible)
    vocabulary = [dict(item) for item in context.get("edit_vocabulary", [])]
    edit_scores: dict[str, float] = {}
    for item in vocabulary:
        token = str(item["token"])
        observed_score = mean(edit_values.get(token, [0.0]))
        contrast_score = mean(edit_contrasts.get(token, [0.0]))
        annotation_score = _annotation_priority(token) * 0.08
        feasibility_score = mean(edit_feasible.get(token, [1.0]))
        edit_scores[token] = observed_score + 0.45 * contrast_score + annotation_score + 0.08 * feasibility_score
    return {
        "base_utility": base_utility,
        "base_contrast": base_contrast,
        "edit_scores": edit_scores,
        "edit_contrasts": {token: mean(values) for token, values in edit_contrasts.items()},
    }


def _reference_score_candidates(
    candidates: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any],
    ablation: str,
) -> list[dict[str, Any]]:
    reference = state.get("reference_state", {}) if isinstance(state, Mapping) else {}
    edit_scores = reference.get("edit_scores", {})
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        row = dict(candidate)
        tokens = _edit_tokens(row)
        score = _to_float(reference.get("base_utility"), 0.45)
        score += sum(_to_float(edit_scores.get(token), 0.0) for token in tokens)
        if len(tokens) >= 2:
            score += 0.06 * _complementarity_bonus(tokens)
        if ablation not in {"no_ph_contrast_objective", "no_contrastive_objective"}:
            score += 0.28 * _to_float(row.get("predicted_pH_contrast"), 0.0)
        if ablation != "no_constraint_guardrail" and not _to_bool(row.get("predicted_feasible")):
            score -= 0.60
        score -= 0.04 * max(_cost(row) - 1.0, 0.0)
        row["score"] = _round_metric(score)
        row["score_components"] = {
            "reference_edit_score": _round_metric(score),
            "predicted_pH_contrast": _to_float(row.get("predicted_pH_contrast"), 0.0),
            "predicted_feasible": 1.0 if _to_bool(row.get("predicted_feasible")) else 0.0,
        }
        rows.append(row)
    return sorted(rows, key=lambda item: (_to_float(item.get("score"), 0.0), str(item["candidate_id"])), reverse=True)


def _coerce_scored_candidates(
    scored: Sequence[Any],
    original_candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {str(row["candidate_id"]): dict(row) for row in original_candidates}
    rows: list[dict[str, Any]] = []
    for item in scored:
        if isinstance(item, Mapping):
            candidate_id = str(item.get("candidate_id") or item.get("variant_id") or "")
            if candidate_id in by_id:
                row = {**by_id[candidate_id]}
                for key in ("score", "score_components", "posterior_mean", "posterior_std"):
                    if key in item:
                        row[key] = _copy_jsonable(item[key])
                rows.append(row)
        else:
            candidate_id = str(item)
            if candidate_id in by_id:
                rows.append(by_id[candidate_id])
    if not rows:
        rows = [dict(candidate) for candidate in original_candidates]
    _assert_candidate_list_blind(rows)
    return rows


def _score_mccbd_pool(
    observed: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    try:
        from design_scientist.algorithms import mccbd

        state = mccbd.fit_state(observed, candidates)
        return list(mccbd.score_candidates(observed, candidates, fitted_state=state))
    except Exception:
        return _reference_score_candidates(
            candidates,
            {"reference_state": _fit_reference_state(observed, {"edit_vocabulary": _visible_edit_vocabulary()})},
            "full",
        )


def _score_evidence_calibrated_ucb(
    observed: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    try:
        from design_scientist.algorithms import evidence_calibrated_ucb

        scored = evidence_calibrated_ucb.score_candidates(observed, candidates, round_index=0)
        return _coerce_scored_candidates(scored, candidates)
    except Exception:
        return _reference_score_candidates(
            candidates,
            {"reference_state": _fit_reference_state(observed, {"edit_vocabulary": _visible_edit_vocabulary()})},
            "full",
        )


def _mccbd_pool_select(
    observed: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    budget: int,
    rng: random.Random,
) -> list[str]:
    try:
        from design_scientist.algorithms import mccbd

        return list(mccbd.select_batch(observed, candidates, budget, 0, rng))
    except Exception:
        return _greedy_score_select(candidates, budget, diversity=True)


def _evidence_calibrated_ucb_select(
    observed: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    budget: int,
    rng: random.Random,
) -> list[str]:
    try:
        from design_scientist.algorithms import evidence_calibrated_ucb

        selected = evidence_calibrated_ucb.select_batch(observed, candidates, budget, 0, rng)
        return _resolve_selected(selected, candidates, budget)
    except Exception:
        return _greedy_score_select(candidates, budget, diversity=True)


def _policy_select(
    policy_name: str,
    observed: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    budget: int,
    rng: random.Random,
) -> list[str]:
    try:
        from design_scientist import policies

        policy = getattr(policies, policy_name)
        selected = policy(observed, candidates, budget, 0, rng)
        return _resolve_selected(selected, candidates, budget)
    except Exception:
        return []


def _random_feasible_select(
    candidates: Sequence[Mapping[str, Any]],
    budget: int,
    rng: random.Random,
) -> list[str]:
    pool = [candidate for candidate in candidates if _to_bool(candidate.get("predicted_feasible", True))]
    if not pool:
        pool = list(candidates)
    rng.shuffle(pool)
    return _resolve_selected(pool, pool, budget)


def _fixed_mix_select(
    candidates: Sequence[Mapping[str, Any]],
    budget: int,
    rng: random.Random,
) -> list[str]:
    ranked = sorted(
        candidates,
        key=lambda row: (
            _to_float(row.get("score"), _to_float(row.get("prior_utility"), 0.0)),
            _to_float(row.get("predicted_pH_contrast"), 0.0),
            str(row.get("candidate_id")),
        ),
        reverse=True,
    )
    selected: list[Mapping[str, Any]] = []
    selected_ids: set[str] = set()
    spent = 0.0
    greedy_budget = max(1.0, float(budget) * 0.55)
    for row in ranked:
        if spent >= greedy_budget:
            break
        spent = _append_affordable(row, selected, selected_ids, spent, budget)
    remaining = [row for row in candidates if str(row.get("candidate_id")) not in selected_ids]
    rng.shuffle(remaining)
    for row in remaining:
        if spent >= float(budget):
            break
        spent = _append_affordable(row, selected, selected_ids, spent, budget)
    return [str(row["candidate_id"]) for row in selected]


def _append_affordable(
    row: Mapping[str, Any],
    selected: list[Mapping[str, Any]],
    selected_ids: set[str],
    spent: float,
    budget: int,
) -> float:
    candidate_id = str(row.get("candidate_id") or row.get("variant_id") or "")
    if not candidate_id or candidate_id in selected_ids:
        return spent
    cost = _cost(row)
    if spent + cost > float(budget):
        return spent
    selected.append(row)
    selected_ids.add(candidate_id)
    return spent + cost


def _greedy_score_select(
    candidates: Sequence[Mapping[str, Any]],
    budget: int,
    *,
    diversity: bool,
) -> list[str]:
    remaining = [dict(candidate) for candidate in candidates]
    selected_rows: list[Mapping[str, Any]] = []
    selected_ids: list[str] = []
    spent = 0.0
    while remaining:
        affordable = [row for row in remaining if spent + _cost(row) <= float(budget)]
        if not affordable:
            break
        best = max(
            affordable,
            key=lambda row: (
                _selection_value(row, selected_rows, diversity),
                _to_float(row.get("score"), _to_float(row.get("prior_utility"), 0.0)),
                -_cost(row),
                str(row["candidate_id"]),
            ),
        )
        selected_rows.append(best)
        selected_ids.append(str(best["candidate_id"]))
        spent += _cost(best)
        remaining.remove(best)
        if spent >= float(budget):
            break
    return selected_ids


def _selection_value(
    row: Mapping[str, Any],
    selected_rows: Sequence[Mapping[str, Any]],
    diversity: bool,
) -> float:
    value = _to_float(row.get("score"), _to_float(row.get("prior_utility"), 0.0))
    if not _to_bool(row.get("predicted_feasible")):
        value -= 0.35
    if diversity and selected_rows:
        value -= 0.16 * max((_edit_jaccard(row, selected) for selected in selected_rows), default=0.0)
    return value


def _all_world_records(world: SequenceWorldSpec, seed: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    vocabulary = [edit.token for edit in _edit_vocabulary()]
    for size in range(0, world.max_edits + 1):
        for edit_tokens in combinations(vocabulary, size):
            if _has_conflicting_edits(edit_tokens):
                continue
            records.append(_truth_record(world, tuple(sorted(edit_tokens)), seed))
    return sorted(records, key=lambda item: str(item["variant_id"]))


def _truth_record(
    world: SequenceWorldSpec,
    edit_tokens: Sequence[str],
    seed: int,
) -> dict[str, Any]:
    edit_set = tuple(sorted(dict.fromkeys(str(token) for token in edit_tokens)))
    heavy, light = _apply_edits(world.base_heavy_sequence, world.base_light_sequence, edit_set)
    endpoint_values = _oracle_endpoints(world, edit_set, seed)
    pH_contrast = _clip(endpoint_values["neutral_binding"] - endpoint_values["acidic_binding"], -1.0, 1.0)
    true_utility = _utility(endpoint_values, pH_contrast)
    true_feasible, constraint_terms = _true_feasibility(world, edit_set, endpoint_values)
    return {
        "variant_id": _variant_id(world.world_id, edit_set),
        "world_id": world.world_id,
        "heavy_sequence": heavy,
        "light_sequence": light,
        "edit_tokens": list(edit_set),
        "endpoint_values": _rounded_mapping(endpoint_values),
        "pH_contrast_score": _round_metric(pH_contrast),
        "true_utility": _round_metric(true_utility),
        "true_feasible": bool(true_feasible),
        "cost": _round_metric(_candidate_cost(edit_set)),
        "oracle": {
            "constraint_terms": constraint_terms,
            "seed": int(seed),
        },
    }


def _oracle_endpoints(
    world: SequenceWorldSpec,
    edit_tokens: Sequence[str],
    seed: int,
) -> dict[str, float]:
    endpoints = {
        "neutral_binding": 0.48,
        "acidic_binding": 0.38,
        "expression": 0.66,
        "specificity": 0.62,
    }
    edits = _edit_by_token()
    for token in edit_tokens:
        edit = edits.get(token)
        if edit is None:
            endpoints["expression"] -= 0.04
            endpoints["specificity"] -= 0.03
            continue
        endpoints["neutral_binding"] += edit.neutral_binding_delta
        endpoints["acidic_binding"] += edit.acidic_binding_delta
        endpoints["expression"] += edit.expression_delta
        endpoints["specificity"] += edit.specificity_delta
    for pair, effect in _interaction_effects(world.world_id).items():
        if set(pair).issubset(set(edit_tokens)):
            for endpoint, delta in effect.items():
                endpoints[endpoint] += delta
    for endpoint in endpoints:
        endpoints[endpoint] += 0.008 * _noise(seed, world.world_id, edit_tokens, endpoint)
    if world.world_id == "escape_risk":
        if set(edit_tokens) & {"H8W", "L6N"}:
            endpoints["acidic_binding"] += 0.06
            endpoints["expression"] -= 0.04
        if {"H13D", "L9R"}.issubset(set(edit_tokens)):
            endpoints["acidic_binding"] -= 0.05
            endpoints["specificity"] += 0.04
    return {key: _clip(value) for key, value in endpoints.items()}


def _true_feasibility(
    world: SequenceWorldSpec,
    edit_tokens: Sequence[str],
    endpoints: Mapping[str, float],
) -> tuple[bool, dict[str, Any]]:
    edit_set = set(edit_tokens)
    forbidden_hits = [pair for pair in world.forbidden_pairs if set(pair).issubset(edit_set)]
    risky_hits = sorted(edit_set & set(world.risky_edits))
    margins = {
        "expression": float(endpoints["expression"]) - world.expression_threshold,
        "specificity": float(endpoints["specificity"]) - world.specificity_threshold,
        "acidic_binding": world.acidic_binding_max - float(endpoints["acidic_binding"]),
    }
    feasible = all(value >= 0.0 for value in margins.values()) and not forbidden_hits
    return feasible, {
        "threshold_margins": _rounded_mapping(margins),
        "forbidden_pair_hits": [list(pair) for pair in forbidden_hits],
        "risky_edit_hits": risky_hits,
    }


def _initial_observed_records(
    world: SequenceWorldSpec,
    records_by_id: Mapping[str, Mapping[str, Any]],
    seed: int,
) -> list[Mapping[str, Any]]:
    del seed
    records = []
    for edit_set in world.initial_edit_sets:
        variant_id = _variant_id(world.world_id, tuple(sorted(edit_set)))
        if variant_id in records_by_id:
            records.append(records_by_id[variant_id])
    return sorted(records, key=lambda item: str(item["variant_id"]))


def _design_context(
    world: SequenceWorldSpec,
    observed: Sequence[Mapping[str, Any]],
    seed: int,
    generation_budget: int,
    ablation: str,
) -> dict[str, Any]:
    cmdgd_config = _cmdgd_config_for_ablation(generation_budget, ablation)
    return {
        "world_id": world.world_id,
        "description": world.description,
        "base_heavy_chain_seq": world.base_heavy_sequence,
        "base_light_chain_seq": world.base_light_sequence,
        "base_heavy_sequence": world.base_heavy_sequence,
        "base_light_sequence": world.base_light_sequence,
        "observed_records": list(observed),
        "edit_vocabulary": _visible_edit_vocabulary(world),
        "observed_mutation_sites": _mutation_site_records(_mutation_sites_from_records(observed, world)),
        "visible_mutation_sites": _mutation_site_records(_visible_mutation_sites(world)),
        "new_site_generation_required": bool(world.new_site_target_tokens),
        "forbidden_pairs": [list(pair) for pair in world.forbidden_pairs],
        "risky_edits": list(world.risky_edits),
        "generation_budget": int(generation_budget),
        "max_generated_candidates": int(generation_budget),
        "cmdgd_config": dict(cmdgd_config),
        "config": dict(cmdgd_config),
        "seed": int(seed),
        "ablation": ablation,
    }


def _candidate_space_type(mechanism: str, ablation: str, generation_counted: bool) -> str:
    if ablation in CMDGD_BENCHMARK_BOUNDARY_ABLATIONS:
        return "fixed_pool_boundary"
    if mechanism in FIXED_POOL_ONLY:
        return "fixed_pool"
    if mechanism in OBSERVED_REUSE_ONLY:
        return "observed_recombination"
    if generation_counted:
        return "generated"
    return "not_generated"


def _cmdgd_config_for_ablation(generation_budget: int, ablation: str) -> dict[str, Any]:
    config: dict[str, Any] = {
        "max_generated_candidates": int(generation_budget),
        "max_edits_per_candidate": 3,
    }
    config.update(CMDGD_COMPONENT_ABLATION_CONFIGS.get(str(ablation), {}))
    return config


def _cmdgd_config_from_context(context: Mapping[str, Any]) -> dict[str, Any]:
    config = context.get("cmdgd_config", context.get("config"))
    if isinstance(config, Mapping):
        return dict(config)
    return _cmdgd_config_for_ablation(
        int(context.get("generation_budget", DEFAULT_GENERATION_BUDGET)),
        str(context.get("ablation", "full")),
    )


def _cmdgd_ablation_type(ablation: str) -> str:
    if ablation in CMDGD_BENCHMARK_BOUNDARY_ABLATIONS:
        return "benchmark_boundary"
    return "cmdgd_component"


def _world_visible_edit_tokens(world: SequenceWorldSpec | None) -> tuple[str, ...]:
    if world is not None and world.visible_edit_tokens:
        return tuple(str(token) for token in world.visible_edit_tokens)
    return tuple(
        edit.token
        for edit in _edit_vocabulary()
        if edit.token not in DEFAULT_HIDDEN_DE_NOVO_EDIT_TOKENS
    )


def _visible_edit_specs(world: SequenceWorldSpec | None = None) -> tuple[EditSpec, ...]:
    visible_tokens = set(_world_visible_edit_tokens(world))
    return tuple(edit for edit in _edit_vocabulary() if edit.token in visible_tokens)


def _world_suite() -> tuple[SequenceWorldSpec, ...]:
    base_heavy = "EVQLVESGGGLVQPGGSL"
    base_light = "DIQMTQSPSSLSASVGDR"
    return (
        SequenceWorldSpec(
            world_id="ph_contrast",
            description="Useful designs preserve neutral binding while reducing acidic binding.",
            base_heavy_sequence=base_heavy,
            base_light_sequence=base_light,
            initial_edit_sets=(
                (),
                ("H4F",),
                ("L4E",),
                ("H13D",),
                ("H7Y",),
                ("H8W",),
                ("L16P",),
            ),
            forbidden_pairs=(("H8W", "L6N"),),
            risky_edits=("H8W", "L6N"),
            expression_threshold=0.47,
            specificity_threshold=0.48,
            acidic_binding_max=0.67,
        ),
        SequenceWorldSpec(
            world_id="escape_risk",
            description="High apparent binders can fail acidic release and developability constraints.",
            base_heavy_sequence=base_heavy,
            base_light_sequence=base_light,
            initial_edit_sets=(
                (),
                ("H4F",),
                ("L4E",),
                ("H13D",),
                ("L9R",),
                ("H8W",),
                ("L6N",),
                ("H15K",),
            ),
            forbidden_pairs=(("H8W", "L6N"), ("H8W", "H15K")),
            risky_edits=("H8W", "L6N"),
            expression_threshold=0.52,
            specificity_threshold=0.52,
            acidic_binding_max=0.61,
        ),
        SequenceWorldSpec(
            world_id="vocabulary_extension",
            description=(
                "The best pH-switching design requires adding a plausible "
                "design-vocabulary edit that has not appeared in the observed variants."
            ),
            base_heavy_sequence=base_heavy,
            base_light_sequence=base_light,
            initial_edit_sets=(
                (),
                ("H4F",),
                ("L4E",),
                ("H13D",),
                ("H7Y",),
                ("H8W",),
                ("L16P",),
            ),
            forbidden_pairs=(("H8W", "L6N"),),
            risky_edits=("H8W", "L6N"),
            expression_threshold=0.49,
            specificity_threshold=0.49,
            acidic_binding_max=0.63,
        ),
        SequenceWorldSpec(
            world_id="de_novo_site_generalization",
            description=(
                "The hidden optimum requires proposing a mutation at a heavy-chain "
                "position absent from both observed variants and the visible edit vocabulary."
            ),
            base_heavy_sequence=base_heavy,
            base_light_sequence=base_light,
            initial_edit_sets=(
                (),
                ("H4F",),
                ("L4E",),
                ("H13D",),
                ("L9R",),
                ("L16P",),
                ("H4F", "L4E"),
            ),
            forbidden_pairs=(("H8W", "L6N"), ("H8W", "H15K")),
            risky_edits=("H8W", "L6N"),
            expression_threshold=0.50,
            specificity_threshold=0.50,
            acidic_binding_max=0.60,
            visible_edit_tokens=(
                "H4F",
                "H7Y",
                "H13D",
                "H15K",
                "H8W",
                "L4E",
                "L9R",
                "L13S",
                "L16P",
                "L6N",
                "L18K",
            ),
            new_site_target_tokens=("H10Q",),
        ),
    )


def _edit_vocabulary() -> tuple[EditSpec, ...]:
    return (
        EditSpec("H4F", "H", 4, "F", "contrast", 0.15, -0.04, 0.02, 0.03),
        EditSpec("H7Y", "H", 7, "Y", "binder", 0.12, 0.02, 0.02, 0.00),
        EditSpec("H10Q", "H", 10, "Q", "contrast", 0.18, -0.15, 0.03, 0.02),
        EditSpec("H13D", "H", 13, "D", "contrast", 0.05, -0.11, -0.01, 0.04),
        EditSpec("H15K", "H", 15, "K", "specificity", 0.02, -0.03, 0.01, 0.10),
        EditSpec("H8W", "H", 8, "W", "liability", 0.25, 0.21, -0.18, -0.22),
        EditSpec("L4E", "L", 4, "E", "contrast", 0.10, -0.08, 0.03, 0.02),
        EditSpec("L9R", "L", 9, "R", "contrast", 0.08, -0.04, 0.06, 0.02),
        EditSpec("L13S", "L", 13, "S", "stability", 0.02, -0.01, 0.07, 0.03),
        EditSpec("L16P", "L", 16, "P", "expression", -0.01, 0.00, 0.11, 0.00),
        EditSpec("L6N", "L", 6, "N", "liability", 0.09, 0.12, -0.13, -0.04),
        EditSpec("L18K", "L", 18, "K", "contrast", 0.04, -0.07, 0.02, 0.02),
    )


def _visible_edit_vocabulary(world: SequenceWorldSpec | None = None) -> list[dict[str, Any]]:
    return [
        {
            "token": edit.token,
            "chain": edit.chain,
            "position": edit.position,
            "residue": edit.residue,
            "annotation": edit.annotation,
        }
        for edit in _visible_edit_specs(world)
    ]


def _interaction_effects(world_id: str) -> dict[tuple[str, str] | tuple[str, str, str], dict[str, float]]:
    base: dict[tuple[str, str] | tuple[str, str, str], dict[str, float]] = {
        ("H4F", "L4E"): {
            "neutral_binding": 0.08,
            "acidic_binding": -0.05,
            "expression": 0.02,
        },
        ("H13D", "L9R"): {
            "neutral_binding": 0.07,
            "acidic_binding": -0.08,
            "specificity": 0.05,
        },
        ("H8W", "L6N"): {
            "neutral_binding": 0.18,
            "acidic_binding": 0.20,
            "expression": -0.08,
            "specificity": -0.10,
        },
        ("H4F", "H13D", "L4E"): {
            "neutral_binding": 0.04,
            "acidic_binding": -0.04,
            "expression": 0.01,
        },
    }
    if world_id == "escape_risk":
        base[("H13D", "L9R", "H15K")] = {
            "neutral_binding": 0.05,
            "acidic_binding": -0.06,
            "specificity": 0.07,
        }
    if world_id == "vocabulary_extension":
        base[("H4F", "H13D", "L4E")] = {
            "neutral_binding": 0.00,
            "acidic_binding": -0.01,
            "expression": 0.00,
        }
        base[("H4F", "L18K", "L4E")] = {
            "neutral_binding": 0.12,
            "acidic_binding": -0.16,
            "expression": 0.03,
            "specificity": 0.03,
        }
    if world_id == "de_novo_site_generalization":
        base[("H4F", "H13D", "L4E")] = {
            "neutral_binding": 0.00,
            "acidic_binding": -0.01,
            "expression": 0.00,
        }
        base[("H10Q", "L4E")] = {
            "neutral_binding": 0.13,
            "acidic_binding": -0.17,
            "expression": 0.03,
            "specificity": 0.03,
        }
        base[("H10Q", "H13D")] = {
            "neutral_binding": 0.07,
            "acidic_binding": -0.10,
            "specificity": 0.04,
        }
    return base


def _utility(endpoints: Mapping[str, float], pH_contrast: float) -> float:
    return _clip(
        0.38 * float(endpoints["neutral_binding"])
        + 0.28 * _clip(pH_contrast, 0.0, 1.0)
        + 0.22 * float(endpoints["expression"])
        + 0.12 * float(endpoints["specificity"]),
        0.0,
        1.25,
    )


def _predicted_feasibility_from_annotations(edit_tokens: Sequence[str]) -> tuple[bool, float]:
    edit_set = set(edit_tokens)
    risk_count = len(edit_set & _risky_edit_set())
    forbidden = {"H8W", "L6N"}.issubset(edit_set) or {"H8W", "H15K"}.issubset(edit_set)
    margin = 0.55 - 0.22 * risk_count - (0.35 if forbidden else 0.0) - 0.03 * max(len(edit_set) - 2, 0)
    return margin >= 0.16 and not forbidden, margin


def _prior_utility_from_annotations(edit_tokens: Sequence[str]) -> float:
    if not edit_tokens:
        return 0.45
    value = 0.42
    for token in edit_tokens:
        value += 0.08 * _annotation_priority(token)
    value += 0.025 * _complementarity_bonus(edit_tokens)
    value -= 0.035 * max(len(edit_tokens) - 2, 0)
    return _clip(value, 0.0, 1.0)


def _predicted_contrast_from_annotations(edit_tokens: Sequence[str]) -> float:
    value = 0.10
    for token in edit_tokens:
        annotation = _edit_by_token().get(token).annotation if token in _edit_by_token() else ""
        if annotation == "contrast":
            value += 0.12
        elif annotation in {"specificity", "expression", "stability"}:
            value += 0.03
        elif annotation == "liability":
            value -= 0.10
    value += 0.04 * _complementarity_bonus(edit_tokens)
    return _clip(value, -0.5, 1.0)


def _annotation_priority(token: str) -> float:
    edit = _edit_by_token().get(token)
    if edit is None:
        return -0.2
    priorities = {
        "contrast": 1.0,
        "specificity": 0.72,
        "expression": 0.58,
        "stability": 0.54,
        "binder": 0.42,
        "liability": -1.0,
    }
    return priorities.get(edit.annotation, 0.0)


def _complementarity_bonus(edit_tokens: Sequence[str]) -> float:
    annotations = {
        _edit_by_token()[token].annotation
        for token in edit_tokens
        if token in _edit_by_token()
    }
    bonus = 0.0
    if "contrast" in annotations and "expression" in annotations:
        bonus += 0.7
    if "contrast" in annotations and "specificity" in annotations:
        bonus += 0.7
    if "contrast" in annotations and "stability" in annotations:
        bonus += 0.4
    if "liability" in annotations:
        bonus -= 1.0
    return bonus


def _apply_edits(
    base_heavy: str,
    base_light: str,
    edit_tokens: Sequence[str],
) -> tuple[str, str]:
    heavy = list(base_heavy)
    light = list(base_light)
    for token in edit_tokens:
        edit = _edit_by_token().get(str(token))
        if edit is None:
            continue
        target = heavy if edit.chain == "H" else light
        index = int(edit.position) - 1
        if 0 <= index < len(target):
            target[index] = edit.residue
    return "".join(heavy), "".join(light)


def _extract_edit_tokens(raw: Mapping[str, Any], world: SequenceWorldSpec) -> tuple[str, ...] | None:
    for key in ("edit_tokens", "edits", "modules", "mutation_tokens", "mutations"):
        value = raw.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, str):
            tokens = [part.strip() for part in value.replace(",", ";").split(";") if part.strip()]
        elif isinstance(value, Sequence):
            tokens = []
            for item in value:
                if isinstance(item, Mapping):
                    token = item.get("id", item.get("edit_id"))
                    if token not in (None, ""):
                        tokens.append(str(token).strip())
                elif str(item).strip():
                    tokens.append(str(item).strip())
        else:
            continue
        return tuple(sorted(dict.fromkeys(tokens)))
    heavy = raw.get("heavy_sequence")
    light = raw.get("light_sequence")
    if heavy not in (None, "") and light not in (None, ""):
        return _diff_to_known_edit_tokens(world, str(heavy), str(light))
    return None


def _diff_to_known_edit_tokens(
    world: SequenceWorldSpec,
    heavy: str,
    light: str,
) -> tuple[str, ...]:
    tokens = []
    for edit in _edit_vocabulary():
        sequence = heavy if edit.chain == "H" else light
        base = world.base_heavy_sequence if edit.chain == "H" else world.base_light_sequence
        index = edit.position - 1
        if 0 <= index < len(sequence) and 0 <= index < len(base):
            if sequence[index] != base[index] and sequence[index] == edit.residue:
                tokens.append(edit.token)
    return tuple(sorted(tokens))


def _new_site_candidate_ids(
    candidates: Sequence[Mapping[str, Any]],
    *,
    world: SequenceWorldSpec,
    observed_mutation_sites: set[tuple[str, int]],
    visible_mutation_sites: set[tuple[str, int]],
) -> set[str]:
    excluded_sites = set(observed_mutation_sites) | set(visible_mutation_sites)
    ids: set[str] = set()
    for candidate in candidates:
        if bool(candidate.get("observed_duplicate")):
            continue
        candidate_sites = _candidate_mutation_sites(candidate, world)
        if any(site not in excluded_sites for site in candidate_sites):
            ids.add(str(candidate.get("candidate_id") or candidate.get("variant_id") or ""))
    ids.discard("")
    return ids


def _mutation_sites_from_records(
    records: Sequence[Mapping[str, Any]],
    world: SequenceWorldSpec,
) -> set[tuple[str, int]]:
    sites: set[tuple[str, int]] = set()
    for record in records:
        sites.update(_candidate_mutation_sites(record, world))
    return sites


def _visible_mutation_sites(world: SequenceWorldSpec) -> set[tuple[str, int]]:
    sites: set[tuple[str, int]] = set()
    for edit in _visible_edit_vocabulary(world):
        chain = str(edit.get("chain", "")).upper()
        position = _to_int(edit.get("position"))
        if chain in {"H", "L"} and position is not None:
            sites.add((chain, int(position)))
    return sites


def _mutation_site_records(sites: set[tuple[str, int]]) -> list[dict[str, Any]]:
    return [
        {"chain": chain, "position": int(position)}
        for chain, position in sorted(sites, key=lambda item: (item[0], item[1]))
    ]


def _candidate_mutation_sites(
    record: Mapping[str, Any],
    world: SequenceWorldSpec,
) -> set[tuple[str, int]]:
    sites: set[tuple[str, int]] = set()
    for token in _edit_tokens(record):
        site = _mutation_site_from_token(token)
        if site is not None:
            sites.add(site)
    heavy = str(record.get("heavy_sequence", record.get("heavy_chain_seq", "")))
    light = str(record.get("light_sequence", record.get("light_chain_seq", "")))
    sites.update(_sequence_mutation_sites("H", world.base_heavy_sequence, heavy))
    sites.update(_sequence_mutation_sites("L", world.base_light_sequence, light))
    return sites


def _sequence_mutation_sites(
    chain: str,
    base_sequence: str,
    candidate_sequence: str,
) -> set[tuple[str, int]]:
    if not candidate_sequence:
        return set()
    return {
        (chain, index)
        for index, (base_residue, candidate_residue) in enumerate(
            zip(base_sequence, candidate_sequence),
            start=1,
        )
        if base_residue != candidate_residue
    }


def _mutation_site_from_token(token: str) -> tuple[str, int] | None:
    edit = _edit_by_token().get(str(token))
    if edit is not None:
        return edit.chain, int(edit.position)
    match = re.fullmatch(r"([HL])(\d+)[A-Za-z*]+", str(token).strip())
    if not match:
        return None
    chain, position = match.groups()
    return chain, int(position)


def _new_site_generation_capable(mechanism: str, ablation: str) -> bool:
    return (
        mechanism == "cmdgd"
        and ablation not in CMDGD_BENCHMARK_BOUNDARY_ABLATIONS
        and ablation not in CMDGD_DE_NOVO_SITE_DISABLED_ABLATIONS
    )


def _resolve_selected(
    selected: Sequence[Any] | None,
    candidates: Sequence[Mapping[str, Any]],
    budget: int,
) -> list[str]:
    if selected is None:
        return []
    by_id = {
        str(candidate.get("candidate_id") or candidate.get("variant_id")): candidate
        for candidate in candidates
    }
    resolved: list[str] = []
    spent = 0.0
    for item in selected:
        if isinstance(item, Mapping):
            candidate_id = str(item.get("candidate_id") or item.get("variant_id") or "")
        else:
            candidate_id = str(item)
        if not candidate_id or candidate_id in resolved or candidate_id not in by_id:
            continue
        cost = _cost(by_id[candidate_id])
        if spent + cost > float(budget):
            continue
        resolved.append(candidate_id)
        spent += cost
        if spent >= float(budget):
            break
    return resolved


def _project_candidate_example(candidate: Mapping[str, Any]) -> dict[str, Any]:
    if not candidate:
        return {}
    return {
        key: _copy_jsonable(candidate[key])
        for key in CANDIDATE_VISIBLE_FIELDS
        if key in candidate
    }


def _assert_candidate_list_blind(candidates: Sequence[Mapping[str, Any]]) -> None:
    for candidate in candidates:
        _assert_candidate_record_blind(candidate)


def _assert_candidate_record_blind(candidate: Mapping[str, Any]) -> None:
    leaked = sorted(_candidate_leakage_paths(candidate))
    if leaked:
        raise AssertionError("selector candidate leaks oracle truth: " + ", ".join(leaked))


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


def _load_cmdgd_module() -> Any | None:
    try:
        return importlib.import_module("design_scientist.algorithms.cmdgd")
    except ImportError:
        return None


def _cmdgd_lifecycle(module: Any) -> Any | None:
    factory = getattr(module, "mechanism_lifecycle", None)
    if callable(factory):
        return factory()
    lifecycle_type = getattr(module, "CMDGDLifecycle", None)
    if callable(lifecycle_type):
        return lifecycle_type()
    return None


def _try_call(
    fn: Any,
    positional_attempts: Sequence[tuple[Any, ...]],
    keyword_attempt: Mapping[str, Any],
) -> Any | None:
    for args in positional_attempts:
        try:
            return fn(*args)
        except TypeError:
            continue
    try:
        return fn(**dict(keyword_attempt))
    except TypeError:
        return None


def _normalize_mechanisms(mechanisms: Iterable[str] | str | None) -> list[str]:
    if mechanisms is None:
        names = list(DEFAULT_MECHANISMS)
    elif isinstance(mechanisms, str):
        names = [mechanisms]
    else:
        names = [str(mechanism) for mechanism in mechanisms]
    valid = set(SUPPORTED_MECHANISMS)
    unknown = [name for name in names if name not in valid]
    if unknown:
        raise ValueError(f"Unknown generative benchmark mechanisms: {', '.join(unknown)}")
    return _dedupe(names)


def _select_worlds(worlds: Iterable[str] | str | None) -> list[SequenceWorldSpec]:
    if worlds is None:
        world_ids = list(DEFAULT_WORLDS)
    elif isinstance(worlds, str):
        world_ids = [worlds]
    else:
        world_ids = [str(world) for world in worlds]
    return [_world_by_id(world_id) for world_id in world_ids]


def _world_by_id(world_id: str) -> SequenceWorldSpec:
    by_id = {world.world_id: world for world in _world_suite()}
    try:
        return by_id[str(world_id)]
    except KeyError as exc:
        raise ValueError(f"Unknown generative benchmark world: {world_id}") from exc


def _edit_by_token() -> dict[str, EditSpec]:
    return {edit.token: edit for edit in _edit_vocabulary()}


def _risky_edit_set() -> set[str]:
    return {edit.token for edit in _edit_vocabulary() if edit.annotation == "liability"}


def _risk_flags(edit_tokens: Sequence[str]) -> list[str]:
    flags = []
    if set(edit_tokens) & _risky_edit_set():
        flags.append("liability_edit")
    if {"H8W", "L6N"}.issubset(set(edit_tokens)):
        flags.append("forbidden_pair")
    if {"H8W", "H15K"}.issubset(set(edit_tokens)):
        flags.append("escape_specificity_risk")
    return sorted(flags)


def _mechanism_tags(edit_tokens: Sequence[str]) -> list[str]:
    tags = []
    annotations = [
        _edit_by_token()[token].annotation
        for token in edit_tokens
        if token in _edit_by_token()
    ]
    if not edit_tokens:
        tags.append("base_sequence")
    if len(edit_tokens) >= 2:
        tags.append("combination")
    tags.extend(sorted(set(f"annotation:{annotation}" for annotation in annotations)))
    return tags


def _has_conflicting_edits(edit_tokens: Sequence[str]) -> bool:
    positions: set[tuple[str, int]] = set()
    for token in edit_tokens:
        edit = _edit_by_token().get(token)
        if edit is None:
            continue
        key = (edit.chain, edit.position)
        if key in positions:
            return True
        positions.add(key)
    return False


def _edit_tokens(row: Mapping[str, Any]) -> tuple[str, ...]:
    value = row.get("edit_tokens", row.get("modules", ()))
    if isinstance(value, str):
        return tuple(part.strip() for part in value.replace(",", ";").split(";") if part.strip())
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return ()


def _cost(row: Mapping[str, Any]) -> float:
    return max(_to_float(row.get("cost"), 1.0), 0.01)


def _candidate_cost(edit_tokens: Sequence[str]) -> float:
    return 1.0 + 0.20 * max(len(edit_tokens) - 1, 0) + 0.08 * len(set(edit_tokens) & _risky_edit_set())


def _edit_jaccard(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    left_edits = set(_edit_tokens(left))
    right_edits = set(_edit_tokens(right))
    union = left_edits | right_edits
    if not union:
        return 1.0
    return len(left_edits & right_edits) / len(union)


def _sequence_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row.get("heavy_sequence", "")), str(row.get("light_sequence", ""))


def _sequence_hash(heavy: str, light: str) -> str:
    return hashlib.sha256(f"{heavy}|{light}".encode("utf-8")).hexdigest()[:16]


def _variant_id(world_id: str, edit_tokens: Sequence[str]) -> str:
    if not edit_tokens:
        suffix = "WT"
    else:
        suffix = "_".join(_slug(token) for token in edit_tokens)
    return f"{world_id}__{suffix}"


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", value)


def _safe_run_id(run_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_id)).strip("._")
    return safe or DEFAULT_RUN_ID


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _noise(seed: int, world_id: str, edit_tokens: Sequence[str], channel: str) -> float:
    rng = random.Random(_stable_seed(seed, world_id, ",".join(sorted(edit_tokens)), channel))
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


def _copy_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _copy_jsonable(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_copy_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _mean_metric(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return _round_metric(mean(_to_float(row.get(field), 0.0) for row in rows) if rows else 0.0)


def _rounded_mapping(values: Mapping[str, float]) -> dict[str, float]:
    return {str(key): _round_metric(float(value)) for key, value in values.items()}


def _round_metric(value: float) -> float:
    return round(float(value), 6)


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(max(float(value), lower), upper)


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


def _to_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "feasible", "pass", "passed"}
    return bool(value)


def _bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"
