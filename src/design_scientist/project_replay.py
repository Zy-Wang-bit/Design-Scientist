"""Project-data retrospective masking benchmark.

This module evaluates active-design mechanisms on real standardized project
observations by hiding observed variants and scoring only the hidden outcomes.
It is deliberately separate from the deterministic stress worlds used by the
framework benchmark.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from design_scientist.io import ensure_dir, read_yaml, write_json, write_yaml


PROJECT_MASKING_RESULT = "project_masking_benchmark_results.csv"
PROJECT_MASKING_SUMMARY = "project_masking_benchmark_summary.csv"
PROJECT_MASKING_ABLATION = "project_masking_ablation_results.csv"
PROJECT_MASKING_CONFIG = "project_masking_config.json"
CMDGD_PROJECT_CANDIDATES = "cmdgd_project_generated_candidates.csv"
CMDGD_PROJECT_SUMMARY = "cmdgd_project_design_summary.json"
PH_SWITCH_GRAPH_PROJECT_CANDIDATES = "ph_switch_graph_project_generated_candidates.csv"
PH_SWITCH_GRAPH_PROJECT_SUMMARY = "ph_switch_graph_project_design_summary.json"
DEFAULT_MECHANISMS = (
    "ph_switch_graph",
    "mccbd",
    "evidence_calibrated_ucb",
    "random_feasible",
    "fixed_mix",
    "greedy_observed",
)
UTILITY_FORMULA = (
    "utility = log1p(mean pH7.4 ELISA signal) - mean(pH6/pH7.4) "
    "+ expression guardrail bonus; pH7.4 retention and pH6/pH7.4 are paired by variant/genotype."
)


@dataclass(frozen=True)
class ProjectVariantRecord:
    variant_id: str
    modules: tuple[str, ...]
    heavy_chain_seq: str
    light_chain_seq: str
    endpoint_values: dict[str, float]
    observed_utility: float
    true_feasible: bool
    expression_concentration: float | None
    dilution_record_count: int
    source_rows: tuple[str, ...]


@dataclass(frozen=True)
class ProjectReplayDataset:
    project_dir: Path
    variants: tuple[ProjectVariantRecord, ...]
    variant_by_id: dict[str, ProjectVariantRecord]
    maskable_variant_ids: tuple[str, ...]
    supplementary_tracks: dict[str, dict[str, float]]
    context: dict[str, Any]


def dataset_is_available(project_dir: str | Path) -> bool:
    """Return True when standardized project artifacts are sufficient."""

    root = Path(project_dir).expanduser().resolve()
    return (
        (root / "standardized" / "observations_long.csv").is_file()
        and (root / "standardized" / "variant_sequences.csv").is_file()
    )


def load_project_replay_dataset(project_dir: str | Path) -> ProjectReplayDataset:
    """Load a real observed-pool replay dataset from standardized artifacts."""

    root = Path(project_dir).expanduser().resolve()
    observations_path = root / "standardized" / "observations_long.csv"
    sequences_path = root / "standardized" / "variant_sequences.csv"
    if not observations_path.is_file():
        raise FileNotFoundError(f"Missing standardized/observations_long.csv under {root}")
    if not sequences_path.is_file():
        raise FileNotFoundError(f"Missing standardized/variant_sequences.csv under {root}")

    observations = _read_csv(observations_path)
    sequences = _read_csv(sequences_path)
    sequence_by_variant = {
        str(row.get("variant_id") or "").strip(): row
        for row in sequences
        if str(row.get("variant_id") or "").strip()
    }
    _merge_optional_mutation_annotations(root, sequence_by_variant)
    grouped: dict[str, list[dict[str, str]]] = {}
    supplementary_kd: dict[str, float] = {}
    for row in observations:
        variant_id = str(row.get("variant_id") or "").strip()
        if not variant_id:
            continue
        if str(row.get("endpoint") or "") == "KD_ratio":
            value = _to_float(row.get("value"))
            if value is not None:
                supplementary_kd[variant_id] = value
            continue
        if variant_id in sequence_by_variant:
            grouped.setdefault(variant_id, []).append(row)

    variants: list[ProjectVariantRecord] = []
    for variant_id in sorted(grouped, key=_natural_sort_key):
        record = _build_variant_record(variant_id, grouped[variant_id], sequence_by_variant[variant_id])
        if record is not None:
            variants.append(record)

    variant_by_id = {record.variant_id: record for record in variants}
    context = {
        "data_contract": _safe_read_yaml(root / "data_contract.yaml"),
        "estimands": _safe_read_yaml(root / "estimands.yaml"),
        "utility_formula": UTILITY_FORMULA,
        "evidence_boundary": "observed_pool_masking_real_project_data",
    }
    return ProjectReplayDataset(
        project_dir=root,
        variants=tuple(variants),
        variant_by_id=variant_by_id,
        maskable_variant_ids=tuple(record.variant_id for record in variants),
        supplementary_tracks={"KD_ratio": supplementary_kd},
        context=context,
    )


def run_project_masking_benchmark(
    project_dir: str | Path,
    run_id: str = "project_masking",
    mechanisms: Sequence[str | Callable[..., list[str]]] | None = None,
    budget: int = 4,
    folds: str = "kfold_5",
) -> dict[str, Any]:
    """Run leave-variant masking over real project observations."""

    root = Path(project_dir).expanduser().resolve()
    try:
        dataset = load_project_replay_dataset(root)
    except FileNotFoundError as exc:
        return {"status": "unavailable", "error": str(exc)}
    if not dataset.maskable_variant_ids:
        return {"status": "unavailable", "error": "No maskable project variants were found."}
    if budget <= 0:
        raise ValueError("budget must be positive")

    requested = list(mechanisms or DEFAULT_MECHANISMS)
    mechanism_fns = [(_mechanism_name(item), _resolve_mechanism(item)) for item in requested]
    run_dir = ensure_dir(root / "runs" / _safe_run_id(run_id))
    result_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    fold_specs = _folds(dataset, folds=folds)
    all_variant_ids = set(dataset.variant_by_id)

    for fold_index, (fold_id, heldout_ids) in enumerate(fold_specs):
        heldout_set = set(heldout_ids)
        train_records = [
            _policy_record(record, reveal=True)
            for record in dataset.variants
            if record.variant_id not in heldout_set
        ]
        candidate_records = [
            _policy_record(dataset.variant_by_id[variant_id], reveal=False)
            for variant_id in heldout_ids
        ]
        candidate_ids = [_record_id(record) for record in candidate_records]
        heldout_truth = [dataset.variant_by_id[variant_id] for variant_id in heldout_ids]
        global_best = max((record.observed_utility for record in heldout_truth), default=0.0)
        feasible_best = max(
            (record.observed_utility for record in heldout_truth if record.true_feasible),
            default=0.0,
        )

        for name, fn in mechanism_fns:
            selected_ids, error = _run_selector(
                fn,
                train_records,
                candidate_records,
                budget=budget,
                fold_index=fold_index,
            )
            selected_set = set(selected_ids)
            scored_heldout = sorted(selected_set & heldout_set)
            selected_truth = [dataset.variant_by_id[variant_id] for variant_id in scored_heldout]
            selected_feasible = [record for record in selected_truth if record.true_feasible]
            selected_infeasible = [record for record in selected_truth if not record.true_feasible]
            best_selected = max((record.observed_utility for record in selected_truth), default=0.0)
            best_feasible_selected = max(
                (record.observed_utility for record in selected_feasible),
                default=0.0,
            )
            hit_rate = (
                sum(1 for record in selected_truth if record.true_feasible) / len(scored_heldout)
                if scored_heldout
                else 0.0
            )
            evidence_coverage = len(scored_heldout) / max(len(heldout_set), 1)
            false_claim_rate = (
                sum(1 for record in selected_truth if not record.true_feasible) / len(scored_heldout)
                if scored_heldout
                else 0.0
            )
            diagnostics = {
                "selected_ids_outside_candidates": sorted(selected_set - set(candidate_ids)),
                "heldout_ids_in_train": sorted(heldout_set & {_record_id(record) for record in train_records}),
                "kd_ratio_joined_to_combo_records": False,
            }
            status = "completed" if error is None and not diagnostics["selected_ids_outside_candidates"] else "failed"
            row = {
                "fold_id": fold_id,
                "mechanism": name,
                "status": status,
                "error": error or "",
                "train_count": len(train_records),
                "candidate_count": len(candidate_records),
                "selector_round_index": fold_index,
                "selected_count": len(selected_ids),
                "heldout_ids": json.dumps(sorted(heldout_ids)),
                "candidate_ids": json.dumps(candidate_ids),
                "selected_ids": json.dumps(selected_ids),
                "scored_heldout_ids": json.dumps(scored_heldout),
                "selected_feasible_ids": json.dumps([record.variant_id for record in selected_feasible]),
                "selected_infeasible_ids": json.dumps([record.variant_id for record in selected_infeasible]),
                "global_best_heldout_utility": _round(global_best),
                "best_feasible_heldout_utility": _round(feasible_best),
                "best_selected_utility": _round(best_selected),
                "best_feasible_utility": _round(best_feasible_selected),
                "hit_rate": _round(hit_rate),
                "regret_proxy": _round(max(feasible_best - best_feasible_selected, 0.0)),
                "false_claim_rate": _round(false_claim_rate),
                "evidence_coverage": _round(evidence_coverage),
                "selected_ids_digest": _digest(selected_ids),
                "leakage_diagnostics": json.dumps(diagnostics, sort_keys=True),
            }
            result_rows.append(row)
            ablation_rows.append(
                {
                    "fold_id": fold_id,
                    "mechanism": name,
                    "ablation": "none",
                    "best_feasible_utility": row["best_feasible_utility"],
                    "delta_from_full_best_feasible_utility": 0.0,
                    "false_claim_rate": row["false_claim_rate"],
                }
            )

    summary_rows = _summary_rows(result_rows)
    result_path = run_dir / PROJECT_MASKING_RESULT
    summary_path = run_dir / PROJECT_MASKING_SUMMARY
    ablation_path = run_dir / PROJECT_MASKING_ABLATION
    config_path = run_dir / PROJECT_MASKING_CONFIG
    _write_csv(result_path, result_rows)
    _write_csv(summary_path, summary_rows)
    _write_csv(ablation_path, ablation_rows)
    write_json(config_path, _config_payload(dataset, folds=folds, mechanisms=[name for name, _ in mechanism_fns], budget=budget))

    return {
        "status": "completed",
        "run_id": _safe_run_id(run_id),
        "benchmark_results_path": str(result_path),
        "summary_results_path": str(summary_path),
        "ablation_results_path": str(ablation_path),
        "project_masking_benchmark_results_path": str(result_path),
        "project_masking_benchmark_summary_path": str(summary_path),
        "project_masking_ablation_results_path": str(ablation_path),
        "config_path": str(config_path),
        "summary": {"mechanism_rankings": summary_rows},
        "dataset": {
            "variant_count": len(dataset.variants),
            "maskable_variant_count": len(dataset.maskable_variant_ids),
            "supplementary_tracks": {key: len(value) for key, value in dataset.supplementary_tracks.items()},
        },
    }


def run_project_cmdgd_design(
    project_dir: str | Path,
    run_id: str = "cmdgd_project_design",
    *,
    budget: int = 5,
    generation_budget: int = 160,
    candidate_edit_vocabulary: Sequence[Mapping[str, Any]] | None = None,
    cmdgd_config: Mapping[str, Any] | None = None,
    seed: int = 1729,
) -> dict[str, Any]:
    """Run prospective CMD-GD candidate generation from visible project data."""

    root = Path(project_dir).expanduser().resolve()
    try:
        dataset = load_project_replay_dataset(root)
    except FileNotFoundError as exc:
        return {"status": "unavailable", "error": str(exc)}
    if budget <= 0:
        raise ValueError("budget must be positive")
    if generation_budget <= 0:
        raise ValueError("generation_budget must be positive")

    from design_scientist.algorithms import cmdgd

    observed_records = [
        _cmdgd_training_record(_policy_record(record, reveal=True))
        for record in dataset.variants
        if record.heavy_chain_seq and record.light_chain_seq
    ]
    if not observed_records:
        return {"status": "unavailable", "error": "No sequence-linked project variants were found."}

    base_heavy, base_light, base_source = _cmdgd_project_base_sequences(root, observed_records)
    endpoint_records = [_cmdgd_endpoint_record(record) for record in observed_records]
    vocabulary = (
        [dict(item) for item in candidate_edit_vocabulary]
        if candidate_edit_vocabulary is not None
        else _cmdgd_allowed_candidate_edit_vocabulary(root)
    )
    config = {"max_generated_candidates": int(generation_budget)}
    if cmdgd_config:
        config.update(dict(cmdgd_config))

    state = cmdgd.fit_state(
        base_heavy,
        base_light,
        observed_records,
        endpoint_records,
        candidate_edit_vocabulary=vocabulary,
        config=config,
    )
    rng = random.Random(seed)
    generated = cmdgd.generate_candidates(state, max_candidates=int(generation_budget), rng=rng)
    scored = cmdgd.score_candidates(state, generated)
    selected_ids = [
        str(candidate_id)
        for candidate_id in cmdgd.select_panel(state, scored, budget, random.Random(seed))
    ]

    observed_signatures = _cmdgd_observed_mutation_signatures(
        observed_records,
        base_heavy=base_heavy,
        base_light=base_light,
    )
    observed_sequences = {
        (
            str(record.get("heavy_chain_seq") or ""),
            str(record.get("light_chain_seq") or ""),
        )
        for record in observed_records
    }
    candidate_rows = _cmdgd_project_candidate_rows(
        scored,
        selected_ids=selected_ids,
        observed_signatures=observed_signatures,
        observed_sequences=observed_sequences,
    )

    run_dir = ensure_dir(root / "runs" / _safe_run_id(run_id))
    candidates_path = run_dir / CMDGD_PROJECT_CANDIDATES
    summary_path = run_dir / CMDGD_PROJECT_SUMMARY
    _write_csv(candidates_path, candidate_rows)
    summary = _cmdgd_project_design_summary(
        dataset,
        run_id=_safe_run_id(run_id),
        base_heavy=base_heavy,
        base_light=base_light,
        base_source=base_source,
        vocabulary=vocabulary,
        generated=generated,
        scored=scored,
        candidate_rows=candidate_rows,
        selected_ids=selected_ids,
        state=state,
        endpoint_count=_cmdgd_observed_endpoint_count(root, {_record_id(record) for record in observed_records}),
    )
    write_json(summary_path, summary)

    return {
        "status": "completed",
        "run_id": _safe_run_id(run_id),
        "generated_candidates_path": str(candidates_path),
        "design_summary_path": str(summary_path),
        "summary": summary,
    }


def run_project_ph_switch_graph_design(
    project_dir: str | Path,
    run_id: str = "ph_switch_graph_project_design",
    *,
    budget: int = 5,
    generation_budget: int = 160,
    candidate_edit_vocabulary: Sequence[Mapping[str, Any]] | None = None,
    config: Mapping[str, Any] | None = None,
    seed: int = 1729,
) -> dict[str, Any]:
    """Run prospective pH-switch graph candidate generation from visible project data."""

    root = Path(project_dir).expanduser().resolve()
    try:
        dataset = load_project_replay_dataset(root)
    except FileNotFoundError as exc:
        return {"status": "unavailable", "error": str(exc)}
    if budget <= 0:
        raise ValueError("budget must be positive")
    if generation_budget <= 0:
        raise ValueError("generation_budget must be positive")

    from design_scientist.algorithms import pcig

    observed_records = [
        _cmdgd_training_record(_policy_record(record, reveal=True))
        for record in dataset.variants
        if record.heavy_chain_seq and record.light_chain_seq
    ]
    if not observed_records:
        return {"status": "unavailable", "error": "No sequence-linked project variants were found."}

    base_heavy, base_light, base_source = _cmdgd_project_base_sequences(root, observed_records)
    endpoint_records = [_cmdgd_endpoint_record(record) for record in observed_records]
    vocabulary = (
        [dict(item) for item in candidate_edit_vocabulary]
        if candidate_edit_vocabulary is not None
        else _cmdgd_allowed_candidate_edit_vocabulary(root)
    )
    pcig_config = {"max_generated_candidates": int(generation_budget)}
    if config:
        pcig_config.update(dict(config))
    observed_signatures = _cmdgd_observed_mutation_signatures(
        observed_records,
        base_heavy=base_heavy,
        base_light=base_light,
    )
    observed_sites = [
        {"chain": key[0], "position": int(key[1:])}
        for key in sorted(observed_signatures["positions"])
        if len(key) > 1 and key[0] in {"H", "L"} and key[1:].isdigit()
    ]

    state = pcig.fit_state(
        base_heavy,
        base_light,
        observed_records,
        endpoint_records,
        candidate_edit_vocabulary=vocabulary,
        observed_mutation_sites=observed_sites,
        visible_mutation_sites=observed_sites,
        config=pcig_config,
    )
    rng = random.Random(seed)
    generated = pcig.generate_candidates(state, max_candidates=int(generation_budget), rng=rng)
    scored = pcig.score_candidates(state, generated)
    selected_ids = [
        str(candidate_id)
        for candidate_id in pcig.select_panel(state, scored, budget, random.Random(seed))
    ]

    observed_sequences = {
        (
            str(record.get("heavy_chain_seq") or ""),
            str(record.get("light_chain_seq") or ""),
        )
        for record in observed_records
    }
    candidate_rows = _ph_switch_graph_project_candidate_rows(
        scored,
        selected_ids=selected_ids,
        observed_signatures=observed_signatures,
        observed_sequences=observed_sequences,
    )

    run_dir = ensure_dir(root / "runs" / _safe_run_id(run_id))
    candidates_path = run_dir / PH_SWITCH_GRAPH_PROJECT_CANDIDATES
    summary_path = run_dir / PH_SWITCH_GRAPH_PROJECT_SUMMARY
    _write_csv(candidates_path, candidate_rows)
    summary = _ph_switch_graph_project_design_summary(
        dataset,
        run_id=_safe_run_id(run_id),
        base_heavy=base_heavy,
        base_light=base_light,
        base_source=base_source,
        vocabulary=vocabulary,
        generated=generated,
        scored=scored,
        candidate_rows=candidate_rows,
        selected_ids=selected_ids,
        state=state,
        endpoint_count=_cmdgd_observed_endpoint_count(root, {_record_id(record) for record in observed_records}),
    )
    write_json(summary_path, summary)
    _advance_project_design_stage(
        root,
        run_id=_safe_run_id(run_id),
        algorithm="ph_switch_graph",
        selected_ids=selected_ids,
        budget=budget,
        generation_budget=generation_budget,
    )
    generic_artifacts = _write_project_design_run_artifacts(
        root,
        run_dir,
        run_id=_safe_run_id(run_id),
        algorithm="ph_switch_graph",
        candidate_rows=candidate_rows,
        selected_ids=selected_ids,
        summary=summary,
        budget=budget,
        generation_budget=generation_budget,
    )
    summary["generic_project_artifacts"] = generic_artifacts
    write_json(summary_path, summary)

    return {
        "status": "completed",
        "run_id": _safe_run_id(run_id),
        "generated_candidates_path": str(candidates_path),
        "design_summary_path": str(summary_path),
        "generic_project_artifacts": generic_artifacts,
        "summary": summary,
    }


def _build_variant_record(
    variant_id: str,
    observations: list[dict[str, str]],
    sequence: dict[str, str],
) -> ProjectVariantRecord | None:
    ph_values: dict[tuple[str, str], list[float]] = {}
    ratio_values: dict[str, list[float]] = {}
    expression: float | None = None
    dilution_count = 0
    source_rows: set[str] = set()
    for row in observations:
        value = _to_float(row.get("value"))
        if value is None or str(row.get("value_status") or "observed") != "observed":
            continue
        endpoint = str(row.get("endpoint") or "")
        genotype = str(row.get("antigen_genotype") or "")
        source_rows.add(f"{row.get('source_file')}:{row.get('source_row')}")
        if endpoint == "summary_elisa_signal":
            ph = str(row.get("pH") or "")
            if ph:
                ph_values.setdefault((genotype, ph), []).append(value)
        elif endpoint == "elisa_signal":
            dilution_count += 1
        elif endpoint == "summary_pH74_over_pH60_ratio":
            ratio_values.setdefault(genotype, []).append(value)
        elif endpoint == "expression_concentration":
            expression = value

    genotypes = sorted({genotype for genotype, _ in ph_values if genotype})
    if not genotypes:
        return None
    neutral_values: list[float] = []
    acid_ratios: list[float] = []
    endpoint_values: dict[str, float] = {}
    for genotype in genotypes:
        ph74 = _mean_or_none(ph_values.get((genotype, "7.4"), []))
        ph6 = _mean_or_none(ph_values.get((genotype, "6.0"), []))
        if ph74 is None or ph6 is None or ph74 <= 0:
            continue
        ratio = ph6 / ph74
        neutral_values.append(ph74)
        acid_ratios.append(ratio)
        endpoint_values[f"{genotype}_pH7.4"] = _round(ph74)
        endpoint_values[f"{genotype}_pH6_over_pH7.4"] = _round(ratio)
        if genotype in ratio_values and ratio_values[genotype]:
            endpoint_values[f"{genotype}_reported_ratio"] = _round(mean(ratio_values[genotype]))
    if not neutral_values or not acid_ratios:
        return None
    expression_bonus = 0.05 if expression is not None and expression >= 0.5 else -0.08
    utility = _log1p(mean(neutral_values)) - mean(acid_ratios) + expression_bonus
    feasible = min(neutral_values) >= 0.05 and mean(acid_ratios) <= 1.05 and expression_bonus >= 0.0
    endpoint_values["mean_pH7.4"] = _round(mean(neutral_values))
    endpoint_values["mean_pH6_over_pH7.4"] = _round(mean(acid_ratios))
    if expression is not None:
        endpoint_values["expression_concentration"] = _round(expression)
    return ProjectVariantRecord(
        variant_id=variant_id,
        modules=tuple(_mutation_tokens(sequence)),
        heavy_chain_seq=str(
            sequence.get("heavy_chain_seq")
            or sequence.get("heavy_sequence")
            or sequence.get("vh_sequence")
            or ""
        ),
        light_chain_seq=str(
            sequence.get("light_chain_seq")
            or sequence.get("light_sequence")
            or sequence.get("vl_sequence")
            or ""
        ),
        endpoint_values=endpoint_values,
        observed_utility=_round(utility),
        true_feasible=bool(feasible),
        expression_concentration=expression,
        dilution_record_count=dilution_count,
        source_rows=tuple(sorted(source_rows)),
    )


def _policy_record(record: ProjectVariantRecord, *, reveal: bool) -> dict[str, Any]:
    out: dict[str, Any] = {
        "variant_id": record.variant_id,
        "candidate_id": record.variant_id,
        "background": "project_observed_pool",
        "modules": list(record.modules),
        "heavy_chain_seq": record.heavy_chain_seq,
        "light_chain_seq": record.light_chain_seq,
        "endpoint_values": dict(record.endpoint_values) if reveal else {},
        "required_measurements": ["pH7.4_retention", "pH6_release", "expression"],
        "missing_endpoints": [] if reveal else ["masked_project_outcome"],
        "cost": 1.0,
        "feasibility_status": "feasible",
        "source_refs": [
            {"kind": "standardized_project_row", "id": source_row}
            for source_row in record.source_rows
        ],
    }
    if reveal:
        out["observed_utility"] = record.observed_utility
        out["true_feasible"] = record.true_feasible
    return out


def _resolve_mechanism(name_or_fn: str | Callable[..., list[str]]) -> Callable[..., list[str]]:
    if callable(name_or_fn):
        return name_or_fn
    name = str(name_or_fn)
    if name == "mccbd":
        from design_scientist.algorithms.mccbd import select_batch

        return select_batch
    if name == "evidence_calibrated_ucb":
        from design_scientist.algorithms.evidence_calibrated_ucb import select_batch

        return select_batch
    if name in {"ph_switch_graph", "ph_switch_graph_observed_pool"}:
        return _ph_switch_graph_observed_pool
    if name in {"cmdgd", "cmd_gd", "cmdgd_generative", "cmdgd_observed_pool"}:
        return _cmdgd_observed_pool
    if name == "random_feasible":
        return _random_feasible
    if name == "fixed_mix":
        return _fixed_mix
    if name == "greedy_observed":
        return _greedy_observed
    raise ValueError(f"Unknown project masking mechanism: {name}")


def _mechanism_name(name_or_fn: str | Callable[..., list[str]]) -> str:
    if isinstance(name_or_fn, str):
        return name_or_fn
    return getattr(name_or_fn, "__name__", name_or_fn.__class__.__name__)


def _random_feasible(
    observed: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    budget: int,
    round_index: int,
    rng: random.Random,
) -> list[str]:
    del observed, round_index
    pool = [dict(item) for item in candidates]
    rng.shuffle(pool)
    return [_record_id(item) for item in pool[:budget]]


def _greedy_observed(
    observed: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    budget: int,
    round_index: int,
    rng: random.Random,
) -> list[str]:
    del round_index, rng
    observed_modules: dict[str, list[float]] = {}
    for record in observed:
        utility = _to_float(record.get("observed_utility"))
        if utility is None:
            continue
        for module in _as_list(record.get("modules")):
            observed_modules.setdefault(str(module), []).append(utility)
    def score(candidate: Mapping[str, Any]) -> tuple[float, str]:
        values = [
            mean(observed_modules[module])
            for module in _as_list(candidate.get("modules"))
            if module in observed_modules
        ]
        return (mean(values) if values else 0.0, _record_id(candidate))
    return [_record_id(item) for item in sorted(candidates, key=score, reverse=True)[:budget]]


def _fixed_mix(
    observed: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    budget: int,
    round_index: int,
    rng: random.Random,
) -> list[str]:
    greedy = _greedy_observed(observed, candidates, max(1, budget // 2), round_index, rng)
    remaining = [item for item in candidates if _record_id(item) not in set(greedy)]
    random_part = _random_feasible(observed, remaining, budget - len(greedy), round_index, rng)
    return [*greedy, *random_part]


def _ph_switch_graph_observed_pool(
    observed: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    budget: int,
    round_index: int,
    rng: random.Random,
) -> list[str]:
    """Rank masked observed variants with the V3 pH-switch graph without endpoint leakage."""

    del round_index
    if budget <= 0 or not candidates:
        return []
    from design_scientist.algorithms import pcig

    train_records = [
        _cmdgd_training_record(record)
        for record in observed
        if _has_cmdgd_sequences(record)
    ]
    if not train_records:
        return []
    base = _cmdgd_base_record(train_records)
    base_heavy = str(base["heavy_chain_seq"])
    base_light = str(base["light_chain_seq"])
    endpoint_records = [_cmdgd_endpoint_record(record) for record in train_records]
    observed_signatures = _cmdgd_observed_mutation_signatures(
        train_records,
        base_heavy=base_heavy,
        base_light=base_light,
    )
    observed_sites = [
        {"chain": key[0], "position": int(key[1:])}
        for key in sorted(observed_signatures["positions"])
        if len(key) > 1 and key[0] in {"H", "L"} and key[1:].isdigit()
    ]
    state = pcig.fit_state(
        base_heavy,
        base_light,
        train_records,
        endpoint_records,
        observed_mutation_sites=observed_sites,
        visible_mutation_sites=observed_sites,
    )
    hidden_candidates = [
        _ph_switch_graph_hidden_candidate(record, base_heavy=base_heavy, base_light=base_light)
        for record in candidates
        if _has_cmdgd_sequences(record)
    ]
    if not hidden_candidates:
        return []
    scored = pcig.score_candidates(state, hidden_candidates)
    return pcig.select_panel(state, scored, budget, rng)


def _cmdgd_observed_pool(
    observed: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    budget: int,
    round_index: int,
    rng: random.Random,
) -> list[str]:
    """Use CMD-GD state to rank masked observed variants without endpoint truth."""

    del round_index
    if budget <= 0 or not candidates:
        return []
    from design_scientist.algorithms import cmdgd

    train_records = [
        _cmdgd_training_record(record)
        for record in observed
        if _has_cmdgd_sequences(record)
    ]
    if not train_records:
        return []
    base = _cmdgd_base_record(train_records)
    base_heavy = str(base["heavy_chain_seq"])
    base_light = str(base["light_chain_seq"])
    endpoint_records = [_cmdgd_endpoint_record(record) for record in train_records]
    state = cmdgd.fit_state(
        base_heavy,
        base_light,
        train_records,
        endpoint_records,
    )
    hidden_candidates = [
        _cmdgd_hidden_candidate(record, base_heavy=base_heavy, base_light=base_light)
        for record in candidates
        if _has_cmdgd_sequences(record)
    ]
    if not hidden_candidates:
        return []
    scored = cmdgd.score_candidates(state, hidden_candidates)
    return cmdgd.select_panel(state, scored, budget, rng)


def _ph_switch_graph_hidden_candidate(
    record: Mapping[str, Any],
    *,
    base_heavy: str,
    base_light: str,
) -> dict[str, Any]:
    candidate = _cmdgd_hidden_candidate(record, base_heavy=base_heavy, base_light=base_light)
    candidate["design_context"] = {
        "algorithm": "ph_switch_graph",
        "benchmark_boundary": "retrospective_observed_pool_masking",
        "heldout_endpoint_values_visible": False,
        "legacy_rejected_method_reused": False,
    }
    return candidate




def _cmdgd_training_record(record: Mapping[str, Any]) -> dict[str, Any]:
    record_id = _record_id(record)
    endpoint_values = _cmdgd_endpoint_values(record)
    return {
        "variant_id": record_id,
        "candidate_id": record_id,
        "heavy_chain_seq": str(record.get("heavy_chain_seq") or ""),
        "light_chain_seq": str(record.get("light_chain_seq") or ""),
        "modules": _as_list(record.get("modules")),
        "endpoint_values": endpoint_values,
        "observed_utility": _to_float(record.get("observed_utility")) or 0.0,
        "feasibility_status": (
            "feasible" if bool(record.get("true_feasible")) else "infeasible"
        ),
        "source_refs": list(record.get("source_refs") or []),
    }


def _cmdgd_endpoint_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "variant_id": _record_id(record),
        **_cmdgd_endpoint_values(record),
        "observed_utility": _to_float(record.get("observed_utility")) or 0.0,
    }


def _cmdgd_endpoint_values(record: Mapping[str, Any]) -> dict[str, float]:
    raw = record.get("endpoint_values")
    endpoint_values = {
        str(key): numeric
        for key, value in (raw.items() if isinstance(raw, Mapping) else ())
        if (numeric := _to_float(value)) is not None
    }
    expression = _to_float(endpoint_values.get("expression_concentration"))
    if expression is not None:
        endpoint_values.setdefault("expression", expression)
    utility = _to_float(record.get("observed_utility"))
    if utility is not None:
        endpoint_values.setdefault("project_utility", utility)
    return endpoint_values


def _cmdgd_base_record(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return min(
        records,
        key=lambda record: (
            len(_as_list(record.get("modules"))),
            _record_id(record),
        ),
    )


def _cmdgd_hidden_candidate(
    record: Mapping[str, Any],
    *,
    base_heavy: str,
    base_light: str,
) -> dict[str, Any]:
    candidate_id = _record_id(record)
    heavy = str(record.get("heavy_chain_seq") or "")
    light = str(record.get("light_chain_seq") or "")
    edits = _cmdgd_sequence_edits(base_heavy, base_light, heavy, light)
    return {
        "variant_id": candidate_id,
        "candidate_id": candidate_id,
        "heavy_chain_seq": heavy,
        "light_chain_seq": light,
        "modules": [edit["id"] for edit in edits],
        "observed_modules": _as_list(record.get("modules")),
        "edits": edits,
        "parent_ids": [],
        "operator": "retrospective_observed_pool_candidate",
        "source_refs": [
            {
                "kind": "project_masking_hidden_descriptor",
                "variant_id": candidate_id,
            }
        ],
        "design_context": {
            "algorithm": "CMD-GD",
            "benchmark_boundary": "retrospective_observed_pool_masking",
            "heldout_endpoint_values_visible": False,
        },
        "required_measurements": _as_list(record.get("required_measurements")),
        "missing_endpoints": _as_list(record.get("missing_endpoints")),
        "cost": 1.0,
        "feasibility_status": "review",
    }


def _cmdgd_project_base_sequences(
    root: Path,
    observed_records: Sequence[Mapping[str, Any]],
) -> tuple[str, str, str]:
    allowed_files = _cmdgd_allowed_source_files(root)
    heavy = _cmdgd_read_allowed_fasta(root, allowed_files, "raw/base_sequences/heavy.fasta")
    light = _cmdgd_read_allowed_fasta(root, allowed_files, "raw/base_sequences/light.fasta")
    if heavy and light:
        return heavy, light, "data_contract_base_sequences"

    context = _read_json_if_exists(root / "standardized" / "project_context.json")
    if isinstance(context, Mapping):
        context_heavy = str(
            context.get("base_heavy_chain_seq")
            or context.get("base_heavy_sequence")
            or ""
        ).strip()
        context_light = str(
            context.get("base_light_chain_seq")
            or context.get("base_light_sequence")
            or ""
        ).strip()
        if context_heavy and context_light:
            return context_heavy, context_light, "standardized_project_context"

    base = _cmdgd_base_record(observed_records)
    return (
        str(base["heavy_chain_seq"]),
        str(base["light_chain_seq"]),
        "minimal_observed_edit_record",
    )


def _cmdgd_allowed_source_files(root: Path) -> set[str]:
    contract = _safe_read_yaml(root / "data_contract.yaml")
    allowed = contract.get("allowed_sources")
    files: set[str] = set()
    if not isinstance(allowed, Mapping):
        return files
    for spec in allowed.values():
        if not isinstance(spec, Mapping):
            continue
        raw_files = spec.get("files")
        if not isinstance(raw_files, Sequence) or isinstance(raw_files, str):
            continue
        for item in raw_files:
            clean = str(item).strip().lstrip("./")
            if clean:
                files.add(clean)
    return files


def _cmdgd_read_allowed_fasta(root: Path, allowed_files: set[str], relative_path: str) -> str:
    if relative_path not in allowed_files:
        return ""
    path = root / relative_path
    if not path.is_file():
        return ""
    sequence_parts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if not clean or clean.startswith(">"):
            continue
        sequence_parts.append(clean)
    return "".join(sequence_parts).strip()


def _cmdgd_allowed_candidate_edit_vocabulary(root: Path) -> list[dict[str, Any]]:
    payload_paths = (
        root / "standardized" / "project_context.json",
        root / "framework" / "mechanism_library.json",
        root / "framework" / "mechanism_cards.json",
    )
    vocabulary: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in payload_paths:
        payload = _read_json_if_exists(path)
        for raw in _cmdgd_extract_candidate_edit_vocabulary(payload):
            if not _cmdgd_is_allowed_mechanism_prior(raw):
                continue
            item = dict(raw)
            key = json.dumps(_json_safe(item), sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            vocabulary.append(item)
    return vocabulary


def _cmdgd_extract_candidate_edit_vocabulary(payload: Any) -> list[Mapping[str, Any]]:
    keys = {
        "candidate_edit_vocabulary",
        "cmdgd_candidate_edit_vocabulary",
        "edit_vocabulary",
        "mechanism_priors",
    }
    found: list[Mapping[str, Any]] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if str(key) in keys and isinstance(value, Sequence) and not isinstance(value, str):
                found.extend(item for item in value if isinstance(item, Mapping))
            if isinstance(value, Mapping | list | tuple):
                found.extend(_cmdgd_extract_candidate_edit_vocabulary(value))
    elif isinstance(payload, list | tuple):
        for item in payload:
            found.extend(_cmdgd_extract_candidate_edit_vocabulary(item))
    return found


def _cmdgd_is_allowed_mechanism_prior(raw: Mapping[str, Any]) -> bool:
    if not _cmdgd_prior_has_edit_fields(raw):
        return False
    if bool(raw.get("project_specific")):
        return False
    scope = str(raw.get("scope") or raw.get("specificity") or "").lower()
    if scope in {"project", "project_specific", "project-specific"}:
        return False
    provenance = " ".join(
        str(raw.get(key) or "").lower()
        for key in ("source", "source_system", "method", "generator", "provenance")
    )
    return "optim-pipe" not in provenance and "optim_pipe" not in provenance


def _cmdgd_prior_has_edit_fields(raw: Mapping[str, Any]) -> bool:
    if str(raw.get("token") or raw.get("edit_id") or raw.get("id") or "").strip():
        return True
    return bool(
        str(raw.get("chain") or "").strip()
        and _to_int(raw.get("position")) is not None
        and str(raw.get("residue") or raw.get("to") or raw.get("to_residue") or "").strip()
    )


def _cmdgd_observed_mutation_signatures(
    records: Sequence[Mapping[str, Any]],
    *,
    base_heavy: str,
    base_light: str,
) -> dict[str, set[str]]:
    edit_ids: set[str] = set()
    positions: set[str] = set()
    variant_ids: set[str] = set()
    for record in records:
        variant_ids.add(_record_id(record))
        edits = _cmdgd_sequence_edits(
            base_heavy,
            base_light,
            str(record.get("heavy_chain_seq") or ""),
            str(record.get("light_chain_seq") or ""),
        )
        for edit in edits:
            edit_id = str(edit.get("id") or "")
            position = _cmdgd_position_key(edit)
            if edit_id:
                edit_ids.add(edit_id)
            if position:
                positions.add(position)
    return {"edit_ids": edit_ids, "positions": positions, "variant_ids": variant_ids}


def _cmdgd_project_candidate_rows(
    candidates: Sequence[Mapping[str, Any]],
    *,
    selected_ids: Sequence[str],
    observed_signatures: Mapping[str, set[str]],
    observed_sequences: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    selected_set = {str(item) for item in selected_ids}
    rows: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates, start=1):
        candidate_id = str(candidate.get("candidate_id") or candidate.get("variant_id") or f"rank_{rank}")
        edits = _cmdgd_candidate_edit_dicts(candidate)
        edit_ids = [str(edit.get("id") or "") for edit in edits if str(edit.get("id") or "")]
        positions = [position for edit in edits if (position := _cmdgd_position_key(edit))]
        new_sites = sorted(set(edit_ids) - set(observed_signatures.get("edit_ids", set())))
        new_positions = sorted(set(positions) - set(observed_signatures.get("positions", set())))
        trace = candidate.get("cmdgd_trace") if isinstance(candidate.get("cmdgd_trace"), Mapping) else {}
        vocabulary_ids = _cmdgd_candidate_vocabulary_edit_ids(candidate, trace)
        observed_duplicate = (
            candidate_id in set(observed_signatures.get("variant_ids", set()))
            or (
                str(candidate.get("heavy_chain_seq") or ""),
                str(candidate.get("light_chain_seq") or ""),
            )
            in observed_sequences
        )
        score_components = candidate.get("score_components")
        row = {
            "rank": rank,
            "candidate_id": candidate_id,
            "selected": _bool_text(candidate_id in selected_set),
            "score": _cmdgd_metric(candidate, "score"),
            "predicted_contrastive_ph_objective": _cmdgd_metric(candidate, "predicted_contrastive_ph_objective"),
            "developability_feasibility": _cmdgd_metric(candidate, "developability_feasibility"),
            "novelty": _cmdgd_metric(candidate, "novelty"),
            "uncertainty": _cmdgd_metric(candidate, "uncertainty"),
            "residue_combination_prior": _cmdgd_metric(score_components, "residue_combination_prior"),
            "provenance_prior": _cmdgd_metric(score_components, "provenance_prior"),
            "selection_diversity_prior": _cmdgd_metric(score_components, "selection_diversity_prior"),
            "operator": str(candidate.get("operator") or ""),
            "parent_ids": json.dumps(_as_list(candidate.get("parent_ids"))),
            "modules": json.dumps(_as_list(candidate.get("modules")) or edit_ids),
            "heavy_chain_seq": str(candidate.get("heavy_chain_seq") or ""),
            "light_chain_seq": str(candidate.get("light_chain_seq") or ""),
            "observed_duplicate": _bool_text(observed_duplicate),
            "uses_new_mutation_site": _bool_text(bool(new_sites)),
            "new_mutation_site_count": len(new_sites),
            "new_mutation_sites": json.dumps(new_sites),
            "uses_new_position": _bool_text(bool(new_positions)),
            "new_position_count": len(new_positions),
            "new_positions": json.dumps(new_positions),
            "candidate_vocabulary_edit_count": len(vocabulary_ids),
            "candidate_vocabulary_edit_ids": json.dumps(vocabulary_ids),
            "feasibility_status": str(candidate.get("feasibility_status") or "review"),
            "cost": _cmdgd_metric(candidate, "cost"),
            "cmdgd_trace": json.dumps(_json_safe(trace), sort_keys=True),
        }
        rows.append(row)
    return rows


def _ph_switch_graph_project_candidate_rows(
    candidates: Sequence[Mapping[str, Any]],
    *,
    selected_ids: Sequence[str],
    observed_signatures: Mapping[str, set[str]],
    observed_sequences: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    selected_set = {str(item) for item in selected_ids}
    rows: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates, start=1):
        candidate_id = str(candidate.get("candidate_id") or candidate.get("variant_id") or f"rank_{rank}")
        edits = _cmdgd_candidate_edit_dicts(candidate)
        edit_ids = [str(edit.get("id") or "") for edit in edits if str(edit.get("id") or "")]
        positions = [position for edit in edits if (position := _cmdgd_position_key(edit))]
        new_sites = sorted(set(edit_ids) - set(observed_signatures.get("edit_ids", set())))
        new_positions = sorted(set(positions) - set(observed_signatures.get("positions", set())))
        observed_duplicate = (
            candidate_id in set(observed_signatures.get("variant_ids", set()))
            or (
                str(candidate.get("heavy_chain_seq") or ""),
                str(candidate.get("light_chain_seq") or ""),
            )
            in observed_sequences
        )
        score_components = (
            candidate.get("score_components")
            if isinstance(candidate.get("score_components"), Mapping)
            else {}
        )
        trace = candidate.get("pcig_trace") if isinstance(candidate.get("pcig_trace"), Mapping) else {}
        row = {
            "rank": rank,
            "candidate_id": candidate_id,
            "selected": _bool_text(candidate_id in selected_set),
            "score": _cmdgd_metric(candidate, "score"),
            "posterior_mean": _cmdgd_metric(candidate, "posterior_mean"),
            "posterior_std": _cmdgd_metric(candidate, "posterior_std"),
            "pcig_evidence_effect": _cmdgd_metric(score_components, "pcig_evidence_effect"),
            "counterfactual_site_field": _cmdgd_metric(score_components, "counterfactual_site_field"),
            "pair_program_bonus": _cmdgd_metric(score_components, "pair_program_bonus"),
            "feasibility_prior": _cmdgd_metric(score_components, "feasibility_prior"),
            "operator": str(candidate.get("operator") or ""),
            "parent_ids": json.dumps(_as_list(candidate.get("parent_ids"))),
            "modules": json.dumps(_as_list(candidate.get("modules")) or edit_ids),
            "heavy_chain_seq": str(candidate.get("heavy_chain_seq") or ""),
            "light_chain_seq": str(candidate.get("light_chain_seq") or ""),
            "observed_duplicate": _bool_text(observed_duplicate),
            "uses_new_mutation_site": _bool_text(bool(new_sites)),
            "new_mutation_site_count": len(new_sites),
            "new_mutation_sites": json.dumps(new_sites),
            "uses_new_position": _bool_text(bool(new_positions)),
            "new_position_count": len(new_positions),
            "new_positions": json.dumps(new_positions),
            "feasibility_status": str(candidate.get("feasibility_status") or "review"),
            "cost": _cmdgd_metric(candidate, "cost"),
            "ph_switch_graph_trace": json.dumps(_json_safe(trace), sort_keys=True),
        }
        rows.append(row)
    return rows


def _cmdgd_candidate_edit_dicts(candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_edits = candidate.get("edits")
    edits: list[dict[str, Any]] = []
    if isinstance(raw_edits, Sequence) and not isinstance(raw_edits, str):
        for raw in raw_edits:
            if isinstance(raw, Mapping):
                parsed = _cmdgd_normalized_edit(raw)
                if parsed:
                    edits.append(parsed)
    if edits:
        return edits
    for token in _as_list(candidate.get("modules")):
        parsed = _cmdgd_parse_edit_token(token)
        if parsed:
            edits.append(parsed)
    return edits


def _cmdgd_normalized_edit(raw: Mapping[str, Any]) -> dict[str, Any]:
    token = str(raw.get("id") or raw.get("edit_id") or raw.get("token") or "").strip()
    parsed = _cmdgd_parse_edit_token(token) if token else {}
    chain = str(raw.get("chain") or parsed.get("chain") or "").upper()
    position = _to_int(raw.get("position", parsed.get("position")))
    if not token and chain and position is not None:
        residue = str(raw.get("to") or raw.get("residue") or raw.get("to_residue") or "").strip().upper()
        token = f"{chain}{position}{residue}" if residue else f"{chain}{position}"
    if not token:
        return {}
    out = dict(raw)
    out["id"] = token
    if chain:
        out["chain"] = chain
    if position is not None:
        out["position"] = position
    return out


def _cmdgd_parse_edit_token(token: str) -> dict[str, Any]:
    token = str(token or "").strip()
    if len(token) < 3:
        return {}
    chain = token[0].upper()
    if chain not in {"H", "L"}:
        return {}
    digits = ""
    for char in token[1:]:
        if not char.isdigit():
            break
        digits += char
    if not digits:
        return {"id": token, "chain": chain}
    return {"id": token, "chain": chain, "position": int(digits)}


def _cmdgd_position_key(edit: Mapping[str, Any]) -> str:
    chain = str(edit.get("chain") or "").strip().upper()
    position = _to_int(edit.get("position"))
    if chain not in {"H", "L"} or position is None:
        parsed = _cmdgd_parse_edit_token(str(edit.get("id") or ""))
        chain = str(parsed.get("chain") or chain).upper()
        position = _to_int(parsed.get("position", position))
    if chain not in {"H", "L"} or position is None:
        return ""
    return f"{chain}{position}"


def _cmdgd_candidate_vocabulary_edit_ids(
    candidate: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> list[str]:
    expansion = trace.get("vocabulary_expansion") if isinstance(trace, Mapping) else {}
    if isinstance(expansion, Mapping):
        raw = expansion.get("candidate_vocabulary_edit_ids")
        if isinstance(raw, Sequence) and not isinstance(raw, str):
            return [str(item) for item in raw if str(item)]
    return [
        str(edit.get("id"))
        for edit in _cmdgd_candidate_edit_dicts(candidate)
        if str(edit.get("source") or "").lower() == "design_vocabulary"
    ]


def _cmdgd_metric(source: Any, key: str) -> Any:
    if not isinstance(source, Mapping):
        return ""
    value = _to_float(source.get(key))
    return _round(value) if value is not None else ""


def _cmdgd_project_design_summary(
    dataset: ProjectReplayDataset,
    *,
    run_id: str,
    base_heavy: str,
    base_light: str,
    base_source: str,
    vocabulary: Sequence[Mapping[str, Any]],
    generated: Sequence[Mapping[str, Any]],
    scored: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    selected_ids: Sequence[str],
    state: Any,
    endpoint_count: int,
) -> dict[str, Any]:
    selected_set = {str(item) for item in selected_ids}
    selected_rows = [row for row in candidate_rows if str(row.get("candidate_id")) in selected_set]
    component_trace = getattr(state, "component_trace", {})
    if isinstance(state, Mapping):
        component_trace = state.get("component_trace", component_trace)
    return {
        "schema_version": 1,
        "algorithm": "cmdgd",
        "run_id": run_id,
        "design_mode": "project_cmdgd_sequence_generation",
        "evidence_boundary": (
            "computational candidate generation from startup observations; "
            "generated candidates have no measured project truth"
        ),
        "retrospective_masking_semantics": {
            "generated_candidates_are_not_heldout_truth": True,
            "hidden_measured_variants_not_reused_as_prospective_truth": True,
        },
        "observed_variant_count": len(dataset.variants),
        "observed_endpoint_count": int(endpoint_count),
        "base_heavy_length": len(base_heavy),
        "base_light_length": len(base_light),
        "base_sequence_source": base_source,
        "candidate_edit_vocabulary_count": len(vocabulary),
        "generated_candidate_count": len(generated),
        "scored_candidate_count": len(scored),
        "selected_count": len(selected_ids),
        "selected_candidate_ids": list(selected_ids),
        "generated_new_mutation_site_count": _row_flag_count(candidate_rows, "uses_new_mutation_site"),
        "generated_new_position_count": _row_flag_count(candidate_rows, "uses_new_position"),
        "selected_new_mutation_site_count": _row_flag_count(selected_rows, "uses_new_mutation_site"),
        "selected_new_position_count": _row_flag_count(selected_rows, "uses_new_position"),
        "observed_duplicate_selected_count": _row_flag_count(selected_rows, "observed_duplicate"),
        "selected_parent_id_modes": sorted({str(row.get("parent_ids") or "[]") for row in selected_rows}),
        "unique_score_count": len({str(row.get("score")) for row in candidate_rows if row.get("score") != ""}),
        "component_trace": _json_safe(component_trace),
    }


def _ph_switch_graph_project_design_summary(
    dataset: ProjectReplayDataset,
    *,
    run_id: str,
    base_heavy: str,
    base_light: str,
    base_source: str,
    vocabulary: Sequence[Mapping[str, Any]],
    generated: Sequence[Mapping[str, Any]],
    scored: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    selected_ids: Sequence[str],
    state: Any,
    endpoint_count: int,
) -> dict[str, Any]:
    selected_set = {str(item) for item in selected_ids}
    selected_rows = [row for row in candidate_rows if str(row.get("candidate_id")) in selected_set]
    component_trace = getattr(state, "component_trace", {})
    if isinstance(state, Mapping):
        component_trace = state.get("component_trace", component_trace)
    return {
        "schema_version": 1,
        "algorithm": "ph_switch_graph",
        "run_id": run_id,
        "design_mode": "project_ph_switch_graph_sequence_generation",
        "evidence_boundary": (
            "computational candidate generation from startup observations and literature-derived "
            "mechanism priors; generated candidates have no measured project truth"
        ),
        "legacy_rejected_method_reused": False,
        "retrospective_masking_semantics": {
            "generated_candidates_are_not_heldout_truth": True,
            "hidden_measured_variants_not_reused_as_prospective_truth": True,
        },
        "observed_variant_count": len(dataset.variants),
        "observed_endpoint_count": int(endpoint_count),
        "base_heavy_length": len(base_heavy),
        "base_light_length": len(base_light),
        "base_sequence_source": base_source,
        "candidate_edit_vocabulary_count": len(vocabulary),
        "generated_candidate_count": len(generated),
        "scored_candidate_count": len(scored),
        "selected_count": len(selected_ids),
        "selected_candidate_ids": list(selected_ids),
        "generated_new_mutation_site_count": _row_flag_count(candidate_rows, "uses_new_mutation_site"),
        "generated_new_position_count": _row_flag_count(candidate_rows, "uses_new_position"),
        "selected_new_mutation_site_count": _row_flag_count(selected_rows, "uses_new_mutation_site"),
        "selected_new_position_count": _row_flag_count(selected_rows, "uses_new_position"),
        "observed_duplicate_selected_count": _row_flag_count(selected_rows, "observed_duplicate"),
        "selected_operator_modes": sorted({str(row.get("operator") or "") for row in selected_rows}),
        "unique_score_count": len({str(row.get("score")) for row in candidate_rows if row.get("score") != ""}),
        "component_trace": _json_safe(component_trace),
    }


def _write_project_design_run_artifacts(
    root: Path,
    run_dir: Path,
    *,
    run_id: str,
    algorithm: str,
    candidate_rows: Sequence[Mapping[str, Any]],
    selected_ids: Sequence[str],
    summary: Mapping[str, Any],
    budget: int,
    generation_budget: int,
) -> dict[str, str]:
    """Write the generic project-run artifacts expected by project validation."""

    candidate_pool_rows = [
        _generic_project_candidate_row(row, algorithm=algorithm)
        for row in candidate_rows
    ]
    selected_set = {str(candidate_id) for candidate_id in selected_ids}
    panel_rows = [
        _generic_project_panel_row(row, algorithm=algorithm)
        for row in candidate_rows
        if str(row.get("candidate_id") or "") in selected_set
    ]
    policy_comparison_rows = _project_design_policy_comparison_rows(
        algorithm=algorithm,
        summary=summary,
        panel_rows=panel_rows,
    )
    policy_metrics = _project_design_policy_metrics(
        algorithm=algorithm,
        summary=summary,
        panel_rows=panel_rows,
        budget=budget,
        generation_budget=generation_budget,
    )

    candidate_pool_path = run_dir / "candidate_pool.csv"
    panel_path = run_dir / "panel_recommendation.csv"
    policy_comparison_path = run_dir / "policy_comparison.csv"
    policy_metrics_path = run_dir / "policy_metrics.json"
    validation_path = run_dir / "validation_report.json"
    decision_report_path = run_dir / "decision_report.md"

    _write_csv(candidate_pool_path, candidate_pool_rows)
    _write_csv(panel_path, panel_rows)
    _write_csv(policy_comparison_path, policy_comparison_rows)
    write_json(policy_metrics_path, policy_metrics)
    write_json(
        validation_path,
        {
            "project_id": _project_id(root),
            "run_id": run_id,
            "valid": False,
            "findings": [
                {
                    "severity": "warning",
                    "code": "validation_pending",
                    "message": "Initial placeholder overwritten by project validation.",
                    "artifact": str(validation_path),
                }
            ],
        },
    )
    _write_project_design_decision_report(
        decision_report_path,
        run_id=run_id,
        algorithm=algorithm,
        summary=summary,
        panel_rows=panel_rows,
        budget=budget,
        generation_budget=generation_budget,
    )

    from design_scientist.reporting import write_human_review_packet
    from design_scientist.validators import validate_project

    write_human_review_packet(root, run_id=run_id)
    validation_report = validate_project(root, run_id=run_id, write_report=True)
    human_review_path = write_human_review_packet(
        root,
        run_id=run_id,
        validation_report=validation_report,
    )
    return {
        "candidate_pool": str(candidate_pool_path),
        "panel_recommendation": str(panel_path),
        "policy_comparison": str(policy_comparison_path),
        "policy_metrics": str(policy_metrics_path),
        "validation_report": str(validation_path),
        "decision_report": str(decision_report_path),
        "human_review_packet": str(human_review_path),
    }


def _generic_project_candidate_row(
    row: Mapping[str, Any],
    *,
    algorithm: str,
) -> dict[str, Any]:
    return {
        **dict(row),
        "category": _project_candidate_category(row),
        "target_system": "1E62",
        "background": "1E62_base_sequence",
        "design_algorithm": algorithm,
        "source_refs": json.dumps(
            [
                {"kind": "computational_project_design", "algorithm": algorithm},
                {"kind": "startup_observation_state", "id": "standardized/observations_long.csv"},
            ],
            sort_keys=True,
        ),
        "required_endpoints": json.dumps(
            ["pH7.4_binding_retention", "pH6.0_release", "expression_qc"],
            sort_keys=True,
        ),
        "risk_flags": json.dumps(_project_candidate_risk_flags(row), sort_keys=True),
    }


def _generic_project_panel_row(
    row: Mapping[str, Any],
    *,
    algorithm: str,
) -> dict[str, Any]:
    generic = _generic_project_candidate_row(row, algorithm=algorithm)
    generic["selection_rationale"] = _project_selection_rationale(row)
    return generic


def _project_candidate_category(row: Mapping[str, Any]) -> str:
    if str(row.get("uses_new_mutation_site") or "").lower() == "true":
        return "de_novo_candidate"
    operator = str(row.get("operator") or "")
    if operator.startswith("evidence_guardrail"):
        return "evidence_guardrail_candidate"
    return "generated_candidate"


def _project_candidate_risk_flags(row: Mapping[str, Any]) -> list[str]:
    flags: list[str] = []
    if str(row.get("feasibility_status") or "").lower() not in {"feasible", "review", "needs_review"}:
        flags.append("feasibility_status_not_feasible")
    if str(row.get("uses_new_mutation_site") or "").lower() == "true":
        flags.append("new_mutation_site_requires_wet_lab_validation")
    if str(row.get("observed_duplicate") or "").lower() == "true":
        flags.append("observed_duplicate")
    return flags


def _project_selection_rationale(row: Mapping[str, Any]) -> str:
    operator = str(row.get("operator") or "generated_candidate")
    score = row.get("score", "")
    new_sites = row.get("new_mutation_sites", "[]")
    return (
        f"{operator}; score={score}; new_mutation_sites={new_sites}; "
        "computational recommendation requiring human review before wet-lab submission"
    )


def _project_design_policy_comparison_rows(
    *,
    algorithm: str,
    summary: Mapping[str, Any],
    panel_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected_count = len(panel_rows)
    total_cost = sum((_to_float(row.get("cost")) or 0.0) for row in panel_rows)
    return [
        {
            "policy": algorithm,
            "role": "selected_project_design_policy",
            "selected_count": selected_count,
            "selected_new_mutation_site_count": summary.get("selected_new_mutation_site_count", 0),
            "observed_duplicate_selected_count": summary.get("observed_duplicate_selected_count", 0),
            "total_cost": _round(total_cost),
            "comparison_note": "Selected by project pH-switch graph candidate generation.",
        },
        {
            "policy": "project_masking_benchmark",
            "role": "retrospective_real_pool_check",
            "selected_count": "",
            "selected_new_mutation_site_count": "",
            "observed_duplicate_selected_count": "",
            "total_cost": "",
            "comparison_note": "Use runs/formal_project_masking for observed-pool masking; small measured pool can make methods tied.",
        },
        {
            "policy": "generative_benchmark",
            "role": "synthetic_and_stress_check",
            "selected_count": "",
            "selected_new_mutation_site_count": "",
            "observed_duplicate_selected_count": "",
            "total_cost": "",
            "comparison_note": "Use runs/formal_generative_ph_switch for mechanism-level generation gate and ablations.",
        },
    ]


def _project_design_policy_metrics(
    *,
    algorithm: str,
    summary: Mapping[str, Any],
    panel_rows: Sequence[Mapping[str, Any]],
    budget: int,
    generation_budget: int,
) -> dict[str, Any]:
    total_cost = sum((_to_float(row.get("cost")) or 0.0) for row in panel_rows)
    return {
        "schema_version": 1,
        "algorithm": algorithm,
        "budget_mode": "cost_budget",
        "budget": int(budget),
        "generation_budget": int(generation_budget),
        "selected_count": len(panel_rows),
        "selected_candidate_ids": [str(row.get("candidate_id") or "") for row in panel_rows],
        "selected_total_cost": _round(total_cost),
        "generated_candidate_count": summary.get("generated_candidate_count", 0),
        "generated_new_mutation_site_count": summary.get("generated_new_mutation_site_count", 0),
        "selected_new_mutation_site_count": summary.get("selected_new_mutation_site_count", 0),
        "observed_duplicate_selected_count": summary.get("observed_duplicate_selected_count", 0),
        "baseline_comparison": [
            "Retrospective project masking is stored separately in runs/formal_project_masking.",
            "Generative benchmark and ablations are stored separately in runs/formal_generative_ph_switch.",
            "External antibody and pH-switch checks are stored in runs/formal_external_antibody and runs/formal_external_ph_switch.",
        ],
        "unsupported_claims": [
            "Generated candidates are computational designs, not wet-lab validated antibodies.",
            "Current 1E62 combo data do not prove mutation-level causality for selected edits.",
            "External and synthetic benchmarks support algorithm stress testing but do not replace prospective pH 6.0/7.4 assays.",
            "A selected de novo mutation site requires sequence, expression, binding, and developability review before synthesis.",
        ],
    }


def _write_project_design_decision_report(
    path: Path,
    *,
    run_id: str,
    algorithm: str,
    summary: Mapping[str, Any],
    panel_rows: Sequence[Mapping[str, Any]],
    budget: int,
    generation_budget: int,
) -> None:
    selected_ids = [str(row.get("candidate_id") or "") for row in panel_rows]
    lines = [
        f"# Project Design Decision Report: {run_id}",
        "",
        f"Algorithm: `{algorithm}`",
        f"Budget mode: `cost_budget`; budget={budget}; generation_budget={generation_budget}",
        f"Generated candidates: {summary.get('generated_candidate_count', 0)}",
        f"Selected candidates: {len(panel_rows)}",
        f"Selected candidates with new mutation sites: {summary.get('selected_new_mutation_site_count', 0)}",
        "",
        "## Selected Candidates",
    ]
    for row in panel_rows:
        lines.append(
            "- "
            f"{row.get('candidate_id')} | modules={row.get('modules')} | "
            f"score={row.get('score')} | cost={row.get('cost')} | "
            f"category={row.get('category')}"
        )
    if not selected_ids:
        lines.append("- No candidates selected.")
    lines.extend(
        [
            "",
            "## Evidence Boundary",
            "This is a computational design recommendation from startup observations and literature-derived mechanism priors. It is not wet-lab validation.",
            "",
            "## Required Human Review",
            "- Check sequence liabilities and synthesis feasibility.",
            "- Decide whether assay controls/repeats should be added before wet-lab submission.",
            "- Treat pH 6.0 release and pH 7.4 retention as matched endpoints in the next experiment.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _project_id(root: Path) -> str:
    data = _safe_read_yaml(root / "project.yaml")
    return str(data.get("project_id") or data.get("name") or root.name)


def _advance_project_design_stage(
    root: Path,
    *,
    run_id: str,
    algorithm: str,
    selected_ids: Sequence[str],
    budget: int,
    generation_budget: int,
) -> None:
    project_path = root / "project.yaml"
    project = _safe_read_yaml(project_path)
    if project:
        project["stage"] = "project_design_execution"
        project.setdefault("constraints", {})["no_panel_recommendation_in_startup_stage"] = False
        project.setdefault("constraints", {})["wet_lab_submission_requires_human_review"] = True
        write_yaml(project_path, project)

    state_path = root / "state" / "design_state.json"
    state = _read_json_if_exists(state_path)
    if isinstance(state, Mapping):
        state = dict(state)
    else:
        state = {}
    state["panel_stage"] = {
        "status": "computational_project_design_ready_for_human_review",
        "run_id": run_id,
        "algorithm": algorithm,
        "selected_candidate_ids": list(selected_ids),
        "budget_mode": "cost_budget",
        "budget": int(budget),
        "generation_budget": int(generation_budget),
        "untested_sequences_evidence_tier": "model_derived",
        "wet_lab_submission_requires_human_review": True,
    }
    state.setdefault("policy_history", []).append(
        {
            "event_type": "project_design_run",
            "run_id": run_id,
            "algorithm": algorithm,
            "selected_count": len(selected_ids),
            "budget": int(budget),
            "generation_budget": int(generation_budget),
        }
    )
    write_json(state_path, state)

    from design_scientist.project_history import append_history_event

    append_history_event(
        root,
        {
            "event_type": "project_design_run",
            "status": "completed",
            "run_id": run_id,
            "algorithm": algorithm,
            "selected_count": len(selected_ids),
            "budget": int(budget),
            "generation_budget": int(generation_budget),
        },
    )


def _cmdgd_observed_endpoint_count(root: Path, observed_ids: set[str]) -> int:
    observations_path = root / "standardized" / "observations_long.csv"
    if not observations_path.is_file():
        return 0
    count = 0
    for row in _read_csv(observations_path):
        if str(row.get("variant_id") or "").strip() not in observed_ids:
            continue
        if str(row.get("endpoint") or "") == "KD_ratio":
            continue
        count += 1
    return count


def _row_flag_count(rows: Sequence[Mapping[str, Any]], field: str) -> int:
    return sum(1 for row in rows if str(row.get(field) or "").lower() == "true")


def _cmdgd_sequence_edits(
    base_heavy: str,
    base_light: str,
    heavy: str,
    light: str,
) -> list[dict[str, Any]]:
    return [
        *_cmdgd_chain_edits("H", base_heavy, heavy),
        *_cmdgd_chain_edits("L", base_light, light),
    ]


def _cmdgd_chain_edits(chain: str, base: str, observed: str) -> list[dict[str, Any]]:
    edits: list[dict[str, Any]] = []
    for index, (from_residue, to_residue) in enumerate(zip(base, observed), start=1):
        if from_residue == to_residue:
            continue
        edits.append(
            {
                "id": f"{chain}{index}{to_residue}",
                "chain": chain,
                "position": index,
                "from": from_residue,
                "to": to_residue,
            }
        )
    if len(observed) > len(base):
        start = len(base) + 1
        inserted = observed[len(base) :]
        edits.append(
            {
                "id": f"{chain}{start}ins{inserted}",
                "chain": chain,
                "position": start,
                "from": "",
                "to": inserted,
            }
        )
    elif len(observed) < len(base):
        start = len(observed) + 1
        deleted = base[len(observed) :]
        edits.append(
            {
                "id": f"{chain}{start}del{deleted}",
                "chain": chain,
                "position": start,
                "from": deleted,
                "to": "",
            }
        )
    return edits


def _has_cmdgd_sequences(record: Mapping[str, Any]) -> bool:
    return bool(record.get("heavy_chain_seq")) and bool(record.get("light_chain_seq"))


def _run_selector(
    fn: Callable[..., list[str]],
    observed: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    budget: int,
    fold_index: int,
) -> tuple[list[str], str | None]:
    try:
        selected = fn(observed, candidates, budget, fold_index, random.Random(1729 + fold_index))
    except Exception as exc:  # pragma: no cover - defensive benchmark path.
        return [], f"{type(exc).__name__}: {exc}"
    return [str(item) for item in selected], None


def _folds(dataset: ProjectReplayDataset, *, folds: str) -> list[tuple[str, tuple[str, ...]]]:
    if folds == "leave_one_variant":
        return [(f"leave_one_{variant_id}", (variant_id,)) for variant_id in dataset.maskable_variant_ids]
    if folds.startswith("kfold_"):
        try:
            fold_count = int(folds.split("_", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Invalid kfold strategy: {folds}") from exc
        if fold_count <= 1:
            raise ValueError("kfold strategy requires at least two folds")
        ids = list(dataset.maskable_variant_ids)
        chunks = [[] for _ in range(min(fold_count, len(ids)))]
        for index, variant_id in enumerate(ids):
            chunks[index % len(chunks)].append(variant_id)
        return [
            (f"{folds}_{index + 1}", tuple(chunk))
            for index, chunk in enumerate(chunks)
            if chunk
        ]
    raise ValueError(f"Unsupported project masking fold strategy: {folds}")


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(str(row["mechanism"]), []).append(row)
    summary = []
    for method, method_rows in by_method.items():
        completed = [row for row in method_rows if row.get("status") == "completed"]
        summary.append(
            {
                "mechanism": method,
                "fold_count": len(method_rows),
                "completed_fold_count": len(completed),
                "mean_best_feasible_utility": _round(_mean_field(completed, "best_feasible_utility")),
                "mean_hit_rate": _round(_mean_field(completed, "hit_rate")),
                "mean_regret_proxy": _round(_mean_field(completed, "regret_proxy")),
                "mean_false_claim_rate": _round(_mean_field(completed, "false_claim_rate")),
                "mean_evidence_coverage": _round(_mean_field(completed, "evidence_coverage")),
                "failed_fold_count": len(method_rows) - len(completed),
            }
        )
    return sorted(
        summary,
        key=lambda row: (
            float(row["mean_best_feasible_utility"]),
            -float(row["mean_regret_proxy"]),
            -float(row["mean_false_claim_rate"]),
            row["mechanism"],
        ),
        reverse=True,
    )


def _config_payload(
    dataset: ProjectReplayDataset,
    *,
    folds: str,
    mechanisms: list[str],
    budget: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "benchmark": "project_data_observed_pool_masking",
        "folds": folds,
        "mechanisms": mechanisms,
        "budget": budget,
        "variant_count": len(dataset.variants),
        "maskable_variant_count": len(dataset.maskable_variant_ids),
        "utility_formula": UTILITY_FORMULA,
        "evidence_scope": "retrospective_masked_project_data",
        "leakage_controls": {
            "score_only_masked_heldout_outcomes": True,
            "kd_ratio_kept_as_supplementary_track": True,
            "observed_pool_only": True,
            "heldout_endpoint_values_hidden_from_selectors": True,
            "heldout_sequence_and_module_descriptors_visible": True,
        },
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = read_yaml(path)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _read_json_if_exists(path: Path) -> Any:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _mutation_tokens(sequence_row: Mapping[str, Any]) -> list[str]:
    raw = sequence_row.get("mutations")
    if not raw:
        raw = ";".join(
            str(sequence_row.get(key) or "")
            for key in ("heavy_construct", "light_construct", "heavy_variant_label", "light_variant_label")
        )
    tokens = []
    for part in str(raw or "").replace(",", ";").split(";"):
        token = part.strip()
        if token:
            tokens.append(token)
    return tokens


def _merge_optional_mutation_annotations(
    root: Path,
    sequence_by_variant: dict[str, dict[str, str]],
) -> None:
    """Merge allowlisted variant mutation annotations when standardized rows lack them."""

    summary_path = root / "raw" / "wet_lab" / "elisa_summary.csv"
    if not summary_path.is_file():
        return
    try:
        rows = _read_csv(summary_path)
    except OSError:
        return
    for row in rows:
        variant_id = str(row.get("variant") or row.get("variant_id") or "").strip()
        mutations = str(row.get("mutations") or "").strip()
        if not variant_id or not mutations or variant_id not in sequence_by_variant:
            continue
        if not str(sequence_by_variant[variant_id].get("mutations") or "").strip():
            sequence_by_variant[variant_id]["mutations"] = mutations


def _record_id(record: Mapping[str, Any]) -> str:
    return str(record.get("variant_id") or record.get("candidate_id") or record.get("id") or record.get("name"))


def _as_list(value: Any) -> list[str]:
    if value in (None, "", [], {}, ()):
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.replace(",", ";").split(";") if item.strip()]
    if isinstance(value, Iterable):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted(_json_safe(item) for item in value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _mean_or_none(values: Sequence[float]) -> float | None:
    return mean(values) if values else None


def _mean_field(rows: list[dict[str, Any]], field: str) -> float:
    values = [_to_float(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return mean(values) if values else 0.0


def _log1p(value: float) -> float:
    import math

    return math.log1p(max(value, 0.0))


def _round(value: float) -> float:
    return round(float(value), 6)


def _digest(values: Sequence[str]) -> str:
    payload = json.dumps(list(values), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_run_id(run_id: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(run_id)).strip("._-")
    return clean or "project_masking"


def _natural_sort_key(value: str) -> tuple[str, int]:
    prefix = "".join(ch for ch in value if not ch.isdigit())
    digits = "".join(ch for ch in value if ch.isdigit())
    return prefix, int(digits or 0)


__all__ = [
    "CMDGD_PROJECT_CANDIDATES",
    "CMDGD_PROJECT_SUMMARY",
    "PH_SWITCH_GRAPH_PROJECT_CANDIDATES",
    "PH_SWITCH_GRAPH_PROJECT_SUMMARY",
    "PROJECT_MASKING_ABLATION",
    "PROJECT_MASKING_CONFIG",
    "PROJECT_MASKING_RESULT",
    "PROJECT_MASKING_SUMMARY",
    "ProjectReplayDataset",
    "ProjectVariantRecord",
    "dataset_is_available",
    "load_project_replay_dataset",
    "run_project_cmdgd_design",
    "run_project_ph_switch_graph_design",
    "run_project_masking_benchmark",
]
