from __future__ import annotations

import csv
import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import design_scientist.algorithm_benchmark as algorithm_benchmark
from design_scientist.algorithm_benchmark import run_mccbd_benchmark
from design_scientist.algorithms import mccbd


FORBIDDEN_SELECTOR_KEYS = {
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


def _assert_candidate_blind(record: Mapping[str, Any]) -> None:
    leaked = sorted(_find_forbidden_keys(record))
    assert leaked == []


def _find_forbidden_keys(value: Any, path: str = "") -> set[str]:
    if isinstance(value, Mapping):
        leaked = set()
        for key, nested in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}" if path else key_text
            if key_text in FORBIDDEN_SELECTOR_KEYS:
                leaked.add(child_path)
            leaked.update(_find_forbidden_keys(nested, child_path))
        return leaked
    if isinstance(value, list):
        leaked = set()
        for index, nested in enumerate(value):
            leaked.update(_find_forbidden_keys(nested, f"{path}[{index}]"))
        return leaked
    return set()


def _rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_algorithm_benchmark_writes_required_artifacts_and_generic_candidate_records(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"

    result = run_mccbd_benchmark(
        project,
        run_id="unit",
        seeds=(0,),
        rounds=2,
        budget=4,
        mechanisms=("mccbd", "random_feasible", "fixed_mix"),
        worlds=("additive",),
    )

    run_dir = project / "runs" / "unit"
    assert result["status"] == "completed"
    assert Path(result["benchmark_results_path"]) == run_dir / "mccbd_benchmark_results.csv"
    assert Path(result["summary_results_path"]) == run_dir / "mccbd_benchmark_summary.csv"
    assert Path(result["ablation_results_path"]) == run_dir / "mccbd_ablation_results.csv"
    assert Path(result["config_path"]) == run_dir / "mccbd_benchmark_config.json"
    assert set(result["artifact_paths"]) == {
        "benchmark_results",
        "benchmark_summary",
        "ablation_results",
        "benchmark_config",
    }
    assert all(Path(path).exists() for path in result["artifact_paths"].values())

    benchmark_rows = _rows(result["benchmark_results_path"])
    assert {row["mechanism"] for row in benchmark_rows} == {
        "mccbd",
        "random_feasible",
        "fixed_mix",
    }
    assert {row["world_id"] for row in benchmark_rows} == {"additive"}
    assert {row["status"] for row in benchmark_rows} == {"completed"}
    assert set(benchmark_rows[0]) >= {
        "best_feasible_utility",
        "hit_rate",
        "regret_proxy",
        "false_claim_rate",
        "evidence_coverage",
        "selected_diversity",
        "candidate_record_example",
    }

    example = json.loads(benchmark_rows[0]["candidate_record_example"])
    assert {
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
    } <= set(example)
    _assert_candidate_blind(example)

    config = json.loads(Path(result["config_path"]).read_text(encoding="utf-8"))
    assert config["benchmark"] == "generic_mccbd_algorithm_benchmark"
    assert config["worlds"] == ["additive"]
    assert config["mechanisms"] == ["mccbd", "random_feasible", "fixed_mix"]


def test_algorithm_benchmark_is_deterministic_for_same_seed_suite(tmp_path: Path) -> None:
    project = tmp_path / "project"

    first = run_mccbd_benchmark(
        project,
        run_id="deterministic",
        seeds=(3, 7),
        rounds=2,
        budget=3,
        mechanisms=("mccbd", "evidence_calibrated_ucb", "random_feasible", "fixed_mix"),
    )
    snapshots = {
        name: Path(path).read_text(encoding="utf-8")
        for name, path in first["artifact_paths"].items()
    }

    second = run_mccbd_benchmark(
        project,
        run_id="deterministic",
        seeds=(3, 7),
        rounds=2,
        budget=3,
        mechanisms=("mccbd", "evidence_calibrated_ucb", "random_feasible", "fixed_mix"),
    )

    assert second["selected_mechanism"] == first["selected_mechanism"]
    assert second["selected_passes_gate"] == first["selected_passes_gate"]
    assert {
        name: Path(path).read_text(encoding="utf-8")
        for name, path in second["artifact_paths"].items()
    } == snapshots


def test_default_suite_selects_mccbd_and_passes_winner_gate(tmp_path: Path) -> None:
    result = run_mccbd_benchmark(
        tmp_path / "project",
        run_id="gate",
        seeds=(0, 1, 2),
        rounds=3,
        budget=4,
    )

    assert result["selected_mechanism"] == "mccbd"
    assert result["selected_passes_gate"] is True

    summary_rows = _rows(result["summary_results_path"])
    constrained = {
        row["mechanism"]: row
        for row in summary_rows
        if row["world_id"] == "constrained_risky_decoy"
    }
    assert float(constrained["mccbd"]["mean_best_feasible_utility"]) > float(
        constrained["random_feasible"]["mean_best_feasible_utility"]
    )
    assert float(constrained["mccbd"]["mean_best_feasible_utility"]) > float(
        constrained["fixed_mix"]["mean_best_feasible_utility"]
    )


def test_ablation_artifact_contains_mccbd_component_removals(tmp_path: Path) -> None:
    result = run_mccbd_benchmark(
        tmp_path / "project",
        run_id="ablations",
        seeds=(0,),
        rounds=2,
        budget=4,
        mechanisms=("mccbd",),
        worlds=("constrained_risky_decoy",),
    )

    rows = _rows(result["ablation_results_path"])
    assert {row["mechanism"] for row in rows} == {"mccbd"}
    assert {row["ablation"] for row in rows} >= {
        "full",
        "mccbd_no_constraint",
        "mccbd_no_information_value",
        "mccbd_no_batch_diversity",
    }
    assert all(row["status"] == "completed" for row in rows)


def test_selector_visible_candidate_examples_and_selector_inputs_are_truth_blind(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_candidates: list[Mapping[str, Any]] = []

    def spy_select_for_mechanism(
        mechanism: str,
        observed: list[Mapping[str, Any]],
        candidates: list[Mapping[str, Any]],
        budget: int,
        round_index: int,
        rng: random.Random,
    ) -> list[str]:
        del mechanism, observed, budget, round_index, rng
        captured_candidates.extend(dict(candidate) for candidate in candidates)
        return [str(candidates[0]["variant_id"])] if candidates else []

    monkeypatch.setattr(
        algorithm_benchmark,
        "_select_for_mechanism",
        spy_select_for_mechanism,
    )

    result = run_mccbd_benchmark(
        tmp_path / "project",
        run_id="blind",
        seeds=(0,),
        rounds=2,
        budget=2,
        mechanisms=("mccbd",),
        worlds=("constrained_risky_decoy",),
    )

    benchmark_rows = _rows(result["benchmark_results_path"])
    assert benchmark_rows
    for row in benchmark_rows:
        _assert_candidate_blind(json.loads(row["candidate_record_example"]))
    assert captured_candidates
    for candidate in captured_candidates:
        _assert_candidate_blind(candidate)


def test_mccbd_ablation_mechanisms_use_real_select_batch_with_ablation_config(
    monkeypatch,
) -> None:
    records = algorithm_benchmark.generate_synthetic_mechanism_world("additive", seed=0)
    records_by_id = {str(record["variant_id"]): record for record in records}
    world = algorithm_benchmark._world_by_id("additive")
    observed_ids = set(algorithm_benchmark._initial_observation_ids(world, records_by_id))
    observed = [
        algorithm_benchmark._observed_record(records_by_id[variant_id])
        for variant_id in sorted(observed_ids)
    ]
    candidates = [
        algorithm_benchmark._candidate_record(record)
        for record in records
        if str(record["variant_id"]) not in observed_ids
    ]
    seen_ablation_modes: list[str | None] = []

    def spy_select_batch(
        observed_records,
        candidate_records,
        budget,
        round_index,
        rng,
        config=None,
        fitted_state=None,
    ) -> list[str]:
        del observed_records, budget, round_index, rng, fitted_state
        for candidate in candidate_records:
            _assert_candidate_blind(candidate)
        seen_ablation_modes.append(getattr(config, "ablation_mode", None))
        return [str(candidate_records[0]["variant_id"])]

    monkeypatch.setattr(mccbd, "select_batch", spy_select_batch)

    for mechanism in (
        "mccbd",
        "mccbd_no_constraint",
        "mccbd_no_information_value",
        "mccbd_no_batch_diversity",
    ):
        selected = algorithm_benchmark._select_for_mechanism(
            mechanism,
            observed,
            candidates,
            budget=2,
            round_index=0,
            rng=random.Random(0),
        )
        assert selected

    assert seen_ablation_modes == [
        "full",
        "no_constraint",
        "no_information_value",
        "no_batch_diversity",
    ]


def test_summary_aggregates_by_world_and_overall(tmp_path: Path) -> None:
    result = run_mccbd_benchmark(
        tmp_path / "project",
        run_id="summary",
        seeds=(0, 1),
        rounds=2,
        budget=3,
        mechanisms=("mccbd", "random_feasible", "fixed_mix"),
        worlds=("additive", "epistatic"),
    )

    rows = _rows(result["summary_results_path"])
    by_world = {(row["mechanism"], row["world_id"]) for row in rows}
    assert ("mccbd", "additive") in by_world
    assert ("mccbd", "epistatic") in by_world
    assert ("mccbd", "overall") in by_world
    assert ("random_feasible", "overall") in by_world
    assert ("fixed_mix", "overall") in by_world
    assert all(int(row["replicate_count"]) >= 1 for row in rows)
