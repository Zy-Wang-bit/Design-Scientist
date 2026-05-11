from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from design_scientist import policies
from design_scientist.mechanism_replay import (
    make_policy_mechanism,
    run_mechanism_benchmark,
)


EXPECTED_SUMMARY_COLUMNS = {
    "mechanism",
    "rank",
    "worlds_tested",
    "majority_win_vs_random_feasible",
    "majority_win_vs_fixed_mix",
    "mean_best_feasible_utility",
    "mean_false_claim_rate",
    "key_ablation_delta",
    "architecture_clone",
    "selected_eligible",
}


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _selected_eligible(rows: list[dict[str, str]], mechanism: str) -> str:
    return next(row for row in rows if row["mechanism"] == mechanism)["selected_eligible"]


def _clone_lifecycle() -> dict[str, Any]:
    return {
        "name": "random_clone",
        "claims": ["same architecture as random feasible baseline"],
        "architecture_clone": True,
        "fit_state": lambda context: context,
        "generate_candidates": lambda state: state["candidate_records"],
        "score_candidates": lambda state, candidates: list(candidates),
        "select_panel": (
            lambda state, candidates, budget, rng: policies.random_feasible(
                state["observed_records"],
                candidates,
                budget,
                state["round_index"],
                rng,
            )
        ),
        "plan_ablations": lambda state: [
            {"name": "none"},
            {"name": "key_component_removed"},
        ],
    }


def _no_positive_ablation_lifecycle() -> dict[str, Any]:
    return {
        "name": "no_positive_ablation",
        "claims": ["general mechanism claim"],
        "fit_state": lambda context: context,
        "generate_candidates": lambda state: state["candidate_records"],
        "score_candidates": lambda state, candidates: list(candidates),
        "select_panel": (
            lambda state, candidates, budget, rng: policies.mechanism_aware(
                state["observed_records"],
                candidates,
                budget,
                state["round_index"],
                rng,
            )
        ),
        "plan_ablations": lambda state: [
            {"name": "none"},
            {"name": "key_component_removed"},
        ],
    }


def test_baseline_mechanisms_write_all_v3_csv_artifacts(tmp_path: Path) -> None:
    project = tmp_path / "project"

    result = run_mechanism_benchmark(project, run_id="unit", rounds=2, budget=5)

    benchmark_path = project / "runs" / "unit" / "mechanism_benchmark_results.csv"
    summary_path = project / "runs" / "unit" / "mechanism_benchmark_summary.csv"
    ablation_path = project / "runs" / "unit" / "mechanism_ablation_results.csv"
    assert Path(result["benchmark_results_path"]) == benchmark_path
    assert Path(result["summary_results_path"]) == summary_path
    assert Path(result["ablation_results_path"]) == ablation_path
    assert benchmark_path.exists()
    assert summary_path.exists()
    assert ablation_path.exists()

    benchmark_rows = _read_rows(benchmark_path)
    summary_rows = _read_rows(summary_path)
    ablation_rows = _read_rows(ablation_path)
    assert {row["mechanism"] for row in benchmark_rows} == {
        "random_feasible",
        "fixed_mix",
        "mechanism_aware",
    }
    assert {row["mechanism"] for row in summary_rows} == {
        "random_feasible",
        "fixed_mix",
        "mechanism_aware",
    }
    assert EXPECTED_SUMMARY_COLUMNS <= set(summary_rows[0])
    assert {row["world_id"] for row in benchmark_rows} == {"additive", "epistatic"}
    assert {row["ablation"] for row in ablation_rows} >= {"none", "key_component_removed"}


def test_claim_stress_mapping_chooses_expected_worlds(tmp_path: Path) -> None:
    claim_mechanism = make_policy_mechanism(
        "claim_driven_mechanism",
        policies.mechanism_aware,
        claims=[
            "Transfer and confound correction should survive noisy assay endpoints.",
            "Sparse early-round epistasis and interaction evidence should be tested.",
        ],
    )

    result = run_mechanism_benchmark(
        tmp_path / "project",
        run_id="claims",
        mechanisms=[claim_mechanism],
        rounds=1,
        budget=3,
    )

    assert result["config"]["worlds"] == [
        "confounded_transfer",
        "sparse_early_round",
        "epistatic",
        "noisy_endpoint",
    ]


def test_clone_mechanism_is_not_selected_eligible(tmp_path: Path) -> None:
    result = run_mechanism_benchmark(
        tmp_path / "project",
        run_id="clone",
        mechanisms=[_clone_lifecycle()],
        rounds=1,
        budget=4,
    )

    summary_rows = _read_rows(result["summary_results_path"])
    clone_row = next(row for row in summary_rows if row["mechanism"] == "random_clone")
    assert clone_row["architecture_clone"] == "true"
    assert clone_row["selected_eligible"] == "false"


def test_no_positive_ablation_mechanism_is_not_selected_eligible(tmp_path: Path) -> None:
    result = run_mechanism_benchmark(
        tmp_path / "project",
        run_id="ablation",
        mechanisms=[_no_positive_ablation_lifecycle()],
        rounds=2,
        budget=5,
    )

    summary_rows = _read_rows(result["summary_results_path"])
    target_row = next(row for row in summary_rows if row["mechanism"] == "no_positive_ablation")
    assert float(target_row["key_ablation_delta"]) <= 0.0
    assert target_row["selected_eligible"] == "false"


def test_lifecycle_object_is_accepted(tmp_path: Path) -> None:
    class ObjectLifecycle:
        name = "object_lifecycle"
        claims = "tiny early assay claim"

        def fit_state(self, context):
            return context

        def generate_candidates(self, state):
            return state["candidate_records"]

        def score_candidates(self, state, candidates):
            return list(candidates)

        def select_panel(self, state, candidates, budget, rng):
            return policies.fixed_mix(
                state["observed_records"],
                candidates,
                budget,
                state["round_index"],
                rng,
            )

        def plan_ablations(self, state):
            return [{"name": "none"}]

    result = run_mechanism_benchmark(
        tmp_path / "project",
        run_id="object",
        mechanisms=ObjectLifecycle(),
        rounds=1,
        budget=3,
    )

    rows = _read_rows(result["summary_results_path"])
    assert any(row["mechanism"] == "object_lifecycle" for row in rows)
    assert result["config"]["worlds"] == ["sparse_early_round", "noisy_endpoint"]


def test_fixed_seed_gives_stable_summary(tmp_path: Path) -> None:
    project = tmp_path / "project"

    first = run_mechanism_benchmark(project, run_id="stable_a", rounds=3, budget=6)
    second = run_mechanism_benchmark(project, run_id="stable_b", rounds=3, budget=6)

    first_rows = _read_rows(first["summary_results_path"])
    second_rows = _read_rows(second["summary_results_path"])
    assert first_rows == second_rows
    assert _selected_eligible(first_rows, "random_feasible") == "false"
    assert _selected_eligible(first_rows, "fixed_mix") == "false"
