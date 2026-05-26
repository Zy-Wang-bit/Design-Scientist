from __future__ import annotations

import csv
import json
import tempfile
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


def _crashing_key_ablation_lifecycle() -> dict[str, Any]:
    def select_panel(state, candidates, budget, rng):
        if state.get("policy_ablation") == "crash_key_ablation":
            raise RuntimeError("key ablation crashed")
        return policies.mechanism_aware(
            state["observed_records"],
            candidates,
            budget,
            state["round_index"],
            rng,
        )

    return {
        "name": "crashing_key_ablation",
        "claims": [
            "Transfer and confound correction should survive noisy assay endpoints.",
            "Sparse early-round epistasis and interaction evidence should be tested.",
        ],
        "fit_state": lambda context: context,
        "generate_candidates": lambda state: state["candidate_records"],
        "score_candidates": lambda state, candidates: list(candidates),
        "select_panel": select_panel,
        "plan_ablations": lambda state: [
            {"name": "none"},
            {
                "name": "key_component_removed",
                "policy_ablation": "crash_key_ablation",
            },
        ],
    }


def _mechanism_aware_wrapper_lifecycle() -> dict[str, Any]:
    return {
        "name": "mechanism_aware_wrapper",
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
            {"name": "key_component_removed", "policy_ablation": "interaction_prior"},
        ],
    }


def _legal_replay_lifecycle() -> dict[str, Any]:
    return {
        "name": "legal_replay_lifecycle",
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
            {"name": "key_component_removed", "policy_ablation": "interaction_prior"},
        ],
    }


def _fit_state_escape_lifecycle(outside_file: Path) -> dict[str, Any]:
    def fit_state(context):
        outside_file.write_text("fit_state escape", encoding="utf-8")
        return context

    return {
        "name": "fit_state_escape",
        "claims": ["general mechanism claim"],
        "fit_state": fit_state,
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
        "plan_ablations": lambda state: [{"name": "none"}],
    }


def _select_panel_escape_lifecycle(outside_file: Path) -> dict[str, Any]:
    def select_panel(state, candidates, budget, rng):
        outside_file.write_text("select_panel escape", encoding="utf-8")
        return policies.mechanism_aware(
            state["observed_records"],
            candidates,
            budget,
            state["round_index"],
            rng,
        )

    return {
        "name": "select_panel_escape",
        "claims": ["general mechanism claim"],
        "fit_state": lambda context: context,
        "generate_candidates": lambda state: state["candidate_records"],
        "score_candidates": lambda state, candidates: list(candidates),
        "select_panel": select_panel,
        "plan_ablations": lambda state: [{"name": "none"}],
    }


def _project_write_lifecycle(project_file: Path) -> dict[str, Any]:
    def fit_state(context):
        project_file.parent.mkdir(parents=True, exist_ok=True)
        project_file.write_text("project tree mutation", encoding="utf-8")
        return context

    return {
        "name": "project_write_escape",
        "claims": ["general mechanism claim"],
        "fit_state": fit_state,
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
        "plan_ablations": lambda state: [{"name": "none"}],
    }


def _fork_escape_lifecycle() -> dict[str, Any]:
    def fit_state(context):
        import os

        os.fork()
        return context

    return {
        "name": "fork_escape",
        "claims": ["general mechanism claim"],
        "fit_state": fit_state,
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
        "plan_ablations": lambda state: [{"name": "none"}],
    }


def test_baseline_mechanisms_write_all_v3_csv_artifacts(tmp_path: Path) -> None:
    project = tmp_path / "project"

    result = run_mechanism_benchmark(project, run_id="unit", rounds=2, budget=5)

    benchmark_path = project / "runs" / "unit" / "mechanism_benchmark_results.csv"
    summary_path = project / "runs" / "unit" / "mechanism_benchmark_summary.csv"
    ablation_path = project / "runs" / "unit" / "mechanism_ablation_results.csv"
    saturation_path = project / "runs" / "unit" / "benchmark_saturation.json"
    assert Path(result["benchmark_results_path"]) == benchmark_path
    assert Path(result["summary_results_path"]) == summary_path
    assert Path(result["ablation_results_path"]) == ablation_path
    assert Path(result["benchmark_saturation_path"]) == saturation_path
    assert benchmark_path.exists()
    assert summary_path.exists()
    assert ablation_path.exists()
    assert saturation_path.exists()

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
    saturation = json.loads(saturation_path.read_text(encoding="utf-8"))
    assert "saturated" in saturation
    assert saturation["required_baselines"] == ["random_feasible", "fixed_mix"]
    assert saturation["row_counts"]["benchmark_results"] == len(benchmark_rows)


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


def test_project_context_reaches_lifecycle_fit_state(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "standardized").mkdir(parents=True)
    (project / "standardized" / "project_context.json").write_text(
        json.dumps(
            {
                "objective": "Improve validated binding selectivity.",
                "endpoints": ["neutral_binding", "acid_release"],
                "observed_row_counts": {"standardized_primary": 7},
                "constraints": {"max_batch": 12},
                "allowed_sources": ["standardized/primary.csv"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    captured_contexts: list[dict[str, Any]] = []

    def fit_state(context: dict[str, Any]) -> dict[str, Any]:
        captured_contexts.append(dict(context["project_context"]))
        return context

    lifecycle = {
        "name": "context_probe",
        "claims": ["general mechanism claim"],
        "fit_state": fit_state,
        "generate_candidates": lambda state: state["candidate_records"],
        "score_candidates": lambda state, candidates: list(candidates),
        "select_panel": (
            lambda state, candidates, budget, rng: policies.fixed_mix(
                state["observed_records"],
                candidates,
                budget,
                state["round_index"],
                rng,
            )
        ),
        "plan_ablations": lambda state: [{"name": "none"}],
    }

    result = run_mechanism_benchmark(
        project,
        run_id="project-context",
        mechanisms=[lifecycle],
        rounds=1,
        budget=3,
    )

    assert captured_contexts
    project_context = captured_contexts[0]
    assert project_context["objective"] == "Improve validated binding selectivity."
    assert project_context["endpoints"] == ["neutral_binding", "acid_release"]
    assert project_context["observed_row_counts"] == {"standardized_primary": 7}
    assert project_context["constraints"] == {"max_batch": 12}
    assert project_context["allowed_sources"] == ["standardized/primary.csv"]
    assert result["config"]["worlds"] == ["additive", "epistatic"]


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


def test_failed_key_ablation_does_not_create_positive_delta_or_eligibility(tmp_path: Path) -> None:
    result = run_mechanism_benchmark(
        tmp_path / "project",
        run_id="failed-key-ablation",
        mechanisms=[_crashing_key_ablation_lifecycle(), "mechanism_aware"],
        rounds=2,
        budget=5,
    )

    ablation_rows = _read_rows(result["mechanism_ablation_results_path"])
    failed_key_rows = [
        row
        for row in ablation_rows
        if row["mechanism"] == "crashing_key_ablation"
        and row["ablation"] == "key_component_removed"
    ]
    assert failed_key_rows
    assert {row["status"] for row in failed_key_rows} == {"failed"}

    summary_rows = _read_rows(result["summary_results_path"])
    target_row = next(row for row in summary_rows if row["mechanism"] == "crashing_key_ablation")
    assert float(target_row["key_ablation_delta"]) == 0.0
    assert target_row["selected_eligible"] == "false"


def test_mechanism_aware_wrapper_is_marked_clone_and_ineligible(tmp_path: Path) -> None:
    result = run_mechanism_benchmark(
        tmp_path / "project",
        run_id="mechanism-aware-wrapper",
        mechanisms=[_mechanism_aware_wrapper_lifecycle(), "mechanism_aware"],
        rounds=2,
        budget=5,
    )

    summary_rows = _read_rows(result["summary_results_path"])
    target_row = next(row for row in summary_rows if row["mechanism"] == "mechanism_aware_wrapper")
    assert target_row["architecture_clone"] == "true"
    assert target_row["selected_eligible"] == "false"


def test_replay_lifecycle_outside_writes_fail_only_that_mechanism(tmp_path: Path) -> None:
    fit_escape = (
        Path(tempfile.gettempdir())
        / f"design_scientist_replay_fit_escape_{tmp_path.name}.txt"
    )
    select_escape = (
        Path(tempfile.gettempdir())
        / f"design_scientist_replay_select_escape_{tmp_path.name}.txt"
    )
    fit_escape.unlink(missing_ok=True)
    select_escape.unlink(missing_ok=True)
    try:
        result = run_mechanism_benchmark(
            tmp_path / "project",
            run_id="lifecycle-escape",
            mechanisms=[
                _fit_state_escape_lifecycle(fit_escape),
                _select_panel_escape_lifecycle(select_escape),
                _legal_replay_lifecycle(),
            ],
            rounds=1,
            budget=3,
        )
    finally:
        fit_escape.unlink(missing_ok=True)
        select_escape.unlink(missing_ok=True)

    benchmark_rows = _read_rows(result["mechanism_benchmark_results_path"])
    statuses = {
        row["mechanism"]: row["status"]
        for row in benchmark_rows
        if row["mechanism"] in {
            "fit_state_escape",
            "select_panel_escape",
            "legal_replay_lifecycle",
        }
    }
    errors = {
        row["mechanism"]: row["error"]
        for row in benchmark_rows
        if row["mechanism"] in {"fit_state_escape", "select_panel_escape"}
    }

    assert statuses["fit_state_escape"] == "failed"
    assert statuses["select_panel_escape"] == "failed"
    assert statuses["legal_replay_lifecycle"] == "completed"
    assert "outside replay workspace" in errors["fit_state_escape"]
    assert "outside replay workspace" in errors["select_panel_escape"]


def test_replay_lifecycle_project_tree_write_fails_only_that_mechanism(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project_file = project / "framework" / "rogue_write.txt"

    result = run_mechanism_benchmark(
        project,
        run_id="project-tree-escape",
        mechanisms=[
            _project_write_lifecycle(project_file),
            _legal_replay_lifecycle(),
        ],
        rounds=1,
        budget=3,
    )

    benchmark_rows = _read_rows(result["mechanism_benchmark_results_path"])
    statuses = {
        row["mechanism"]: row["status"]
        for row in benchmark_rows
        if row["mechanism"] in {"project_write_escape", "legal_replay_lifecycle"}
    }
    errors = {
        row["mechanism"]: row["error"]
        for row in benchmark_rows
        if row["mechanism"] == "project_write_escape"
    }

    assert statuses["project_write_escape"] == "failed"
    assert statuses["legal_replay_lifecycle"] == "completed"
    assert "outside replay workspace" in errors["project_write_escape"]
    assert not project_file.exists()


def test_replay_lifecycle_fork_is_blocked_and_fails_only_that_mechanism(tmp_path: Path) -> None:
    result = run_mechanism_benchmark(
        tmp_path / "project",
        run_id="fork-escape",
        mechanisms=[
            _fork_escape_lifecycle(),
            _legal_replay_lifecycle(),
        ],
        rounds=1,
        budget=3,
    )

    benchmark_rows = _read_rows(result["mechanism_benchmark_results_path"])
    statuses = {
        row["mechanism"]: row["status"]
        for row in benchmark_rows
        if row["mechanism"] in {"fork_escape", "legal_replay_lifecycle"}
    }
    errors = {
        row["mechanism"]: row["error"]
        for row in benchmark_rows
        if row["mechanism"] == "fork_escape"
    }

    assert statuses["fork_escape"] == "failed"
    assert statuses["legal_replay_lifecycle"] == "completed"
    assert "os.fork" in errors["fork_escape"]


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
