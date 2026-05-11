from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from design_scientist.synthetic_replay import DEFAULT_METHODS, run_synthetic_benchmark


EXPECTED_WORLD_ORDER = (
    "additive",
    "epistatic",
    "confounded_transfer",
    "noisy_endpoint",
    "sparse_early_round",
)
EXPECTED_WORLDS = set(EXPECTED_WORLD_ORDER)


def _read_rows(path: str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_synthetic_benchmark_writes_deterministic_artifacts(tmp_path: Path) -> None:
    project = tmp_path / "project"

    first = run_synthetic_benchmark(project, run_id="unit", rounds=3, budget=12)
    benchmark_path = Path(first["benchmark_results_path"])
    ablation_path = Path(first["ablation_results_path"])
    config_path = Path(first["config_path"])

    assert benchmark_path == project / "runs" / "unit" / "benchmark_results.csv"
    assert ablation_path == project / "runs" / "unit" / "ablation_results.csv"
    assert config_path == project / "benchmarks" / "synthetic_replay" / "config.yaml"
    assert benchmark_path.exists()
    assert ablation_path.exists()
    assert config_path.exists()

    rows = _read_rows(first["benchmark_results_path"])
    assert {row["world_id"] for row in rows} == EXPECTED_WORLDS
    assert {row["method"] for row in rows} == {
        "mechanism_aware",
        "random_feasible",
        "top_observed",
        "greedy_utility",
        "fixed_mix",
        "pure_uncertainty",
        "pure_lattice_repair",
    }
    assert all(row["novelty_score"] for row in rows)
    assert all(row["baseline_overlap"] for row in rows)
    expected_overlap_columns = {f"selection_overlap_{method}" for method in DEFAULT_METHODS}
    assert expected_overlap_columns <= set(rows[0])

    by_world_method = {
        (row["world_id"], row["method"]): row
        for row in rows
    }

    mechanism_wins = 0
    for world_id in EXPECTED_WORLDS:
        mechanism_best = float(
            by_world_method[(world_id, "mechanism_aware")]["best_feasible_utility"]
        )
        if (
            mechanism_best
            > float(by_world_method[(world_id, "random_feasible")]["best_feasible_utility"])
            and mechanism_best
            > float(by_world_method[(world_id, "fixed_mix")]["best_feasible_utility"])
        ):
            mechanism_wins += 1
    assert mechanism_wins >= 3
    assert float(
        first["summary"]["method_rankings"][0]["mean_best_feasible_utility"]
    ) >= float(
        first["summary"]["method_rankings"][-1]["mean_best_feasible_utility"]
    )
    summary_rows = _read_rows(first["summary_results_path"])
    mechanism_summary = next(row for row in summary_rows if row["method"] == "mechanism_aware")
    assert {
        "beats_random_feasible_worlds",
        "beats_random_feasible_majority",
        "beats_fixed_mix_worlds",
        "beats_fixed_mix_majority",
    } <= set(mechanism_summary)
    assert int(mechanism_summary["world_count"]) == len(EXPECTED_WORLDS)
    assert int(mechanism_summary["beats_random_feasible_worlds"]) >= 3
    assert int(mechanism_summary["beats_fixed_mix_worlds"]) >= 3
    assert mechanism_summary["beats_random_feasible_majority"] == "true"
    assert mechanism_summary["beats_fixed_mix_majority"] == "true"

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["seed"] == 1729
    assert set(config["worlds"]) == EXPECTED_WORLDS
    assert config["target_background"] == "transfer_bg"
    assert config["latent_features"]["confounding"].startswith("Initial singleton")

    benchmark_text = benchmark_path.read_text(encoding="utf-8")
    ablation_text = ablation_path.read_text(encoding="utf-8")
    second = run_synthetic_benchmark(project, run_id="unit", rounds=3, budget=12)
    assert Path(second["benchmark_results_path"]).read_text(encoding="utf-8") == benchmark_text
    assert Path(second["ablation_results_path"]).read_text(encoding="utf-8") == ablation_text
    assert second["summary"] == first["summary"]


def test_synthetic_benchmark_respects_method_subset_and_budget(tmp_path: Path) -> None:
    project = tmp_path / "project"

    result = run_synthetic_benchmark(
        project,
        run_id="subset",
        rounds=2,
        budget=5,
        methods=["mechanism_aware", "random_feasible"],
    )

    rows = _read_rows(result["benchmark_results_path"])
    assert [row["world_id"] for row in rows] == [
        world_id
        for world_id in EXPECTED_WORLD_ORDER
        for _ in range(2)
    ]
    assert [row["method"] for row in rows] == [
        method
        for _ in EXPECTED_WORLD_ORDER
        for method in ["mechanism_aware", "random_feasible"]
    ]
    assert {int(row["selected_count"]) for row in rows} == {10}
    assert {int(row["rounds"]) for row in rows} == {2}
    assert {int(row["budget_per_round"]) for row in rows} == {5}

    ablation_rows = _read_rows(result["ablation_results_path"])
    assert [row["removed_component"] for row in ablation_rows] == [
        removed_component
        for _ in EXPECTED_WORLD_ORDER
        for removed_component in [
            "none",
            "interaction_prior",
            "uncertainty",
            "lattice_repair",
            "confounding_correction",
            "evidence_guardrail",
        ]
    ]
    assert {row["world_id"] for row in ablation_rows} == EXPECTED_WORLDS


def test_synthetic_benchmark_accepts_policy_callables(tmp_path: Path) -> None:
    def first_two_policy(observed, candidates, budget, round_index, rng) -> list[str]:
        del observed, round_index, rng
        return [str(candidate["variant_id"]) for candidate in candidates[:budget]]

    result = run_synthetic_benchmark(
        tmp_path / "project",
        run_id="callable",
        rounds=1,
        budget=2,
        methods=[first_two_policy],
    )

    rows = _read_rows(result["benchmark_results_path"])
    assert [row["method"] for row in rows] == ["first_two_policy"] * len(EXPECTED_WORLD_ORDER)
    assert {int(row["selected_count"]) for row in rows} == {2}
    assert result["summary"]["method_rankings"][0]["method"] == "first_two_policy"


def test_synthetic_benchmark_accepts_policy_callables_returning_candidate_records(
    tmp_path: Path,
) -> None:
    def first_two_records_policy(observed, candidates, budget, round_index, rng) -> list[dict[str, object]]:
        del observed, round_index, rng
        return [dict(candidate) for candidate in candidates[:budget]]

    result = run_synthetic_benchmark(
        tmp_path / "project",
        run_id="callable_records",
        rounds=1,
        budget=2,
        methods=[first_two_records_policy],
    )

    rows = _read_rows(result["benchmark_results_path"])
    assert [row["method"] for row in rows] == ["first_two_records_policy"] * len(EXPECTED_WORLD_ORDER)
    assert {int(row["selected_count"]) for row in rows} == {2}


def test_synthetic_benchmark_isolates_policy_failures_by_method(tmp_path: Path) -> None:
    def raising_policy(observed, candidates, budget, round_index, rng) -> list[str]:
        del observed, candidates, budget, round_index, rng
        raise RuntimeError("policy exploded")

    def invalid_id_policy(observed, candidates, budget, round_index, rng) -> list[str]:
        del observed, candidates, budget, round_index, rng
        return ["missing_candidate_id"]

    def first_two_policy(observed, candidates, budget, round_index, rng) -> list[str]:
        del observed, round_index, rng
        return [str(candidate["variant_id"]) for candidate in candidates[:budget]]

    raising_policy.policy_name = "generated_raising_policy"  # type: ignore[attr-defined]
    invalid_id_policy.policy_name = "generated_invalid_id_policy"  # type: ignore[attr-defined]

    result = run_synthetic_benchmark(
        tmp_path / "project",
        run_id="policy_failures",
        rounds=1,
        budget=2,
        methods=[raising_policy, invalid_id_policy, first_two_policy, "random_feasible"],
    )

    rows = _read_rows(result["benchmark_results_path"])
    rows_by_method: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        rows_by_method.setdefault(row["method"], []).append(row)

    assert {row["status"] for row in rows_by_method["first_two_policy"]} == {"completed"}
    assert {row["status"] for row in rows_by_method["random_feasible"]} == {"completed"}
    assert {row["status"] for row in rows_by_method["generated_raising_policy"]} == {"failed"}
    assert {row["status"] for row in rows_by_method["generated_invalid_id_policy"]} == {"failed"}
    assert all("policy exploded" in row["error"] for row in rows_by_method["generated_raising_policy"])
    assert all("unavailable candidate" in row["error"] for row in rows_by_method["generated_invalid_id_policy"])
    assert {int(row["selected_count"]) for row in rows_by_method["generated_raising_policy"]} == {0}
    assert {int(row["selected_count"]) for row in rows_by_method["generated_invalid_id_policy"]} == {0}

    summary_by_method = {
        row["method"]: row
        for row in result["summary"]["method_rankings"]
    }
    assert summary_by_method["first_two_policy"]["failed_worlds"] == 0
    assert summary_by_method["generated_raising_policy"]["failed_worlds"] == len(EXPECTED_WORLDS)
    assert summary_by_method["generated_invalid_id_policy"]["failed_worlds"] == len(EXPECTED_WORLDS)


def test_synthetic_benchmark_rejects_unknown_method_and_unsafe_run_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown synthetic replay method"):
        run_synthetic_benchmark(tmp_path / "project", methods=["mechanism_aware", "not_a_method"])

    with pytest.raises(ValueError, match="path separators"):
        run_synthetic_benchmark(tmp_path / "project", run_id="../escape")
