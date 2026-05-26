from __future__ import annotations

import csv
import json
from pathlib import Path

from design_scientist.project_replay import run_project_masking_benchmark
from test_project_replay import write_project_fixture


def _rows(path: str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_project_masking_benchmark_writes_deterministic_artifacts_and_guardrails(
    tmp_path: Path,
) -> None:
    project = write_project_fixture(tmp_path / "project")

    first = run_project_masking_benchmark(project, run_id="unit", budget=2)

    assert Path(first["benchmark_results_path"]) == (
        project / "runs" / "unit" / "project_masking_benchmark_results.csv"
    )
    assert Path(first["summary_results_path"]) == (
        project / "runs" / "unit" / "project_masking_benchmark_summary.csv"
    )
    assert Path(first["ablation_results_path"]) == (
        project / "runs" / "unit" / "project_masking_ablation_results.csv"
    )
    assert Path(first["config_path"]) == project / "runs" / "unit" / "project_masking_config.json"

    rows = _rows(first["benchmark_results_path"])
    methods = {row["mechanism"] for row in rows}
    assert {"mccbd", "evidence_calibrated_ucb", "random_feasible", "fixed_mix", "greedy_observed"} <= methods
    assert {row["status"] for row in rows} == {"completed"}
    assert {int(row["train_count"]) for row in rows} == {4}
    assert {int(row["candidate_count"]) for row in rows} == {1}
    assert all(int(row["selected_count"]) <= 1 for row in rows)

    for row in rows:
        heldout_ids = set(json.loads(row["heldout_ids"]))
        candidate_ids = set(json.loads(row["candidate_ids"]))
        selected_ids = set(json.loads(row["selected_ids"]))
        scored_heldout_ids = set(json.loads(row["scored_heldout_ids"]))
        diagnostics = json.loads(row["leakage_diagnostics"])

        assert heldout_ids <= candidate_ids
        assert selected_ids <= candidate_ids
        assert scored_heldout_ids <= heldout_ids
        assert diagnostics["selected_ids_outside_candidates"] == []
        assert diagnostics["heldout_ids_in_train"] == []
        assert diagnostics["kd_ratio_joined_to_combo_records"] is False
        assert row["selected_ids_digest"]

    summary_rows = _rows(first["summary_results_path"])
    summary_by_method = {row["mechanism"]: row for row in summary_rows}
    assert {"mccbd", "evidence_calibrated_ucb", "random_feasible", "fixed_mix", "greedy_observed"} <= set(
        summary_by_method
    )
    assert int(summary_by_method["mccbd"]["fold_count"]) == 5
    assert int(summary_by_method["evidence_calibrated_ucb"]["fold_count"]) == 5
    assert int(summary_by_method["random_feasible"]["fold_count"]) == 5

    config = json.loads(Path(first["config_path"]).read_text(encoding="utf-8"))
    assert config["folds"] == "kfold_5"
    assert "pH7.4" in config["utility_formula"]
    assert "pH6/pH7.4" in config["utility_formula"]
    assert "expression" in config["utility_formula"]
    assert config["leakage_controls"]["score_only_masked_heldout_outcomes"] is True

    benchmark_text = Path(first["benchmark_results_path"]).read_text(encoding="utf-8")
    summary_text = Path(first["summary_results_path"]).read_text(encoding="utf-8")
    ablation_text = Path(first["ablation_results_path"]).read_text(encoding="utf-8")

    second = run_project_masking_benchmark(project, run_id="unit", budget=2)

    assert Path(second["benchmark_results_path"]).read_text(encoding="utf-8") == benchmark_text
    assert Path(second["summary_results_path"]).read_text(encoding="utf-8") == summary_text
    assert Path(second["ablation_results_path"]).read_text(encoding="utf-8") == ablation_text
    assert second["summary"] == first["summary"]


def test_project_masking_benchmark_supports_kfold_comparison(tmp_path: Path) -> None:
    project = write_project_fixture(tmp_path / "project")

    result = run_project_masking_benchmark(project, run_id="kfold", budget=2, folds="kfold_2")

    rows = _rows(result["benchmark_results_path"])
    assert {int(row["candidate_count"]) for row in rows} == {2, 3}
    assert all(int(row["selected_count"]) <= 2 for row in rows)
    assert {int(row["selector_round_index"]) for row in rows} == {0, 1}
    assert all(
        set(json.loads(row["selected_ids"])) <= set(json.loads(row["candidate_ids"]))
        for row in rows
    )
    config = json.loads(Path(result["config_path"]).read_text(encoding="utf-8"))
    assert config["folds"] == "kfold_2"


def test_project_masking_best_feasible_utility_excludes_infeasible_selected_variant(
    tmp_path: Path,
) -> None:
    project = write_project_fixture(tmp_path / "project")

    def selects_low_expression_variant(_observed, _candidates, _budget, _round_index, _rng):
        return ["com5"]

    result = run_project_masking_benchmark(
        project,
        run_id="feasible_only_metric",
        budget=1,
        folds="kfold_2",
        mechanisms=[selects_low_expression_variant],
    )

    rows = [
        row
        for row in _rows(result["benchmark_results_path"])
        if "com5" in json.loads(row["candidate_ids"])
    ]
    assert len(rows) == 1
    row = rows[0]
    assert json.loads(row["selected_infeasible_ids"]) == ["com5"]
    assert json.loads(row["selected_feasible_ids"]) == []
    assert float(row["best_selected_utility"]) > 0.0
    assert float(row["best_feasible_utility"]) == 0.0
    assert float(row["false_claim_rate"]) == 1.0


def test_project_masking_selector_receives_fold_round_index_not_result_row_count(
    tmp_path: Path,
) -> None:
    project = write_project_fixture(tmp_path / "project")
    seen_round_indices: list[int] = []

    def records_round_index(_observed, candidates, budget, round_index, _rng):
        seen_round_indices.append(round_index)
        return [row["variant_id"] for row in candidates[:budget]]

    run_project_masking_benchmark(
        project,
        run_id="round_index",
        budget=1,
        folds="kfold_2",
        mechanisms=[records_round_index],
    )

    assert seen_round_indices == [0, 1]


def test_project_masking_benchmark_reports_clear_error_for_missing_dataset(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()

    result = run_project_masking_benchmark(project, run_id="missing")

    assert result["status"] == "unavailable"
    assert "standardized/observations_long.csv" in result["error"]
    assert not (project / "runs" / "missing").exists()
