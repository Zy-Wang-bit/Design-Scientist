from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from design_scientist import generative_benchmark
from design_scientist.generative_benchmark import run_generative_benchmark


FORBIDDEN_SELECTOR_KEYS = {
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


def _rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _find_forbidden_keys(value: Any, path: str = "") -> set[str]:
    if isinstance(value, Mapping):
        leaked: set[str] = set()
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


def _assert_blind(record: Mapping[str, Any]) -> None:
    assert sorted(_find_forbidden_keys(record)) == []


def test_generative_benchmark_writes_required_artifacts(tmp_path: Path) -> None:
    result = run_generative_benchmark(
        tmp_path,
        mechanisms=(
            "cmdgd",
            "random_edit_generator",
            "observed_recombination_baseline",
            "single_edit_scan",
            "mccbd_pool_selector",
        ),
        seeds=(0,),
        worlds=("ph_contrast",),
    )

    assert result["status"] == "completed"
    run_dir = tmp_path / "runs" / "generative_benchmark"
    assert Path(result["benchmark_results_path"]) == run_dir / "generative_benchmark_results.csv"
    assert Path(result["summary_results_path"]) == run_dir / "generative_benchmark_summary.csv"
    assert Path(result["ablation_results_path"]) == run_dir / "generative_ablation_results.csv"
    assert Path(result["design_examples_path"]) == run_dir / "generative_design_examples.csv"
    assert Path(result["statistical_summary_path"]) == run_dir / "generative_statistical_summary.csv"
    assert Path(result["pairwise_comparisons_path"]) == run_dir / "generative_pairwise_comparisons.csv"
    assert Path(result["config_path"]) == run_dir / "generative_benchmark_config.json"
    assert Path(result["selection_gate_report_path"]) == run_dir / "generative_selection_gate_report.json"
    assert set(result["artifact_paths"]) == {
        "benchmark_results",
        "benchmark_summary",
        "ablation_results",
        "design_examples",
        "statistical_summary",
        "pairwise_comparisons",
        "benchmark_config",
        "selection_gate_report",
    }
    assert all(Path(path).exists() for path in result["artifact_paths"].values())

    rows = _rows(result["benchmark_results_path"])
    assert {row["mechanism"] for row in rows} == {
        "cmdgd",
        "random_edit_generator",
        "observed_recombination_baseline",
        "single_edit_scan",
        "mccbd_pool_selector",
    }
    assert {row["world_id"] for row in rows} == {"ph_contrast"}
    assert {row["status"] for row in rows} == {"completed"}
    assert set(rows[0]) >= {
        "generated_novel_count",
        "best_generated_utility",
        "best_selected_utility",
        "candidate_space_type",
        "generation_counted",
        "generated_utility_applicable",
        "constraint_pass_rate",
        "pH_contrast_score",
        "novel_sequence_rate",
        "false_claim_rate",
        "cost_spent",
        "selector_candidate_example",
    }

    config = json.loads(Path(result["config_path"]).read_text(encoding="utf-8"))
    assert config["benchmark"] == "generative_design_algorithm_benchmark"
    assert config["worlds"] == ["ph_contrast"]
    assert config["seeds"] == [0]
    assert config["leakage_controls"]["selector_inputs_exclude_oracle_truth"] is True
    assert config["selection_gate_report"]["selection_basis"] == "generative_candidate_space_expansion"


def test_generative_benchmark_reports_seed_world_statistical_uncertainty(
    tmp_path: Path,
) -> None:
    result = run_generative_benchmark(
        tmp_path,
        mechanisms=(
            "cmdgd",
            "random_feasible",
            "fixed_mix",
            "evidence_calibrated_ucb",
            "random_edit_generator",
            "observed_recombination_baseline",
            "mccbd_pool_selector",
        ),
        seeds=(0, 1, 2),
        worlds=("ph_contrast", "escape_risk"),
    )

    statistical_rows = _rows(result["statistical_summary_path"])
    assert set(statistical_rows[0]) >= {
        "mechanism",
        "world_id",
        "metric",
        "n",
        "mean",
        "sd",
        "sem",
        "ci95_low",
        "ci95_high",
    }
    by_key = {
        (row["mechanism"], row["world_id"], row["metric"]): row
        for row in statistical_rows
    }
    cmdgd_world = by_key[("cmdgd", "ph_contrast", "best_selected_utility")]
    assert cmdgd_world["n"] == "3"
    assert float(cmdgd_world["ci95_low"]) <= float(cmdgd_world["mean"])
    assert float(cmdgd_world["ci95_high"]) >= float(cmdgd_world["mean"])
    assert by_key[("cmdgd", "overall", "best_selected_utility")]["n"] == "6"
    assert by_key[("mccbd_pool_selector", "overall", "best_generated_utility")]["n"] == "0"

    pairwise_rows = _rows(result["pairwise_comparisons_path"])
    assert set(pairwise_rows[0]) >= {
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
    }
    pairwise = {
        (row["baseline"], row["world_id"], row["metric"]): row
        for row in pairwise_rows
        if row["mechanism"] == "cmdgd"
    }
    for baseline in (
        "random_edit_generator",
        "random_feasible",
        "fixed_mix",
        "evidence_calibrated_ucb",
        "observed_recombination_baseline",
        "mccbd_pool_selector",
    ):
        row = pairwise[(baseline, "overall", "best_selected_utility")]
        assert row["n_pairs"] == "6"
        assert 0.0 <= float(row["win_fraction"]) <= 1.0
        assert float(row["paired_delta_ci95_low"]) <= float(row["paired_delta_mean"])
        assert float(row["paired_delta_ci95_high"]) >= float(row["paired_delta_mean"])


def test_cmdgd_generates_novel_non_observed_sequences_and_examples(tmp_path: Path) -> None:
    result = run_generative_benchmark(
        tmp_path,
        mechanisms=("cmdgd", "observed_recombination_baseline"),
        seeds=(0,),
        worlds=("ph_contrast",),
    )

    cmdgd_rows = [
        row for row in _rows(result["benchmark_results_path"]) if row["mechanism"] == "cmdgd"
    ]
    assert cmdgd_rows
    assert all(int(row["generated_novel_count"]) > 0 for row in cmdgd_rows)
    assert all(float(row["novel_sequence_rate"]) > 0.0 for row in cmdgd_rows)

    examples = _rows(result["design_examples_path"])
    assert any(
        row["generated_by"] == "cmdgd" and row["observed_duplicate"] == "false"
        for row in examples
    )


def test_selector_visible_inputs_do_not_contain_oracle_truth(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: list[Mapping[str, Any]] = []
    original = generative_benchmark._select_panel_for_mechanism

    def spy_select_panel(
        mechanism: str,
        observed: list[Mapping[str, Any]],
        candidates: list[Mapping[str, Any]],
        budget: int,
        rng,
        state: Any,
    ) -> list[str]:
        del observed
        captured.extend(dict(candidate) for candidate in candidates)
        return original(mechanism, [], candidates, budget, rng, state)

    monkeypatch.setattr(generative_benchmark, "_select_panel_for_mechanism", spy_select_panel)

    result = run_generative_benchmark(
        tmp_path,
        mechanisms=("cmdgd", "mccbd_pool_selector"),
        seeds=(0,),
        worlds=("ph_contrast",),
    )

    assert captured
    for candidate in captured:
        _assert_blind(candidate)
    for row in _rows(result["benchmark_results_path"]):
        _assert_blind(json.loads(row["selector_candidate_example"]))


def test_generative_benchmark_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    first = run_generative_benchmark(
        tmp_path,
        mechanisms=("cmdgd", "random_edit_generator", "single_edit_scan", "mccbd_pool_selector"),
        seeds=(0,),
        worlds=("ph_contrast", "escape_risk"),
    )
    snapshot = {
        name: Path(path).read_text(encoding="utf-8")
        for name, path in first["artifact_paths"].items()
    }

    second = run_generative_benchmark(
        tmp_path,
        mechanisms=("cmdgd", "random_edit_generator", "single_edit_scan", "mccbd_pool_selector"),
        seeds=(0,),
        worlds=("ph_contrast", "escape_risk"),
    )

    assert second["selected_mechanism"] == first["selected_mechanism"]
    assert {
        name: Path(path).read_text(encoding="utf-8")
        for name, path in second["artifact_paths"].items()
    } == snapshot


def test_summary_selects_generative_mechanism_not_observed_reuse_baseline(
    tmp_path: Path,
) -> None:
    result = run_generative_benchmark(
        tmp_path,
        mechanisms=(
            "cmdgd",
            "observed_recombination_baseline",
            "mccbd_pool_selector",
        ),
        seeds=(0, 1),
        worlds=("ph_contrast", "escape_risk"),
    )

    assert result["selected_mechanism"]
    assert result["selected_mechanism"] != "observed_recombination_baseline"

    summary_rows = _rows(result["summary_results_path"])
    selected_rows = [row for row in summary_rows if row["selected_mechanism"] == "true"]
    assert len(selected_rows) == 1
    assert selected_rows[0]["mechanism"] == result["selected_mechanism"]
    assert selected_rows[0]["observed_reuse_only"] == "false"


def test_vocabulary_extension_world_rewards_true_design_space_expansion(
    tmp_path: Path,
) -> None:
    result = run_generative_benchmark(
        tmp_path,
        mechanisms=("cmdgd", "observed_recombination_baseline", "random_edit_generator"),
        seeds=(0, 1, 2),
        worlds=("vocabulary_extension",),
    )

    summary_rows = _rows(result["summary_results_path"])
    overall = {row["mechanism"]: row for row in summary_rows if row["world_id"] == "overall"}

    assert float(overall["cmdgd"]["mean_best_generated_utility"]) > float(
        overall["observed_recombination_baseline"]["mean_best_generated_utility"]
    )


def test_de_novo_site_world_requires_cmdgd_to_generate_unseen_mutation_site(
    tmp_path: Path,
) -> None:
    world = generative_benchmark._world_by_id("de_novo_site_generalization")
    records = {
        str(record["variant_id"]): record
        for record in generative_benchmark.generate_synthetic_sequence_world(
            "de_novo_site_generalization",
            seed=0,
        )
    }
    observed = generative_benchmark._initial_observed_records(world, records, seed=0)
    context = generative_benchmark._design_context(
        world,
        [generative_benchmark._observed_record(record, seed=0) for record in observed],
        seed=0,
        generation_budget=64,
        ablation="full",
    )

    observed_sites = generative_benchmark._mutation_sites_from_records(observed, world)
    visible_sites = {
        (str(edit["chain"]), int(edit["position"])) for edit in context["edit_vocabulary"]
    }
    assert ("H", 10) not in observed_sites
    assert ("H", 10) not in visible_sites

    result = run_generative_benchmark(
        tmp_path,
        mechanisms=(
            "cmdgd",
            "observed_recombination_baseline",
            "random_edit_generator",
            "single_edit_scan",
            "mccbd_pool_selector",
            "fixed_mix",
        ),
        seeds=(0,),
        worlds=("de_novo_site_generalization",),
        generation_budget=64,
    )

    rows = {
        row["mechanism"]: row
        for row in _rows(result["benchmark_results_path"])
    }
    assert set(rows["cmdgd"]) >= {
        "generated_new_site_count",
        "selected_new_site_count",
        "new_site_rate",
        "new_site_generation_capable",
    }

    assert rows["cmdgd"]["new_site_generation_capable"] == "true"
    assert int(rows["cmdgd"]["generated_new_site_count"]) > 0
    assert int(rows["cmdgd"]["selected_new_site_count"]) >= 0
    assert float(rows["cmdgd"]["new_site_rate"]) > 0.0

    for mechanism in (
        "observed_recombination_baseline",
        "random_edit_generator",
        "single_edit_scan",
        "mccbd_pool_selector",
        "fixed_mix",
    ):
        assert rows[mechanism]["new_site_generation_capable"] == "false"
        assert int(rows[mechanism]["generated_new_site_count"]) == 0
        assert int(rows[mechanism]["selected_new_site_count"]) == 0
        assert float(rows[mechanism]["new_site_rate"]) == 0.0


def test_cmdgd_ablation_results_include_de_novo_site_and_vocabulary_controls(
    tmp_path: Path,
) -> None:
    result = run_generative_benchmark(
        tmp_path,
        mechanisms=("cmdgd",),
        seeds=(0,),
        worlds=("vocabulary_extension", "de_novo_site_generalization"),
        generation_budget=64,
    )

    rows = _rows(result["ablation_results_path"])
    assert {
        "no_de_novo_site_proposal",
        "no_vocabulary_expansion",
        "no_site_or_vocabulary_generation",
    } <= {row["ablation"] for row in rows}
    by_key = {
        (row["world_id"], int(row["seed"]), row["ablation"]): row
        for row in rows
    }

    de_novo_full = by_key[("de_novo_site_generalization", 0, "full")]
    no_de_novo = by_key[("de_novo_site_generalization", 0, "no_de_novo_site_proposal")]
    assert int(de_novo_full["generated_new_site_count"]) > 0
    assert no_de_novo["new_site_generation_capable"] == "false"
    assert int(no_de_novo["generated_new_site_count"]) == 0
    assert int(no_de_novo["selected_new_site_count"]) == 0

    vocabulary_full = by_key[("vocabulary_extension", 0, "full")]
    no_vocabulary = by_key[("vocabulary_extension", 0, "no_vocabulary_expansion")]
    assert float(vocabulary_full["best_generated_utility"]) > float(
        no_vocabulary["best_generated_utility"]
    )
    assert float(vocabulary_full["best_selected_utility"]) > float(
        no_vocabulary["best_selected_utility"]
    )
    assert float(no_vocabulary["delta_from_full_best_generated_utility"]) > 0.0
    assert float(no_vocabulary["delta_from_full_best_selected_utility"]) > 0.0


def test_cmdgd_component_ablations_use_cmdgd_scorer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from design_scientist.algorithms import cmdgd

    calls: list[tuple[bool, bool, bool, bool, bool]] = []
    original_score_candidates = cmdgd.score_candidates

    def spy_score_candidates(state: Any, candidates: list[Mapping[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
        algorithm_state = state.get("algorithm_state") if isinstance(state, Mapping) else state
        calls.append(
            (
                bool(algorithm_state.config.enable_contrastive_objective),
                bool(algorithm_state.config.enable_grammar_recombination),
                bool(algorithm_state.config.enable_novelty_uncertainty),
                bool(algorithm_state.config.enable_vocabulary_expansion),
                bool(algorithm_state.config.enable_de_novo_site_proposal),
            )
        )
        return original_score_candidates(state, candidates, **kwargs)

    monkeypatch.setattr(cmdgd, "score_candidates", spy_score_candidates)

    result = run_generative_benchmark(
        tmp_path,
        mechanisms=("cmdgd",),
        seeds=(0,),
        worlds=("ph_contrast",),
    )

    assert (False, True, True, True, True) in calls
    assert (True, False, True, True, True) in calls
    assert (True, True, False, True, True) in calls
    assert (True, True, True, False, True) in calls
    assert (True, True, True, True, False) in calls
    assert (True, True, True, False, False) in calls

    rows = _rows(result["ablation_results_path"])
    ablations = {row["ablation"] for row in rows}
    assert {
        "full",
        "no_contrastive_objective",
        "no_grammar_recombination",
        "no_novelty_uncertainty",
        "no_vocabulary_expansion",
        "no_de_novo_site_proposal",
        "no_site_or_vocabulary_generation",
        "benchmark_no_generation_boundary",
    } <= ablations
    assert "no_ph_contrast_objective" not in ablations
    assert "no_constraint_guardrail" not in ablations
    for row in rows:
        if row["ablation"] == "benchmark_no_generation_boundary":
            assert row["ablation_type"] == "benchmark_boundary"
            assert row["fixed_pool_only"] == "true"
        else:
            assert row["ablation_type"] == "cmdgd_component"
            assert row["cmdgd_adapter"] == "design_scientist.algorithms.cmdgd.CMDGDLifecycle"


def test_fixed_pool_reports_selected_utility_without_generated_utility(
    tmp_path: Path,
) -> None:
    result = run_generative_benchmark(
        tmp_path,
        mechanisms=("cmdgd", "mccbd_pool_selector"),
        seeds=(0,),
        worlds=("ph_contrast",),
    )

    rows = {
        row["mechanism"]: row
        for row in _rows(result["benchmark_results_path"])
    }
    fixed_pool = rows["mccbd_pool_selector"]

    assert fixed_pool["fixed_pool_only"] == "true"
    assert fixed_pool["candidate_space_type"] == "fixed_pool"
    assert fixed_pool["generation_counted"] == "false"
    assert fixed_pool["generated_utility_applicable"] == "false"
    assert fixed_pool["generated_count"] == "0"
    assert fixed_pool["best_generated_utility"] == ""
    assert float(fixed_pool["best_selected_utility"]) > 0.0

    overall = {
        row["mechanism"]: row
        for row in _rows(result["summary_results_path"])
        if row["world_id"] == "overall"
    }
    assert overall["mccbd_pool_selector"]["mean_best_generated_utility"] == "0.0"
    assert float(overall["mccbd_pool_selector"]["mean_best_selected_utility"]) > 0.0


def test_gate_report_separates_generative_selection_from_utility_ranking(
    tmp_path: Path,
) -> None:
    result = run_generative_benchmark(
        tmp_path,
        mechanisms=("cmdgd", "mccbd_pool_selector"),
        seeds=(0, 1, 2),
        worlds=("ph_contrast",),
    )

    gate_report = json.loads(Path(result["selection_gate_report_path"]).read_text(encoding="utf-8"))

    assert gate_report["selection_basis"] == "generative_candidate_space_expansion"
    assert gate_report["selected_mechanism"] == "cmdgd"
    assert gate_report["utility_rank_basis"] == "mean_best_selected_utility"
    assert gate_report["comparative_utility_rankings"]

    summary = {
        row["mechanism"]: row
        for row in _rows(result["summary_results_path"])
        if row["world_id"] == "overall"
    }
    assert summary["cmdgd"]["selected_mechanism"] == "true"
    assert summary["cmdgd"]["comparative_utility_rank"]

    fixed_pool_beats_cmdgd = generative_benchmark._selection_gate_report(
        [
            {
                "mechanism": "cmdgd",
                "world_id": "overall",
                "mean_best_generated_utility": 0.70,
                "mean_best_selected_utility": 0.70,
                "mean_pH_contrast_score": 0.50,
                "mean_constraint_pass_rate": 1.0,
                "mean_false_claim_rate": 0.0,
                "mean_generated_novel_count": 12.0,
                "mean_novel_sequence_rate": 1.0,
                "observed_reuse_only": False,
                "fixed_pool_only": False,
                "selected_eligible": True,
            },
            {
                "mechanism": "mccbd_pool_selector",
                "world_id": "overall",
                "mean_best_generated_utility": 0.0,
                "mean_best_selected_utility": 0.90,
                "mean_pH_contrast_score": 0.40,
                "mean_constraint_pass_rate": 1.0,
                "mean_false_claim_rate": 0.0,
                "mean_generated_novel_count": 0.0,
                "mean_novel_sequence_rate": 0.0,
                "observed_reuse_only": False,
                "fixed_pool_only": True,
                "selected_eligible": False,
            },
        ]
    )

    assert fixed_pool_beats_cmdgd["selected_mechanism"] == "cmdgd"
    assert fixed_pool_beats_cmdgd["utility_winner"] == "mccbd_pool_selector"
    assert fixed_pool_beats_cmdgd["selected_utility_rank"] > 1

    utility_rankings = {
        row["mechanism"]: row
        for row in fixed_pool_beats_cmdgd["comparative_utility_rankings"]
    }
    assert utility_rankings["mccbd_pool_selector"]["rank"] == 1
    assert utility_rankings["mccbd_pool_selector"]["utility_exceeds_selected"] is True
    assert utility_rankings["mccbd_pool_selector"]["eligible_for_generative_selection"] is False
