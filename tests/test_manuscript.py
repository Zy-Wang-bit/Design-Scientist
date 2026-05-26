from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path

from design_scientist.cli import main
from design_scientist.algorithm_benchmark import run_mccbd_benchmark
from design_scientist.manuscript import (
    _bar_svg,
    generate_algorithm_manuscript,
    generate_short_paper,
    validate_paper_readiness,
)
from design_scientist.method_report import write_method_report
from design_scientist.project_replay import run_project_masking_benchmark
from test_framework_validation_v3 import _build_v3_framework_run
from test_project_replay import write_project_fixture


def _legacy_wet_lab_proof_phrase() -> str:
    return " ".join(("not", "prospective", "wet-lab", "proof"))


def _legacy_wet_lab_proof_sentence() -> str:
    return f"This is {_legacy_wet_lab_proof_phrase()}"


def test_generate_short_paper_writes_readiness_checked_bundle(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)
    write_method_report(project, run_id="v3_unit")

    result = generate_short_paper(project, run_id="v3_unit")

    paper_dir = project / "runs" / "v3_unit" / "paper"
    expected_files = {
        "short_paper.md",
        "references.bib",
        "results_summary.json",
        "claim_evidence_map.json",
        "reproducibility.md",
        "data_availability.md",
        "code_availability.md",
        "paper_readiness_report.json",
    }
    assert set(result["artifacts"]) == expected_files
    for filename in expected_files:
        assert (paper_dir / filename).exists()

    short_paper = (paper_dir / "short_paper.md").read_text(encoding="utf-8")
    assert "## Abstract" in short_paper
    assert "## Evidence Boundary" in short_paper
    assert "computational and synthetic replay evidence" in short_paper
    assert "wet-lab validation" in short_paper
    assert "does not report wet-lab validation" in short_paper
    assert "[@paper_active_design]" in short_paper
    assert "[mechanism_benchmark_summary.csv](../mechanism_benchmark_summary.csv)" in short_paper
    assert "prospective wet-lab superiority" not in short_paper

    references = (paper_dir / "references.bib").read_text(encoding="utf-8")
    assert "@misc{paper_active_design" in references

    summary = json.loads((paper_dir / "results_summary.json").read_text(encoding="utf-8"))
    assert summary["run_id"] == "v3_unit"
    assert summary["evidence_scope"] == "computational_synthetic_only"
    assert summary["no_wet_lab_validation_claimed"] is True
    assert summary["selected_mechanism"] == "literature_kernel"

    claim_map = json.loads((paper_dir / "claim_evidence_map.json").read_text(encoding="utf-8"))
    assert claim_map["evidence_scope"] == "computational_synthetic_only"
    assert claim_map["claims"]
    assert all(claim["evidence_type"] == "computational/synthetic" for claim in claim_map["claims"])

    readiness = json.loads((paper_dir / "paper_readiness_report.json").read_text(encoding="utf-8"))
    assert readiness["valid"]
    assert readiness["summary"]["errors"] == 0


def test_generate_short_paper_has_publishable_algorithm_manuscript_structure(
    tmp_path: Path,
) -> None:
    project = _build_v3_framework_run(tmp_path)
    write_method_report(project, run_id="v3_unit")

    generate_short_paper(project, run_id="v3_unit")

    short_paper = (
        project / "runs" / "v3_unit" / "paper" / "short_paper.md"
    ).read_text(encoding="utf-8")
    for section in (
        "## Formal Problem Setup",
        "## Literature Basis",
        "## Algorithmic Contribution",
        "## Evaluation Design",
        "## Contribution Boundaries",
    ):
        assert section in short_paper

    assert "design space" in short_paper
    assert "observations" in short_paper
    assert "objectives" in short_paper
    assert "constraints" in short_paper
    assert "baselines" in short_paper
    assert "ablations" in short_paper
    assert "real algorithmic contribution" in short_paper
    assert "does not establish real experimental performance" in short_paper


def test_generate_short_paper_ingests_project_masking_artifacts_when_present(
    tmp_path: Path,
) -> None:
    project = _build_v3_framework_run(tmp_path)
    run_dir = project / "runs" / "v3_unit"
    write_method_report(project, run_id="v3_unit")
    _write_csv(
        run_dir / "project_masking_benchmark_summary.csv",
        [
            {
                "mechanism": "evidence_calibrated_ucb",
                "fold_count": "2",
                "completed_fold_count": "2",
                "mean_best_feasible_utility": "0.810",
                "mean_hit_rate": "0.750",
                "mean_regret_proxy": "0.120",
                "mean_false_claim_rate": "0.100",
                "mean_evidence_coverage": "0.500",
                "failed_fold_count": "0",
            },
            {
                "mechanism": "random_feasible",
                "fold_count": "2",
                "completed_fold_count": "2",
                "mean_best_feasible_utility": "0.420",
                "mean_hit_rate": "0.250",
                "mean_regret_proxy": "0.440",
                "mean_false_claim_rate": "0.300",
                "mean_evidence_coverage": "0.500",
                "failed_fold_count": "0",
            },
        ],
    )
    _write_csv(
        run_dir / "project_masking_benchmark_results.csv",
        [
            {
                "fold_id": "kfold_2_1",
                "mechanism": "evidence_calibrated_ucb",
                "status": "completed",
                "best_feasible_utility": "0.82",
                "hit_rate": "1.0",
                "regret_proxy": "0.1",
                "false_claim_rate": "0.0",
                "evidence_coverage": "0.5",
            }
        ],
    )
    _write_csv(
        run_dir / "project_masking_ablation_results.csv",
        [
            {
                "fold_id": "kfold_2_1",
                "mechanism": "evidence_calibrated_ucb",
                "ablation": "none",
                "best_feasible_utility": "0.82",
                "delta_from_full_best_feasible_utility": "0.0",
                "false_claim_rate": "0.0",
            }
        ],
    )
    (run_dir / "project_masking_config.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark": "project_data_observed_pool_masking",
                "folds": "kfold_2",
                "mechanisms": ["evidence_calibrated_ucb", "random_feasible"],
                "budget": 2,
                "evidence_scope": "retrospective_masked_project_data",
                "leakage_controls": {
                    "score_only_masked_heldout_outcomes": True,
                    "observed_pool_only": True,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    generate_short_paper(project, run_id="v3_unit")

    paper_dir = run_dir / "paper"
    short_paper = (paper_dir / "short_paper.md").read_text(encoding="utf-8")
    assert "retrospective masked project-data evidence" in short_paper
    assert "observed-pool masking" in short_paper
    assert "does not constitute prospective wet-lab validation" in short_paper
    assert _legacy_wet_lab_proof_phrase() not in short_paper
    assert "[project_masking_benchmark_summary.csv](../project_masking_benchmark_summary.csv)" in short_paper
    assert "[project_masking_benchmark_results.csv](../project_masking_benchmark_results.csv)" in short_paper
    assert "[project_masking_ablation_results.csv](../project_masking_ablation_results.csv)" in short_paper
    assert "[project_masking_config.json](../project_masking_config.json)" in short_paper

    summary = json.loads((paper_dir / "results_summary.json").read_text(encoding="utf-8"))
    assert summary["evidence_scope"] == "computational_synthetic_and_retrospective_masked_project_data"
    assert summary["project_masking"]["available"] is True
    assert summary["project_masking"]["summary_row_count"] == 2
    assert summary["project_masking"]["result_row_count"] == 1
    assert summary["project_masking"]["ablation_row_count"] == 1
    assert summary["project_masking"]["config"]["evidence_scope"] == "retrospective_masked_project_data"

    claim_map = json.loads((paper_dir / "claim_evidence_map.json").read_text(encoding="utf-8"))
    assert any(
        claim["evidence_type"] == "retrospective/masked_project_data"
        for claim in claim_map["claims"]
    )


def test_validate_paper_readiness_reports_missing_sections_citations_and_artifact_links(
    tmp_path: Path,
) -> None:
    project = _build_v3_framework_run(tmp_path)
    paper_dir = project / "runs" / "v3_unit" / "paper"
    paper_dir.mkdir(parents=True)
    (paper_dir / "short_paper.md").write_text(
        "# Incomplete Paper\n\n"
        "## Abstract\n\n"
        "Unsupported text with a missing citation [@missing_key] and "
        "[missing artifact](../missing_artifact.json).\n",
        encoding="utf-8",
    )
    (paper_dir / "references.bib").write_text(
        "@misc{different_key,\n  title = {Different paper}\n}\n",
        encoding="utf-8",
    )

    report = validate_paper_readiness(project, run_id="v3_unit")

    assert not report["valid"]
    error_codes = {finding["code"] for finding in report["findings"] if finding["severity"] == "error"}
    assert "missing_section" in error_codes
    assert "missing_citation_key" in error_codes
    assert "missing_artifact_link" in error_codes


def test_generate_algorithm_manuscript_centers_real_algorithm_and_masked_project_evidence(
    tmp_path: Path,
) -> None:
    project = _build_v3_framework_run(tmp_path)
    fixture_project = write_project_fixture(project)
    assert fixture_project == project
    run_project_masking_benchmark(
        project,
        run_id="algorithm_unit",
        budget=2,
        folds="kfold_2",
    )

    result = generate_algorithm_manuscript(
        project,
        run_id="algorithm_unit",
        algorithm="evidence_calibrated_ucb",
    )

    paper_dir = project / "runs" / "algorithm_unit" / "paper"
    manuscript = (paper_dir / "manuscript.md").read_text(encoding="utf-8")
    assert result["artifacts"]["manuscript.md"] == str(paper_dir / "manuscript.md")
    assert "Evidence-Calibrated UCB" in manuscript
    assert "real algorithmic contribution" in manuscript
    assert "observed-neighborhood calibration" in manuscript
    assert "motif enrichment" in manuscript
    assert "batch diversity" in manuscript
    assert "retrospective masked project-data evidence" in manuscript
    assert "does not constitute prospective wet-lab validation" in manuscript
    assert _legacy_wet_lab_proof_phrase() not in manuscript
    assert "[project_masking_benchmark_summary.csv](../project_masking_benchmark_summary.csv)" in manuscript
    assert "prospective wet-lab superiority" not in manuscript

    summary = json.loads((paper_dir / "algorithm_results_summary.json").read_text(encoding="utf-8"))
    summary_json = json.dumps(summary)
    assert "/Users/" not in summary_json
    assert str(tmp_path) not in summary_json
    assert summary["algorithm"] == "evidence_calibrated_ucb"
    assert summary["evidence_scope"] == "retrospective_masked_project_data"
    assert summary["paper_card_count"] >= 1
    assert summary["algorithm_summary_row"]["mechanism"] == "evidence_calibrated_ucb"
    assert summary["baseline_comparisons"]


def test_generate_algorithm_manuscript_centers_mccbd_benchmark_with_figures_and_case_study(
    tmp_path: Path,
) -> None:
    project = _build_v3_framework_run(tmp_path)
    write_project_fixture(project)
    run_mccbd_benchmark(
        project,
        run_id="mccbd_algorithm_unit",
        seeds=(0, 1),
        rounds=2,
        budget=3,
    )
    run_project_masking_benchmark(
        project,
        run_id="mccbd_project_case",
        budget=2,
        folds="kfold_2",
        mechanisms=["mccbd", "evidence_calibrated_ucb", "random_feasible", "fixed_mix"],
    )

    result = generate_algorithm_manuscript(
        project,
        run_id="mccbd_algorithm_unit",
        algorithm="mccbd",
    )

    paper_dir = project / "runs" / "mccbd_algorithm_unit" / "paper"
    manuscript = (paper_dir / "manuscript.md").read_text(encoding="utf-8")
    assert result["readiness"]["valid"]
    assert "Mechanism-Calibrated Constrained Bayesian Design" in manuscript
    assert "constrained expected improvement" in manuscript
    assert "Bayesian posterior" in manuscript
    assert "algorithmic benchmark" in manuscript
    assert "1E62 retrospective masking case study" in manuscript
    assert "does not constitute prospective wet-lab validation" in manuscript
    assert _legacy_wet_lab_proof_phrase() not in manuscript
    assert "[Figure 1](figures/figure_1_mccbd_workflow.svg)" in manuscript
    assert "[Figure 2](figures/figure_2_benchmark_summary.svg)" in manuscript
    assert "[Figure 3](figures/figure_3_ablation_summary.svg)" in manuscript

    for filename in (
        "figure_1_mccbd_workflow.svg",
        "figure_2_benchmark_summary.svg",
        "figure_3_ablation_summary.svg",
        "figure_4_project_case_study.svg",
    ):
        assert (paper_dir / "figures" / filename).exists()

    summary = json.loads((paper_dir / "algorithm_results_summary.json").read_text(encoding="utf-8"))
    summary_json = json.dumps(summary)
    assert "/Users/" not in summary_json
    assert str(tmp_path) not in summary_json
    assert summary["algorithm"] == "mccbd"
    assert summary["mccbd_benchmark"]["available"] is True
    assert summary["mccbd_benchmark"]["selected_mechanism"] == "mccbd"
    assert summary["mccbd_benchmark"]["selected_passes_gate"] is True
    assert summary["project_masking"]["available"] is True


def test_generate_algorithm_manuscript_builds_cmdgd_paper_bundle_from_v6_literature(
    tmp_path: Path,
) -> None:
    project = _copy_cmdgd_fixture(tmp_path)

    result = generate_algorithm_manuscript(
        project,
        run_id="cmdgd_paper_v6",
        algorithm="cmdgd",
    )

    paper_dir = project / "runs" / "cmdgd_paper_v6" / "paper"
    manuscript_path = paper_dir / "manuscript.md"
    references_path = paper_dir / "references.bib"
    readiness_path = paper_dir / "paper_readiness_report.json"
    assert manuscript_path.exists()
    assert references_path.exists()
    assert readiness_path.exists()

    figures = sorted((paper_dir / "figures").glob("*.svg"))
    tables = sorted((paper_dir / "tables").glob("*.csv"))
    assert len(figures) >= 4
    assert len(tables) >= 3
    for expected in (
        "figure_1_cmdgd_workflow.svg",
        "figure_2_cmdgd_benchmark_comparison.svg",
        "figure_3_cmdgd_vocabulary_extension.svg",
        "figure_4_cmdgd_ablation_claim_boundary.svg",
        "figure_5_cmdgd_seed_level_stability.svg",
    ):
        assert (paper_dir / "figures" / expected).exists()
    for expected in (
        "related_work_matrix.csv",
        "benchmark_summary.csv",
        "statistical_summary.csv",
        "pairwise_comparisons.csv",
        "design_examples.csv",
        "project_candidate_rationale.csv",
    ):
        assert (paper_dir / "tables" / expected).exists()

    manuscript = manuscript_path.read_text(encoding="utf-8")
    abstract = manuscript.split("## Introduction", 1)[0]
    results_boundary = manuscript.split("## Results", 1)[1].split("## Statistical Stability", 1)[0]
    assert "pH-Sensitive Antibody Design" in manuscript
    assert "## Literature Review" in manuscript or "## Related Work" in manuscript
    assert "learns an edit grammar" in manuscript
    assert "design vocabulary" in manuscript
    assert "mechanism priors" in manuscript
    assert "grammar-guided combinatorial generator" in manuscript
    assert "mechanism-prior grammar expansion" in manuscript
    assert "vocabulary-guided recombination" in manuscript
    assert "grammar recombination" in manuscript
    assert "new heavy/light chain sequences" in manuscript
    assert "selects a panel" in manuscript
    assert "Algorithm 1" in manuscript
    assert "Algorithm 2" in manuscript
    assert "grammar learning" in manuscript
    assert "candidate enumeration" in manuscript
    assert "scoring" in manuscript
    assert "selection" in manuscript
    assert "tie-breaking" in manuscript
    assert "generation_budget" in manuscript
    assert "generated novel count" in manuscript
    assert "best generated utility" in manuscript
    assert "constraint pass rate" in manuscript
    assert "pH contrast" in manuscript
    assert "novel sequence rate" in manuscript
    assert "false claim rate" in manuscript
    assert "observed recombination" in manuscript
    assert "fixed-pool" in manuscript
    assert "mccbd_pool_selector" in manuscript
    assert "observed recombination is a strong baseline" in manuscript
    assert "fixed-pool selectors may exceed CMD-GD utility" in manuscript
    assert "ablation rows support the comparison" in manuscript
    assert "endpoint gap" in manuscript
    assert "decision-policy gap" in manuscript
    assert "representation gap" in manuscript
    assert "alignment gap" in manuscript
    assert "evaluation gap" in manuscript
    assert "AntBO-like constrained search" in manuscript
    assert "protein language models" in manuscript
    assert "structure-aware generation" in manuscript
    assert "auditable grammar-guided expansion" in abstract
    assert "does not claim broad selected-utility superiority" in abstract
    assert "did not outperform the best observed-pool selector" in abstract
    assert "Real-data masking boundary" in results_boundary
    assert "did not outperform the best observed-pool selector" in results_boundary
    assert "Figure 2 caption:" in manuscript
    assert "Figure 5 caption:" in manuscript
    assert "n =" in manuscript
    assert "95% CI" in manuscript
    assert _legacy_wet_lab_proof_sentence() not in manuscript
    assert _legacy_wet_lab_proof_phrase() not in manuscript
    assert "does not constitute prospective wet-lab validation" in manuscript
    assert "benchmark no-generation boundary" in manuscript
    assert "fixed-pool benchmark boundary" in manuscript
    assert "Removing generation" not in manuscript
    assert "benchmark includes vocabulary-extension stress tests" in manuscript
    assert "external mechanism-vocabulary edits recorded" in manuscript
    assert "## Statistical Stability" in manuscript
    assert "## Real-Data Retrospective Masking" in manuscript
    assert "## Biophysical Plausibility and Panel Rationale" in manuscript
    assert "observed-recombination boundary is intentionally strong" in manuscript
    assert "not simply the top five ranks" in manuscript
    assert "plausibility screens rather than proof of pH switching" in manuscript
    assert "CMD-GD did not outperform the best observed-pool selector" in manuscript
    assert "## New Mutation-Site Generation" not in manuscript
    for forbidden in (
        "v4",
        "artifact",
        "trace",
        "current run",
        "Reviewer-facing",
        "selected_passes_gate",
        "selected_eligible",
        "observed_reuse_only",
        "fixed_pool_only",
        "generative selection gate passed",
        "selected mechanism:",
        "generative_benchmark_results.csv",
        "generative_benchmark_config.json",
        "generative_selection_gate_report.json",
        "cmdgd_project_generated_candidates.csv",
        "cmdgd_project_design_summary.json",
        "BibTeX references are provided",
        "local outputs",
        "manuscript generator",
        "supplementary run files",
        "run identifier",
        "uv run",
        "<project_dir>",
        "src/design_scientist",
        "/Users/",
    ):
        assert forbidden not in manuscript
    for raw_result_bullet in (
        "- generated novel count:",
        "- best generated utility:",
        "- constraint pass rate:",
        "- novel sequence rate:",
        "- false claim rate:",
        "- benchmark_no_generation_boundary:",
        "- no_contrastive_objective:",
        "- no_grammar_recombination:",
    ):
        assert raw_result_bullet not in manuscript
    assert "CMD-GD generated 32.0 novel candidates per benchmark replicate on average" in manuscript
    assert "constraint pass rate was 1.0" in manuscript
    assert "false-claim rate was 0.0" in manuscript
    assert "selected-panel utility rank was 1 among eligible mechanisms" in manuscript
    assert "not as a general utility-dominance claim" in manuscript
    assert "2432 standardized observation rows overall" in manuscript
    assert "2380 endpoint rows linked to the 20 observed sequence records" in manuscript
    assert "52 Ae KD-ratio rows" in manuscript
    assert "The cited literature is listed in" in manuscript
    assert "[Figure 1](figures/figure_1_cmdgd_workflow.svg)" in manuscript
    assert "[benchmark_summary.csv](tables/benchmark_summary.csv)" in manuscript
    assert "[design_examples.csv](tables/design_examples.csv)" in manuscript
    assert "## 1E62 Computational Design Output" in manuscript

    selected_table = paper_dir / "tables" / "project_selected_candidates.csv"
    assert selected_table.exists()
    selected_rows = list(csv.DictReader(selected_table.open(encoding="utf-8")))
    assert selected_rows
    assert "cmdgd_trace" not in selected_rows[0]
    assert any(row.get("selected", "").lower() == "true" for row in selected_rows)
    rationale_table = paper_dir / "tables" / "project_candidate_rationale.csv"
    rationale_rows = list(csv.DictReader(rationale_table.open(encoding="utf-8")))
    assert rationale_rows
    assert {row["rank"] for row in rationale_rows} >= {"1", "6", "8", "20", "59"}
    assert any(row["histidine_edit_count"] != "0" for row in rationale_rows)
    assert all(row["selection_rationale"] for row in rationale_rows)

    citation_keys = set(re.findall(r"@([A-Za-z0-9_:.+-]+)", manuscript))
    assert len(citation_keys) >= 8

    source_bib = (
        _repo_root()
        / "projects"
        / "1e62_startup_data"
        / "framework"
        / "v4_references.bib"
    ).read_text(encoding="utf-8")
    references = references_path.read_text(encoding="utf-8")
    assert references == source_bib
    assert len(re.findall(r"@\w+\{", references)) >= 15

    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    assert result["readiness"]["valid"]
    assert readiness["valid"]

    summary = json.loads((paper_dir / "algorithm_results_summary.json").read_text(encoding="utf-8"))
    summary_json = json.dumps(summary)
    assert "/Users/" not in summary_json
    assert str(tmp_path) not in summary_json
    assert summary["algorithm"] == "cmdgd"
    assert summary["generative_benchmark"]["available"] is True
    assert summary["generative_benchmark"]["selected_mechanism"] == "cmdgd"
    assert summary["generative_benchmark"]["selected_passes_gate"] is True
    assert summary["generative_benchmark"]["top_summary_row"]["mean_generated_novel_count"] == "32.0"
    assert summary["generative_benchmark"]["statistical_summary_row_count"] > 0
    assert summary["generative_benchmark"]["pairwise_comparison_row_count"] > 0
    fixed_pool_comparison = next(
        item
        for item in summary["baseline_comparisons"]
        if item["baseline"] == "mccbd_pool_selector"
    )
    assert fixed_pool_comparison["generated_utility_delta"] is None
    assert fixed_pool_comparison["selected_utility_delta"] == "+0.018369"


def test_cmdgd_bar_svg_clamps_negative_widths() -> None:
    svg = _bar_svg(
        "Signed comparison",
        [
            {"label": "negative", "value": "-0.25"},
            {"label": "positive", "value": "0.50"},
        ],
        label_key="label",
        value_key="value",
        color="#0ea5e9",
    )

    assert 'width="-"' not in svg
    assert 'width="-280"' not in svg


def test_generate_algorithm_manuscript_reports_cmdgd_new_site_metrics_when_artifacts_exist(
    tmp_path: Path,
) -> None:
    project = _write_minimal_cmdgd_new_site_fixture(tmp_path)

    generate_algorithm_manuscript(
        project,
        run_id="cmdgd_new_sites",
        algorithm="cmdgd",
    )

    manuscript = (
        project / "runs" / "cmdgd_new_sites" / "paper" / "manuscript.md"
    ).read_text(encoding="utf-8")
    assert "## New Mutation-Site Generation" not in manuscript
    assert "mechanism-prior grammar expansion" in manuscript
    assert "Design-space diagnostics" in manuscript
    assert "not a standalone algorithmic feature" in manuscript
    assert "grammar recombination" in manuscript
    assert "mean unobserved-edit-position count was 7.0" in manuscript
    assert "mean unobserved-edit-position rate was 0.35" in manuscript
    assert "no_de_novo_site_proposal" in manuscript
    assert "experimental activity, structural tolerance, and pH-sensitive release remain to be measured prospectively" in manuscript
    assert "does not constitute prospective wet-lab validation" in manuscript
    assert _legacy_wet_lab_proof_phrase() not in manuscript


def test_cli_generate_short_paper_writes_bundle(tmp_path: Path, capsys) -> None:
    project = _build_v3_framework_run(tmp_path)
    write_method_report(project, run_id="v3_unit")

    exit_code = main(["generate-short-paper", str(project), "--run-id", "v3_unit"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Wrote short paper:" in output
    assert "Paper readiness passed:" in output
    assert (project / "runs" / "v3_unit" / "paper" / "short_paper.md").exists()


def test_cli_generate_algorithm_paper_writes_algorithm_manuscript(tmp_path: Path, capsys) -> None:
    project = _build_v3_framework_run(tmp_path)
    write_project_fixture(project)
    run_project_masking_benchmark(project, run_id="algorithm_unit", budget=2, folds="kfold_2")

    exit_code = main(
        [
            "generate-algorithm-paper",
            str(project),
            "--run-id",
            "algorithm_unit",
            "--algorithm",
            "evidence_calibrated_ucb",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Wrote algorithm manuscript:" in output
    assert (project / "runs" / "algorithm_unit" / "paper" / "manuscript.md").exists()


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _copy_cmdgd_fixture(tmp_path: Path) -> Path:
    source_project = _repo_root() / "projects" / "1e62_startup_data"
    project = tmp_path / "cmdgd_project"
    framework_dir = project / "framework"
    standardized_dir = project / "standardized"
    source_run_id = "cmdgd_paper_v6"
    run_dir = project / "runs" / source_run_id
    framework_dir.mkdir(parents=True)
    standardized_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    for filename in (
        "v4_literature_review.md",
        "v4_references.bib",
        "v4_related_work_matrix.csv",
        "v4_literature_trace.json",
    ):
        shutil.copyfile(source_project / "framework" / filename, framework_dir / filename)
    for filename in (
        "observations_long.csv",
        "project_context.json",
        "variant_sequences.csv",
    ):
        shutil.copyfile(source_project / "standardized" / filename, standardized_dir / filename)
    for filename in (
        "generative_benchmark_results.csv",
        "generative_benchmark_summary.csv",
        "generative_ablation_results.csv",
        "generative_design_examples.csv",
        "generative_benchmark_config.json",
        "generative_selection_gate_report.json",
        "cmdgd_project_generated_candidates.csv",
        "cmdgd_project_design_summary.json",
        "project_masking_benchmark_summary.csv",
        "project_masking_benchmark_results.csv",
        "project_masking_ablation_results.csv",
        "project_masking_config.json",
    ):
        shutil.copyfile(source_project / "runs" / source_run_id / filename, run_dir / filename)
    return project


def _write_minimal_cmdgd_new_site_fixture(tmp_path: Path) -> Path:
    project = tmp_path / "cmdgd_new_site_project"
    run_dir = project / "runs" / "cmdgd_new_sites"
    run_dir.mkdir(parents=True)
    _write_csv(
        run_dir / "generative_benchmark_summary.csv",
        [
            {
                "mechanism": "cmdgd",
                "world_id": "overall",
                "replicate_count": "2",
                "mean_generated_novel_count": "18.0",
                "mean_best_generated_utility": "0.73",
                "mean_best_selected_utility": "0.69",
                "mean_best_selected_pH_contrast_score": "0.61",
                "mean_constraint_pass_rate": "1.0",
                "mean_pH_contrast_score": "0.61",
                "mean_novel_sequence_rate": "1.0",
                "mean_false_claim_rate": "0.0",
                "mean_generated_new_mutation_site_count": "7.0",
                "mean_new_site_generation_rate": "0.35",
                "selected_eligible": "true",
                "selected_mechanism": "true",
                "comparative_utility_rank": "1",
            },
            {
                "mechanism": "observed_recombination_baseline",
                "world_id": "overall",
                "mean_best_generated_utility": "0.64",
                "mean_best_selected_utility": "0.62",
                "mean_novel_sequence_rate": "0.8",
                "mean_false_claim_rate": "0.0",
                "mean_generated_new_mutation_site_count": "0.0",
                "mean_new_site_generation_rate": "0.0",
                "observed_reuse_only": "true",
            },
        ],
    )
    _write_csv(
        run_dir / "generative_ablation_results.csv",
        [
            {
                "status": "completed",
                "seed": "0",
                "world_id": "new_site_context",
                "mechanism": "cmdgd",
                "ablation": "full",
                "generated_novel_count": "18",
                "generated_new_mutation_site_count": "7",
                "new_site_generation_rate": "0.35",
                "best_selected_utility": "0.69",
                "pH_contrast_score": "0.61",
                "false_claim_rate": "0.0",
                "delta_from_full_best_selected_utility": "0.0",
            },
            {
                "status": "completed",
                "seed": "0",
                "world_id": "new_site_context",
                "mechanism": "cmdgd",
                "ablation": "no_de_novo_site_proposal",
                "ablation_component": "de_novo_site_proposal",
                "generated_novel_count": "11",
                "generated_new_mutation_site_count": "0",
                "new_site_generation_rate": "0.0",
                "best_selected_utility": "0.58",
                "pH_contrast_score": "0.49",
                "false_claim_rate": "0.0",
                "delta_from_full_best_selected_utility": "0.11",
            },
        ],
    )
    (run_dir / "generative_benchmark_config.json").write_text(
        json.dumps(
            {
                "selected_mechanism": "cmdgd",
                "selected_passes_gate": True,
                "budget": 3,
                "generation_budget": 18,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "generative_selection_gate_report.json").write_text(
        json.dumps({"selected_utility_rank": 1}, indent=2) + "\n",
        encoding="utf-8",
    )
    return project


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
