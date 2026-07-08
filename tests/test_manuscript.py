from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

from design_scientist.cli import main
from design_scientist.algorithm_benchmark import run_mccbd_benchmark
from design_scientist.manuscript import (
    PH_SWITCH_GRAPH_RELEASE_SOURCE_FILES,
    _algorithm_claim_evidence_map,
    _bar_svg,
    _ph_switch_graph_ablation_summary_sentence,
    _ph_switch_graph_algorithm_formal_definition,
    _ph_switch_graph_candidate_rationale_rows,
    _ph_switch_graph_candidate_annotation_rows,
    _ph_switch_graph_claim_boundary_analysis,
    _ph_switch_graph_configuration_contract_rows,
    _ph_switch_graph_external_antibody_rows,
    _ph_switch_graph_external_benchmark_sentence,
    _ph_switch_graph_external_ph_abstract_clause,
    _ph_switch_graph_external_ph_benchmark_sentence,
    _ph_switch_graph_pareto_claim_boundary_rows,
    _ph_switch_graph_developability_risk_summary,
    _ph_switch_graph_quality_metric_rows,
    _ph_switch_graph_quality_metric_sentence,
    _render_submission_readiness_checklist,
    _ph_switch_graph_world_block_rows,
    _ph_switch_graph_world_block_sentence,
    _render_references_bib,
    _validate_bioinformatics_render_artifacts,
    _validate_ph_switch_graph_submission_metadata,
    _write_ph_switch_graph_supplementary_package,
    _load_cmdgd_literature_artifacts,
    _load_external_antibody_benchmark_artifacts,
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


def test_cmdgd_literature_artifacts_fall_back_to_v3_paper_cards(tmp_path: Path) -> None:
    project = tmp_path / "project"
    framework = project / "framework"
    framework.mkdir(parents=True)
    (framework / "paper_cards.json").write_text(
        json.dumps(
            [
                {
                    "paper_id": "pubmed:20953198",
                    "source": "pubmed",
                    "source_id": "20953198",
                    "title": "Antibody recycling by engineered pH-dependent antigen binding improves the duration of antigen neutralization.",
                    "year": 2010,
                    "doi": "10.1038/nbt.1691",
                    "url": "https://pubmed.ncbi.nlm.nih.gov/20953198/",
                    "authors": ["Tomoyuki Igawa"],
                },
                {
                    "paper_id": "arxiv:2412.07763v1",
                    "source": "arxiv",
                    "source_id": "2412.07763v1",
                    "title": "Bayesian Optimization of Antibodies Informed by a Generative Model of Evolving Sequences",
                    "year": 2024,
                    "url": "https://arxiv.org/abs/2412.07763",
                    "authors": ["Alan Amin"],
                },
            ]
        ),
        encoding="utf-8",
    )

    artifacts = _load_cmdgd_literature_artifacts(project)

    assert artifacts["available"] is True
    assert artifacts["reference_count"] == 2
    assert artifacts["related_work_row_count"] == 2
    assert "@article{pubmed_20953198" in artifacts["references_bib_text"]
    assert "journal = {" not in artifacts["references_bib_text"]
    assert "url = {https://pubmed.ncbi.nlm.nih.gov/20953198/}" not in artifacts["references_bib_text"]
    assert {row["citation_key"] for row in artifacts["related_work_rows"]} == {
        "pubmed_20953198",
        "arxiv_2412_07763v1",
    }


def test_references_bib_corrects_known_doi_venue_metadata() -> None:
    references = _render_references_bib(
        {
            "paper_cards": [
                {
                    "paper_id": "semantic_scholar:43fc5d1174560e905e17caa27fb5ab0451666003",
                    "title": "ALLM-Ab: Active Learning-Driven Antibody Optimization Using Fine-Tuned Protein Language Models",
                    "authors": ["Kairi Furui", "Masahito Ohue"],
                    "year": 2025,
                    "venue": "bioRxiv",
                    "doi": "10.1021/acs.jcim.5c01577",
                }
            ]
        }
    )

    assert "journal = {Journal of Chemical Information and Modeling}" in references
    assert "journal = {bioRxiv}" not in references


def test_ph_switch_graph_supplementary_package_includes_declared_tables(tmp_path: Path) -> None:
    paper_dir = tmp_path / "paper"
    table_dir = paper_dir / "tables"
    table_dir.mkdir(parents=True)
    for relative in (
        "supplementary_data.md",
        "supplement_manifest.md",
        "algorithm_formal_definition.md",
        "data_dictionary.json",
        "tables/external_ph_switch_variant_replay.csv",
        "tables/external_ph_switch_variant_method_summary.csv",
        "tables/world_block_comparison.csv",
    ):
        path = paper_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder\n", encoding="utf-8")

    package = _write_ph_switch_graph_supplementary_package(paper_dir)

    assert package.exists()
    with zipfile.ZipFile(package) as archive:
        names = set(archive.namelist())
    assert "supplementary_data.md" in names
    assert "tables/external_ph_switch_variant_replay.csv" in names
    assert "tables/external_ph_switch_variant_method_summary.csv" in names
    assert "tables/world_block_comparison.csv" in names


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


def test_short_paper_readiness_flags_broken_required_bundle(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)
    write_method_report(project, run_id="v3_unit")

    result = generate_short_paper(project, run_id="v3_unit")

    paper_dir = project / "runs" / "v3_unit" / "paper"
    required_bundle_files = {
        "short_paper.md",
        "references.bib",
        "results_summary.json",
        "claim_evidence_map.json",
        "paper_readiness_report.json",
    }
    assert required_bundle_files <= set(result["artifacts"])
    for filename in required_bundle_files:
        assert (paper_dir / filename).exists()

    for filename in ("references.bib", "results_summary.json", "claim_evidence_map.json"):
        (paper_dir / filename).unlink()
    (paper_dir / "short_paper.md").write_text(
        "# Broken Paper\n\n"
        "## Abstract\n\n"
        "This corrupted bundle has no citation and links to "
        "[a missing artifact](../missing_artifact.json).\n",
        encoding="utf-8",
    )

    report = validate_paper_readiness(project, run_id="v3_unit")

    assert not report["valid"]
    errors = [finding for finding in report["findings"] if finding["severity"] == "error"]
    error_codes = {finding["code"] for finding in errors}
    assert {
        "missing_output_file",
        "missing_section",
        "missing_citation",
        "missing_artifact_link",
    } <= error_codes
    missing_outputs = {
        Path(finding["artifact"]).name
        for finding in errors
        if finding["code"] == "missing_output_file"
    }
    assert {"references.bib", "results_summary.json", "claim_evidence_map.json"} <= missing_outputs


def test_validate_paper_readiness_runs_algorithm_specific_gate_when_summary_exists(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    paper_dir = project / "runs" / "algorithm_unit" / "paper"
    (paper_dir / "tables").mkdir(parents=True)
    (paper_dir / "manuscript.md").write_text(
        "# pH-Switch Graph Search\n\n"
        "## Abstract\n\nComputational draft.\n\n"
        "## Introduction\n\nText.\n\n"
        "## Formal Problem Setup\n\nText.\n\n"
        "## Algorithmic Contribution\n\nText.\n\n"
        "## Evaluation Design\n\nText.\n\n"
        "## Results\n\nText.\n\n"
        "## Limitations\n\nText.\n\n"
        "## Contribution Boundaries\n\nText.\n\n"
        "## Literature Review\n\nText.\n\n"
        "## Methods\n\nText.\n\n"
        "## Data Availability\n\nText.\n\n"
        "## Code Availability\n\nText.\n\n"
        "## Supplementary Information\n\nText.\n\n"
        "## Funding\n\npending author confirmation\n\n"
        "## Conflict of Interest\n\npending author confirmation\n\n"
        "## Acknowledgements and AI Use Disclosure\n\nAI/Codex assisted draft.\n\n"
        "## References\n\n[@paper_active_design]\n",
        encoding="utf-8",
    )
    (paper_dir / "short_paper.md").write_text(
        "## Abstract\n\nComputational synthetic wet-lab boundary.\n\n"
        "## Evidence Boundary\n\ncomputational synthetic wet-lab.\n\n"
        "## Introduction\n\nText.\n\n"
        "## Formal Problem Setup\n\nText.\n\n"
        "## Literature Basis\n\nText.\n\n"
        "## Algorithmic Contribution\n\nText.\n\n"
        "## Methods\n\nText.\n\n"
        "## Evaluation Design\n\nText.\n\n"
        "## Results\n\nText.\n\n"
        "## Discussion\n\nText.\n\n"
        "## Limitations\n\nText.\n\n"
        "## Contribution Boundaries\n\nText.\n\n"
        "## Reproducibility\n\n[artifact](references.bib)\n\n"
        "## Data Availability\n\nText.\n\n"
        "## Code Availability\n\nText.\n\n"
        "## References\n\n[@paper_active_design]\n",
        encoding="utf-8",
    )
    (paper_dir / "references.bib").write_text(
        "@misc{paper_active_design,title={Active design}}\n",
        encoding="utf-8",
    )
    (paper_dir / "results_summary.json").write_text("{}\n", encoding="utf-8")
    (paper_dir / "claim_evidence_map.json").write_text("{}\n", encoding="utf-8")
    (paper_dir / "data_availability.md").write_text("data\n", encoding="utf-8")
    (paper_dir / "code_availability.md").write_text("no archival DOI is claimed\n", encoding="utf-8")
    (paper_dir / "algorithm_results_summary.json").write_text(
        json.dumps(
            {
                "algorithm": "ph_switch_graph",
                "generative_benchmark": {
                    "available": True,
                    "selected_mechanism": "ph_switch_graph",
                    "selected_passes_gate": True,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (paper_dir / "submission_metadata.md").write_text(
        "- corresponding_author: pending author confirmation\n"
        "- repository_url: `https://github.com/Zy-Wang-bit/Design-Scientist`\n"
        "- archival_release: `https://archive.softwareheritage.org/swh:1:snp:test`\n"
        "- license: `not yet selected`\n"
        "- funding: pending author confirmation\n"
        "- conflicts_of_interest: pending author confirmation\n"
        "- ai_use_disclosure: AI/Codex assisted draft.\n",
        encoding="utf-8",
    )

    report = validate_paper_readiness(project, run_id="algorithm_unit")

    codes = {finding["code"] for finding in report["findings"]}
    assert "unresolved_submission_placeholder" in codes
    assert "missing_corresponding_author_email" in codes


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
    assert result["readiness"]["artifacts"]["manuscript.md"] == str(paper_dir / "manuscript.md")
    assert Path(result["readiness"]["artifacts"]["references.bib"]).exists()
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
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    assert readiness["valid"]
    assert readiness["artifacts"]["manuscript.md"] == str(manuscript_path)
    assert readiness["artifacts"]["references.bib"] == str(references_path)

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


def test_ph_switch_graph_formal_definition_and_quality_metrics_are_auditable() -> None:
    context = {
        "generative_benchmark": {
            "config": {
                "budget": 5,
                "generation_budget": 128,
                "worlds": ["ph_contrast", "de_novo_site_generalization"],
            },
            "summary_rows": [
                {
                    "mechanism": "ph_switch_graph",
                    "world_id": "overall",
                    "replicate_count": "20",
                    "comparative_utility_rank": "2",
                    "selected_eligible": "true",
                    "selected_mechanism": "true",
                    "mean_best_generated_utility": "0.827",
                    "mean_best_selected_utility": "0.810",
                    "mean_pH_contrast_score": "0.734",
                    "mean_constraint_pass_rate": "1.0",
                    "mean_false_claim_rate": "0.0",
                    "mean_cost_spent": "4.2",
                    "mean_generated_new_site_count": "88.0",
                    "mean_selected_new_site_count": "1.0",
                    "mean_new_site_rate": "0.688",
                    "new_site_generation_capable": "true",
                },
                {
                    "mechanism": "same_pool_reference_scorer",
                    "world_id": "overall",
                    "replicate_count": "20",
                    "mean_best_selected_utility": "0.810",
                    "mean_false_claim_rate": "0.0",
                    "new_site_generation_capable": "false",
                },
            ],
        }
    }

    formal = _ph_switch_graph_algorithm_formal_definition(context)
    assert "G=(V_s union V_c, A)" in formal
    assert "x=apply(b,P)" in formal
    assert "Update Contract" in formal
    assert "ph_contrast, de_novo_site_generalization" in formal

    rows = _ph_switch_graph_quality_metric_rows(context)
    ph_row = next(row for row in rows if row["mechanism"] == "ph_switch_graph")
    assert ph_row["mean_generated_new_site_count"] == "88.0"
    assert ph_row["interpretation"].startswith("reported generated mechanism")
    sentence = _ph_switch_graph_quality_metric_sentence(context)
    assert "selected utility 0.81" in sentence
    assert "generated new-site count 88" in sentence


def test_ph_switch_graph_candidate_annotations_are_sequence_only_and_auditable(
    tmp_path: Path,
) -> None:
    context = {
        "root": tmp_path,
        "run_id": "unit_run",
        "standardized_context": {
            "base_heavy_chain_seq": "A" * 29 + "S" + "A" * 89,
            "base_light_chain_seq": "A" * 23 + "K" + "A" * 19 + "Q" + "A" * 65,
        },
    }
    project_rows = [
        {
            "candidate_id": "phsg_test",
            "rank": "1",
            "selected": "true",
            "operator": "counterfactual_triplet_program",
            "score": "0.28",
            "standard_edit_notation": json.dumps(["VH:S30H", "VL:K24H", "VL:Q44H"]),
            "standard_new_mutation_sites": json.dumps(["VH:S30H"]),
        }
    ]

    rows = _ph_switch_graph_candidate_annotation_rows(project_rows, context)
    summary = _ph_switch_graph_developability_risk_summary(rows, context)

    assert len(rows) == 3
    assert rows[0]["standard_mutation"] == "VH:S30H"
    assert rows[0]["is_new_site"] is True
    assert rows[0]["region_is_heuristic"] is True
    assert "not IMGT/Kabat" in rows[0]["region_note"]
    assert rows[0]["numbering_source"] == "sequence_index_fallback"
    assert rows[0]["annotation_scope"].startswith("sequence_only_screen")
    assert summary["schema_version"] == 1
    assert summary["candidate_count"] == 1
    assert summary["mutation_annotation_count"] == 3
    assert summary["declared_new_site_annotation_count"] == 1
    assert summary["flag_counts"]["histidine_switch"] == 3
    assert summary["standard_numbering_source"] == "unavailable"
    assert "IMGT numbering is reported" in summary["region_annotation_boundary"]


def test_ph_switch_graph_candidate_rationale_does_not_label_anchor_as_new_site(
    tmp_path: Path,
) -> None:
    context = {
        "root": tmp_path,
        "standardized_context": {
            "base_heavy_chain_seq": "A" * 29 + "S" + "A" * 89,
            "base_light_chain_seq": "A" * 23 + "K" + "A" * 19 + "Q" + "A" * 65,
        },
    }
    rows = _ph_switch_graph_candidate_rationale_rows(
        [
            {
                "candidate_id": "new_site",
                "operator": "counterfactual_triplet_program",
                "modules": '["H30H", "L24H", "L44H"]',
                "new_mutation_sites": '["H30H"]',
                "counterfactual_site_field": "1.1",
                "pair_program_bonus": "0.65",
                "feasibility_prior": "0.64",
            },
            {
                "candidate_id": "anchor",
                "operator": "evidence_guardrail_triplet",
                "modules": '["L24H", "L44H", "L62R"]',
                "new_mutation_sites": "[]",
                "counterfactual_site_field": "0.0",
                "pair_program_bonus": "0.0",
                "feasibility_prior": "0.64",
            },
        ],
        context,
    )

    new_site = next(row for row in rows if row["candidate_id"] == "new_site")
    anchor = next(row for row in rows if row["candidate_id"] == "anchor")

    assert new_site["candidate_role"] == "counterfactual_new_site_pair_or_triplet_probe"
    assert "counterfactual new-site hypothesis" in new_site["rationale"]
    assert anchor["candidate_role"] == "empirical_anchor_guardrail"
    assert "does not contain a declared counterfactual new mutation site" in anchor["rationale"]
    assert "combines a counterfactual" not in anchor["rationale"]
    assert anchor["neutral_retention_constraint_type"] == "soft_proxy_not_hard_constraint"


def test_ph_switch_graph_ablation_summary_uses_mean_loss_not_row_max() -> None:
    context = {
        "generative_benchmark": {
            "ablation_rows": [
                {
                    "mechanism": "ph_switch_graph",
                    "ablation": "full",
                    "generated_new_site_count": "88",
                    "best_selected_utility": "0.83",
                    "delta_from_full_best_selected_utility": "0.0",
                },
                {
                    "mechanism": "ph_switch_graph",
                    "ablation": "full",
                    "generated_new_site_count": "88",
                    "best_selected_utility": "0.83",
                    "delta_from_full_best_selected_utility": "0.0",
                },
                {
                    "mechanism": "ph_switch_graph",
                    "ablation": "no_counterfactual_site_map",
                    "generated_new_site_count": "0",
                    "best_selected_utility": "0.82",
                    "delta_from_full_best_selected_utility": "0.000",
                },
                {
                    "mechanism": "ph_switch_graph",
                    "ablation": "no_counterfactual_site_map",
                    "generated_new_site_count": "0",
                    "best_selected_utility": "0.75",
                    "delta_from_full_best_selected_utility": "0.056",
                },
                {
                    "mechanism": "ph_switch_graph",
                    "ablation": "no_pair_programs",
                    "generated_new_site_count": "32",
                    "best_selected_utility": "0.80",
                    "delta_from_full_best_selected_utility": "0.024",
                },
            ],
        }
    }

    sentence = _ph_switch_graph_ablation_summary_sentence(context)

    assert "largest new-site generation losses were no_counterfactual_site_map (88.0 candidates)" in sentence
    assert "no_pair_programs (0.024)" in sentence
    assert "no_counterfactual_site_map (0.056)" not in sentence
    assert "changed mean selected utility by only 0.028" in sentence
    assert "rather than selected-utility superiority" in sentence


def test_ph_switch_graph_configuration_contract_separates_benchmark_and_project_limits() -> None:
    context = {
        "generative_benchmark": {
            "config": {
                "generation_budget": 128,
                "max_edits_per_candidate": 999,
            },
            "project_design_summary": {
                "generated_candidate_count": 256,
                "selected_count": 4,
            },
            "project_design_candidate_rows": [
                {
                    "selected": "true",
                    "modules": '["H30H", "L24H", "L44H"]',
                },
                {
                    "selected": "true",
                    "modules": '["L24H", "L44H", "L62R"]',
                },
            ],
        }
    }

    rows = _ph_switch_graph_configuration_contract_rows(context)
    by_setting = {row["setting"]: row for row in rows}

    assert by_setting["benchmark_generation_budget"]["value"] == 128
    assert by_setting["benchmark_max_edits_per_candidate"]["value"] == 999
    assert "Sentinel" in by_setting["benchmark_max_edits_per_candidate"]["interpretation"]
    assert by_setting["project_generated_candidate_count"]["value"] == 256
    assert by_setting["selected_candidate_program_widths"]["value"] == "3"
    assert by_setting["selection_mode"]["value"] == "balanced"


def test_ph_switch_graph_pareto_claim_boundary_marks_supported_and_unsupported_claims() -> None:
    context = {
        "run_id": "unit_run",
        "generative_benchmark": {
            "selection_gate_report": {
                "selected_mechanism": "ph_switch_graph",
                "selected_utility_rank": 3,
                "utility_winner": "same_pool_reference_scorer",
            },
            "summary_rows": [
                {
                    "mechanism": "same_pool_reference_scorer",
                    "world_id": "overall",
                    "mean_best_selected_utility": "0.809726",
                    "mean_selected_new_site_count": "0.0",
                    "mean_false_claim_rate": "0.0",
                    "mean_generated_new_site_count": "88.0",
                    "mean_new_site_rate": "0.6875",
                    "selected_eligible": "false",
                },
                {
                    "mechanism": "observed_recombination_baseline",
                    "world_id": "overall",
                    "mean_best_selected_utility": "0.809726",
                    "mean_selected_new_site_count": "0.0",
                    "mean_false_claim_rate": "0.0",
                    "mean_generated_new_site_count": "0.0",
                    "mean_new_site_rate": "0.0",
                    "observed_reuse_only": "true",
                },
                {
                    "mechanism": "ph_switch_graph",
                    "world_id": "overall",
                    "mean_best_generated_utility": "0.827286",
                    "mean_best_selected_utility": "0.809588",
                    "mean_selected_new_site_count": "1.0",
                    "mean_false_claim_rate": "0.0",
                    "mean_generated_new_site_count": "88.0",
                    "mean_new_site_rate": "0.6875",
                    "selected_eligible": "true",
                },
                {
                    "mechanism": "random_new_site_scan",
                    "world_id": "overall",
                    "mean_best_generated_utility": "0.905778",
                    "mean_best_selected_utility": "0.737866",
                    "mean_selected_new_site_count": "3.633333",
                    "mean_false_claim_rate": "0.0",
                    "mean_generated_new_site_count": "116.0",
                    "mean_new_site_rate": "1.0",
                },
            ],
        },
        "project_masking": {
            "summary_rows": [
                {
                    "mechanism": "a",
                    "mean_best_feasible_utility": "0.081313",
                    "mean_hit_rate": "0.3",
                    "mean_false_claim_rate": "0.7",
                },
                {
                    "mechanism": "b",
                    "mean_best_feasible_utility": "0.081313",
                    "mean_hit_rate": "0.3",
                    "mean_false_claim_rate": "0.7",
                },
            ]
        },
    }

    rows = _ph_switch_graph_pareto_claim_boundary_rows(context)
    analysis = _ph_switch_graph_claim_boundary_analysis(context)

    ph_row = next(row for row in rows if row["mechanism"] == "ph_switch_graph")
    assert ph_row["supports_candidate_space_expansion_claim"] is True
    assert ph_row["supports_selected_utility_superiority_claim"] is False
    assert analysis["utility_winner"] == "same_pool_reference_scorer"
    assert analysis["selected_utility_rank"] == 3
    assert analysis["selected_vs_same_pool_reference_utility_delta"] == "-0.000138"
    assert analysis["project_masking_discriminative"] is False
    assert "selected-utility superiority" in analysis["unsupported_claims"][0]


def test_ph_switch_graph_pareto_claim_boundary_does_not_upgrade_rank_one_to_superiority() -> None:
    context = {
        "run_id": "unit_run",
        "generative_benchmark": {
            "selection_gate_report": {
                "selected_mechanism": "ph_switch_graph",
                "selected_utility_rank": 1,
                "utility_winner": "ph_switch_graph",
            },
            "summary_rows": [
                {
                    "mechanism": "ph_switch_graph",
                    "world_id": "overall",
                    "mean_best_generated_utility": "0.854977",
                    "mean_best_selected_utility": "0.833352",
                    "mean_selected_new_site_count": "1.0",
                    "mean_false_claim_rate": "0.0",
                    "mean_generated_new_site_count": "88.0",
                    "mean_new_site_rate": "0.6875",
                    "selected_eligible": "true",
                },
                {
                    "mechanism": "same_pool_reference_scorer",
                    "world_id": "overall",
                    "mean_best_selected_utility": "0.814616",
                    "mean_selected_new_site_count": "0.0",
                    "mean_false_claim_rate": "0.0",
                    "mean_generated_new_site_count": "88.0",
                    "mean_new_site_rate": "0.6875",
                    "selected_eligible": "false",
                },
            ],
        },
        "project_masking": {"summary_rows": []},
    }

    rows = _ph_switch_graph_pareto_claim_boundary_rows(context)
    ph_row = next(row for row in rows if row["mechanism"] == "ph_switch_graph")
    assert ph_row["utility_rank"] == 1
    assert ph_row["supports_candidate_space_expansion_claim"] is True
    assert ph_row["supports_selected_utility_superiority_claim"] is False
    assert "same-pool near-tie control" in ph_row["interpretation"]


def test_ph_switch_graph_world_block_rows_report_same_pool_boundary() -> None:
    context = {
        "generative_benchmark": {
            "summary_rows": [
                {
                    "mechanism": "ph_switch_graph",
                    "world_id": "ph_contrast",
                    "replicate_count": "10",
                    "mean_best_selected_utility": "0.850",
                    "mean_selected_new_site_count": "1.0",
                    "mean_false_claim_rate": "0.0",
                },
                {
                    "mechanism": "same_pool_reference_scorer",
                    "world_id": "ph_contrast",
                    "mean_best_selected_utility": "0.840",
                },
                {
                    "mechanism": "observed_recombination_baseline",
                    "world_id": "ph_contrast",
                    "mean_best_selected_utility": "0.810",
                },
                {
                    "mechanism": "random_feasible",
                    "world_id": "ph_contrast",
                    "mean_best_selected_utility": "0.700",
                },
                {
                    "mechanism": "fixed_mix",
                    "world_id": "ph_contrast",
                    "mean_best_selected_utility": "0.500",
                },
                {
                    "mechanism": "ph_switch_graph",
                    "world_id": "escape_risk",
                    "replicate_count": "10",
                    "mean_best_selected_utility": "0.800",
                    "mean_selected_new_site_count": "1.0",
                    "mean_false_claim_rate": "0.0",
                },
                {
                    "mechanism": "same_pool_reference_scorer",
                    "world_id": "escape_risk",
                    "mean_best_selected_utility": "0.803",
                },
            ],
        }
    }

    rows = _ph_switch_graph_world_block_rows(context)
    assert len(rows) == 2
    by_world = {row["world_id"]: row for row in rows}
    assert by_world["ph_contrast"]["delta_vs_same_pool_reference"] == "+0.010000"
    assert by_world["escape_risk"]["interpretation"] == "near-tie versus same-pool reference in this world"
    sentence = _ph_switch_graph_world_block_sentence(context)
    assert "Across 2 stress worlds" in sentence
    assert "above the same-pool reference in 1" in sentence
    assert "near-tied in 1" in sentence


def test_external_antibody_benchmark_artifacts_feed_ph_switch_manuscript(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "external_unit"
    run_dir.mkdir(parents=True)
    (run_dir / "external_antibody_benchmark_summary.csv").write_text(
        "\n".join(
            [
                "method,dataset_id,source,replicate_count,dataset_count,mean_best_selected_fitness,mean_best_selected_normalized_fitness,mean_hit_rate_top_decile,mean_regret_vs_oracle,selector_uses_oracle_truth,interpretation",
                "pcig_external_replay,overall,public_flab,15,3,8.75,0.937,0.34,0.013,false,limited external ranking check",
                "random_measured,overall,public_flab,15,3,8.21,0.834,0.20,0.056,false,random measured candidate baseline",
                "fewest_edits,overall,public_flab,15,3,8.50,0.901,0.25,0.037,false,simple edit-distance baseline",
                "oracle_top_measured,overall,public_flab,15,3,8.80,0.958,1.0,0.0,true,measured upper bound",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "external_antibody_benchmark_results.csv").write_text(
        "dataset_id,method,seed,best_selected_fitness\nflab_unit,pcig_external_replay,0,8.75\n",
        encoding="utf-8",
    )
    (run_dir / "external_antibody_benchmark_config.json").write_text(
        json.dumps({"budget": 5, "source": "fixture"}),
        encoding="utf-8",
    )
    (run_dir / "external_antibody_source_trace.json").write_text(
        json.dumps({"datasets": [{"dataset_id": "flab_unit", "network_fetch": False}]}),
        encoding="utf-8",
    )
    (run_dir / "external_antibody_claims.json").write_text(
        json.dumps({"validated_claim_scope": ["external antibody affinity ranking"]}),
        encoding="utf-8",
    )

    artifacts = _load_external_antibody_benchmark_artifacts(run_dir)
    context = {"external_antibody_benchmark": artifacts}

    assert artifacts["available"] is True
    assert artifacts["summary_rows"][0]["method"] == "pcig_external_replay"
    assert len(_ph_switch_graph_external_antibody_rows(context)) == 4
    sentence = _ph_switch_graph_external_benchmark_sentence(context)
    assert "External antibody replay adds an independent held-out affinity-ranking check" in sentence
    assert "0.937" in sentence
    assert "does not validate 1E62 pH 6.0 dissociation" in sentence


def test_ph_switch_graph_claim_map_includes_public_ph_switch_replay() -> None:
    claim_map = _algorithm_claim_evidence_map(
        {"algorithm": "ph_switch_graph", "run_id": "ph_switch_graph_unit"}
    )

    public_ph_claim = next(
        claim
        for claim in claim_map["claims"]
        if claim["evidence_type"] == "external_public_ph_switch_table_replay"
    )
    assert public_ph_claim["evidence_scope"] == "external_pH_switch_prior_sanity_check_not_1E62_validation"
    assert "external_ph_switch_variant_method_summary" in public_ph_claim["evidence_artifacts"]
    assert any("not prospective 1E62 measurements" in limitation for limitation in public_ph_claim["limitations"])


def test_ph_switch_graph_submission_metadata_rejects_journal_placeholders(
    tmp_path: Path,
) -> None:
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    (paper_dir / "submission_metadata.md").write_text(
        "\n".join(
            [
                "# Submission Metadata",
                "",
                "- authors: author details to be inserted before submission",
                "- corresponding_author: Correspondence details withheld for review",
                "- repository_url: `https://github.com/Zy-Wang-bit/Design-Scientist`",
                "- archival_release: `archival DOI or Software Heritage URL to be inserted before submission`",
                "- license: `repository license to be confirmed before submission`",
                "- data_availability_statement: data release and reviewer-access statement to be inserted before submission",
                "- funding: funding statement to be inserted before submission",
                "- conflicts_of_interest: conflict of interest statement to be inserted before submission",
                "- ai_use_disclosure: Codex was used as an author-side assistant.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (paper_dir / "code_availability.md").write_text(
        "no archival DOI is claimed for this draft\n",
        encoding="utf-8",
    )

    findings: list[dict[str, str]] = []
    _validate_ph_switch_graph_submission_metadata(paper_dir, findings)

    codes = {finding["code"] for finding in findings}
    assert "unresolved_submission_placeholder" in codes
    assert "missing_archival_software_url" in codes
    assert "missing_corresponding_author_email" in codes


def test_ph_switch_graph_external_ph_clause_reports_limited_public_table_anchor() -> None:
    context = {
        "external_ph_switch_benchmark": {
            "available": True,
            "summary_rows": [
                {
                    "method": "leave_one_study_transition_calibration",
                    "dataset_id": "overall",
                    "dataset_count": 5,
                    "mean_best_selected_normalized_log_ratio": 0.846,
                    "mean_best_selected_ph_ratio": 987.6,
                    "mean_hit_rate_top_tertile": 0.3,
                    "mean_regret_vs_oracle": 835.2,
                },
                {
                    "method": "histidine_count",
                    "dataset_id": "overall",
                    "mean_best_selected_normalized_log_ratio": 0.709,
                    "mean_best_selected_ph_ratio": 1004.9,
                },
                {
                    "method": "transition_context_prior",
                    "dataset_id": "overall",
                    "mean_best_selected_normalized_log_ratio": 0.758,
                    "mean_best_selected_ph_ratio": 953.2,
                },
                {
                    "method": "ph_switch_residue_prior",
                    "dataset_id": "overall",
                    "mean_best_selected_ph_ratio": 1004.9,
                },
                {
                    "method": "parent_reference",
                    "dataset_id": "overall",
                    "mean_best_selected_ph_ratio": 8.0,
                },
                {
                    "method": "ionizable_count",
                    "dataset_id": "overall",
                    "mean_best_selected_ph_ratio": 1004.9,
                },
                {
                    "method": "oracle_top_pH_ratio",
                    "dataset_id": "overall",
                    "mean_best_selected_ph_ratio": 1822.8,
                },
            ],
            "variant_replay_summary": {
                "method": "leave_one_variant_transition_calibration",
                "variant_count": 35,
                "predicted_vs_observed_normalized_log_ratio_pearson": 0.42,
                "predicted_vs_observed_normalized_log_ratio_spearman": 0.39,
                "top_tertile_hit_rate_at_top_third_by_score": 0.58,
                "top_tertile_base_rate": 0.34,
            },
        }
    }

    abstract_clause = _ph_switch_graph_external_ph_abstract_clause(context)
    sentence = _ph_switch_graph_external_ph_benchmark_sentence(context)

    assert "5 antibody study tables" in abstract_clause
    assert "0.846 versus 0.709" in abstract_clause
    assert "leave-one-variant replay over 35 public variants" in abstract_clause
    assert "limited external sanity check rather than independent validation" in abstract_clause
    assert "Pearson correlation 0.42" in sentence
    assert "top-tertile hit rate 0.58 versus base rate 0.34" in sentence
    assert "raw mean ratio falls below the histidine-count baseline" in sentence
    assert "not validation of the full 1E62 generator" in sentence


def test_submission_readiness_checklist_separates_author_actions_from_artifacts() -> None:
    checklist = _render_submission_readiness_checklist(
        authors="author details to be inserted before submission",
        contact="corresponding author email to be inserted before submission",
        repository_url="https://github.com/Zy-Wang-bit/Design-Scientist",
        archive_url="https://archive.softwareheritage.org/swh:1:snp:test",
        license_text="repository license to be confirmed before submission",
        data_availability_statement=(
            "Data release and reviewer-access statement to be inserted before submission"
        ),
        funding="funding statement to be inserted before submission",
        conflicts="conflict of interest statement to be inserted before submission",
        ai_disclosure="AI/Codex assisted code implementation under author responsibility.",
    )

    assert "Submission Readiness Checklist" in checklist
    assert "| `repository_url` | ready |" in checklist
    assert "| `authors` | author action required |" in checklist
    assert "| `license` | author action required |" in checklist
    assert "must continue to fail until all author-action rows are resolved" in checklist


def test_ph_switch_graph_submission_metadata_accepts_explicit_submit_ready_fields(
    tmp_path: Path,
) -> None:
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    (paper_dir / "submission_metadata.md").write_text(
        "\n".join(
            [
                "# Submission Metadata",
                "",
                "- authors: Ziyang Wang",
                "- corresponding_author: ziyang@example.org",
                "- repository_url: `https://github.com/Zy-Wang-bit/Design-Scientist`",
                "- archival_release: `https://archive.softwareheritage.org/api/1/origin/save/2386008/`",
                "- license: `MIT`",
                "- data_availability_statement: Standardized input data and computational run artifacts are available in the archived repository; reviewer access is provided by the archive URL.",
                "- funding: No specific funding was received for this computational draft.",
                "- conflicts_of_interest: The author declares no competing interests.",
                "- ai_use_disclosure: AI/Codex assisted code generation, artifact consistency checks, and editorial revision under author review.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (paper_dir / "code_availability.md").write_text(
        "repository URL and Software Heritage save request are recorded.\n",
        encoding="utf-8",
    )

    findings: list[dict[str, str]] = []
    _validate_ph_switch_graph_submission_metadata(paper_dir, findings)

    assert findings == []


def test_ph_switch_graph_submission_metadata_rejects_untracked_release_sources(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    paper_dir = repo / "runs" / "unit" / "paper"
    paper_dir.mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='unit'\n", encoding="utf-8")
    (repo / "src").mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.DEVNULL)
    for relative in PH_SWITCH_GRAPH_RELEASE_SOURCE_FILES:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# unit release file\n", encoding="utf-8")
    (paper_dir / "submission_metadata.md").write_text(
        "\n".join(
            [
                "# Submission Metadata",
                "",
                "- authors: Ziyang Wang",
                "- corresponding_author: ziyang@example.org",
                "- repository_url: `https://github.com/Zy-Wang-bit/Design-Scientist`",
                "- repository_commit: `unrecorded`",
                "- archival_release: `https://archive.softwareheritage.org/swh:1:snp:test`",
                "- license: `MIT`",
                "- data_availability_statement: Standardized input data and computational run artifacts are available in the archived repository.",
                "- funding: No specific funding was received for this computational draft.",
                "- conflicts_of_interest: The author declares no competing interests.",
                "- ai_use_disclosure: AI/Codex assisted code generation and artifact consistency checks under author review.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (paper_dir / "code_availability.md").write_text(
        "- primary algorithm module: `src/design_scientist/algorithms/pcig.py`\n"
        "- one-command smoke test: `uv run pytest tests/test_pcig.py -q`\n",
        encoding="utf-8",
    )
    (paper_dir / "manuscript.md").write_text("Manuscript body.\n", encoding="utf-8")

    findings: list[dict[str, str]] = []
    _validate_ph_switch_graph_submission_metadata(paper_dir, findings)

    codes = {finding["code"] for finding in findings}
    assert "untracked_release_source_file" in codes
    assert "unresolved_submission_placeholder" not in codes


def test_bioinformatics_render_artifact_checks_structured_abstract(
    tmp_path: Path,
) -> None:
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    long_results = " ".join(["word"] * 151)
    (paper_dir / "bioinformatics_preamble.tex").write_text(
        "\\abstract{"
        "\\textbf{Motivation:} Test. "
        f"\\textbf{{Results:}} {long_results}. "
        "\\textbf{Availability and Implementation:} no archival DOI is claimed. "
        "\\textbf{Contact:} Correspondence details withheld for review. "
        "\\textbf{Supplementary information:} Supplementary Data are available."
        "}\\keywords{test}\n",
        encoding="utf-8",
    )
    (paper_dir / "bioinformatics_manuscript.log").write_text(
        "Overfull \\hbox (165.21239pt too wide) in paragraph at lines 145--146\n",
        encoding="utf-8",
    )

    findings: list[dict[str, str]] = []
    _validate_bioinformatics_render_artifacts(paper_dir, findings)

    codes = {finding["code"] for finding in findings}
    assert "bioinformatics_abstract_too_long" in codes
    assert "bioinformatics_abstract_placeholder" in codes
    assert "bioinformatics_pdf_overfull_hbox" in codes


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
