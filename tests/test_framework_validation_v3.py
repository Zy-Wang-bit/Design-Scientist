from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from design_scientist.framework_validation import review_framework_run
from design_scientist.method_report import write_method_report
from design_scientist.reference_manifest import write_reference_data_sources
from design_scientist.scientist_search_v3 import SCIENTIST_V3_STAGES


def test_complete_v3_framework_run_validates(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)
    _write_accepted_claim_gate_artifacts(project)

    write_method_report(project, run_id="v3_unit")
    report = review_framework_run(project, run_id="v3_unit")

    assert report["valid"]
    assert report["summary"]["errors"] == 0


def test_v3_validation_errors_when_paper_readiness_report_is_missing(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)

    write_method_report(project, run_id="v3_unit")
    report = review_framework_run(project, run_id="v3_unit")

    assert not report["valid"]
    assert "missing_paper_readiness_report" in _error_codes(report)
    assert {
        "missing_claim_cap",
        "missing_benchmark_saturation",
        "missing_novelty_review",
    } <= _warning_codes(report)


def test_v3_validation_errors_on_missing_claim_gate_for_ready_paper(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)
    _write_ready_paper_bundle(project)

    write_method_report(project, run_id="v3_unit")
    report = review_framework_run(project, run_id="v3_unit")

    assert not report["valid"]
    assert {
        "missing_claim_cap",
        "missing_benchmark_saturation",
        "missing_novelty_review",
    } <= _error_codes(report)


def test_v3_validation_accepts_claim_gate_artifacts_and_reports_them(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)
    _write_accepted_claim_gate_artifacts(project)

    write_method_report(project, run_id="v3_unit")
    report = review_framework_run(project, run_id="v3_unit")
    method_report = (project / "runs" / "v3_unit" / "method_report.md").read_text(encoding="utf-8")

    assert report["valid"]
    assert not _error_codes(report)
    assert "missing_claim_cap" not in _warning_codes(report)
    assert "## Claim Gate" in method_report
    assert "claim_cap.json" in method_report
    assert "computational_benchmark_only" in method_report
    assert "benchmark_saturation.json" in method_report
    assert "all_benchmark_worlds_saturated" in method_report
    assert "Reviewer verdicts" in method_report


def test_v3_validation_accepts_reference_data_sources_manifest(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)
    _write_accepted_claim_gate_artifacts(project)

    write_method_report(project, run_id="v3_unit")
    report = review_framework_run(project, run_id="v3_unit")

    assert report["valid"]
    assert "missing_reference_data_sources" not in _error_codes(report)
    assert "reference_data_sources" in report["artifacts"]


def test_v3_validation_rejects_reference_data_sources_missing_source(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)
    _write_accepted_claim_gate_artifacts(project)
    manifest_path = project / "runs" / "v3_unit" / "reference_data_sources.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["claim_dependencies"][0]["required_sources"].append(
        "runs/v3_unit/nodes/node_literature_kernel/missing_trace.json"
    )
    _write_json(manifest_path, manifest)

    write_method_report(project, run_id="v3_unit")
    report = review_framework_run(project, run_id="v3_unit")

    assert not report["valid"]
    assert "reference_data_sources_missing_source" in _error_codes(report)


def test_v3_validation_rejects_reference_data_sources_selected_node_mismatch(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)
    _write_accepted_claim_gate_artifacts(project)
    manifest_path = project / "runs" / "v3_unit" / "reference_data_sources.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["selected_node_id"] = "other_node"
    _write_json(manifest_path, manifest)

    write_method_report(project, run_id="v3_unit")
    report = review_framework_run(project, run_id="v3_unit")

    assert not report["valid"]
    assert "reference_data_sources_selected_node_mismatch" in _error_codes(report)


def test_v3_method_report_contains_mechanism_spec_kernel_sections(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)

    report_path = write_method_report(project, run_id="v3_unit")

    text = report_path.read_text(encoding="utf-8")
    assert "MechanismSpec" in text
    assert "MechanismSpec Kernel" in text
    assert "components" in text
    assert "Stress Tests" in text
    assert "Ablation" in text
    assert "Mechanism Library" in text
    assert "literature_kernel" in text


def test_v4_validation_requires_research_harness_artifacts(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)
    _write_accepted_claim_gate_artifacts(project)
    _mark_journal_v4(project)

    write_method_report(project, run_id="v3_unit")
    report = review_framework_run(project, run_id="v3_unit")

    assert not report["valid"]
    assert {
        "missing_v4_claim_ledger",
        "missing_v4_research_harness_summary",
        "missing_v4_novelty_audit",
        "missing_v4_verification_ladder",
    } <= _error_codes(report)


def test_v4_validation_accepts_harness_artifacts_and_reports_them(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)
    _write_accepted_claim_gate_artifacts(project)
    _mark_journal_v4(project)
    _write_v4_research_harness_artifacts(project)

    report_path = write_method_report(project, run_id="v3_unit")
    report = review_framework_run(project, run_id="v3_unit")
    text = report_path.read_text(encoding="utf-8")

    assert report["valid"]
    assert "## Research Harness V4" in text
    assert "claim_ledger.jsonl" in text
    assert "novelty_audit.json" in text


def test_v3_validation_rejects_reviewer_reject_verdict(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)
    _write_accepted_claim_gate_artifacts(project)
    _write_json(
        project / "runs" / "v3_unit" / "biology_review.json",
        {
            "verdict": "reject",
            "reasons": ["Biology claims exceed the evidence boundary."],
        },
    )

    write_method_report(project, run_id="v3_unit")
    report = review_framework_run(project, run_id="v3_unit")
    method_report = (project / "runs" / "v3_unit" / "method_report.md").read_text(encoding="utf-8")

    assert not report["valid"]
    assert "reviewer_verdict_rejected" in _error_codes(report)
    assert "biology_review.json" in method_report
    assert "Biology claims exceed the evidence boundary." in method_report


def test_v3_validation_rejects_structured_unfair_baseline_audit(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)
    _write_accepted_claim_gate_artifacts(project)
    _write_json(
        project / "runs" / "v3_unit" / "baseline_audit.json",
        {
            "verdict": "accept",
            "fairness": {
                "same_budget": False,
                "same_rounds": True,
                "same_candidate_pool": False,
                "no_oracle_features": False,
                "same_baseline_inputs": False,
            },
            "issues": [
                "method used budget=64 while baselines used budget=16",
                "method saw oracle labels unavailable to fixed_mix",
                "baseline inputs used a smaller candidate pool",
            ],
        },
    )

    write_method_report(project, run_id="v3_unit")
    report = review_framework_run(project, run_id="v3_unit")

    assert not report["valid"]
    assert "baseline_audit_unfair_comparison" in _error_codes(report)


def test_v3_validation_rejects_biology_overclaim_review(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)
    _write_accepted_claim_gate_artifacts(project)
    _write_json(
        project / "runs" / "v3_unit" / "biology_review.json",
        {
            "verdict": "accept",
            "evidence_boundary_violations": [
                {
                    "source_evidence": "synthetic replay",
                    "overclaim": "reported as prospective wet-lab validation",
                },
                {
                    "source_evidence": "computational result",
                    "overclaim": "biological mechanism confirmed",
                },
            ],
        },
    )

    write_method_report(project, run_id="v3_unit")
    report = review_framework_run(project, run_id="v3_unit")

    assert not report["valid"]
    assert "biology_overclaim_detected" in _error_codes(report)


def test_v3_validation_rejects_invalid_paper_readiness_report(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)
    _write_accepted_claim_gate_artifacts(project)
    _write_json(
        project / "runs" / "v3_unit" / "paper" / "paper_readiness_report.json",
        {
            "valid": False,
            "status": "failed",
            "findings": [
                {
                    "severity": "error",
                    "code": "biology_overclaim_detected",
                    "message": "Retrospective masking is written as prospective biological mechanism confirmed.",
                }
            ],
        },
    )

    write_method_report(project, run_id="v3_unit")
    report = review_framework_run(project, run_id="v3_unit")

    assert not report["valid"]
    assert {
        "paper_readiness_report_invalid",
        "biology_overclaim_detected",
    } <= _error_codes(report)


def test_v3_scientific_only_validation_ignores_submission_metadata_only_readiness_errors(
    tmp_path: Path,
) -> None:
    project = _build_v3_framework_run(tmp_path)
    _write_accepted_claim_gate_artifacts(project)
    paper_path = project / "runs" / "v3_unit" / "paper" / "paper_readiness_report.json"
    payload = _ready_paper_payload(project)
    payload.update(
        {
            "ready": False,
            "valid": False,
            "status": "failed",
            "scientific_ready": True,
            "scientific_valid": True,
            "scientific_status": "passed",
            "summary": {
                "errors": 1,
                "warnings": 0,
                "scientific_errors": 0,
                "scientific_warnings": 0,
                "submission_metadata_errors": 1,
            },
            "findings": [
                {
                    "severity": "error",
                    "code": "missing_corresponding_author_email",
                    "message": "Bioinformatics structured abstracts require a corresponding-author contact email.",
                    "artifact": "submission_metadata.md",
                }
            ],
        }
    )
    _write_json(paper_path, payload)
    write_method_report(project, run_id="v3_unit")

    strict_report = review_framework_run(project, run_id="v3_unit")
    scientific_report = review_framework_run(
        project,
        run_id="v3_unit",
        scientific_only=True,
    )

    assert not strict_report["valid"]
    assert strict_report["validation_mode"] == "full_submission"
    assert scientific_report["valid"]
    assert scientific_report["validation_mode"] == "scientific_only"
    assert scientific_report["summary"]["effective_errors"] == 0
    assert "paper_submission_metadata_incomplete" in _warning_codes(scientific_report)


def test_v3_validation_requires_literature_corpus_and_mechanism_library(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)
    (project / "framework" / "literature_corpus.jsonl").unlink()
    (project / "framework" / "mechanism_library.json").unlink()

    write_method_report(project, run_id="v3_unit")
    report = review_framework_run(project, run_id="v3_unit")

    assert not report["valid"]
    assert {"missing_literature_corpus", "missing_mechanism_library"} <= _error_codes(report)


def test_v3_validation_rejects_selected_node_missing_mechanism_spec(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)
    (_selected_workspace(project) / "mechanism_spec.json").unlink()

    write_method_report(project, run_id="v3_unit")
    report = review_framework_run(project, run_id="v3_unit")

    assert not report["valid"]
    assert "missing_selected_node_mechanism_spec" in _error_codes(report)


def test_v3_validation_rejects_selected_node_that_is_not_completed(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)
    journal_path = project / "runs" / "v3_unit" / "scientist_journal.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    selected = next(node for node in journal["nodes"] if node["node_id"] == journal["selected_node_id"])
    selected["status"] = "failed"
    selected["contract"] = {"valid": True}
    selected["valid"] = True
    _write_json(journal_path, journal)

    write_method_report(project, run_id="v3_unit")
    report = review_framework_run(project, run_id="v3_unit")

    assert not report["valid"]
    assert "selected_node_not_completed" in _error_codes(report)


def test_v3_validation_rejects_selected_node_failing_static_lifecycle_contract(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)
    workspace = _selected_workspace(project)
    sentinel = workspace / "executed_untrusted_code.txt"
    (workspace / "mechanism.py").write_text(
        "from pathlib import Path\n"
        "def select_batch(*args, **kwargs):\n"
        "    return []\n"
        "def run(workspace):\n"
        "    Path(workspace, 'executed_untrusted_code.txt').write_text('executed', encoding='utf-8')\n",
        encoding="utf-8",
    )

    write_method_report(project, run_id="v3_unit")
    report = review_framework_run(project, run_id="v3_unit")

    assert not report["valid"]
    assert "selected_node_lifecycle_contract_invalid" in _error_codes(report)
    assert not sentinel.exists()


def test_v3_validation_rejects_selected_node_empty_semantic_artifact_shells(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)
    workspace = _selected_workspace(project)
    for filename in (
        "proposal.json",
        "ablation_plan.json",
        "stress_test_plan.json",
        "mechanism_metrics.json",
    ):
        _write_json(workspace / filename, {"notes": []})

    write_method_report(project, run_id="v3_unit")
    report = review_framework_run(project, run_id="v3_unit")

    assert not report["valid"]
    assert {
        "selected_node_proposal_semantic_empty",
        "selected_node_ablation_plan_semantic_empty",
        "selected_node_stress_test_plan_semantic_empty",
        "selected_node_mechanism_metrics_semantic_empty",
    } <= _error_codes(report)


def test_v3_validation_accepts_runner_shaped_mechanism_metrics(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)
    _write_accepted_claim_gate_artifacts(project)
    workspace = _selected_workspace(project)
    _write_json(
        workspace / "mechanism_metrics.json",
        {
            "node_id": "node_literature_kernel",
            "mechanism_benchmark_summary": {
                "mechanism": "literature_kernel",
                "mean_best_feasible_utility": "0.875",
                "mean_false_claim_rate": "0.025",
                "key_ablation_delta": "0.12",
                "selected_eligible": "true",
            },
            "ranking_score": 1.2,
            "selected_eligible": True,
        },
    )

    write_method_report(project, run_id="v3_unit")
    report = review_framework_run(project, run_id="v3_unit")

    assert report["valid"]
    assert "selected_node_mechanism_metrics_semantic_empty" not in _error_codes(report)


@pytest.mark.parametrize(
    ("updates", "expected_code"),
    [
        ({"valid": False}, "paper_readiness_report_invalid"),
        ({"ready": False}, "paper_readiness_report_invalid"),
        (
            {
                "valid": True,
                "ready": True,
                "findings": [{"severity": "error", "code": "missing_section"}],
            },
            "paper_readiness_report_has_error_findings",
        ),
        (
            {"valid": True, "ready": True, "summary": {"errors": 1, "warnings": 0}},
            "paper_readiness_report_has_error_findings",
        ),
        ({"valid": None, "ready": None}, "paper_readiness_report_invalid"),
    ],
)
def test_v3_validation_rejects_readiness_report_that_is_not_ready_or_has_errors(
    tmp_path: Path,
    updates: dict[str, object],
    expected_code: str,
) -> None:
    project = _build_v3_framework_run(tmp_path)
    _write_accepted_claim_gate_artifacts(project)
    payload = _ready_paper_payload(project)
    payload.update(updates)
    _write_json(project / "runs" / "v3_unit" / "paper" / "paper_readiness_report.json", payload)

    write_method_report(project, run_id="v3_unit")
    report = review_framework_run(project, run_id="v3_unit")

    assert not report["valid"]
    assert expected_code in _error_codes(report)


@pytest.mark.parametrize(
    ("case", "expected_code"),
    [
        ("outside_paper_dir", "paper_readiness_artifact_outside_paper_dir"),
        ("empty", "empty_paper_readiness_artifact"),
    ],
)
def test_v3_validation_rejects_bad_paper_readiness_artifact_paths(
    tmp_path: Path,
    case: str,
    expected_code: str,
) -> None:
    project = _build_v3_framework_run(tmp_path)
    _write_accepted_claim_gate_artifacts(project)
    payload = _ready_paper_payload(project)
    if case == "outside_paper_dir":
        bad_path = project / "runs" / "v3_unit" / "method_report.md"
        bad_path.write_text("outside paper dir\n", encoding="utf-8")
    else:
        bad_path = project / "runs" / "v3_unit" / "paper" / "empty_artifact.md"
        bad_path.write_text("", encoding="utf-8")
    payload["artifacts"]["short_paper.md"] = str(bad_path)
    _write_json(project / "runs" / "v3_unit" / "paper" / "paper_readiness_report.json", payload)

    write_method_report(project, run_id="v3_unit")
    report = review_framework_run(project, run_id="v3_unit")

    assert not report["valid"]
    assert expected_code in _error_codes(report)


def test_v3_validation_rejects_non_contract_stage_progress_json(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)
    write_method_report(project, run_id="v3_unit")
    _write_json(project / "runs" / "v3_unit" / "stage_progress.json", {"arbitrary": "json"})

    report = review_framework_run(project, run_id="v3_unit")

    assert not report["valid"]
    assert "stage_progress_invalid_contract" in _error_codes(report)


def test_v3_validation_rejects_non_contract_route_tree_json(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)
    write_method_report(project, run_id="v3_unit")
    _write_json(project / "runs" / "v3_unit" / "route_tree.json", {"arbitrary": "json"})

    report = review_framework_run(project, run_id="v3_unit")

    assert not report["valid"]
    assert "route_tree_invalid_contract" in _error_codes(report)


def test_v3_validation_rejects_stage_progress_without_selected_node(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)
    write_method_report(project, run_id="v3_unit")
    stage_path = project / "runs" / "v3_unit" / "stage_progress.json"
    stage_progress = json.loads(stage_path.read_text(encoding="utf-8"))
    stage_progress.pop("selected_node_id")
    _write_json(stage_path, stage_progress)

    report = review_framework_run(project, run_id="v3_unit")

    assert not report["valid"]
    assert "stage_progress_invalid_contract" in _error_codes(report)


def test_v3_validation_rejects_route_tree_selected_node_not_selected_or_connected(
    tmp_path: Path,
) -> None:
    project = _build_v3_framework_run(tmp_path)
    write_method_report(project, run_id="v3_unit")
    route_path = project / "runs" / "v3_unit" / "route_tree.json"
    route_tree = json.loads(route_path.read_text(encoding="utf-8"))
    route_tree["nodes"][0]["selected"] = False
    route_tree["nodes"][0]["status"] = "failed"
    route_tree["edges"] = []
    _write_json(route_path, route_tree)

    report = review_framework_run(project, run_id="v3_unit")

    assert not report["valid"]
    assert "route_tree_invalid_contract" in _error_codes(report)


def test_v3_validation_rejects_selected_mechanism_missing_from_benchmark_results_and_summary(
    tmp_path: Path,
) -> None:
    project = _build_v3_framework_run(tmp_path)
    run_dir = project / "runs" / "v3_unit"
    for filename in ("mechanism_benchmark_results.csv", "mechanism_benchmark_summary.csv"):
        path = run_dir / filename
        rows = [row for row in _read_csv_rows(path) if row["mechanism"] != "literature_kernel"]
        _write_csv(path, rows)

    write_method_report(project, run_id="v3_unit")
    report = review_framework_run(project, run_id="v3_unit")

    assert not report["valid"]
    assert {
        "selected_mechanism_missing_benchmark_results",
        "selected_mechanism_missing_benchmark_summary",
    } <= _error_codes(report)


def test_v3_validation_rejects_strong_claim_cap_when_benchmark_unsaturated(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)
    _write_accepted_claim_gate_artifacts(project)
    run_dir = project / "runs" / "v3_unit"
    _write_json(
        run_dir / "claim_cap.json",
        {
            "verdict": "accept",
            "claim_cap": "algorithm_superiority",
            "allowed_claims": [
                "algorithm superiority",
                "prospective biological validation",
            ],
            "blocked_claims": [],
        },
    )
    _write_json(
        run_dir / "benchmark_saturation.json",
        {
            "verdict": "reject",
            "saturated": False,
            "summary": "benchmark worlds are not saturated",
            "blocked_claims": [
                "algorithm superiority",
                "generative superiority",
                "prospective biological validation",
            ],
        },
    )

    write_method_report(project, run_id="v3_unit")
    report = review_framework_run(project, run_id="v3_unit")

    assert not report["valid"]
    assert {
        "benchmark_saturation_not_met",
        "claim_cap_conflicts_with_benchmark_saturation",
    } <= _error_codes(report)


def test_v3_validation_rejects_claim_cap_allowed_claim_blocked_by_saturation(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)
    _write_accepted_claim_gate_artifacts(project)
    run_dir = project / "runs" / "v3_unit"
    _write_json(
        run_dir / "claim_cap.json",
        {
            "verdict": "accept",
            "claim_cap": "computational_benchmark_only",
            "allowed_claims": ["prospective biological validation"],
            "blocked_claims": [],
        },
    )
    _write_json(
        run_dir / "benchmark_saturation.json",
        {
            "verdict": "accept",
            "saturated": True,
            "summary": "replay benchmark saturated; prospective wet-lab evidence absent",
            "blocked_claims": ["prospective biological validation"],
        },
    )

    write_method_report(project, run_id="v3_unit")
    report = review_framework_run(project, run_id="v3_unit")

    assert not report["valid"]
    assert "claim_cap_conflicts_with_benchmark_saturation" in _error_codes(report)


def test_v3_validation_rejects_selected_node_artifact_in_run_dir_outside_workspace(
    tmp_path: Path,
) -> None:
    project = _build_v3_framework_run(tmp_path)
    run_dir = project / "runs" / "v3_unit"
    run_level_proposal = run_dir / "proposal.json"
    _write_json(
        run_level_proposal,
        {
            "mechanism_id": "literature_kernel",
            "hypothesis": "A run-level artifact must not satisfy the selected node workspace contract.",
            "claims": ["reduces unsupported transfer claims"],
        },
    )
    _set_selected_artifact(project, "proposal", run_level_proposal)

    write_method_report(project, run_id="v3_unit")
    report = review_framework_run(project, run_id="v3_unit")

    assert not report["valid"]
    assert "selected_node_artifact_outside_allowed_roots" in _error_codes(report)


@pytest.mark.parametrize(
    ("case", "expected_code"),
    [
        ("external", "selected_node_artifact_outside_allowed_roots"),
        ("directory", "selected_node_artifact_not_file"),
        ("empty", "empty_selected_node_proposal"),
    ],
)
def test_v3_validation_rejects_selected_node_bad_artifact_paths(
    tmp_path: Path,
    case: str,
    expected_code: str,
) -> None:
    project = _build_v3_framework_run(tmp_path)
    workspace = _selected_workspace(project)
    if case == "external":
        artifact_path = tmp_path / "outside_workspace" / "proposal.json"
        _write_json(
            artifact_path,
            {
                "mechanism_id": "literature_kernel",
                "hypothesis": "External proposal path must be rejected.",
                "claims": ["reduces unsupported transfer claims"],
            },
        )
    elif case == "directory":
        artifact_path = workspace / "proposal_directory"
        artifact_path.mkdir()
    else:
        artifact_path = workspace / "empty_proposal.json"
        artifact_path.write_text("", encoding="utf-8")
    _set_selected_artifact(project, "proposal", artifact_path)

    write_method_report(project, run_id="v3_unit")
    report = review_framework_run(project, run_id="v3_unit")

    assert not report["valid"]
    assert expected_code in _error_codes(report)


def test_v3_validation_rejects_selected_architecture_clone(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)
    metrics_path = _selected_workspace(project) / "mechanism_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["architecture_clone"] = True
    _write_json(metrics_path, metrics)
    _rewrite_summary_row(project, {"architecture_clone": "true", "selected_eligible": "false"})

    write_method_report(project, run_id="v3_unit")
    report = review_framework_run(project, run_id="v3_unit")

    assert not report["valid"]
    assert "selected_node_architecture_clone" in _error_codes(report)


def test_v3_validation_rejects_legacy_cmdgd_as_selected_mechanism(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)
    run_dir = project / "runs" / "v3_unit"
    workspace = _selected_workspace(project)

    for filename in ("mechanism_spec.json", "proposal.json", "mechanism_metrics.json"):
        path = workspace / filename
        data = json.loads(path.read_text(encoding="utf-8"))
        data["mechanism"] = "cmdgd"
        data["mechanism_id"] = "cmdgd"
        _write_json(path, data)

    journal_path = run_dir / "scientist_journal.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    journal["selected_node"]["mechanism"] = "cmdgd"
    for node in journal["nodes"]:
        if node.get("node_id") == journal["selected_node_id"]:
            node["mechanism"] = "cmdgd"
    _write_json(journal_path, journal)

    for filename in ("mechanism_benchmark_results.csv", "mechanism_benchmark_summary.csv"):
        path = run_dir / filename
        rows = _read_csv_rows(path)
        for row in rows:
            if row.get("mechanism") == "literature_kernel":
                row["mechanism"] = "cmdgd"
        _write_csv(path, rows)
    _rewrite_summary_row(project, {"selected_eligible": "true"})

    write_method_report(project, run_id="v3_unit")
    report = review_framework_run(project, run_id="v3_unit")

    assert not report["valid"]
    assert "selected_v3_mechanism_is_legacy_cmdgd" in _error_codes(report)


def test_v3_validation_rejects_selected_eligible_false(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)
    metrics_path = _selected_workspace(project) / "mechanism_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["selected_eligible"] = False
    _write_json(metrics_path, metrics)
    _rewrite_summary_row(project, {"selected_eligible": "false"})

    write_method_report(project, run_id="v3_unit")
    report = review_framework_run(project, run_id="v3_unit")

    assert not report["valid"]
    assert "selected_mechanism_not_eligible" in _error_codes(report)


def test_v3_validation_rejects_metrics_eligible_when_summary_ineligible(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)
    metrics_path = _selected_workspace(project) / "mechanism_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["selected_eligible"] = True
    _write_json(metrics_path, metrics)
    _rewrite_summary_row(project, {"selected_eligible": "false"})

    write_method_report(project, run_id="v3_unit")
    report = review_framework_run(project, run_id="v3_unit")

    assert not report["valid"]
    assert {
        "selected_mechanism_not_eligible",
        "selected_mechanism_eligibility_mismatch",
    } <= _error_codes(report)


def test_v3_validation_requires_required_baseline_summary_rows(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)
    summary_path = project / "runs" / "v3_unit" / "mechanism_benchmark_summary.csv"
    rows = [
        row
        for row in _read_csv_rows(summary_path)
        if row["mechanism"] != "fixed_mix"
    ]
    _write_csv(summary_path, rows)

    write_method_report(project, run_id="v3_unit")
    report = review_framework_run(project, run_id="v3_unit")

    assert not report["valid"]
    assert "mechanism_benchmark_summary_missing_required_baselines" in _error_codes(report)


def _build_v3_framework_run(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    framework_dir = project / "framework"
    run_dir = project / "runs" / "v3_unit"
    workspace = run_dir / "nodes" / "node_literature_kernel"
    framework_dir.mkdir(parents=True)
    workspace.mkdir(parents=True)

    (framework_dir / "framework_spec.yaml").write_text(
        "framework_id: v3_unit\n"
        "domain: protein variant design\n"
        "artifact_root: framework\n",
        encoding="utf-8",
    )
    (framework_dir / "literature_queries.yaml").write_text(
        "queries:\n"
        "  - query_id: q1\n"
        "    domain: protein variant design\n"
        "    query: mechanism-aware active design\n",
        encoding="utf-8",
    )
    (framework_dir / "quest.yaml").write_text(
        "schema_version: 1\n"
        "quest_id: v3_unit\n"
        "domain: protein variant design\n"
        "status: active\n",
        encoding="utf-8",
    )
    _write_json(
        framework_dir / "research_map.json",
        {
            "loops": [{"loop_id": "mechanism_spec_kernel", "status": "active"}],
            "memory": {
                "findings": "framework/findings_memory.jsonl",
                "failures": "framework/failure_memory.jsonl",
            },
        },
    )
    (framework_dir / "findings_memory.jsonl").write_text(
        json.dumps({"summary": "Literature Engine V3 found a reusable mechanism pattern."}) + "\n",
        encoding="utf-8",
    )
    (framework_dir / "failure_memory.jsonl").write_text(
        json.dumps({"summary": "A clone-like mechanism was rejected."}) + "\n",
        encoding="utf-8",
    )
    _write_json(
        framework_dir / "paper_cards.json",
        [
            {
                "paper_id": "paper_active_design",
                "title": "Mechanism-aware active design",
                "source": "fixture",
                "year": 2026,
            }
        ],
    )
    _write_json(
        framework_dir / "literature_search_plan.json",
        {
            "queries": [
                {
                    "query_id": "q1",
                    "query": "mechanism-aware active design",
                    "purpose": "method discovery",
                }
            ]
        },
    )
    _write_json(
        framework_dir / "literature_search_trace.json",
        {"events": [{"event": "offline_fixture_loaded", "paper_id": "paper_active_design"}]},
    )
    _write_json(
        framework_dir / "citation_graph.json",
        {"nodes": [{"paper_id": "paper_active_design"}], "edges": []},
    )
    _write_csv(
        framework_dir / "paper_scores.csv",
        [{"paper_id": "paper_active_design", "total_score": "1.0"}],
    )
    (framework_dir / "literature_corpus.jsonl").write_text(
        json.dumps(
            {
                "paper_id": "paper_active_design",
                "title": "Mechanism-aware active design",
                "chunks": ["Explicit mechanism components support stress-test mapping."],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        framework_dir / "literature_reading_trace.json",
        {"events": [{"event": "read_fulltext", "paper_id": "paper_active_design"}]},
    )
    _write_json(
        framework_dir / "mechanism_cards.json",
        [
            {
                "mechanism_id": "literature_kernel",
                "name": "Literature-guided mechanism kernel",
                "components": ["state_update"],
                "claims": ["reduces unsupported transfer claims"],
            }
        ],
    )
    _write_json(
        framework_dir / "mechanism_library.json",
        {
            "mechanisms": [
                {
                    "mechanism_id": "literature_kernel",
                    "name": "Literature-guided mechanism kernel",
                    "components": [{"component_id": "state_update"}],
                    "claims": ["reduces unsupported transfer claims"],
                }
            ]
        },
    )
    _write_csv(
        framework_dir / "mechanism_gap_matrix.csv",
        [{"mechanism_id": "literature_kernel", "gap": "needs sparse early-round stress test"}],
    )
    _write_json(
        framework_dir / "operator_specs.json",
        [
            {
                "operator_id": "operator_literature_kernel",
                "mechanism_id": "literature_kernel",
                "source_paper_ids": ["paper_active_design"],
                "evidence_strength": "candidate_from_text",
                "objective": {
                    "description": "Stress-testable mechanism acquisition.",
                    "acquisition_formula": "score = utility + uncertainty - claim_penalty",
                },
                "update_rule": {"state_model": "literature_kernel_state"},
                "candidate_generation": {"description": "Generate stress-testable candidates."},
                "required_baselines": ["random_feasible", "fixed_mix"],
                "ablation_hypotheses": [
                    {
                        "ablation_id": "remove_literature_kernel",
                        "removed_component": "literature_kernel",
                    }
                ],
                "negative_controls": [
                    {
                        "control_id": "label_permutation",
                        "expected_result": "advantage collapses",
                    }
                ],
                "implementation_tests": [
                    {
                        "test_id": "operator_contract",
                        "assertion": "operator is traced into code",
                    }
                ],
                "claim_limits": ["computational benchmark only"],
            }
        ],
    )
    _write_csv(
        framework_dir / "operator_gap_matrix.csv",
        [
            {
                "operator_id": "operator_literature_kernel",
                "mechanism_id": "literature_kernel",
                "missing_source_paper_ids": "false",
                "claim_limits": "computational benchmark only",
            }
        ],
    )
    _write_json(
        framework_dir / "operator_evidence_map.json",
        {
            "operators": {
                "operator_literature_kernel": {
                    "mechanism_id": "literature_kernel",
                    "source_paper_ids": ["paper_active_design"],
                }
            }
        },
    )
    _write_json(
        framework_dir / "operator_negative_controls.json",
        {
            "negative_controls": {
                "operator_literature_kernel": [
                    {"control_id": "label_permutation", "expected_result": "advantage collapses"}
                ]
            }
        },
    )

    _write_json(
        workspace / "mechanism_spec.json",
        {
            "mechanism_id": "literature_kernel",
            "name": "Literature-guided MechanismSpec",
            "version": "v3",
            "components": [
                {
                    "component_id": "state_update",
                    "component_type": "state_model",
                    "description": "Update state from observed assay evidence.",
                }
            ],
            "claims": ["reduces unsupported transfer claims"],
            "literature_basis": ["paper_active_design"],
            "stress_test_requirements": ["sparse_early_round"],
            "ablation_targets": ["state_update"],
            "operator_refs": ["operator_literature_kernel"],
            "operator_specs": [{"operator_id": "operator_literature_kernel"}],
        },
    )
    (workspace / "mechanism.py").write_text(
        "def fit_state(context=None):\n"
        "    return context or {}\n"
        "\n"
        "def generate_candidates(state=None, budget=1):\n"
        "    return []\n"
        "\n"
        "def score_candidates(state=None, candidates=None):\n"
        "    return []\n"
        "\n"
        "def select_panel(state=None, scored_candidates=None, budget=1):\n"
        "    return []\n"
        "\n"
        "def plan_ablations(mechanism_spec=None):\n"
        "    return []\n"
        "\n"
        "def run(workspace):\n"
        "    return None\n",
        encoding="utf-8",
    )
    _write_json(
        workspace / "proposal.json",
        {
            "mechanism_id": "literature_kernel",
            "hypothesis": "MechanismSpec components can map claims to stress tests.",
            "claims": ["reduces unsupported transfer claims"],
            "operator_refs": ["operator_literature_kernel"],
        },
    )
    _write_json(
        workspace / "ablation_plan.json",
        {
            "ablations": [
                {
                    "name": "key_component_removed",
                    "ablation_id": "remove_state_update",
                    "removed_components": ["state_update"],
                    "removed_operator_ids": ["operator_literature_kernel"],
                }
            ]
        },
    )
    _write_json(
        workspace / "stress_test_plan.json",
        {
            "worlds": [{"world_id": "sparse_early_round"}],
            "required_baselines": ["random_feasible", "fixed_mix"],
        },
    )
    _write_json(
        workspace / "mechanism_metrics.json",
        {
            "mechanism": "literature_kernel",
            "selected_eligible": True,
            "architecture_clone": False,
            "metrics": {
                "mean_best_feasible_utility": 0.875,
                "mean_false_claim_rate": 0.025,
                "key_ablation_delta": 0.12,
            },
            "benchmark_summary": {
                "mean_best_feasible_utility": 0.875,
                "mean_false_claim_rate": 0.025,
            },
            "unsupported_claims": ["Synthetic replay does not prove wet-lab superiority."],
        },
    )
    _write_json(
        workspace / "validation_report.json",
        {
            "valid": True,
            "caveats": ["Synthetic replay is not prospective wet-lab evidence."],
        },
    )
    _write_json(
        workspace / "operator_to_code_trace.json",
        {
            "operator_to_code_trace": {
                "operator_literature_kernel": [
                    "fit_state.state_update",
                    "generate_candidates.design_space",
                    "score_candidates.mechanism_score",
                    "select_panel.mechanism_score_rank",
                    "plan_ablations.removed_operator_ids",
                ]
            }
        },
    )

    _write_json(
        run_dir / "stage_progress.json",
        {
            "version": "v3",
            "stage_sequence": list(SCIENTIST_V3_STAGES),
            "stages": [
                {"stage": stage, "status": "completed", "artifacts": [], "errors": []}
                for stage in SCIENTIST_V3_STAGES
            ],
            "selected_node_id": "node_literature_kernel",
        },
    )
    _write_json(
        run_dir / "route_tree.json",
        {
            "version": "v3",
            "stage_sequence": list(SCIENTIST_V3_STAGES),
            "root": {
                "node_id": "root",
                "stage": "mechanism_ideation",
                "status": "completed",
            },
            "nodes": [
                {
                    "node_id": "node_literature_kernel",
                    "parent_id": "root",
                    "stage": "mechanism_implementation",
                    "status": "completed",
                    "workspace": str(workspace),
                    "selected": True,
                    "artifacts": {},
                }
            ],
            "edges": [{"source": "root", "target": "node_literature_kernel"}],
            "selected_node_id": "node_literature_kernel",
        },
    )
    _write_csv(
        run_dir / "mechanism_benchmark_results.csv",
        [
            {
                "world_id": "sparse_early_round",
                "mechanism": "literature_kernel",
                "best_feasible_utility": "0.90",
                "false_claim_rate": "0.02",
            },
            {
                "world_id": "sparse_early_round",
                "mechanism": "random_feasible",
                "best_feasible_utility": "0.40",
                "false_claim_rate": "0.05",
            },
            {
                "world_id": "sparse_early_round",
                "mechanism": "fixed_mix",
                "best_feasible_utility": "0.60",
                "false_claim_rate": "0.06",
            },
            {
                "world_id": "noisy_endpoint",
                "mechanism": "literature_kernel",
                "best_feasible_utility": "0.85",
                "false_claim_rate": "0.03",
            },
            {
                "world_id": "noisy_endpoint",
                "mechanism": "random_feasible",
                "best_feasible_utility": "0.50",
                "false_claim_rate": "0.04",
            },
            {
                "world_id": "noisy_endpoint",
                "mechanism": "fixed_mix",
                "best_feasible_utility": "0.70",
                "false_claim_rate": "0.05",
            },
        ],
    )
    _write_csv(
        run_dir / "mechanism_benchmark_summary.csv",
        [
            {
                "mechanism": "literature_kernel",
                "rank": "1",
                "worlds_tested": "2",
                "majority_win_vs_random_feasible": "true",
                "majority_win_vs_fixed_mix": "true",
                "mean_best_feasible_utility": "0.875",
                "mean_false_claim_rate": "0.025",
                "key_ablation_delta": "0.12",
                "architecture_clone": "false",
                "selected_eligible": "true",
            },
            {
                "mechanism": "random_feasible",
                "rank": "2",
                "worlds_tested": "2",
                "majority_win_vs_random_feasible": "false",
                "majority_win_vs_fixed_mix": "false",
                "mean_best_feasible_utility": "0.45",
                "mean_false_claim_rate": "0.045",
                "key_ablation_delta": "0.0",
                "architecture_clone": "false",
                "selected_eligible": "false",
            },
            {
                "mechanism": "fixed_mix",
                "rank": "3",
                "worlds_tested": "2",
                "majority_win_vs_random_feasible": "true",
                "majority_win_vs_fixed_mix": "false",
                "mean_best_feasible_utility": "0.65",
                "mean_false_claim_rate": "0.055",
                "key_ablation_delta": "0.0",
                "architecture_clone": "false",
                "selected_eligible": "false",
            },
        ],
    )
    _write_csv(
        run_dir / "mechanism_ablation_results.csv",
        [
            {
                "world_id": "sparse_early_round",
                "mechanism": "literature_kernel",
                "ablation": "none",
                "best_feasible_utility": "0.90",
                "delta_from_full_best_feasible_utility": "0.0",
                "false_claim_rate": "0.02",
            },
            {
                "world_id": "sparse_early_round",
                "mechanism": "literature_kernel",
                "ablation": "key_component_removed",
                "best_feasible_utility": "0.78",
                "delta_from_full_best_feasible_utility": "-0.12",
                "false_claim_rate": "0.03",
            },
        ],
    )
    _write_json(
        run_dir / "scientist_journal.json",
        {
            "version": "v3",
            "run_id": "v3_unit",
            "selected_node_id": "node_literature_kernel",
            "selected_node": {
                "node_id": "node_literature_kernel",
                "mechanism": "literature_kernel",
            },
            "nodes": [
                {
                    "node_id": "node_literature_kernel",
                    "status": "completed",
                    "valid": True,
                    "contract": {"valid": True},
                    "mechanism": "literature_kernel",
                    "workspace": str(workspace),
                    "artifacts": {
                        "mechanism_spec": str(workspace / "mechanism_spec.json"),
                        "mechanism": str(workspace / "mechanism.py"),
                        "proposal": str(workspace / "proposal.json"),
                        "ablation_plan": str(workspace / "ablation_plan.json"),
                        "stress_test_plan": str(workspace / "stress_test_plan.json"),
                        "mechanism_metrics": str(workspace / "mechanism_metrics.json"),
                        "validation_report": str(workspace / "validation_report.json"),
                        "operator_to_code_trace": str(workspace / "operator_to_code_trace.json"),
                    },
                },
                {
                    "node_id": "node_failed",
                    "status": "failed",
                    "valid": False,
                    "reasons": ["contract missing plan_ablations"],
                },
            ],
            "unsupported_claims": ["Synthetic replay does not prove wet-lab superiority."],
        },
    )
    return project


def _selected_workspace(project: Path) -> Path:
    journal = json.loads((project / "runs" / "v3_unit" / "scientist_journal.json").read_text(encoding="utf-8"))
    selected_id = journal["selected_node_id"]
    selected = next(node for node in journal["nodes"] if node["node_id"] == selected_id)
    return Path(selected["workspace"])


def _set_selected_artifact(project: Path, key: str, path: Path) -> None:
    journal_path = project / "runs" / "v3_unit" / "scientist_journal.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    selected_id = journal["selected_node_id"]
    selected = next(node for node in journal["nodes"] if node["node_id"] == selected_id)
    selected["artifacts"][key] = str(path)
    _write_json(journal_path, journal)


def _mark_journal_v4(project: Path) -> None:
    journal_path = project / "runs" / "v3_unit" / "scientist_journal.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    journal["research_harness_version"] = "v4"
    journal["research_harness"] = {
        "claim_ledger_path": str(project / "framework" / "claim_ledger.jsonl"),
        "evidence_ledger_path": str(project / "framework" / "evidence_ledger.jsonl"),
        "mechanism_ledger_path": str(project / "framework" / "mechanism_ledger.jsonl"),
        "summary_path": str(project / "framework" / "research_harness_summary.json"),
        "novelty_audit_path": str(project / "runs" / "v3_unit" / "novelty_audit.json"),
        "verification_ladder_path": str(project / "runs" / "v3_unit" / "verification_ladder.json"),
    }
    _write_json(journal_path, journal)


def _write_v4_research_harness_artifacts(project: Path) -> None:
    framework_dir = project / "framework"
    run_dir = project / "runs" / "v3_unit"
    (framework_dir / "claim_ledger.jsonl").write_text(
        json.dumps(
            {
                "claim_id": "v3_unit:selected_mechanism_replay_supported",
                "subject": "literature_kernel",
                "claim_type": "algorithmic",
                "text": "Selected mechanism passed replay selection gate.",
                "support_status": "supported",
                "evidence_ids": ["v3_unit:mechanism_benchmark_summary"],
                "limitations": ["computational benchmark only"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (framework_dir / "evidence_ledger.jsonl").write_text(
        json.dumps(
            {
                "evidence_id": "v3_unit:mechanism_benchmark_summary",
                "source_type": "benchmark",
                "path": str(run_dir / "mechanism_benchmark_summary.csv"),
                "summary": "Selected mechanism beat random_feasible and fixed_mix in replay.",
                "strength": "moderate",
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (framework_dir / "mechanism_ledger.jsonl").write_text(
        json.dumps(
            {
                "mechanism_id": "literature_kernel",
                "name": "Literature-guided MechanismSpec",
                "components": ["state_update", "claim_gate"],
                "literature_refs": ["paper_active_design"],
                "implementation_path": str(run_dir / "nodes" / "node_literature_kernel" / "mechanism.py"),
                "status": "selected",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        framework_dir / "research_harness_summary.json",
        {
            "valid": True,
            "errors": [],
            "claim_count": 1,
            "evidence_count": 1,
            "mechanism_count": 1,
            "unsupported_claim_ids": [],
            "missing_evidence_by_claim": {},
            "empty_component_mechanism_ids": [],
            "selected_unsupported_claim_ids": [],
        },
    )
    _write_json(
        framework_dir / "mechanism_cards_v4.json",
        [{"mechanism_id": "literature_kernel", "components": ["state_update"]}],
    )
    _write_json(
        framework_dir / "mechanism_library_v4.json",
        {"mechanisms": [{"mechanism_id": "literature_kernel", "components": ["state_update"]}]},
    )
    _write_csv(
        framework_dir / "mechanism_gap_matrix_v4.csv",
        [{"mechanism_id": "literature_kernel", "gap": "needs human review"}],
    )
    _write_json(
        framework_dir / "literature_mine_trace_v4.json",
        {"status": "ok", "source": "test_fixture", "card_count": 1},
    )
    _write_json(
        framework_dir / "mechanism_graph_v4.json",
        {"components": [{"component_id": "state_update"}], "gaps": [], "edges": []},
    )
    _write_json(
        run_dir / "novelty_audit.json",
        {
            "schema_version": 1,
            "verdict": "pass",
            "baseline_clone": False,
            "weak_delta": False,
            "findings": [],
        },
    )
    _write_json(
        run_dir / "verification_ladder.json",
        {
            "schema_version": 1,
            "mechanism_name": "literature_kernel",
            "passed": True,
            "blocking_stage": None,
            "stages": [{"name": "contract", "status": "passed", "passed": True}],
            "claim_findings": [],
        },
    )


def _rewrite_summary_row(project: Path, updates: dict[str, str]) -> None:
    summary_path = project / "runs" / "v3_unit" / "mechanism_benchmark_summary.csv"
    rows = _read_csv_rows(summary_path)
    for row in rows:
        if row["mechanism"] == "literature_kernel":
            row.update(updates)
    _write_csv(summary_path, rows)


PAPER_BUNDLE_FILENAMES = (
    "short_paper.md",
    "references.bib",
    "results_summary.json",
    "claim_evidence_map.json",
    "reproducibility.md",
    "data_availability.md",
    "code_availability.md",
)


def _ready_paper_payload(project: Path) -> dict[str, object]:
    run_dir = project / "runs" / "v3_unit"
    paper_dir = run_dir / "paper"
    artifacts = {"paper_dir": str(paper_dir)}
    for filename in PAPER_BUNDLE_FILENAMES:
        artifacts[filename] = str(paper_dir / filename)
    artifacts["paper_readiness_report.json"] = str(paper_dir / "paper_readiness_report.json")
    return {
        "schema_version": 1,
        "ready": True,
        "valid": True,
        "status": "passed",
        "scientific_ready": True,
        "scientific_valid": True,
        "scientific_status": "passed",
        "run_id": "v3_unit",
        "project_dir": str(project),
        "run_dir": str(run_dir),
        "summary": {"errors": 0, "warnings": 0},
        "findings": [],
        "artifacts": artifacts,
    }


def _write_ready_paper_bundle(project: Path) -> None:
    paper_dir = project / "runs" / "v3_unit" / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    for filename in PAPER_BUNDLE_FILENAMES:
        path = paper_dir / filename
        if filename.endswith(".json"):
            _write_json(path, {"fixture": filename})
        else:
            path.write_text(f"{filename} fixture\n", encoding="utf-8")
    _write_json(paper_dir / "paper_readiness_report.json", _ready_paper_payload(project))


def _write_accepted_claim_gate_artifacts(project: Path) -> None:
    run_dir = project / "runs" / "v3_unit"
    _write_ready_paper_bundle(project)
    _write_json(
        run_dir / "claim_cap.json",
        {
            "verdict": "accept",
            "claim_cap": "computational_benchmark_only",
            "allowed_claims": ["algorithmic replay evidence"],
            "blocked_claims": [],
        },
    )
    _write_json(
        run_dir / "benchmark_saturation.json",
        {
            "verdict": "accept",
            "saturated": True,
            "summary": "all_benchmark_worlds_saturated",
        },
    )
    for filename in (
        "novelty_review.json",
        "baseline_audit.json",
        "experiment_review.json",
        "biology_review.json",
        "paper_contribution_review.json",
    ):
        _write_json(
            run_dir / filename,
            {
                "verdict": "accept",
                "summary": f"{filename} accepted the bounded claim.",
            },
        )
    _write_reference_manifest(project)


def _write_reference_manifest(project: Path) -> Path:
    run_dir = project / "runs" / "v3_unit"
    journal = json.loads((run_dir / "scientist_journal.json").read_text(encoding="utf-8"))
    selected_node = next(
        node for node in journal["nodes"] if node["node_id"] == journal["selected_node_id"]
    )
    return write_reference_data_sources(
        project,
        run_dir,
        run_id="v3_unit",
        selected_node=selected_node,
    )


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    assert rows
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


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _error_codes(report: dict[str, object]) -> set[str]:
    findings = report["findings"]
    assert isinstance(findings, list)
    return {
        finding["code"]
        for finding in findings
        if isinstance(finding, dict) and finding.get("severity") == "error"
    }


def _warning_codes(report: dict[str, object]) -> set[str]:
    findings = report["findings"]
    assert isinstance(findings, list)
    return {
        finding["code"]
        for finding in findings
        if isinstance(finding, dict) and finding.get("severity") == "warning"
    }
