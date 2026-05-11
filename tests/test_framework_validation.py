from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from design_scientist.framework import init_framework
from design_scientist.framework_validation import METHOD_REPORT_REQUIRED_SECTIONS, review_framework_run
from design_scientist.literature_pipeline import run_literature_search
from design_scientist.method_extraction import extract_methods
from design_scientist.method_report import write_method_report
from design_scientist.reference_audit import audit_references
from design_scientist.research_os import append_failure, append_finding
from design_scientist.scientist_search import develop_method


def test_review_framework_run_reports_missing_method_report_as_error(tmp_path: Path) -> None:
    project = _build_framework_run(tmp_path)

    report = review_framework_run(project, run_id="framework_unit")

    assert not report["valid"]
    assert report["summary"]["errors"] == 1
    assert _codes(report) == {"missing_method_report"}
    assert report["artifacts"]["scientist_journal"].endswith("scientist_journal.json")


def test_write_method_report_completes_framework_validation(tmp_path: Path) -> None:
    project = _build_framework_run(tmp_path)
    audit_references(project)
    append_finding(
        project,
        {
            "summary": "Reference audit identified staged journals and durable quest state.",
            "source": "reference_audit",
        },
    )
    append_failure(
        project,
        {
            "summary": "One candidate method was rejected for weak baseline separation.",
            "source": "method_search",
        },
    )
    _write_run_json(
        project,
        "stage_progress.json",
        {
            "stages": [
                {"stage_id": "reference_ingestion", "status": "completed"},
                {"stage_id": "method_development", "status": "completed"},
            ]
        },
    )
    _write_run_json(
        project,
        "route_tree.json",
        {
            "root": "offline_rd",
            "selected_path": ["node_01_mechanism_aware"],
            "routes": [{"node_id": "node_01_mechanism_aware", "status": "selected"}],
        },
    )
    _write_selected_v2_artifacts(
        project,
        proposal={
            "schema_version": 3,
            "proposal_id": "proposal_node_01",
            "title": "Mechanism-aware active allocation",
            "method_claim": "Mechanism-aware allocation should beat simple baselines.",
            "method_hypothesis": "A productized generated policy can add a repair slot without cloning baselines.",
            "literature_basis": [
                "Active design literature motivates exploitation plus coverage repair.",
            ],
            "literature_gap": "Prior replay policies do not expose a clone guard for selected batches.",
            "algorithm_mechanism": "Rank by mechanism-aware utility, then reserve one low-order repair slot.",
            "expected_advantage": "Lower baseline overlap while preserving best feasible utility.",
            "architecture_delta": "Adds a clone-guarded repair slot to the baseline ranking architecture.",
            "no_baseline_rationale": "Synthetic replay overlap stays below the clone threshold.",
            "generator_limited_diagnostics": {
                "status": "limited",
                "reason": "local deterministic generator",
            },
            "failure_modes": ["repair slot can displace a high-utility design"],
            "planned_ablations": ["disable_repair_slot"],
        },
        novelty_report={
            "novelty_score": 0.82,
            "baseline_overlap": 0.18,
            "selection_overlap_vs_baselines": 0.18,
            "baseline_clone": False,
            "nearest_baseline": "fixed_mix",
            "architecture_novelty": "Clone guard changes the selection architecture.",
            "architecture_delta": "Adds a repair slot after baseline-family ranking.",
        },
    )

    report_path = write_method_report(project, run_id="framework_unit")
    report = review_framework_run(project, run_id="framework_unit")

    assert report_path == project / "runs" / "framework_unit" / "method_report.md"
    assert report["valid"]
    assert report["summary"]["errors"] == 0
    text = report_path.read_text(encoding="utf-8")
    for heading in METHOD_REPORT_REQUIRED_SECTIONS:
        assert heading in text
    assert "Reference Audit" in text
    assert "framework/quest.yaml" in text
    assert "Stage Progress" in text
    assert "novelty_score" in text
    assert "## No-Baseline Rationale" in text
    assert "no_baseline_rationale" in text
    assert "## Architecture Novelty" in text
    assert "architecture_delta" in text
    assert "## Generator-Limited Diagnostics" in text
    assert "generator_limited_diagnostics" in text
    assert "Synthetic replay ranking does not establish prospective wet-lab superiority." in text
    assert "runs/framework_unit/scientist_journal.json" in text


def test_review_framework_run_accepts_fresh_local_generated_nodes_without_warnings(
    tmp_path: Path,
) -> None:
    project = _build_complete_framework_run(tmp_path)

    report = review_framework_run(project, run_id="framework_unit")

    assert report["valid"]
    assert report["summary"]["errors"] == 0
    assert report["summary"]["warnings"] == 0
    assert "benchmark_method_not_registered" not in _warning_codes(report)
    assert "selected_node_proposal_sparse" not in _warning_codes(report)
    assert "selected_node_baseline_overlap_missing" not in _warning_codes(report)


def test_review_framework_run_still_warns_for_truly_unknown_benchmark_method(
    tmp_path: Path,
) -> None:
    project = _build_complete_framework_run(tmp_path)
    benchmark_path = project / "runs" / "framework_unit" / "benchmark_results.csv"
    with benchmark_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.append({**rows[0], "method": "unproven_external_method"})
    with benchmark_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = review_framework_run(project, run_id="framework_unit")

    assert report["valid"]
    assert "benchmark_method_not_registered" in _warning_codes(report)


def test_review_framework_run_rejects_selected_v2_node_missing_proposal_and_novelty(
    tmp_path: Path,
) -> None:
    project = _build_framework_run(tmp_path)
    write_method_report(project, run_id="framework_unit")
    workspace = _selected_workspace(project)
    manifest_path = workspace / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 2
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (workspace / "proposal.json").unlink(missing_ok=True)
    (workspace / "novelty_report.json").unlink(missing_ok=True)

    report = review_framework_run(project, run_id="framework_unit")

    assert not report["valid"]
    assert {"missing_selected_node_proposal", "missing_selected_node_novelty_report"} <= _codes(report)


def test_review_framework_run_rejects_selected_baseline_clone(tmp_path: Path) -> None:
    project = _build_framework_run(tmp_path)
    _write_selected_v2_artifacts(
        project,
        novelty_report={
            "novelty_score": 0.05,
            "baseline_overlap": 0.98,
            "baseline_clone": True,
            "nearest_baseline": "random_feasible",
        },
    )
    write_method_report(project, run_id="framework_unit")

    report = review_framework_run(project, run_id="framework_unit")

    assert not report["valid"]
    assert "selected_node_baseline_clone" in _codes(report)


def test_review_framework_run_rejects_selected_high_overlap_to_simple_baseline(tmp_path: Path) -> None:
    project = _build_framework_run(tmp_path)
    _write_selected_v2_artifacts(
        project,
        novelty_report={
            "schema_version": 3,
            "novelty_score": 0.47,
            "baseline_overlap": 0.91,
            "selection_overlap_vs_baselines": 0.91,
            "baseline_clone": False,
            "nearest_baseline": "random_feasible",
        },
    )
    write_method_report(project, run_id="framework_unit")

    report = review_framework_run(project, run_id="framework_unit")

    assert not report["valid"]
    assert "selected_node_baseline_overlap_high" in _codes(report)


def test_review_framework_run_rejects_full_chain_proposal_missing_literature_gap(tmp_path: Path) -> None:
    project = _build_framework_run(tmp_path)
    _write_selected_v2_artifacts(
        project,
        proposal={
            "schema_version": 3,
            "proposal_id": "proposal_node_01",
            "title": "Mechanism-aware active allocation",
            "method_claim": "Mechanism-aware allocation should beat simple baselines.",
            "method_hypothesis": "A productized generated policy can add a repair slot without cloning baselines.",
            "literature_basis": ["Active design literature motivates exploitation plus coverage repair."],
            "algorithm_mechanism": "Rank by mechanism-aware utility, then reserve one low-order repair slot.",
            "expected_advantage": "Lower baseline overlap while preserving best feasible utility.",
            "architecture_delta": "Adds a clone-guarded repair slot to the baseline ranking architecture.",
            "failure_modes": ["repair slot can displace a high-utility design"],
            "planned_ablations": ["disable_repair_slot"],
        },
    )
    write_method_report(project, run_id="framework_unit")

    report = review_framework_run(project, run_id="framework_unit")

    assert not report["valid"]
    assert "selected_node_literature_gap_missing" in _codes(report)


def test_review_framework_run_rejects_full_chain_proposal_missing_architecture_delta(tmp_path: Path) -> None:
    project = _build_framework_run(tmp_path)
    _write_selected_v2_artifacts(
        project,
        proposal={
            "schema_version": 3,
            "proposal_id": "proposal_node_01",
            "title": "Mechanism-aware active allocation",
            "method_claim": "Mechanism-aware allocation should beat simple baselines.",
            "method_hypothesis": "A productized generated policy can add a repair slot without cloning baselines.",
            "literature_basis": ["Active design literature motivates exploitation plus coverage repair."],
            "literature_gap": "Prior replay policies do not expose a clone guard for selected batches.",
            "algorithm_mechanism": "Rank by mechanism-aware utility, then reserve one low-order repair slot.",
            "expected_advantage": "Lower baseline overlap while preserving best feasible utility.",
            "failure_modes": ["repair slot can displace a high-utility design"],
            "planned_ablations": ["disable_repair_slot"],
        },
    )
    write_method_report(project, run_id="framework_unit")

    report = review_framework_run(project, run_id="framework_unit")

    assert not report["valid"]
    assert "selected_node_architecture_delta_missing" in _codes(report)


def test_review_framework_run_accepts_structured_architecture_delta(tmp_path: Path) -> None:
    project = _build_framework_run(tmp_path)
    _write_selected_v2_artifacts(
        project,
        proposal={
            "schema_version": 3,
            "proposal_id": "proposal_node_01",
            "title": "Mechanism-aware active allocation",
            "method_claim": "Mechanism-aware allocation should beat simple baselines.",
            "method_hypothesis": "A productized generated policy can add a repair slot without cloning baselines.",
            "literature_basis": ["Active design literature motivates exploitation plus coverage repair."],
            "literature_gap": "Prior replay policies do not expose a clone guard for selected batches.",
            "architecture_delta": {
                "type": "non_enum_policy_callable",
                "changes": ["adds evidence-tiered allocation before utility ranking"],
            },
            "new_mechanism_claim": "Evidence-tiered allocation changes the acquisition architecture.",
            "algorithm_mechanism": "Rank by mechanism-aware utility, then reserve one low-order repair slot.",
            "expected_advantage": "Lower baseline overlap while preserving best feasible utility.",
            "failure_modes": ["repair slot can displace a high-utility design"],
            "planned_ablations": ["disable_repair_slot"],
        },
    )
    write_method_report(project, run_id="framework_unit")

    report = review_framework_run(project, run_id="framework_unit")

    assert "selected_node_architecture_delta_missing" not in _codes(report)


def test_review_framework_run_validates_selected_design_space_and_candidate_artifacts_when_present(
    tmp_path: Path,
) -> None:
    project = _build_framework_run(tmp_path)
    workspace = _selected_workspace(project)
    design_space_path = workspace / "design_space.json"
    candidate_artifacts_path = workspace / "candidate_artifacts.json"
    _write_json(design_space_path, {"schema_version": 1})
    _write_json(candidate_artifacts_path, {})
    _write_selected_v2_artifacts(
        project,
        extra_artifacts={
            "design_space": design_space_path,
            "candidate_artifacts": candidate_artifacts_path,
        },
    )
    write_method_report(project, run_id="framework_unit")

    report = review_framework_run(project, run_id="framework_unit")

    assert not report["valid"]
    assert {
        "selected_node_design_space_missing_structure",
        "selected_node_candidate_artifacts_empty",
    } <= _codes(report)


def test_review_framework_run_rejects_selected_method_without_majority_summary_wins(
    tmp_path: Path,
) -> None:
    project = _build_framework_run(tmp_path)
    _write_selected_v2_artifacts(project)
    selected_policy = _selected_policy_name(project)
    _write_benchmark_summary(
        project,
        [
            {
                "method": selected_policy,
                "world_count": "5",
                "beats_random_feasible_worlds": "2",
                "beats_fixed_mix_worlds": "2",
            },
            {
                "method": "random_feasible",
                "world_count": "5",
                "beats_random_feasible_worlds": "0",
                "beats_fixed_mix_worlds": "1",
            },
            {
                "method": "fixed_mix",
                "world_count": "5",
                "beats_random_feasible_worlds": "3",
                "beats_fixed_mix_worlds": "0",
            },
        ],
    )
    write_method_report(project, run_id="framework_unit")

    report = review_framework_run(project, run_id="framework_unit")

    assert not report["valid"]
    assert "selected_method_missing_majority_world_wins" in _codes(report)


def test_review_framework_run_rejects_selected_policy_majority_loss_from_benchmark_results(
    tmp_path: Path,
) -> None:
    project = _build_complete_framework_run(tmp_path)
    selected_policy = _selected_policy_name(project)
    run_dir = project / "runs" / "framework_unit"
    benchmark_path = run_dir / "benchmark_results.csv"
    with benchmark_path.open(encoding="utf-8", newline="") as handle:
        benchmark_rows = list(csv.DictReader(handle))

    for row in benchmark_rows:
        if row["method"] == selected_policy:
            row["best_feasible_utility"] = "0.0"
        elif row["method"] in {"random_feasible", "fixed_mix"}:
            row["best_feasible_utility"] = "1.0"
    with benchmark_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(benchmark_rows[0]))
        writer.writeheader()
        writer.writerows(benchmark_rows)

    summary_path = run_dir / "benchmark_summary.csv"
    with summary_path.open(encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    fieldnames = list(summary_rows[0])
    for field in (
        "beats_random_feasible_worlds",
        "beats_random_feasible_majority",
        "beats_fixed_mix_worlds",
        "beats_fixed_mix_majority",
    ):
        if field not in fieldnames:
            fieldnames.append(field)
    for row in summary_rows:
        row.setdefault("beats_random_feasible_worlds", "0")
        row.setdefault("beats_random_feasible_majority", "false")
        row.setdefault("beats_fixed_mix_worlds", "0")
        row.setdefault("beats_fixed_mix_majority", "false")
        if row["method"] == selected_policy:
            row["beats_random_feasible_worlds"] = "5"
            row["beats_random_feasible_majority"] = "true"
            row["beats_fixed_mix_worlds"] = "5"
            row["beats_fixed_mix_majority"] = "true"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    report = review_framework_run(project, run_id="framework_unit")

    assert not report["valid"]
    assert "selected_method_missing_majority_world_wins" in _codes(report)


def test_review_framework_run_rejects_path_traversal_run_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="path separators"):
        review_framework_run(tmp_path, run_id="../escape")


def _build_framework_run(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / "project.yaml").write_text(
        "project_id: unit\n"
        "name: antibody active design\n"
        "goal: improve framework methods for active antibody design\n",
        encoding="utf-8",
    )
    init_framework(project, domain="antibody active design")
    run_literature_search(project, max_papers=5, offline_fixtures=True)
    extraction = extract_methods(project)
    assert extraction["status"] == "ok"
    develop_method(project, nodes=2, use_codex=False, run_id="framework_unit")
    return project


def _build_complete_framework_run(tmp_path: Path) -> Path:
    project = _build_framework_run(tmp_path)
    audit_references(project)
    append_finding(
        project,
        {
            "summary": "Reference audit identified staged journals and durable quest state.",
            "source": "reference_audit",
        },
    )
    append_failure(
        project,
        {
            "summary": "One candidate method was rejected for weak baseline separation.",
            "source": "method_search",
        },
    )
    write_method_report(project, run_id="framework_unit")
    return project


def _write_selected_v2_artifacts(
    project: Path,
    *,
    proposal: dict[str, object] | None = None,
    novelty_report: dict[str, object] | None = None,
    extra_artifacts: dict[str, Path] | None = None,
) -> None:
    workspace = _selected_workspace(project)
    _write_json(
        workspace / "proposal.json",
        proposal
        or {
            "schema_version": 3,
            "proposal_id": "proposal_node_01",
            "title": "Mechanism-aware active allocation",
            "method_claim": "Mechanism-aware allocation should beat simple baselines.",
            "method_hypothesis": "A productized generated policy can add a repair slot without cloning baselines.",
            "literature_basis": [
                "Active design literature motivates exploitation plus coverage repair.",
            ],
            "literature_gap": "Prior replay policies do not expose a clone guard for selected batches.",
            "algorithm_mechanism": "Rank by mechanism-aware utility, then reserve one low-order repair slot.",
            "expected_advantage": "Lower baseline overlap while preserving best feasible utility.",
            "expected_differentiator": "Uses contrast and uncertainty terms together.",
            "architecture_delta": "Adds a clone-guarded repair slot to the baseline ranking architecture.",
            "no_baseline_rationale": "Synthetic replay overlap stays below the clone threshold.",
            "failure_modes": ["repair slot can displace a high-utility design"],
            "planned_ablations": ["disable_repair_slot"],
        },
    )
    _write_json(
        workspace / "novelty_report.json",
        novelty_report
        or {
            "schema_version": 2,
            "novelty_score": 0.74,
            "baseline_overlap": 0.21,
            "selection_overlap_vs_baselines": 0.21,
            "baseline_clone": False,
            "nearest_baseline": "fixed_mix",
            "architecture_delta": "Adds a repair slot after baseline-family ranking.",
        },
    )
    manifest_path = workspace / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 2
    _write_json(manifest_path, manifest)

    journal_path = project / "runs" / "framework_unit" / "scientist_journal.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    selected_id = journal["selected_node_id"]
    for node in journal["nodes"]:
        if node["node_id"] == selected_id:
            node["artifacts"]["proposal"] = str(workspace / "proposal.json")
            node["artifacts"]["novelty_report"] = str(workspace / "novelty_report.json")
            for key, path in (extra_artifacts or {}).items():
                node["artifacts"][key] = str(path)
    journal["selected_node"]["proposal"] = str(workspace / "proposal.json")
    journal["selected_node"]["novelty_report"] = str(workspace / "novelty_report.json")
    for key, path in (extra_artifacts or {}).items():
        journal["selected_node"][key] = str(path)
    _write_json(journal_path, journal)


def _selected_workspace(project: Path) -> Path:
    journal = json.loads((project / "runs" / "framework_unit" / "scientist_journal.json").read_text(encoding="utf-8"))
    selected_id = journal["selected_node_id"]
    selected = next(node for node in journal["nodes"] if node["node_id"] == selected_id)
    return Path(selected["workspace"])


def _selected_policy_name(project: Path) -> str:
    journal = json.loads((project / "runs" / "framework_unit" / "scientist_journal.json").read_text(encoding="utf-8"))
    selected_id = journal["selected_node_id"]
    selected = next(node for node in journal["nodes"] if node["node_id"] == selected_id)
    policy_name = selected.get("benchmark_policy_name")
    assert isinstance(policy_name, str) and policy_name
    return policy_name


def _write_run_json(project: Path, filename: str, data: dict[str, object]) -> None:
    _write_json(project / "runs" / "framework_unit" / filename, data)


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _write_benchmark_summary(project: Path, rows: list[dict[str, str]]) -> None:
    path = project / "runs" / "framework_unit" / "benchmark_summary.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _codes(report: dict[str, object]) -> set[str]:
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
