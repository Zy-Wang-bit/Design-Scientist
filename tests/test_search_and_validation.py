from __future__ import annotations

from pathlib import Path

from design_scientist.io import write_json
from design_scientist.pipeline import run_dry_panel
from design_scientist.search import run_controlled_search
from design_scientist.validators import validate_project


def test_controlled_search_writes_required_artifacts_and_validates(tmp_path: Path) -> None:
    project = tmp_path / "anti_hbsag"
    _write_minimal_project(project)

    run_id = run_dry_panel(project, budget=3)
    run_dir = project / "runs" / run_id
    report = validate_project(project, run_id=run_id)

    assert (run_dir / "candidate_pool.csv").exists()
    assert (run_dir / "panel_recommendation.csv").exists()
    assert (run_dir / "policy_comparison.csv").exists()
    assert (run_dir / "policy_metrics.json").exists()
    assert (run_dir / "decision_report.md").exists()
    assert (run_dir / "human_review_packet.md").exists()
    assert report.valid


def test_controlled_search_records_policy_comparison_in_each_journal_node(tmp_path: Path) -> None:
    project = tmp_path / "anti_hbsag"
    _write_minimal_project(project)

    _run_id, journal = run_controlled_search(project, budget=3)

    assert journal.nodes
    assert all("policy_comparison" in node.artifacts for node in journal.nodes)


def test_validate_project_write_report_false_is_read_only(tmp_path: Path) -> None:
    project = tmp_path / "anti_hbsag"
    _write_minimal_project(project)

    report = validate_project(project, write_report=False)

    assert report.valid
    assert not (project / "state" / "validation_report.json").exists()


def test_validate_project_rejects_stale_onee62_in_run_artifacts(tmp_path: Path) -> None:
    project = tmp_path / "anti_hbsag"
    _write_minimal_project(project)
    run_dir = project / "runs" / "run_with_stale_identifier"
    run_dir.mkdir()
    (run_dir / "candidate_pool.csv").write_text(
        "candidate_id,target_system,base_variant\nc1,1e+62,com1\n",
        encoding="utf-8",
    )
    (run_dir / "panel_recommendation.csv").write_text(
        "candidate_id,target_system,base_variant\nc1,1E62,com1\n",
        encoding="utf-8",
    )
    (run_dir / "policy_comparison.csv").write_text("baseline,summary\nfixed,ok\n", encoding="utf-8")
    write_json(
        run_dir / "policy_metrics.json",
        {
            "baseline_comparison": ["fixed: ok"],
            "unsupported_claims": [
                "Current 1E62 combo data cannot support module causal claims; exact primary matched edges are missing."
            ],
        },
    )
    write_json(
        run_dir / "validation_report.json",
        {
            "project_id": "anti_hbsag",
            "run_id": "run_with_stale_identifier",
            "valid": True,
            "findings": [],
            "artifact_inputs": [{"system": "1e+62"}],
        },
    )
    (run_dir / "decision_report.md").write_text("Decision report\n", encoding="utf-8")
    (run_dir / "human_review_packet.md").write_text("Review packet\n", encoding="utf-8")

    report = validate_project(project, run_id="run_with_stale_identifier", write_report=False)

    assert not report.valid
    assert any(finding.code == "stale_onee62_identifier" for finding in report.findings)


def test_validate_project_rejects_invalid_panel_membership_duplicates_and_feasibility(tmp_path: Path) -> None:
    project = tmp_path / "anti_hbsag"
    _write_minimal_project(project)
    run_dir = project / "runs" / "run_with_invalid_panel"
    run_dir.mkdir()
    _write_required_run_artifacts(
        run_dir,
        candidate_pool_csv=(
            "candidate_id,operator,category,target_system,background,modules,feasibility_status,cost\n"
            "valid,add_module,champion,1E62,real_bg,HD110H,feasible,1\n"
            "infeasible,add_module,champion,1E62,real_bg,HG56H,infeasible,1\n"
        ),
        panel_csv=(
            "candidate_id,operator,category,target_system,background,modules,feasibility_status,cost\n"
            "valid,add_module,champion,1E62,real_bg,HD110H,feasible,1\n"
            "valid,add_module,champion,1E62,real_bg,HD110H,feasible,1\n"
            "infeasible,add_module,champion,1E62,real_bg,HG56H,feasible,1\n"
            "outside_pool,add_module,champion,1E62,real_bg,HN54H,feasible,1\n"
        ),
    )

    report = validate_project(project, run_id="run_with_invalid_panel", write_report=False)
    codes = {finding.code for finding in report.findings}

    assert not report.valid
    assert "panel_candidate_not_in_pool" in codes
    assert "panel_duplicate_candidate" in codes
    assert "panel_infeasible_candidate" in codes


def test_validate_project_rejects_panel_artifacts_missing_required_columns(tmp_path: Path) -> None:
    project = tmp_path / "anti_hbsag"
    _write_minimal_project(project)
    run_dir = project / "runs" / "run_with_missing_panel_columns"
    run_dir.mkdir()
    _write_required_run_artifacts(
        run_dir,
        candidate_pool_csv="candidate_id,feasibility_status,cost\nvalid,feasible,1\n",
        panel_csv="operator,category,target_system\nadd_module,champion,1E62\n",
    )

    report = validate_project(project, run_id="run_with_missing_panel_columns", write_report=False)
    codes = {finding.code for finding in report.findings}

    assert not report.valid
    assert "missing_panel_column" in codes


def _write_minimal_project(project: Path) -> None:
    (project / "state").mkdir(parents=True)
    (project / "runs").mkdir()
    (project / "project.yaml").write_text("project_id: anti_hbsag\n", encoding="utf-8")
    (project / "estimands.yaml").write_text("project_id: anti_hbsag\n", encoding="utf-8")
    (project / "data_contract.yaml").write_text(
        "project_id: anti_hbsag\nraw_access_policy: schema_audit_only\nallowed_tables: []\n",
        encoding="utf-8",
    )
    write_json(
        project / "state" / "design_state.json",
        {
            "project_id": "anti_hbsag",
            "data_version": "test",
            "system_roles": {
                "sdAb": "module_learning",
                "1E62": "target_system_design_genotype_coverage",
            },
            "module_status": [{"module_id": "HD110H"}, {"module_id": "HG56H"}],
            "unresolved_edges": [{"system": "1E62", "module_id": "HA23K", "base_variant": "com1"}],
            "unsupported_claims": [
                "Current 1E62 combo data cannot support module causal claims; exact primary matched edges are missing."
            ],
        },
    )
    write_json(project / "state" / "evidence_cards.json", [])


def _write_required_run_artifacts(
    run_dir: Path,
    *,
    candidate_pool_csv: str,
    panel_csv: str,
) -> None:
    (run_dir / "candidate_pool.csv").write_text(candidate_pool_csv, encoding="utf-8")
    (run_dir / "panel_recommendation.csv").write_text(panel_csv, encoding="utf-8")
    (run_dir / "policy_comparison.csv").write_text("baseline,summary\nfixed,ok\n", encoding="utf-8")
    write_json(
        run_dir / "policy_metrics.json",
        {
            "budget": 3,
            "baseline_comparison": ["fixed: ok"],
            "unsupported_claims": [
                "Current 1E62 combo data cannot support module causal claims; exact primary matched edges are missing."
            ],
        },
    )
    write_json(
        run_dir / "validation_report.json",
        {"project_id": "anti_hbsag", "run_id": run_dir.name, "valid": True, "findings": []},
    )
    (run_dir / "decision_report.md").write_text("Decision report\n", encoding="utf-8")
    (run_dir / "human_review_packet.md").write_text("Review packet\n", encoding="utf-8")
