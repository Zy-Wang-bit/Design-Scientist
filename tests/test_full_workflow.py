from __future__ import annotations

from pathlib import Path

from design_scientist.pipeline import run_full_workflow
from design_scientist.projects import initialize_project
from design_scientist.project_history import load_project_history
from design_scientist.validators import validate_project


def test_full_workflow_creates_framework_state_and_review_packet(tmp_path: Path) -> None:
    project = tmp_path / "anti_hbsag"
    initialize_project(project)

    run_id = run_full_workflow(project, budget=6)
    run_dir = project / "runs" / run_id
    report = validate_project(project, run_id=run_id)

    assert (project / "framework" / "legacy_seed_paper_cards.json").exists()
    assert not (project / "framework" / "paper_cards.json").exists()
    assert (project / "framework" / "research_gap_matrix.csv").exists()
    assert (project / "framework" / "method_hypotheses.md").exists()
    assert (project / "framework" / "policy_registry.yaml").exists()
    assert (project / "framework" / "benchmark_results.csv").exists()
    assert (project / "framework" / "ablation_results.csv").exists()
    assert (project / "framework" / "evaluation_protocol.md").exists()
    assert (project / "state" / "hypothesis_board.json").exists()
    history = load_project_history(project)
    assert history[-1]["event_type"] == "dry_panel_run"
    assert history[-1]["run_id"] == run_id
    assert (run_dir / "decision_report.md").exists()
    assert (run_dir / "human_review_packet.md").exists()
    assert report.valid
