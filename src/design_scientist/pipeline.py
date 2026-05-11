"""High-level local pipelines."""

from __future__ import annotations

from pathlib import Path

from design_scientist.project_history import append_history_event
from design_scientist.reporting import write_human_review_packet
from design_scientist.search import run_controlled_search
from design_scientist.validators import validate_project


def run_dry_panel(project_dir: str | Path, budget: int = 24, use_codex: bool = False) -> str:
    run_id, _journal = run_controlled_search(project_dir, budget=budget, use_codex=use_codex)
    _write_decision_report(project_dir, run_id, budget)
    write_human_review_packet(project_dir, run_id=run_id)
    append_history_event(
        project_dir,
        {
            "event_type": "dry_panel_run",
            "status": "completed",
            "run_id": run_id,
            "budget": budget,
            "use_codex": use_codex,
        },
    )
    report = validate_project(project_dir, run_id=run_id)
    write_human_review_packet(project_dir, run_id=run_id, validation_report=report)
    print(f"Created dry-run panel: {Path(project_dir).resolve() / 'runs' / run_id}")
    return run_id


def run_full_workflow(project_dir: str | Path, budget: int = 24, use_codex: bool = False) -> str:
    """Run literature -> state -> hypotheses -> retrospective -> dry panel -> review."""
    root = Path(project_dir).expanduser().resolve()

    from design_scientist.hypotheses import build_hypothesis_board
    from design_scientist.literature import build_research_gap_matrix, create_seed_paper_cards
    from design_scientist.methods import propose_method_hypotheses
    from design_scientist.retrospective import run_retrospective_eval
    from design_scientist.state import build_state

    create_seed_paper_cards(root)
    build_research_gap_matrix(root)
    propose_method_hypotheses(root)
    build_state(root)
    build_hypothesis_board(root)
    run_retrospective_eval(root, budget=budget)
    run_id = run_dry_panel(root, budget=budget, use_codex=use_codex)
    print(f"Completed full workflow for {root}; latest_run={run_id}")
    return run_id


def _write_decision_report(project_dir: str | Path, run_id: str, budget: int) -> Path:
    root = Path(project_dir).expanduser().resolve()
    run_dir = root / "runs" / run_id
    report = run_dir / "decision_report.md"
    report.write_text(
        "# Decision Report\n\n"
        f"Run: `{run_id}`\n\n"
        f"Budget: `{budget}`\n\n"
        "This is a dry-run panel recommendation. It is not a wet-lab-final panel until "
        "the validation report and human review packet are reviewed.\n\n"
        "Required checks: baseline comparison, unsupported claims, provenance guard, "
        "and evidence-tier separation.\n",
        encoding="utf-8",
    )
    return report
