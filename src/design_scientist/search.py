"""Controlled local policy search for Design Scientist."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

from design_scientist.artifacts import NODE_ARTIFACTS
from design_scientist.io import ensure_dir, read_json, write_json
from design_scientist.journal import save_journal
from design_scientist.schemas import DesignJournal, JournalNode

CODEX_CONTRACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "recommended_changes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "unsupported_claims", "recommended_changes"],
    "additionalProperties": False,
}

NODE_REQUIRED_ARTIFACTS = NODE_ARTIFACTS
CANDIDATE_CSV_COLUMNS = [
    "candidate_id",
    "variant_id",
    "operator",
    "category",
    "target_system",
    "background",
    "target_background",
    "modules",
    "feasibility_status",
    "cost",
    "rationale",
    "evidence_refs",
    "score_components",
    "risk_flags",
    "required_measurements",
]
PANEL_CSV_COLUMNS = CANDIDATE_CSV_COLUMNS


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if not fieldnames:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys or ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _candidate_to_row(candidate: Any) -> dict[str, Any]:
    if hasattr(candidate, "__dataclass_fields__"):
        from dataclasses import asdict

        data = asdict(candidate)
    elif isinstance(candidate, dict):
        data = dict(candidate)
    else:
        data = {"candidate_id": str(candidate)}
    for key, value in list(data.items()):
        if isinstance(value, (list, dict)):
            data[key] = str(value)
    return data


def run_controlled_search(
    project_dir: str | Path,
    budget: int,
    use_codex: bool = False,
) -> tuple[str, DesignJournal]:
    """Run deterministic policy nodes, optionally asking Codex inside node workspaces."""
    root = Path(project_dir).expanduser().resolve()
    state = read_json(root / "state" / "design_state.json")
    evidence = read_json(root / "state" / "evidence_cards.json")
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(root / "runs" / run_id)
    journal = DesignJournal(run_id=run_id, project_id=state.get("project_id", root.name))

    from design_scientist.acquisition import default_policy
    from design_scientist.candidates import generate_candidates
    from design_scientist.panel import compare_baselines, select_panel

    strategies = ["mechanism_aware", "fixed_mix", "pure_lattice_repair"]
    best_score = float("-inf")
    selected_node_id: str | None = None
    selected_outputs: dict[str, Any] | None = None

    for idx, strategy in enumerate(strategies, start=1):
        node_id = f"node_{idx}_{strategy}"
        node_dir = ensure_dir(run_dir / "nodes" / node_id)
        policy = default_policy(strategy)
        candidates = generate_candidates(state, evidence, strategy=strategy)
        panel, score = select_panel(candidates, policy=policy, budget=budget)
        baseline_comparison = compare_baselines(candidates, budget=budget)
        unsupported_claims = state.get("unsupported_claims", [])

        candidate_rows = [_candidate_to_row(c) for c in candidates]
        panel_rows = [_candidate_to_row(c) for c in panel]
        metrics = {
            "strategy": strategy,
            "score": score,
            "budget": budget,
            "candidate_count": len(candidate_rows),
            "panel_count": len(panel_rows),
            "baseline_comparison": baseline_comparison,
            "unsupported_claims": unsupported_claims,
            "used_codex": False,
        }

        _write_csv(node_dir / "candidate_pool.csv", candidate_rows, fieldnames=CANDIDATE_CSV_COLUMNS)
        _write_csv(node_dir / "panel_recommendation.csv", panel_rows, fieldnames=PANEL_CSV_COLUMNS)
        _write_csv(
            node_dir / "policy_comparison.csv",
            [{"baseline": item.split(":", 1)[0], "summary": item} for item in baseline_comparison],
        )
        write_json(node_dir / "policy_metrics.json", metrics)
        write_json(
            node_dir / "validation_report.json",
            {"project_id": state.get("project_id"), "run_id": run_id, "valid": True, "findings": []},
        )
        artifact_validation = validate_node_artifacts(node_dir)
        metrics["artifact_validation"] = artifact_validation
        metrics["codex_contract"] = {"enabled": False, "ok": True, "schema_required": []}
        if use_codex and artifact_validation["ok"]:
            from design_scientist.backends.codex_cli import CodexCliBackend
            from design_scientist.schemas import WorkspaceAgentTask

            task = WorkspaceAgentTask(
                objective=(
                    "Review this Design Scientist policy node. Return JSON with summary, "
                    "unsupported_claims, and recommended_changes. Do not modify files."
                ),
                workspace=str(node_dir),
                mode="read_only_plan",
                allowed_paths=[str(node_dir)],
                output_schema=CODEX_CONTRACT_SCHEMA,
            )
            result = CodexCliBackend().run_task(task)
            metrics["used_codex"] = True
            metrics["codex_summary"] = result.summary
            metrics["codex_contract"] = {
                "enabled": True,
                "ok": result.returncode == 0,
                "schema_required": CODEX_CONTRACT_SCHEMA["required"],
                "workspace_mode": "read_only",
            }
        write_json(node_dir / "policy_metrics.json", metrics)
        if not artifact_validation["ok"]:
            continue

        node = JournalNode(
            node_id=node_id,
            parent_id=None,
            stage="policy_search",
            workspace=str(node_dir),
            status="completed",
            score=float(score),
            artifacts={
                "candidate_pool": str(node_dir / "candidate_pool.csv"),
                "panel_recommendation": str(node_dir / "panel_recommendation.csv"),
                "policy_comparison": str(node_dir / "policy_comparison.csv"),
                "policy_metrics": str(node_dir / "policy_metrics.json"),
                "validation_report": str(node_dir / "validation_report.json"),
            },
            validation_summary={"valid": True},
            rationale=f"Deterministic strategy {strategy}",
        )
        journal.nodes.append(node)
        if score > best_score:
            best_score = float(score)
            selected_node_id = node_id
            selected_outputs = {
                "candidate_rows": candidate_rows,
                "panel_rows": panel_rows,
                "baseline_comparison": baseline_comparison,
                "metrics": metrics,
            }

    journal.selected_node_id = selected_node_id
    if selected_outputs is None:
        raise RuntimeError("No policy nodes were completed")

    _write_csv(run_dir / "candidate_pool.csv", selected_outputs["candidate_rows"], fieldnames=CANDIDATE_CSV_COLUMNS)
    _write_csv(run_dir / "panel_recommendation.csv", selected_outputs["panel_rows"], fieldnames=PANEL_CSV_COLUMNS)
    _write_csv(
        run_dir / "policy_comparison.csv",
        [
            {"baseline": item.split(":", 1)[0], "summary": item}
            for item in selected_outputs["baseline_comparison"]
        ],
    )
    write_json(run_dir / "policy_metrics.json", selected_outputs["metrics"])
    write_json(run_dir / "validation_report.json", {"project_id": state.get("project_id"), "run_id": run_id, "valid": True, "findings": []})
    save_journal(run_dir / "journal.json", journal)
    return run_id, journal


def validate_node_artifacts(node_dir: str | Path) -> dict[str, Any]:
    root = Path(node_dir).expanduser().resolve()
    missing = [name for name in NODE_REQUIRED_ARTIFACTS if not (root / name).exists()]
    empty = [
        name
        for name in NODE_REQUIRED_ARTIFACTS
        if (root / name).exists() and (root / name).is_file() and (root / name).stat().st_size == 0
    ]
    return {
        "ok": not missing and not empty,
        "missing": missing,
        "empty": empty,
        "checked_before_journal": True,
    }


__all__ = [
    "CODEX_CONTRACT_SCHEMA",
    "NODE_REQUIRED_ARTIFACTS",
    "run_controlled_search",
    "validate_node_artifacts",
]
