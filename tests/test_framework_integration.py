from __future__ import annotations

import json
from pathlib import Path

from design_scientist.framework import init_framework
from design_scientist.framework_validation import main as validation_main
from design_scientist.framework_validation import review_framework_run
from design_scientist.manuscript import generate_short_paper
from design_scientist.method_report import main as method_report_main
from design_scientist.reference_audit import audit_references
from design_scientist.scientist_search_v3 import run_scientist_v3


def test_offline_framework_rd_flow_generates_report_and_validates_via_module_cli(
    tmp_path: Path,
    capsys,
) -> None:
    project = tmp_path / "framework_project"
    project.mkdir()
    (project / "project.yaml").write_text(
        "project_id: framework_project\n"
        "name: anti-HBsAg pH-dependent antibody design methods\n"
        "goal: benchmark mechanism-aware active design policies\n",
        encoding="utf-8",
    )

    init_framework(project, domain="anti-HBsAg active antibody design")
    audit_references(project)
    _write_minimal_design_state(project)
    scientist = run_scientist_v3(
        project,
        max_papers=6,
        nodes=3,
        rounds=2,
        use_codex=False,
        offline_fixtures=True,
    )
    run_id = scientist["run_id"]

    assert (project / "runs" / run_id / "scientist_journal.json").exists()
    assert scientist["selected_node"]["mechanism"]

    assert method_report_main([str(project), "--run-id", run_id]) == 0
    out = capsys.readouterr().out
    assert "Wrote method report:" in out

    paper = generate_short_paper(project, run_id=run_id)
    assert paper["readiness"]["valid"]

    assert validation_main([str(project), "--run-id", run_id]) == 0
    out = capsys.readouterr().out
    assert "Framework validation passed" in out

    report = review_framework_run(project, run_id=run_id)
    method_report = project / "runs" / run_id / "method_report.md"
    text = method_report.read_text(encoding="utf-8")

    assert report["valid"]
    assert "## Literature Engine V3" in text
    assert "## Mechanism Library" in text
    assert "## MechanismSpec Kernel" in text
    assert "framework/mechanism_library.json" in text
    assert "## Baseline Comparison" in text
    assert "## Ablation" in text
    assert "## Selected Mechanism" in text
    assert f"Selected node: `{scientist['selected_node']['node_id']}`" in text
    assert "framework/literature_search_trace.json" in text
    assert f"runs/{run_id}/mechanism_benchmark_results.csv" in text
    assert (project / "runs" / run_id / "scientist_journal.json").exists()


def _write_minimal_design_state(project: Path) -> None:
    state_dir = project / "state"
    state_dir.mkdir(exist_ok=True)
    (state_dir / "design_state.json").write_text(
        json.dumps(
            {
                "project_id": "framework_project",
                "module_status": [
                    {"module_id": "HD110H", "status": "available"},
                    {"module_id": "HG56H", "status": "available"},
                ],
                "backgrounds": ["target_bg"],
                "known_champions": ["target_bg__HD110H"],
                "unresolved_edges": [
                    {
                        "module_id": "HD110H",
                        "base_variant": "target_bg",
                        "system": "1E62",
                        "evidence_id": "role_split_sdab_1E62",
                        "reason": "integration fixture lattice edge",
                    }
                ],
                "unsupported_claims": [
                    "Synthetic replay ranking does not establish prospective wet-lab superiority."
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (state_dir / "evidence_cards.json").write_text(
        json.dumps(
            [
                {
                    "evidence_id": "role_split_sdab_1E62",
                    "summary": "Integration fixture evidence for candidate generation.",
                }
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
