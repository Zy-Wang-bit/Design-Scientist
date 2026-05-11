from __future__ import annotations

from pathlib import Path

from design_scientist.framework import init_framework
from design_scientist.framework_validation import main as validation_main
from design_scientist.framework_validation import review_framework_run
from design_scientist.literature_pipeline import run_literature_search
from design_scientist.method_extraction import extract_methods
from design_scientist.method_report import main as method_report_main
from design_scientist.reference_audit import audit_references
from design_scientist.scientist_search import develop_method


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
    run_literature_search(project, max_papers=6, offline_fixtures=True)
    extraction = extract_methods(project)
    scientist = develop_method(project, nodes=3, use_codex=False, run_id="offline_rd")

    assert extraction["status"] == "ok"
    assert scientist["selected_node"]["benchmark_method"]

    assert method_report_main([str(project), "--run-id", "offline_rd"]) == 0
    out = capsys.readouterr().out
    assert "Wrote method report:" in out

    assert validation_main([str(project), "--run-id", "offline_rd"]) == 0
    out = capsys.readouterr().out
    assert "Framework validation passed" in out

    report = review_framework_run(project, run_id="offline_rd")
    method_report = project / "runs" / "offline_rd" / "method_report.md"
    text = method_report.read_text(encoding="utf-8")

    assert report["valid"]
    assert "## Reference Basis" in text
    assert "framework/reference_audit.md" in text
    assert "## Research OS" in text
    assert "framework/quest.yaml" in text
    assert "## Literature Basis" in text
    assert "## Baseline Comparison" in text
    assert "## Ablation" in text
    assert f"Selected node: `{scientist['selected_node']['node_id']}`" in text
    assert "framework/method_registry.yaml" in text
    assert "framework/literature_search_trace.json" in text
    assert "runs/offline_rd/benchmark_results.csv" in text
    assert (project / "runs" / "offline_rd" / "scientist_journal.json").exists()
