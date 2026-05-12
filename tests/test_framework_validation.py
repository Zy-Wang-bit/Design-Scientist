from __future__ import annotations

from pathlib import Path

import pytest

from design_scientist.framework import init_framework
from design_scientist.framework_validation import main as validation_main
from design_scientist.framework_validation import review_framework_run
from design_scientist.literature_pipeline import run_literature_search
from design_scientist.method_extraction import extract_methods
from design_scientist.method_report import write_method_report
from design_scientist.scientist_search import develop_method


def test_review_framework_run_fresh_init_uses_v3_contract(tmp_path: Path) -> None:
    project = tmp_path / "project"
    init_framework(project, domain="protein variant design")

    report = review_framework_run(project)
    codes = _codes(report)
    rendered_artifacts = "\n".join(str(finding.get("artifact", "")) for finding in report["findings"])

    assert not report["valid"]
    assert "missing_literature_corpus" in codes
    assert "missing_mechanism_library" in codes
    assert "framework/method_modules.json" not in rendered_artifacts
    assert "framework/algorithm_spec.md" not in rendered_artifacts
    assert "framework/method_registry.yaml" not in rendered_artifacts


def test_review_framework_run_rejects_legacy_v2_scientist_run(tmp_path: Path) -> None:
    project = _build_legacy_v2_framework_run(tmp_path)
    write_method_report(project, run_id="legacy_v2")

    report = review_framework_run(project, run_id="legacy_v2")
    codes = _codes(report)

    assert not report["valid"]
    assert "scientist_journal_version_not_v3" in codes
    assert "missing_mechanism_benchmark_results" in codes
    assert "missing_selected_node_mechanism_spec" in codes


def test_review_framework_run_rejects_path_traversal_run_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="path separators"):
        review_framework_run(tmp_path, run_id="../escape")


def test_framework_validation_module_cli_rejects_unsafe_run_id_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        validation_main([str(tmp_path), "--run-id", "../escape"])

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "run_id must not contain path separators or '..'" in captured.err
    assert "Traceback" not in captured.err


def _build_legacy_v2_framework_run(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / "project.yaml").write_text(
        "project_id: unit\n"
        "name: antibody active design\n"
        "goal: legacy v2 framework run should not validate under V3\n",
        encoding="utf-8",
    )
    init_framework(project, domain="antibody active design")
    run_literature_search(project, max_papers=5, offline_fixtures=True)
    extraction = extract_methods(project)
    assert extraction["status"] == "ok"
    _write_minimal_design_state(project)
    develop_method(project, nodes=2, use_codex=False, run_id="legacy_v2")
    return project


def _write_minimal_design_state(project: Path) -> None:
    state_dir = project / "state"
    state_dir.mkdir(exist_ok=True)
    (state_dir / "design_state.json").write_text(
        """{
  "project_id": "unit",
  "module_status": [
    {"module_id": "HD110H", "status": "available"},
    {"module_id": "HG56H", "status": "available"}
  ],
  "backgrounds": ["target_bg"],
  "known_champions": ["target_bg__HD110H"],
  "unresolved_edges": [
    {
      "module_id": "HD110H",
      "base_variant": "target_bg",
      "system": "1E62",
      "evidence_id": "role_split_sdab_1E62",
      "reason": "unit fixture lattice edge"
    }
  ],
  "unsupported_claims": [
    "Synthetic replay ranking does not establish prospective wet-lab superiority."
  ]
}
""",
        encoding="utf-8",
    )
    (state_dir / "evidence_cards.json").write_text(
        """[
  {
    "evidence_id": "role_split_sdab_1E62",
    "summary": "Unit fixture evidence for candidate generation."
  }
]
""",
        encoding="utf-8",
    )


def _codes(report: dict[str, object]) -> set[str]:
    findings = report["findings"]
    assert isinstance(findings, list)
    return {
        finding["code"]
        for finding in findings
        if isinstance(finding, dict) and finding.get("severity") == "error"
    }
