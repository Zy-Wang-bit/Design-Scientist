from __future__ import annotations

import json
from pathlib import Path

from design_scientist.cli import main
from design_scientist.reference_audit import audit_references


def _write(path: Path, text: str = "# placeholder\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _seed_reference_tree(root: Path) -> None:
    _write(root / "references" / "AI-Scientist" / "templates" / "nanoGPT" / "prompt.json", "{}\n")
    _write(root / "references" / "AI-Scientist" / "ai_scientist" / "perform_review.py")
    _write(root / "references" / "AI-Scientist" / "ai_scientist" / "perform_writeup.py")
    _write(root / "references" / "AI-Scientist-v2" / "ai_scientist" / "treesearch" / "bfts_utils.py")
    _write(root / "references" / "AI-Scientist-v2" / "ai_scientist" / "treesearch" / "journal.py")
    _write(root / "references" / "AI-Scientist-v2" / "ai_scientist" / "treesearch" / "agent_manager.py")
    _write(root / "references" / "DeepScientist" / "src" / "deepscientist" / "quest" / "service.py")
    _write(root / "references" / "DeepScientist" / "src" / "deepscientist" / "memory" / "service.py")
    _write(root / "references" / "DeepScientist" / "src" / "skills" / "optimize" / "SKILL.md")
    _write(root / "references" / "DeepScientist" / "docs" / "en" / "20_WORKSPACE_MODES_GUIDE.md")


def test_audit_references_writes_component_json_and_markdown(tmp_path: Path) -> None:
    _seed_reference_tree(tmp_path)

    result = audit_references(tmp_path)

    assert result["summary"]["total"] == 10
    assert result["summary"]["present"] == 10
    assert (tmp_path / "framework" / "reference_audit.md").exists()
    assert (tmp_path / "framework" / "reference_components.json").exists()

    payload = json.loads((tmp_path / "framework" / "reference_components.json").read_text(encoding="utf-8"))
    component_ids = {component["component_id"] for component in payload["components"]}
    assert {
        "ai_scientist_v1.templates",
        "ai_scientist_v1.review",
        "ai_scientist_v1.writeup",
        "ai_scientist_v2.tree_search",
        "ai_scientist_v2.journal",
        "ai_scientist_v2.stage_manager",
        "deep_scientist.quest",
        "deep_scientist.findings_memory",
        "deep_scientist.bo_loop",
        "deep_scientist.human_takeover",
    } == component_ids

    markdown = (tmp_path / "framework" / "reference_audit.md").read_text(encoding="utf-8")
    assert "AI-Scientist v1" in markdown
    assert "DeepScientist" in markdown
    assert "Human takeover" in markdown


def test_audit_references_cli_writes_expected_artifacts(tmp_path: Path, capsys) -> None:
    _seed_reference_tree(tmp_path)

    exit_code = main(["audit-references", str(tmp_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Wrote reference audit:" in output
    assert (tmp_path / "framework" / "reference_audit.md").exists()
    assert (tmp_path / "framework" / "reference_components.json").exists()
