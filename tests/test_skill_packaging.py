from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = ROOT / "skills" / "design-scientist"


def test_design_scientist_skill_is_packaged_for_github_install() -> None:
    required_files = [
        "SKILL.md",
        "agents/openai.yaml",
        "scripts/validate_project.py",
        "references/acquisition.md",
        "references/framework_development.md",
        "references/review_checklist.md",
        "references/workflow.md",
    ]

    missing = [path for path in required_files if not (SKILL_DIR / path).is_file()]
    assert not missing

    skill_text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    assert skill_text.startswith("---\n")
    assert "name: design-scientist" in skill_text
    assert "description:" in skill_text


def test_packaged_skill_has_no_machine_local_paths_or_secret_literals() -> None:
    forbidden_fragments = [
        "/Users/",
        "/Volumes/",
        "/private/",
        "g" + "ho_",
        "s" + "k-",
    ]

    checked_files = [
        path
        for path in SKILL_DIR.rglob("*")
        if path.is_file() and path.suffix in {".md", ".py", ".yaml", ".yml"}
    ]
    assert checked_files

    violations: list[str] = []
    for path in checked_files:
        text = path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            if fragment in text:
                violations.append(f"{path.relative_to(ROOT)} contains {fragment}")

    assert not violations
