"""Deterministic reference-repository audit for framework design patterns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from design_scientist.io import ensure_dir, write_json


@dataclass(frozen=True)
class ComponentSpec:
    component_id: str
    source: str
    component: str
    patterns: tuple[str, ...]
    summary: str
    design_scientist_use: str


COMPONENT_SPECS: tuple[ComponentSpec, ...] = (
    ComponentSpec(
        component_id="ai_scientist_v1.templates",
        source="AI-Scientist v1",
        component="Templates",
        patterns=("references/AI-Scientist/templates",),
        summary="Template directories package prompts, seed ideas, experiment code, and plotting hooks.",
        design_scientist_use="Keep local benchmark projects explicit: prompt, experiment, plotting, and seed artifacts should stay inspectable.",
    ),
    ComponentSpec(
        component_id="ai_scientist_v1.review",
        source="AI-Scientist v1",
        component="Review",
        patterns=(
            "references/AI-Scientist/ai_scientist/perform_review.py",
            "references/AI-Scientist/review_ai_scientist",
        ),
        summary="Review logic turns generated papers into structured critique and optional improvement signals.",
        design_scientist_use="Separate method claims from review judgments and keep critique artifacts machine-readable.",
    ),
    ComponentSpec(
        component_id="ai_scientist_v1.writeup",
        source="AI-Scientist v1",
        component="Writeup",
        patterns=("references/AI-Scientist/ai_scientist/perform_writeup.py",),
        summary="Writeup flow composes LaTeX sections, citations, compilation, and correction loops.",
        design_scientist_use="Treat reports as generated artifacts backed by evidence paths rather than as implicit chat output.",
    ),
    ComponentSpec(
        component_id="ai_scientist_v2.tree_search",
        source="AI-Scientist v2",
        component="Tree search",
        patterns=(
            "references/AI-Scientist-v2/ai_scientist/treesearch/bfts_utils.py",
            "references/AI-Scientist-v2/launch_scientist_bfts.py",
        ),
        summary="Tree search explores multiple implementation nodes and selects stronger branches.",
        design_scientist_use="Represent method variants as comparable nodes with explicit selection and baseline evidence.",
    ),
    ComponentSpec(
        component_id="ai_scientist_v2.journal",
        source="AI-Scientist v2",
        component="Journal",
        patterns=(
            "references/AI-Scientist-v2/ai_scientist/treesearch/journal.py",
            "references/AI-Scientist-v2/ai_scientist/treesearch/journal2report.py",
        ),
        summary="Journal objects preserve node history and support report generation from search traces.",
        design_scientist_use="Persist scientist-loop decisions before summarizing them into reports.",
    ),
    ComponentSpec(
        component_id="ai_scientist_v2.stage_manager",
        source="AI-Scientist v2",
        component="Stage manager",
        patterns=(
            "references/AI-Scientist-v2/ai_scientist/treesearch/agent_manager.py",
            "references/AI-Scientist-v2/bfts_config.yaml",
        ),
        summary="Stage management tracks stage goals, transitions, and readiness for the next stage.",
        design_scientist_use="Model framework R&D as staged loops instead of one flat run.",
    ),
    ComponentSpec(
        component_id="deep_scientist.quest",
        source="DeepScientist",
        component="Quest",
        patterns=("references/DeepScientist/src/deepscientist/quest/service.py",),
        summary="Quest service creates a durable project root with status, prompts, and execution state.",
        design_scientist_use="Initialize framework work as a durable quest with explicit state files.",
    ),
    ComponentSpec(
        component_id="deep_scientist.findings_memory",
        source="DeepScientist",
        component="Findings memory",
        patterns=(
            "references/DeepScientist/src/deepscientist/memory/service.py",
            "references/DeepScientist/docs/en/07_MEMORY_AND_MCP.md",
        ),
        summary="Memory separates reusable findings from operational artifacts to reduce rediscovery.",
        design_scientist_use="Append findings and failures to JSONL logs that can be replayed by later framework loops.",
    ),
    ComponentSpec(
        component_id="deep_scientist.bo_loop",
        source="DeepScientist",
        component="BO loop",
        patterns=(
            "references/DeepScientist/src/skills/optimize/SKILL.md",
            "references/DeepScientist/src/skills/optimize/references/optimization-memory-template.md",
        ),
        summary="Optimization skill manages candidate briefs, durable lines, frontier review, and route changes.",
        design_scientist_use="Use bounded candidate pools, acquisition evidence, and failure memory for active design loops.",
    ),
    ComponentSpec(
        component_id="deep_scientist.human_takeover",
        source="DeepScientist",
        component="Human takeover",
        patterns=(
            "references/DeepScientist/docs/en/20_WORKSPACE_MODES_GUIDE.md",
            "references/DeepScientist/src/deepscientist/quest/service.py",
        ),
        summary="Workspace modes define when the system waits for user steering versus autonomous continuation.",
        design_scientist_use="Keep human checkpoints explicit for costly experiments, external side effects, and final decisions.",
    ),
)


def _base(root: str | Path) -> Path:
    return Path(root).expanduser().resolve()


def _has_glob(pattern: str) -> bool:
    return any(char in pattern for char in "*?[]")


def _matches(base: Path, patterns: tuple[str, ...]) -> list[str]:
    paths: list[Path] = []
    for pattern in patterns:
        if _has_glob(pattern):
            paths.extend(path for path in base.glob(pattern) if path.exists())
            continue
        candidate = base / pattern
        if candidate.exists():
            paths.append(candidate)
    unique = sorted({path.relative_to(base).as_posix() for path in paths})
    return unique


def build_reference_components(root: str | Path) -> dict[str, Any]:
    """Return deterministic component records based on reference file presence."""
    base = _base(root)
    components = []
    for spec in COMPONENT_SPECS:
        paths = _matches(base, spec.patterns)
        components.append(
            {
                "component_id": spec.component_id,
                "source": spec.source,
                "component": spec.component,
                "status": "present" if paths else "missing",
                "paths": paths,
                "summary": spec.summary,
                "design_scientist_use": spec.design_scientist_use,
            }
        )
    present = sum(1 for component in components if component["status"] == "present")
    return {
        "schema_version": 1,
        "root": str(base),
        "summary": {
            "total": len(components),
            "present": present,
            "missing": len(components) - present,
        },
        "components": components,
    }


def _markdown_for(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Reference Audit",
        "",
        "Deterministic audit of external reference repositories. This report records file-presence signals and design takeaways; no reference code is imported.",
        "",
        "## Summary",
        "",
        f"- Components present: {summary['present']}/{summary['total']}",
        f"- Components missing: {summary['missing']}",
        "",
    ]
    sources = []
    for component in payload["components"]:
        if component["source"] not in sources:
            sources.append(component["source"])

    for source in sources:
        lines.extend([f"## {source}", ""])
        for component in payload["components"]:
            if component["source"] != source:
                continue
            lines.extend(
                [
                    f"### {component['component']}",
                    "",
                    f"- Status: {component['status']}",
                    f"- Component ID: `{component['component_id']}`",
                    f"- Summary: {component['summary']}",
                    f"- Design Scientist use: {component['design_scientist_use']}",
                    "- Evidence paths:",
                ]
            )
            if component["paths"]:
                lines.extend(f"  - `{path}`" for path in component["paths"])
            else:
                lines.append("  - Missing")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def audit_references(root: str | Path) -> dict[str, Any]:
    """Write reference audit Markdown and component JSON under framework/."""
    base = _base(root)
    framework_dir = ensure_dir(base / "framework")
    payload = build_reference_components(base)
    markdown_path = framework_dir / "reference_audit.md"
    json_path = framework_dir / "reference_components.json"

    markdown_path.write_text(_markdown_for(payload), encoding="utf-8")
    write_json(json_path, payload)
    payload["artifacts"] = {
        "reference_audit": str(markdown_path),
        "reference_components": str(json_path),
    }
    return payload
