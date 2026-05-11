"""Framework-first bootstrap and lightweight review helpers."""

from __future__ import annotations

from pathlib import Path

from design_scientist.artifacts import (
    FRAMEWORK_ARTIFACTS,
    FRAMEWORK_BOOTSTRAP_DIRS,
    FRAMEWORK_SPEC,
    LITERATURE_QUERIES,
    missing_artifacts,
)
from design_scientist.io import ensure_dir, write_yaml
from design_scientist.research_os import initialize_research_os
from design_scientist.schemas import FrameworkSpec, LiteratureQuery


def _framework_id(domain: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "_" for char in domain).strip("_")
    return normalized or "design_scientist_framework"


def seed_literature_queries(domain: str) -> list[LiteratureQuery]:
    """Create conservative seed queries for later literature adapters."""
    return [
        LiteratureQuery(
            query_id="method_foundations",
            domain=domain,
            query=f"{domain} active learning experimental design method benchmark",
            purpose="identify reusable method families and evaluation patterns",
            sources=["adapter_pending"],
        ),
        LiteratureQuery(
            query_id="synthetic_replay_benchmarks",
            domain=domain,
            query=f"{domain} retrospective replay benchmark sequential design",
            purpose="identify synthetic replay protocols and baselines",
            sources=["adapter_pending"],
        ),
    ]


def init_framework(root: str | Path, domain: str) -> FrameworkSpec:
    """Initialize the framework-first artifact layout without running adapters."""
    base = Path(root).expanduser().resolve()
    for rel_dir in FRAMEWORK_BOOTSTRAP_DIRS:
        ensure_dir(base / rel_dir)

    spec = FrameworkSpec(
        framework_id=_framework_id(domain),
        domain=domain,
        artifact_root=str(base),
    )
    write_yaml(base / FRAMEWORK_SPEC, spec)
    write_yaml(base / LITERATURE_QUERIES, {"queries": seed_literature_queries(domain)})
    initialize_research_os(base, domain=domain)
    return spec


def review_framework(root: str | Path) -> dict[str, list[str]]:
    """Return missing framework artifacts; this is intentionally non-throwing."""
    missing = missing_artifacts(root, FRAMEWORK_ARTIFACTS)
    return {"missing_artifacts": missing}
