"""Research OS bootstrap artifacts and durable memory append helpers."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

try:
    import fcntl
except ImportError:  # pragma: no cover - POSIX path is used in CI/dev.
    fcntl = None

from design_scientist.io import ensure_dir, write_json, write_yaml
from design_scientist.schemas import to_plain_data

QUEST_ARTIFACT = "framework/quest.yaml"
RESEARCH_MAP_ARTIFACT = "framework/research_map.json"
FINDINGS_MEMORY_ARTIFACT = "framework/findings_memory.jsonl"
FAILURE_MEMORY_ARTIFACT = "framework/failure_memory.jsonl"


def _base(root: str | Path) -> Path:
    return Path(root).expanduser().resolve()


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _quest_id(domain: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "_" for char in domain).strip("_")
    return normalized or "design_scientist_quest"


def _default_quest(domain: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "quest_id": _quest_id(domain),
        "domain": domain,
        "status": "active",
        "active_loop": "reference_ingestion",
        "created_by": "design_scientist.init_framework",
        "artifacts": {
            "research_map": RESEARCH_MAP_ARTIFACT,
            "findings_memory": FINDINGS_MEMORY_ARTIFACT,
            "failure_memory": FAILURE_MEMORY_ARTIFACT,
        },
    }


def _default_research_map(domain: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "domain": domain,
        "current_loop": "reference_ingestion",
        "loops": [
            {
                "loop_id": "reference_ingestion",
                "status": "planned",
                "goal": "Audit external research-agent references without importing their code.",
                "inputs": ["references/AI-Scientist", "references/AI-Scientist-v2", "references/DeepScientist"],
                "outputs": [
                    "framework/reference_audit.md",
                    "framework/reference_components.json",
                ],
            },
            {
                "loop_id": "method_development",
                "status": "planned",
                "goal": "Translate audited patterns into local method hypotheses and policy nodes.",
                "inputs": ["framework/reference_components.json", "framework/paper_cards.json"],
                "outputs": ["framework/method_hypotheses.md", "framework/policy_registry.yaml"],
            },
            {
                "loop_id": "benchmark_and_ablation",
                "status": "planned",
                "goal": "Compare candidate policies against baselines and ablations.",
                "inputs": ["framework/policy_registry.yaml"],
                "outputs": ["framework/benchmark_results.csv", "framework/ablation_results.csv"],
            },
            {
                "loop_id": "research_decision",
                "status": "planned",
                "goal": "Convert validated method evidence into a bounded next research action.",
                "inputs": ["framework/findings_memory.jsonl", "framework/failure_memory.jsonl"],
                "outputs": ["runs/<run_id>/method_report.md"],
            },
        ],
        "memory": {
            "findings": FINDINGS_MEMORY_ARTIFACT,
            "failures": FAILURE_MEMORY_ARTIFACT,
        },
    }


def initialize_research_os(root: str | Path, *, domain: str) -> dict[str, Path]:
    """Create Research OS control files without truncating existing memory logs."""
    base = _base(root)
    framework_dir = ensure_dir(base / "framework")
    paths = {
        "quest": base / QUEST_ARTIFACT,
        "research_map": base / RESEARCH_MAP_ARTIFACT,
        "findings_memory": base / FINDINGS_MEMORY_ARTIFACT,
        "failure_memory": base / FAILURE_MEMORY_ARTIFACT,
    }

    if not paths["quest"].exists():
        write_yaml(paths["quest"], _default_quest(domain))
    if not paths["research_map"].exists():
        write_json(paths["research_map"], _default_research_map(domain))

    for memory_path in (paths["findings_memory"], paths["failure_memory"]):
        ensure_dir(memory_path.parent)
        memory_path.touch(exist_ok=True)

    ensure_dir(framework_dir)
    return paths


def _normalize_memory_record(record_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    summary = str(payload.get("summary") or "").strip()
    if not summary:
        raise ValueError(f"{record_type} memory records require a non-empty summary")
    record = {
        "schema_version": 1,
        "record_type": record_type,
        "record_id": f"{record_type}_{uuid4().hex[:12]}",
        "created_at": _now(),
        **payload,
    }
    return to_plain_data(record)


def _append_jsonl(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    ensure_dir(path.parent)
    line = json.dumps(record, sort_keys=False, ensure_ascii=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(line)
            handle.flush()
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return record


def append_finding(root: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Append a durable finding record to framework/findings_memory.jsonl."""
    record = _normalize_memory_record("finding", dict(payload))
    return _append_jsonl(_base(root) / FINDINGS_MEMORY_ARTIFACT, record)


def append_failure(root: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Append a durable failure record to framework/failure_memory.jsonl."""
    record = _normalize_memory_record("failure", dict(payload))
    return _append_jsonl(_base(root) / FAILURE_MEMORY_ARTIFACT, record)
