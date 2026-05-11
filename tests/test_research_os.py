from __future__ import annotations

import json
from pathlib import Path

import yaml
import pytest

from design_scientist.framework import init_framework
from design_scientist.research_os import append_failure, append_finding, initialize_research_os


def test_initialize_research_os_creates_quest_map_and_empty_memory_logs(tmp_path: Path) -> None:
    artifacts = initialize_research_os(tmp_path, domain="protein design")

    assert artifacts["quest"].name == "quest.yaml"
    quest = yaml.safe_load((tmp_path / "framework" / "quest.yaml").read_text(encoding="utf-8"))
    research_map = json.loads((tmp_path / "framework" / "research_map.json").read_text(encoding="utf-8"))

    assert quest["domain"] == "protein design"
    assert quest["status"] == "active"
    assert research_map["schema_version"] == 1
    assert research_map["loops"][0]["loop_id"] == "reference_ingestion"
    assert (tmp_path / "framework" / "findings_memory.jsonl").read_text(encoding="utf-8") == ""
    assert (tmp_path / "framework" / "failure_memory.jsonl").read_text(encoding="utf-8") == ""


def test_init_framework_includes_research_os_artifacts_without_losing_existing_artifacts(
    tmp_path: Path,
) -> None:
    init_framework(tmp_path, domain="antibody design")

    assert (tmp_path / "framework" / "framework_spec.yaml").exists()
    assert (tmp_path / "framework" / "literature_queries.yaml").exists()
    assert (tmp_path / "framework" / "quest.yaml").exists()
    assert (tmp_path / "framework" / "research_map.json").exists()
    assert (tmp_path / "framework" / "findings_memory.jsonl").exists()
    assert (tmp_path / "framework" / "failure_memory.jsonl").exists()


def test_init_framework_rejects_existing_research_os_domain_mismatch(tmp_path: Path) -> None:
    init_framework(tmp_path, domain="protein design")

    with pytest.raises(ValueError, match="Research OS domain mismatch"):
        init_framework(tmp_path, domain="antibody design")


def test_init_framework_force_domain_update_rewrites_research_os_domain(tmp_path: Path) -> None:
    init_framework(tmp_path, domain="protein design")

    init_framework(tmp_path, domain="antibody design", force_domain_update=True)

    quest = yaml.safe_load((tmp_path / "framework" / "quest.yaml").read_text(encoding="utf-8"))
    research_map = json.loads((tmp_path / "framework" / "research_map.json").read_text(encoding="utf-8"))
    assert quest["domain"] == "antibody design"
    assert quest["quest_id"] == "antibody_design"
    assert research_map["domain"] == "antibody design"


def test_append_finding_and_failure_write_jsonl_records(tmp_path: Path) -> None:
    initialize_research_os(tmp_path, domain="protein design")

    finding = append_finding(
        tmp_path,
        {
            "summary": "Reference audit found a reusable tree-search journal pattern.",
            "source": "reference_audit",
            "evidence_paths": ["framework/reference_components.json"],
        },
    )
    failure = append_failure(
        tmp_path,
        {
            "summary": "A policy variant failed retrospective replay.",
            "source": "synthetic_replay",
            "next_action": "tighten acquisition constraints",
        },
    )

    finding_lines = (tmp_path / "framework" / "findings_memory.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    failure_lines = (tmp_path / "framework" / "failure_memory.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()

    assert finding["record_id"].startswith("finding_")
    assert failure["record_id"].startswith("failure_")
    assert json.loads(finding_lines[0])["summary"].startswith("Reference audit")
    assert json.loads(failure_lines[0])["next_action"] == "tighten acquisition constraints"


def test_memory_append_does_not_allow_payload_to_override_reserved_fields(tmp_path: Path) -> None:
    initialize_research_os(tmp_path, domain="protein design")

    finding = append_finding(
        tmp_path,
        {
            "summary": "Reference audit found a reusable tree-search journal pattern.",
            "record_type": "failure",
            "record_id": "external_record",
            "created_at": "2000-01-01T00:00:00+00:00",
        },
    )

    stored = json.loads(
        (tmp_path / "framework" / "findings_memory.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert finding["record_type"] == "finding"
    assert stored["record_type"] == "finding"
    assert finding["record_id"].startswith("finding_")
    assert stored["record_id"].startswith("finding_")
    assert finding["created_at"] != "2000-01-01T00:00:00+00:00"
    assert stored["created_at"] != "2000-01-01T00:00:00+00:00"
