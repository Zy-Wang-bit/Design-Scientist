from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

import design_scientist.literature_pipeline as literature_pipeline
from design_scientist.cli import main
from design_scientist.framework import init_framework
from design_scientist.io import read_json
from design_scientist.literature_pipeline import run_literature_search


FIXTURES = Path(__file__).parent / "fixtures" / "literature"


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_literature_plan_uses_objective_estimands_and_allowed_source_hints(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("DESIGN_SCIENTIST_LITERATURE_FIXTURES", str(FIXTURES))
    project = tmp_path / "project"
    _write_yaml(
        project / "project.yaml",
        {
            "project_id": "ph_sensitive_antibody",
            "domain": "antibody_variant_design",
            "system": "1E62",
            "objective": (
                "Design pH-sensitive anti-HBsAg antibody variants that retain "
                "neutral binding and release at acidic pH."
            ),
            "primary_goal": "Protein engineering for pH-dependent binding selectivity.",
        },
    )
    _write_yaml(
        project / "estimands.yaml",
        {
            "primary": [
                {
                    "name": "neutral binding retention",
                    "endpoint": "pH 7.4 binding",
                    "interpretation": "retain neutral-pH HBsAg binding",
                },
                {
                    "name": "acid release",
                    "endpoint": "pH 6.0 binding reduction",
                    "interpretation": "lower acidic-pH binding relative to neutral binding is better",
                },
            ],
        },
    )
    _write_yaml(
        project / "data_contract.yaml",
        {
            "allowed_sources": {
                "wet_lab": {
                    "evidence_tier": "primary_wet_lab",
                    "files": ["raw/wet_lab/ph_binding.csv"],
                },
                "base_sequences": {
                    "evidence_tier": "sequence_reference",
                    "files": ["raw/base_sequences/antigens.fasta"],
                },
            }
        },
    )

    run_literature_search(project, max_papers=1, offline_fixtures=True, sources=["pubmed"])

    plan = read_json(project / "framework" / "literature_search_plan.json")
    query_text = " ".join(query["query"] for query in plan["queries"]).lower()
    assert plan["queries"][0]["sources"] == ["pubmed"]
    assert "ph-sensitive" in query_text
    assert "ph-dependent" in query_text
    assert "antibody" in query_text
    assert "protein engineering" in query_text
    assert "neutral binding retention" in query_text
    assert "ph 6.0 binding reduction" in query_text
    assert "primary wet lab" in query_text
    assert "sequence reference" in query_text


def test_literature_plan_stays_generic_when_project_context_has_no_ph_or_antibody_terms(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("DESIGN_SCIENTIST_LITERATURE_FIXTURES", str(FIXTURES))
    project = tmp_path / "project"
    _write_yaml(
        project / "project.yaml",
        {
            "project_id": "battery_protocol",
            "domain": "materials_process_optimization",
            "objective": "Optimize battery cycling protocols for capacity retention.",
            "primary_goal": "Sequential experimental design for durable cells.",
        },
    )
    _write_yaml(
        project / "estimands.yaml",
        {
            "primary": [
                {
                    "name": "capacity retention",
                    "endpoint": "cycle 500 capacity",
                    "interpretation": "higher retained capacity is better",
                }
            ]
        },
    )

    run_literature_search(project, max_papers=1, offline_fixtures=True, sources=["pubmed"])

    plan = read_json(project / "framework" / "literature_search_plan.json")
    query_text = " ".join(query["query"] for query in plan["queries"]).lower()
    assert "battery" in query_text
    assert "capacity retention" in query_text
    assert "ph-sensitive" not in query_text
    assert "ph-dependent" not in query_text
    assert "antibody" not in query_text
    assert "protein engineering" not in query_text


def test_init_framework_seeds_project_specific_literature_queries_when_project_yaml_exists(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    _write_yaml(
        project / "project.yaml",
        {
            "domain": "antibody_variant_design",
            "system": "1E62",
            "objective": "Engineer pH-sensitive anti-HBsAg antibody variants.",
            "primary_goal": "Retain neutral binding while reducing acidic binding.",
        },
    )
    _write_yaml(
        project / "estimands.yaml",
        {
            "primary": [
                {
                    "name": "acid release",
                    "endpoint": "pH 6.0 binding reduction",
                    "interpretation": "lower acid binding is better",
                }
            ]
        },
    )

    init_framework(project, domain="protein variant design")

    queries = yaml.safe_load((project / "framework" / "literature_queries.yaml").read_text(encoding="utf-8"))[
        "queries"
    ]
    query_ids = {query["query_id"] for query in queries}
    query_text = " ".join(query["query"] for query in queries).lower()
    assert {"method_foundations", "synthetic_replay_benchmarks", "project_context"}.issubset(query_ids)
    assert "ph-sensitive" in query_text
    assert "ph-dependent" in query_text
    assert "antibody" in query_text
    assert "protein engineering" in query_text
    assert "ph 6.0 binding reduction" in query_text


def test_literature_source_errors_redact_api_key_values(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("S2_API_KEY", "test-secret-key")

    def leaking_adapter(*args, **kwargs):
        raise RuntimeError("semantic scholar failed with test-secret-key in request")

    monkeypatch.setitem(literature_pipeline.SOURCE_ADAPTERS, "leaky", leaking_adapter)

    run_literature_search(project, max_papers=1, sources=["leaky"])

    trace = read_json(project / "framework" / "literature_search_trace.json")
    errors = read_json(project / "framework" / "cache" / "literature_raw" / "source_errors.json")
    serialized = json.dumps({"trace": trace, "errors": errors})
    assert "test-secret-key" not in serialized
    assert "[redacted]" in serialized


def test_cli_literature_search_sources_override_planned_sources(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("DESIGN_SCIENTIST_LITERATURE_FIXTURES", str(FIXTURES))
    project = tmp_path / "project"
    framework = project / "framework"
    framework.mkdir(parents=True)
    (framework / "literature_queries.yaml").write_text(
        "queries:\n"
        "- query_id: planned\n"
        "  query: protein engineering antibody\n"
        "  sources: [semantic_scholar]\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "literature-search",
                str(project),
                "--offline-fixtures",
                "--max-papers",
                "1",
                "--sources",
                "pubmed",
            ]
        )
        == 0
    )

    plan = read_json(project / "framework" / "literature_search_plan.json")
    assert plan["queries"][0]["sources"] == ["pubmed"]


def test_cli_run_scientist_sources_are_scoped_for_v3_literature_chain(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    seen_sources: list[str | None] = []
    monkeypatch.delenv("DESIGN_SCIENTIST_LITERATURE_SOURCES", raising=False)

    def fake_run_scientist_v3(*args, **kwargs):
        seen_sources.append(os.getenv("DESIGN_SCIENTIST_LITERATURE_SOURCES"))
        return {
            "run_id": "run_001",
            "journal_path": str(project / "runs" / "run_001" / "scientist_journal.json"),
            "selected_node": {"artifacts": {}},
        }

    monkeypatch.setattr("design_scientist.scientist_search_v3.run_scientist_v3", fake_run_scientist_v3)
    monkeypatch.setattr(
        "design_scientist.method_report.write_method_report",
        lambda root, run_id: Path(root) / "runs" / run_id / "method_report.md",
    )

    assert (
        main(
            [
                "run-scientist",
                str(project),
                "--sources",
                "pubmed",
                "arxiv",
                "--skip-paper",
            ]
        )
        == 0
    )

    assert seen_sources == ["pubmed,arxiv"]
    assert os.getenv("DESIGN_SCIENTIST_LITERATURE_SOURCES") is None
