from __future__ import annotations

from pathlib import Path

from design_scientist.reference_manifest import (
    build_reference_data_sources,
    expected_reference_sources,
)


def test_reference_manifest_ties_claims_to_selected_node_and_review_sources(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    run_dir = root / "runs" / "unit"
    workspace = run_dir / "nodes" / "node_01"
    selected_node = {
        "node_id": "node_01",
        "artifacts": {
            "mechanism_spec": str(workspace / "mechanism_spec.json"),
            "operator_to_code_trace": str(workspace / "operator_to_code_trace.json"),
            "proposal": str(workspace / "proposal.json"),
            "validation_report": str(workspace / "validation_report.json"),
            "mechanism_metrics": str(workspace / "mechanism_metrics.json"),
        },
    }

    manifest = build_reference_data_sources(
        root,
        run_dir,
        run_id="unit",
        selected_node=selected_node,
    )

    assert manifest["schema_version"] == 1
    assert manifest["selected_node_id"] == "node_01"
    assert "framework/operator_specs.json" in manifest["framework_sources"]["operators"]
    assert "runs/unit/nodes/node_01/operator_to_code_trace.json" in manifest["framework_sources"]["node"]
    assert {
        dependency["claim_id"] for dependency in manifest["claim_dependencies"]
    } == {
        "algorithmic_novelty",
        "baseline_superiority",
        "biological_evidence_boundary",
    }
    flattened = expected_reference_sources(manifest)
    assert "runs/unit/benchmark_saturation.json" in flattened
    assert "runs/unit/biology_review.json" in flattened


def test_reference_manifest_allows_no_selected_node_without_faking_node_sources(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    run_dir = root / "runs" / "unit"

    manifest = build_reference_data_sources(
        root,
        run_dir,
        run_id="unit",
        selected_node=None,
    )

    assert manifest["selected_node_id"] is None
    assert manifest["framework_sources"]["node"] == []
    novelty = next(
        dependency
        for dependency in manifest["claim_dependencies"]
        if dependency["claim_id"] == "algorithmic_novelty"
    )
    assert novelty["required_sources"] == [
        "framework/operator_specs.json",
        "runs/unit/novelty_review.json",
    ]
