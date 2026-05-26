"""Run-level source-of-truth manifest for V3 evidence and claim dependencies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from design_scientist.artifacts import (
    V3_LITERATURE_CORPUS,
    V3_LITERATURE_READING_TRACE,
    V3_MECHANISM_CARDS,
    V3_OPERATOR_EVIDENCE_MAP,
    V3_OPERATOR_NEGATIVE_CONTROLS,
    V3_OPERATOR_SPECS,
    V3_OPERATOR_TO_CODE_TRACE,
    V3_REFERENCE_DATA_SOURCES,
)
from design_scientist.io import write_json


REFERENCE_DATA_SOURCES = V3_REFERENCE_DATA_SOURCES


def write_reference_data_sources(
    root: str | Path,
    run_dir: str | Path,
    *,
    run_id: str,
    selected_node: dict[str, Any] | None,
) -> Path:
    """Write the V3 reference data source manifest for a scientist run."""

    root_path = Path(root).expanduser().resolve()
    run_path = Path(run_dir).expanduser().resolve()
    payload = build_reference_data_sources(
        root_path,
        run_path,
        run_id=run_id,
        selected_node=selected_node,
    )
    output = run_path / REFERENCE_DATA_SOURCES
    write_json(output, payload)
    return output


def build_reference_data_sources(
    root: str | Path,
    run_dir: str | Path,
    *,
    run_id: str,
    selected_node: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a manifest tying claims to concrete framework and run artifacts."""

    root_path = Path(root).expanduser().resolve()
    run_path = Path(run_dir).expanduser().resolve()
    node_artifacts = _selected_node_artifacts(root_path, selected_node)
    framework_sources = {
        "literature": [
            V3_LITERATURE_CORPUS,
            V3_LITERATURE_READING_TRACE,
            V3_MECHANISM_CARDS,
        ],
        "operators": [
            V3_OPERATOR_SPECS,
            V3_OPERATOR_EVIDENCE_MAP,
            V3_OPERATOR_NEGATIVE_CONTROLS,
        ],
        "node": node_artifacts,
        "benchmark": [
            _run_rel(root_path, run_path / "mechanism_benchmark_summary.csv"),
            _run_rel(root_path, run_path / "benchmark_saturation.json"),
            _run_rel(root_path, run_path / "claim_cap.json"),
        ],
        "review": [
            _run_rel(root_path, run_path / "novelty_review.json"),
            _run_rel(root_path, run_path / "baseline_audit.json"),
            _run_rel(root_path, run_path / "experiment_review.json"),
            _run_rel(root_path, run_path / "biology_review.json"),
            _run_rel(root_path, run_path / "paper_contribution_review.json"),
        ],
    }
    return {
        "schema_version": 1,
        "run_id": run_id,
        "selected_node_id": selected_node.get("node_id") if selected_node else None,
        "framework_sources": framework_sources,
        "claim_dependencies": [
            {
                "claim_id": "algorithmic_novelty",
                "required_sources": [
                    V3_OPERATOR_SPECS,
                    *_filter_sources(
                        [
                            _node_artifact(node_artifacts, "mechanism_spec.json"),
                            _node_artifact(node_artifacts, V3_OPERATOR_TO_CODE_TRACE),
                        ]
                    ),
                    _run_rel(root_path, run_path / "novelty_review.json"),
                ],
            },
            {
                "claim_id": "baseline_superiority",
                "required_sources": [
                    _run_rel(root_path, run_path / "mechanism_benchmark_summary.csv"),
                    _run_rel(root_path, run_path / "baseline_audit.json"),
                    _run_rel(root_path, run_path / "benchmark_saturation.json"),
                    _run_rel(root_path, run_path / "claim_cap.json"),
                ],
            },
            {
                "claim_id": "biological_evidence_boundary",
                "required_sources": [
                    _run_rel(root_path, run_path / "biology_review.json"),
                    _run_rel(root_path, run_path / "paper_contribution_review.json"),
                    _run_rel(root_path, run_path / "claim_cap.json"),
                ],
            },
        ],
    }


def expected_reference_sources(manifest: dict[str, Any]) -> list[str]:
    """Flatten all source paths expected by the manifest."""

    sources: list[str] = []
    framework_sources = manifest.get("framework_sources")
    if isinstance(framework_sources, dict):
        for value in framework_sources.values():
            if isinstance(value, list):
                sources.extend(str(item) for item in value if str(item).strip())
    dependencies = manifest.get("claim_dependencies")
    if isinstance(dependencies, list):
        for dependency in dependencies:
            if not isinstance(dependency, dict):
                continue
            required = dependency.get("required_sources")
            if isinstance(required, list):
                sources.extend(str(item) for item in required if str(item).strip())
    return list(dict.fromkeys(sources))


def _selected_node_artifacts(root: Path, selected_node: dict[str, Any] | None) -> list[str]:
    if not selected_node:
        return []
    artifacts = selected_node.get("artifacts")
    if not isinstance(artifacts, dict):
        return []
    names = (
        "mechanism_spec",
        "operator_to_code_trace",
        "proposal",
        "validation_report",
        "mechanism_metrics",
    )
    return _filter_sources(_source_rel(root, artifacts.get(name)) for name in names)


def _node_artifact(node_artifacts: list[str], filename: str) -> str | None:
    for artifact in node_artifacts:
        if Path(artifact).name == filename:
            return artifact
    return None


def _filter_sources(values: Any) -> list[str]:
    return [str(value) for value in values if value]


def _source_rel(root: Path, value: Any) -> str | None:
    if not value:
        return None
    try:
        path = Path(str(value)).expanduser().resolve()
    except (OSError, RuntimeError):
        return str(value)
    return _rel(root, path)


def _run_rel(root: Path, path: Path) -> str:
    return _rel(root, path)


def _rel(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except (OSError, RuntimeError, ValueError):
        return str(path)
