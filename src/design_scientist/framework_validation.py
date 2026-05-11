"""Framework R&D run validation.

This module validates the artifacts produced by the framework-first method
development workflow. It deliberately does not run literature adapters,
benchmarks, or method nodes; it only checks the artifacts already on disk.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from design_scientist import artifacts as artifact_constants
from design_scientist.io import read_json, read_yaml
from design_scientist.method_extraction import REQUIRED_POLICY_NAMES


CRITICAL_FRAMEWORK_ARTIFACTS: dict[str, str] = {
    "framework_spec": "framework/framework_spec.yaml",
    "literature_queries": "framework/literature_queries.yaml",
    "paper_cards": "framework/paper_cards.json",
    "method_modules": "framework/method_modules.json",
    "literature_map": "framework/literature_map.md",
    "research_gap_matrix": "framework/research_gap_matrix.csv",
    "method_hypotheses": "framework/method_hypotheses.md",
    "algorithm_spec": "framework/algorithm_spec.md",
    "method_registry": "framework/method_registry.yaml",
}

REFERENCE_AUDIT_ARTIFACTS: dict[str, str] = {
    "reference_audit": "framework/reference_audit.md",
    "reference_components": "framework/reference_components.json",
}

RESEARCH_OS_ARTIFACTS: dict[str, str] = {
    "quest": "framework/quest.yaml",
    "research_map": "framework/research_map.json",
    "findings_memory": "framework/findings_memory.jsonl",
    "failure_memory": "framework/failure_memory.jsonl",
}

LITERATURE_SEARCH_ARTIFACTS: dict[str, str] = {
    "literature_search_plan": "framework/literature_search_plan.json",
    "literature_search_trace": "framework/literature_search_trace.json",
    "citation_graph": "framework/citation_graph.json",
    "paper_scores": "framework/paper_scores.csv",
}

CRITICAL_RUN_ARTIFACTS: dict[str, str] = {
    "benchmark_results": "benchmark_results.csv",
    "ablation_results": "ablation_results.csv",
    "scientist_journal": "scientist_journal.json",
    "method_report": "method_report.md",
}

V3_CRITICAL_FRAMEWORK_ARTIFACTS: dict[str, str] = {
    "framework_spec": artifact_constants.FRAMEWORK_SPEC,
    "literature_queries": artifact_constants.LITERATURE_QUERIES,
    "paper_cards": "framework/paper_cards.json",
    "literature_search_plan": LITERATURE_SEARCH_ARTIFACTS["literature_search_plan"],
    "literature_search_trace": LITERATURE_SEARCH_ARTIFACTS["literature_search_trace"],
    "citation_graph": LITERATURE_SEARCH_ARTIFACTS["citation_graph"],
    "paper_scores": LITERATURE_SEARCH_ARTIFACTS["paper_scores"],
    "literature_corpus": artifact_constants.V3_LITERATURE_CORPUS,
    "literature_reading_trace": artifact_constants.V3_LITERATURE_READING_TRACE,
    "mechanism_cards": artifact_constants.V3_MECHANISM_CARDS,
    "mechanism_library": artifact_constants.V3_MECHANISM_LIBRARY,
    "mechanism_gap_matrix": artifact_constants.V3_MECHANISM_GAP_MATRIX,
    "quest": RESEARCH_OS_ARTIFACTS["quest"],
    "research_map": RESEARCH_OS_ARTIFACTS["research_map"],
    "findings_memory": RESEARCH_OS_ARTIFACTS["findings_memory"],
    "failure_memory": RESEARCH_OS_ARTIFACTS["failure_memory"],
}

V3_CRITICAL_RUN_ARTIFACTS: dict[str, str] = {
    "scientist_journal": "scientist_journal.json",
    "stage_progress": "stage_progress.json",
    "route_tree": "route_tree.json",
    "mechanism_benchmark_results": "mechanism_benchmark_results.csv",
    "mechanism_benchmark_summary": "mechanism_benchmark_summary.csv",
    "mechanism_ablation_results": "mechanism_ablation_results.csv",
    "method_report": "method_report.md",
}

OPTIONAL_RUN_ARTIFACTS: dict[str, str] = {
    "benchmark_summary": "benchmark_summary.csv",
    "stage_progress": "stage_progress.json",
    "route_tree": "route_tree.json",
}

SELECTED_NODE_ARTIFACTS: dict[str, str] = {
    "method": "method.py",
    "manifest": "manifest.json",
    "candidate_policy": "candidate_policy.json",
    "benchmark_metrics": "benchmark_metrics.json",
    "validation_report": "validation_report.json",
}

SELECTED_NODE_V2_ARTIFACTS: dict[str, str] = {
    "proposal": "proposal.json",
    "novelty_report": "novelty_report.json",
}

SIMPLE_BASELINE_METHODS = {
    "random_feasible",
    "top_observed",
    "greedy_utility",
    "fixed_mix",
    "pure_uncertainty",
    "pure_lattice_repair",
}

REQUIRED_BENCHMARK_BASELINES = ("random_feasible", "fixed_mix")
FALSE_CLAIM_RATE_TOLERANCE = 1e-9
BASELINE_OVERLAP_ERROR_THRESHOLD = 0.85

ALGORITHM_REQUIRED_SECTIONS = (
    "## Design Space",
    "## Observation Model",
    "## Objective",
    "## Constraints",
    "## Round Update",
    "## Policy API",
)

METHOD_REPORT_REQUIRED_SECTIONS = (
    "## Reference Basis",
    "## Research OS",
    "## Literature Basis",
    "## Method Hypotheses",
    "## Algorithm Mechanism",
    "## Stage Progress",
    "## Route Tree",
    "## Baseline Comparison",
    "## No-Baseline Rationale",
    "## Architecture Novelty",
    "## Multi-World Benchmark Summary",
    "## Ablation",
    "## Selected Method",
    "## Generator-Limited Diagnostics",
    "## Failed Methods",
    "## Unsupported Claims",
    "## Next Development Questions",
    "## Artifact Paths",
)

V3_METHOD_REPORT_REQUIRED_SECTIONS = (
    "## Literature Engine V3",
    "## Mechanism Library",
    "## MechanismSpec Kernel",
    "## Stress Tests",
    "## Baseline Comparison",
    "## Ablation",
    "## Selected Mechanism",
    "## Failed Nodes",
    "## Unsupported Claims and Validation Caveats",
    "## Artifact Paths",
)


def review_framework_run(project_dir: str | Path, run_id: str | None = None) -> dict[str, Any]:
    """Validate framework R&D artifacts for a run.

    Parameters
    ----------
    project_dir:
        Project root containing ``framework/`` and ``runs/``.
    run_id:
        Optional run directory under ``runs/``. When omitted, the latest run
        with a ``scientist_journal.json`` is preferred, falling back to the
        latest run directory.

    Returns
    -------
    dict
        Validation-style report with ``valid``, ``findings``, and artifact
        paths. Missing critical artifacts are always ``error`` findings.
    """

    root, selected_run_id, run_dir = resolve_framework_run(project_dir, run_id=run_id)
    if _run_journal_version(run_dir) == "v3":
        return _review_framework_run_v3(root, selected_run_id, run_dir)

    findings: list[dict[str, Any]] = []
    artifacts: dict[str, str] = {}

    framework_spec = _require_file(
        root, CRITICAL_FRAMEWORK_ARTIFACTS["framework_spec"], "missing_framework_spec", findings, artifacts
    )
    spec_data = _load_yaml_mapping(framework_spec, "framework_spec", findings)
    if spec_data is not None:
        _require_mapping_keys(
            spec_data,
            ("framework_id", "domain", "artifact_root"),
            "framework_spec",
            _rel(framework_spec, root),
            findings,
        )

    _validate_reference_audit_artifacts(root, findings, artifacts)
    _validate_research_os_artifacts(root, findings, artifacts)

    queries_path = _require_file(
        root,
        CRITICAL_FRAMEWORK_ARTIFACTS["literature_queries"],
        "missing_literature_queries",
        findings,
        artifacts,
    )
    queries_data = _load_yaml_mapping(queries_path, "literature_queries", findings)
    if queries_data is not None:
        _validate_literature_queries(queries_data, _rel(queries_path, root), findings)

    cards_path = _require_file(
        root, CRITICAL_FRAMEWORK_ARTIFACTS["paper_cards"], "missing_paper_cards", findings, artifacts
    )
    cards = _load_json(cards_path, "paper_cards", findings)
    paper_ids = _validate_paper_cards(cards, _rel(cards_path, root), findings)
    _validate_literature_search_artifacts(root, paper_ids, findings, artifacts)

    modules_path = _require_file(
        root,
        CRITICAL_FRAMEWORK_ARTIFACTS["method_modules"],
        "missing_method_modules",
        findings,
        artifacts,
    )
    modules_data = _load_json(modules_path, "method_modules", findings)
    method_modules = _validate_method_modules(modules_data, paper_ids, _rel(modules_path, root), findings)

    for key in ("literature_map", "research_gap_matrix", "method_hypotheses"):
        _require_file(
            root,
            CRITICAL_FRAMEWORK_ARTIFACTS[key],
            f"missing_{key}",
            findings,
            artifacts,
        )

    algorithm_path = _require_file(
        root,
        CRITICAL_FRAMEWORK_ARTIFACTS["algorithm_spec"],
        "missing_algorithm_spec",
        findings,
        artifacts,
    )
    algorithm_text = _load_text(algorithm_path, "algorithm_spec", findings)
    if algorithm_text is not None:
        for heading in ALGORITHM_REQUIRED_SECTIONS:
            if heading not in algorithm_text:
                _add_finding(
                    findings,
                    "error",
                    "algorithm_spec_missing_section",
                    f"Algorithm spec is missing required section {heading}.",
                    _rel(algorithm_path, root),
                )
        if "Algorithm id:" not in algorithm_text:
            _add_finding(
                findings,
                "warning",
                "algorithm_spec_missing_id",
                "Algorithm spec does not include an Algorithm id line.",
                _rel(algorithm_path, root),
            )

    registry_path = _require_file(
        root,
        CRITICAL_FRAMEWORK_ARTIFACTS["method_registry"],
        "missing_method_registry",
        findings,
        artifacts,
    )
    registry_data = _load_yaml_mapping(registry_path, "method_registry", findings)
    registry_policy_names = _validate_method_registry(registry_data, _rel(registry_path, root), findings)

    if run_dir is None or not run_dir.is_dir():
        artifact = "runs/<latest>" if selected_run_id is None else f"runs/{selected_run_id}"
        _add_finding(
            findings,
            "error",
            "missing_run",
            "No framework run directory was found for validation.",
            artifact,
        )
        return _build_report(root, selected_run_id, run_dir, findings, artifacts)

    artifacts["run_dir"] = str(run_dir)
    benchmark_path = _require_run_file(
        root, run_dir, "benchmark_results", "missing_benchmark_results", findings, artifacts
    )
    benchmark_rows = _load_csv(benchmark_path, "benchmark_results", findings)

    benchmark_summary_path = _optional_run_file(root, run_dir, "benchmark_summary", findings, artifacts)
    benchmark_summary_rows = _load_csv(benchmark_summary_path, "benchmark_summary", findings)
    _validate_benchmark_summary_rows(benchmark_summary_rows, _rel(benchmark_summary_path, root), findings)
    _validate_optional_run_json_artifact(root, run_dir, "stage_progress", findings, artifacts)
    _validate_optional_run_json_artifact(root, run_dir, "route_tree", findings, artifacts)

    ablation_path = _require_run_file(
        root, run_dir, "ablation_results", "missing_ablation_results", findings, artifacts
    )
    ablation_rows = _load_csv(ablation_path, "ablation_results", findings)
    _validate_ablation_rows(ablation_rows, _rel(ablation_path, root), findings)

    journal_path = _require_run_file(
        root, run_dir, "scientist_journal", "missing_scientist_journal", findings, artifacts
    )
    journal = _load_json(journal_path, "scientist_journal", findings)
    selected_record = _validate_scientist_journal(journal, selected_run_id, _rel(journal_path, root), findings)
    generated_policy_names = _journal_generated_benchmark_methods(root, journal)
    _validate_benchmark_rows(
        benchmark_rows,
        registry_policy_names,
        _rel(benchmark_path, root),
        findings,
        generated_policy_names,
    )
    _validate_selected_method_benchmark_evidence(
        benchmark_rows,
        benchmark_summary_rows,
        selected_record,
        _rel(benchmark_path, root),
        _rel(benchmark_summary_path, root),
        findings,
    )
    _validate_selected_method_false_claim_rate(
        benchmark_rows,
        benchmark_summary_rows,
        selected_record,
        _rel(benchmark_path, root),
        _rel(benchmark_summary_path, root),
        findings,
    )
    _validate_selected_node_artifacts(root, run_dir, selected_record, findings, artifacts)
    _validate_selected_trace(root, selected_record, findings)
    _validate_selected_method_majority_wins(
        benchmark_rows,
        benchmark_summary_rows,
        selected_record,
        _rel(benchmark_summary_path, root),
        _rel(benchmark_path, root),
        findings,
    )

    method_report_path = _require_run_file(
        root, run_dir, "method_report", "missing_method_report", findings, artifacts
    )
    method_report_text = _load_text(method_report_path, "method_report", findings)
    if method_report_text is not None:
        _validate_method_report(method_report_text, _rel(method_report_path, root), findings)

    _validate_state_unsupported_claims(root, findings, artifacts)
    if method_modules and not any(module.get("source_papers") for module in method_modules):
        _add_finding(
            findings,
            "error",
            "method_modules_without_sources",
            "No method module is tied to source papers.",
            _rel(modules_path, root),
        )

    return _build_report(root, selected_run_id, run_dir, findings, artifacts)


def _review_framework_run_v3(
    root: Path,
    selected_run_id: str | None,
    run_dir: Path | None,
) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    artifacts: dict[str, str] = {}

    framework_spec = _require_file(
        root,
        V3_CRITICAL_FRAMEWORK_ARTIFACTS["framework_spec"],
        "missing_framework_spec",
        findings,
        artifacts,
    )
    spec_data = _load_yaml_mapping(framework_spec, "framework_spec", findings)
    if spec_data is not None:
        _require_mapping_keys(
            spec_data,
            ("framework_id", "domain", "artifact_root"),
            "framework_spec",
            _rel(framework_spec, root),
            findings,
        )

    _validate_research_os_artifacts(root, findings, artifacts)

    queries_path = _require_file(
        root,
        V3_CRITICAL_FRAMEWORK_ARTIFACTS["literature_queries"],
        "missing_literature_queries",
        findings,
        artifacts,
    )
    queries_data = _load_yaml_mapping(queries_path, "literature_queries", findings)
    if queries_data is not None:
        _validate_literature_queries(queries_data, _rel(queries_path, root), findings)

    cards_path = _require_file(
        root,
        V3_CRITICAL_FRAMEWORK_ARTIFACTS["paper_cards"],
        "missing_paper_cards",
        findings,
        artifacts,
    )
    cards = _load_json(cards_path, "paper_cards", findings)
    paper_ids = _validate_paper_cards(cards, _rel(cards_path, root), findings)
    _validate_literature_search_artifacts(root, paper_ids, findings, artifacts)
    _validate_v3_literature_engine_artifacts(root, findings, artifacts)

    if run_dir is None or not run_dir.is_dir():
        artifact = "runs/<latest>" if selected_run_id is None else f"runs/{selected_run_id}"
        _add_finding(
            findings,
            "error",
            "missing_run",
            "No framework run directory was found for validation.",
            artifact,
        )
        return _build_report(root, selected_run_id, run_dir, findings, artifacts)

    artifacts["run_dir"] = str(run_dir)
    journal_path = _require_v3_run_file(
        root,
        run_dir,
        "scientist_journal",
        "missing_scientist_journal",
        findings,
        artifacts,
    )
    journal = _load_json(journal_path, "scientist_journal", findings)
    selected_record = _validate_v3_scientist_journal(journal, selected_run_id, _rel(journal_path, root), findings)

    _validate_required_v3_run_json(root, run_dir, "stage_progress", findings, artifacts)
    _validate_required_v3_run_json(root, run_dir, "route_tree", findings, artifacts)

    benchmark_path = _require_v3_run_file(
        root,
        run_dir,
        "mechanism_benchmark_results",
        "missing_mechanism_benchmark_results",
        findings,
        artifacts,
    )
    benchmark_rows = _load_csv(benchmark_path, "mechanism_benchmark_results", findings)
    _validate_v3_mechanism_benchmark_results(benchmark_rows, _rel(benchmark_path, root), findings)

    summary_path = _require_v3_run_file(
        root,
        run_dir,
        "mechanism_benchmark_summary",
        "missing_mechanism_benchmark_summary",
        findings,
        artifacts,
    )
    summary_rows = _load_csv(summary_path, "mechanism_benchmark_summary", findings)

    ablation_path = _require_v3_run_file(
        root,
        run_dir,
        "mechanism_ablation_results",
        "missing_mechanism_ablation_results",
        findings,
        artifacts,
    )
    ablation_rows = _load_csv(ablation_path, "mechanism_ablation_results", findings)

    selected_artifacts = _validate_v3_selected_node_artifacts(root, run_dir, selected_record, findings, artifacts)
    selected_mechanism = _selected_v3_mechanism_name(
        selected_record,
        selected_artifacts.get("mechanism_spec"),
        selected_artifacts.get("mechanism_metrics"),
    )
    _validate_v3_mechanism_benchmark_summary(
        summary_rows,
        selected_mechanism,
        selected_record,
        selected_artifacts.get("mechanism_metrics"),
        _rel(summary_path, root),
        findings,
    )
    _validate_v3_mechanism_ablation_results(ablation_rows, selected_mechanism, _rel(ablation_path, root), findings)

    method_report_path = _require_v3_run_file(
        root,
        run_dir,
        "method_report",
        "missing_method_report",
        findings,
        artifacts,
    )
    method_report_text = _load_text(method_report_path, "method_report", findings)
    if method_report_text is not None:
        _validate_v3_method_report(method_report_text, _rel(method_report_path, root), findings)

    return _build_report(root, selected_run_id, run_dir, findings, artifacts)


def resolve_framework_run(
    project_dir: str | Path,
    run_id: str | None = None,
) -> tuple[Path, str | None, Path | None]:
    """Resolve a project root and framework run directory."""

    root = Path(project_dir).expanduser().resolve()
    if run_id is not None:
        selected_run_id = _safe_run_id(run_id)
        return root, selected_run_id, root / "runs" / selected_run_id

    runs_dir = root / "runs"
    if not runs_dir.is_dir():
        return root, None, None

    run_dirs = sorted(path for path in runs_dir.iterdir() if path.is_dir())
    if not run_dirs:
        return root, None, None

    preferred = [path for path in run_dirs if (path / "scientist_journal.json").exists()]
    selected = (preferred or run_dirs)[-1]
    return root, selected.name, selected


def _safe_run_id(run_id: str) -> str:
    if not run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a non-empty relative name")
    if "/" in run_id or "\\" in run_id or ".." in Path(run_id).parts:
        raise ValueError("run_id must not contain path separators or '..'")
    return run_id


def _run_journal_version(run_dir: Path | None) -> str | None:
    if run_dir is None:
        return None
    journal_path = run_dir / "scientist_journal.json"
    if not journal_path.is_file():
        return None
    try:
        journal = read_json(journal_path)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if isinstance(journal, dict) and isinstance(journal.get("version"), str):
        return journal["version"]
    return None


def _validate_literature_queries(
    data: dict[str, Any],
    artifact: str,
    findings: list[dict[str, Any]],
) -> None:
    queries = data.get("queries")
    if not isinstance(queries, list) or not queries:
        _add_finding(
            findings,
            "error",
            "literature_queries_empty",
            "Literature queries must contain a non-empty queries list.",
            artifact,
        )
        return
    for index, query in enumerate(queries):
        if not isinstance(query, dict):
            _add_finding(
                findings,
                "error",
                "literature_query_invalid",
                f"Literature query {index} must be a mapping.",
                artifact,
            )
            continue
        _require_mapping_keys(query, ("query_id", "domain", "query"), "literature_query", artifact, findings)
        if query.get("status") in {"failed", "error"}:
            _add_finding(
                findings,
                "warning",
                "literature_query_failed",
                f"Literature query {query.get('query_id', index)!r} is marked {query.get('status')}.",
                artifact,
            )


def _validate_reference_audit_artifacts(
    root: Path,
    findings: list[dict[str, Any]],
    artifacts: dict[str, str],
) -> None:
    audit_path = root / REFERENCE_AUDIT_ARTIFACTS["reference_audit"]
    components_path = root / REFERENCE_AUDIT_ARTIFACTS["reference_components"]
    artifacts["reference_audit"] = str(audit_path)
    artifacts["reference_components"] = str(components_path)

    audit_exists = audit_path.exists()
    components_exists = components_path.exists()
    audit_requested = audit_exists or components_exists

    if not audit_exists:
        _add_finding(
            findings,
            "error" if audit_requested else "warning",
            "missing_reference_audit",
            (
                "Missing reference audit after audit-references produced companion artifacts."
                if audit_requested
                else "Reference audit has not been run; framework claims should note this provenance gap."
            ),
            REFERENCE_AUDIT_ARTIFACTS["reference_audit"],
        )
    elif audit_path.is_file() and audit_path.stat().st_size == 0:
        _add_finding(
            findings,
            "error",
            "empty_reference_audit",
            "Reference audit Markdown is empty.",
            REFERENCE_AUDIT_ARTIFACTS["reference_audit"],
        )

    if not components_exists:
        severity = "error" if audit_requested else "warning"
        _add_finding(
            findings,
            severity,
            "missing_reference_components",
            (
                "Missing reference component JSON after audit-references produced companion artifacts."
                if audit_requested
                else "Reference component JSON has not been produced by audit-references."
            ),
            REFERENCE_AUDIT_ARTIFACTS["reference_components"],
        )
        return

    components = _load_json(components_path, "reference_components", findings)
    if isinstance(components, dict):
        if not isinstance(components.get("components"), list):
            _add_finding(
                findings,
                "error",
                "reference_components_missing_components",
                "reference_components.json must contain a components list.",
                _rel(components_path, root),
            )
        if not isinstance(components.get("summary"), dict):
            _add_finding(
                findings,
                "warning",
                "reference_components_missing_summary",
                "reference_components.json does not include a summary mapping.",
                _rel(components_path, root),
            )


def _validate_research_os_artifacts(
    root: Path,
    findings: list[dict[str, Any]],
    artifacts: dict[str, str],
) -> None:
    quest_path = _require_file(
        root,
        RESEARCH_OS_ARTIFACTS["quest"],
        "missing_quest",
        findings,
        artifacts,
    )
    quest = _load_yaml_mapping(quest_path, "quest", findings)
    if quest is not None:
        _require_mapping_keys(
            quest,
            ("schema_version", "quest_id", "domain", "status"),
            "quest",
            _rel(quest_path, root),
            findings,
        )

    research_map_path = _require_file(
        root,
        RESEARCH_OS_ARTIFACTS["research_map"],
        "missing_research_map",
        findings,
        artifacts,
    )
    research_map = _load_json(research_map_path, "research_map", findings)
    if isinstance(research_map, dict):
        if not isinstance(research_map.get("loops"), list) or not research_map.get("loops"):
            _add_finding(
                findings,
                "error",
                "research_map_missing_loops",
                "research_map.json must contain a non-empty loops list.",
                _rel(research_map_path, root),
            )
        if not isinstance(research_map.get("memory"), dict):
            _add_finding(
                findings,
                "warning",
                "research_map_missing_memory",
                "research_map.json does not include memory artifact pointers.",
                _rel(research_map_path, root),
            )

    for key in ("findings_memory", "failure_memory"):
        memory_path = root / RESEARCH_OS_ARTIFACTS[key]
        artifacts[key] = str(memory_path)
        if not memory_path.exists():
            _add_finding(
                findings,
                "error",
                f"missing_{key}",
                f"Missing critical framework artifact: {RESEARCH_OS_ARTIFACTS[key]}.",
                RESEARCH_OS_ARTIFACTS[key],
            )
            continue
        if not memory_path.is_file():
            _add_finding(
                findings,
                "error",
                f"invalid_{key}",
                f"Research OS memory artifact is not a file: {RESEARCH_OS_ARTIFACTS[key]}.",
                RESEARCH_OS_ARTIFACTS[key],
            )
            continue
        _validate_jsonl_memory(memory_path, key, _rel(memory_path, root), findings)


def _validate_literature_search_artifacts(
    root: Path,
    paper_ids: set[str],
    findings: list[dict[str, Any]],
    artifacts: dict[str, str],
) -> None:
    plan_path = _require_file(
        root,
        LITERATURE_SEARCH_ARTIFACTS["literature_search_plan"],
        "missing_literature_search_plan",
        findings,
        artifacts,
    )
    plan = _load_json(plan_path, "literature_search_plan", findings)
    if isinstance(plan, dict):
        queries = plan.get("queries")
        if not isinstance(queries, list) or not queries:
            _add_finding(
                findings,
                "error",
                "literature_search_plan_missing_queries",
                "literature_search_plan.json must contain a non-empty queries list.",
                _rel(plan_path, root),
            )

    trace_path = _require_file(
        root,
        LITERATURE_SEARCH_ARTIFACTS["literature_search_trace"],
        "missing_literature_search_trace",
        findings,
        artifacts,
    )
    trace = _load_json(trace_path, "literature_search_trace", findings)
    if isinstance(trace, dict) and not isinstance(trace.get("events"), list):
        _add_finding(
            findings,
            "error",
            "literature_search_trace_missing_events",
            "literature_search_trace.json must contain an events list.",
            _rel(trace_path, root),
        )

    graph_path = _require_file(
        root,
        LITERATURE_SEARCH_ARTIFACTS["citation_graph"],
        "missing_citation_graph",
        findings,
        artifacts,
    )
    graph = _load_json(graph_path, "citation_graph", findings)
    if isinstance(graph, dict):
        if not isinstance(graph.get("nodes"), list):
            _add_finding(
                findings,
                "error",
                "citation_graph_missing_nodes",
                "citation_graph.json must contain a nodes list.",
                _rel(graph_path, root),
            )
        if not isinstance(graph.get("edges"), list):
            _add_finding(
                findings,
                "error",
                "citation_graph_missing_edges",
                "citation_graph.json must contain an edges list.",
                _rel(graph_path, root),
            )

    scores_path = _require_file(
        root,
        LITERATURE_SEARCH_ARTIFACTS["paper_scores"],
        "missing_paper_scores",
        findings,
        artifacts,
    )
    score_rows = _load_csv(scores_path, "paper_scores", findings)
    if score_rows is None:
        return
    if not score_rows:
        _add_finding(
            findings,
            "error",
            "paper_scores_empty",
            "paper_scores.csv must contain at least one scored paper row.",
            _rel(scores_path, root),
        )
        return
    missing_columns = [column for column in ("paper_id", "total_score") if column not in score_rows[0]]
    if missing_columns:
        _add_finding(
            findings,
            "error",
            "paper_scores_missing_columns",
            f"paper_scores.csv is missing columns: {', '.join(missing_columns)}.",
            _rel(scores_path, root),
        )
    if paper_ids:
        scored_ids = {row.get("paper_id", "") for row in score_rows}
        missing_scores = sorted(paper_id for paper_id in paper_ids if paper_id not in scored_ids)
        if missing_scores:
            _add_finding(
                findings,
                "warning",
                "paper_scores_missing_paper_ids",
                f"Paper scores are missing paper ids: {', '.join(missing_scores[:8])}.",
                _rel(scores_path, root),
            )


def _validate_v3_literature_engine_artifacts(
    root: Path,
    findings: list[dict[str, Any]],
    artifacts: dict[str, str],
) -> None:
    corpus_path = _require_file(
        root,
        V3_CRITICAL_FRAMEWORK_ARTIFACTS["literature_corpus"],
        "missing_literature_corpus",
        findings,
        artifacts,
    )
    _validate_jsonl_records(corpus_path, "literature_corpus", _rel(corpus_path, root), findings)

    reading_trace_path = _require_file(
        root,
        V3_CRITICAL_FRAMEWORK_ARTIFACTS["literature_reading_trace"],
        "missing_literature_reading_trace",
        findings,
        artifacts,
    )
    reading_trace = _load_json(reading_trace_path, "literature_reading_trace", findings)
    if reading_trace is not None and reading_trace in ({}, []):
        _add_finding(
            findings,
            "error",
            "literature_reading_trace_empty",
            "literature_reading_trace.json must record reading events or paper-level provenance.",
            _rel(reading_trace_path, root),
        )

    cards_path = _require_file(
        root,
        V3_CRITICAL_FRAMEWORK_ARTIFACTS["mechanism_cards"],
        "missing_mechanism_cards",
        findings,
        artifacts,
    )
    cards = _load_json(cards_path, "mechanism_cards", findings)
    _validate_nonempty_v3_collection(cards, "mechanism_cards", _rel(cards_path, root), findings)

    library_path = _require_file(
        root,
        V3_CRITICAL_FRAMEWORK_ARTIFACTS["mechanism_library"],
        "missing_mechanism_library",
        findings,
        artifacts,
    )
    library = _load_json(library_path, "mechanism_library", findings)
    _validate_nonempty_v3_collection(library, "mechanism_library", _rel(library_path, root), findings)

    gap_path = _require_file(
        root,
        V3_CRITICAL_FRAMEWORK_ARTIFACTS["mechanism_gap_matrix"],
        "missing_mechanism_gap_matrix",
        findings,
        artifacts,
    )
    gap_rows = _load_csv(gap_path, "mechanism_gap_matrix", findings)
    if gap_rows is not None:
        if not gap_rows:
            _add_finding(
                findings,
                "error",
                "mechanism_gap_matrix_empty",
                "mechanism_gap_matrix.csv must contain at least one mechanism gap row.",
                _rel(gap_path, root),
            )
        elif "mechanism_id" not in gap_rows[0]:
            _add_finding(
                findings,
                "error",
                "mechanism_gap_matrix_missing_mechanism_id",
                "mechanism_gap_matrix.csv must contain mechanism_id.",
                _rel(gap_path, root),
            )


def _validate_jsonl_records(
    path: Path,
    label: str,
    artifact: str,
    findings: list[dict[str, Any]],
) -> None:
    if not path.exists() or not path.is_file():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        _add_finding(
            findings,
            "error",
            f"invalid_{label}",
            f"Cannot read JSONL artifact {path.name}: {exc}",
            artifact,
        )
        return
    records = 0
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        records += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            _add_finding(
                findings,
                "error",
                f"invalid_{label}",
                f"JSONL line {index} is not valid JSON: {exc}",
                artifact,
            )
            continue
        if not isinstance(record, dict):
            _add_finding(
                findings,
                "error",
                f"invalid_{label}",
                f"JSONL line {index} must be a JSON object.",
                artifact,
            )
    if records == 0:
        _add_finding(
            findings,
            "error",
            f"{label}_empty",
            f"{path.name} must contain at least one JSONL record.",
            artifact,
        )


def _validate_nonempty_v3_collection(
    data: Any,
    label: str,
    artifact: str,
    findings: list[dict[str, Any]],
) -> None:
    if data is None:
        return
    if isinstance(data, list):
        if not data:
            _add_finding(
                findings,
                "error",
                f"{label}_empty",
                f"{label} must contain at least one record.",
                artifact,
            )
        return
    if isinstance(data, dict):
        if not data:
            _add_finding(
                findings,
                "error",
                f"{label}_empty",
                f"{label} must not be an empty object.",
                artifact,
            )
            return
        collection = data.get("mechanisms") or data.get("cards") or data.get("items")
        if collection is not None and (not isinstance(collection, list) or not collection):
            _add_finding(
                findings,
                "error",
                f"{label}_empty",
                f"{label} collection must contain at least one record.",
                artifact,
            )
        return
    _add_finding(
        findings,
        "error",
        f"{label}_invalid",
        f"{label} must be a JSON object or array.",
        artifact,
    )


def _validate_paper_cards(
    cards: Any,
    artifact: str,
    findings: list[dict[str, Any]],
) -> set[str]:
    if not isinstance(cards, list) or not cards:
        _add_finding(
            findings,
            "error",
            "paper_cards_empty",
            "Paper cards must be a non-empty JSON list.",
            artifact,
        )
        return set()

    paper_ids: set[str] = set()
    seed_unverified = 0
    for index, card in enumerate(cards):
        if not isinstance(card, dict):
            _add_finding(
                findings,
                "error",
                "paper_card_invalid",
                f"Paper card {index} must be a mapping.",
                artifact,
            )
            continue
        paper_id = card.get("paper_id")
        if not isinstance(paper_id, str) or not paper_id.strip():
            _add_finding(
                findings,
                "error",
                "paper_card_missing_id",
                f"Paper card {index} is missing paper_id.",
                artifact,
            )
        else:
            paper_ids.add(paper_id)
        if not _has_text(card.get("title")) and not _has_text(card.get("citation")):
            _add_finding(
                findings,
                "warning",
                "paper_card_missing_title",
                f"Paper card {paper_id or index!r} has neither title nor citation.",
                artifact,
            )
        if card.get("review_status") == "seed_unverified":
            seed_unverified += 1
    if seed_unverified == len([card for card in cards if isinstance(card, dict)]):
        _add_finding(
            findings,
            "warning",
            "paper_cards_seed_only",
            "All paper cards are seed_unverified; method claims should stay provisional.",
            artifact,
        )
    return paper_ids


def _validate_method_modules(
    data: Any,
    paper_ids: set[str],
    artifact: str,
    findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        _add_finding(
            findings,
            "error",
            "method_modules_invalid",
            "Method modules artifact must be a JSON object.",
            artifact,
        )
        return []

    modules = data.get("method_modules") or data.get("modules")
    if not isinstance(modules, list) or not modules:
        _add_finding(
            findings,
            "error",
            "method_modules_empty",
            "Method modules artifact must contain a non-empty method_modules list.",
            artifact,
        )
        return []

    required = (
        "module_id",
        "name",
        "problem_setting",
        "algorithm_family",
        "acquisition_policy_mechanism",
        "baselines",
        "evaluation_protocol",
        "failure_modes",
        "reusable_ideas",
        "source_papers",
    )
    clean_modules: list[dict[str, Any]] = []
    for index, module in enumerate(modules):
        if not isinstance(module, dict):
            _add_finding(
                findings,
                "error",
                "method_module_invalid",
                f"Method module {index} must be a mapping.",
                artifact,
            )
            continue
        clean_modules.append(module)
        _require_mapping_keys(module, required, "method_module", artifact, findings)
        source_papers = module.get("source_papers")
        if not isinstance(source_papers, list) or not source_papers:
            _add_finding(
                findings,
                "error",
                "method_module_missing_sources",
                f"Method module {module.get('module_id', index)!r} is missing source_papers.",
                artifact,
            )
            continue
        unknown = [paper_id for paper_id in source_papers if paper_ids and paper_id not in paper_ids]
        if unknown:
            _add_finding(
                findings,
                "error",
                "method_module_unknown_sources",
                f"Method module {module.get('module_id', index)!r} cites unknown paper ids: {', '.join(map(str, unknown))}.",
                artifact,
            )
    return clean_modules


def _validate_method_registry(
    data: dict[str, Any] | None,
    artifact: str,
    findings: list[dict[str, Any]],
) -> set[str]:
    if data is None:
        return set()
    policies = data.get("policies")
    if not isinstance(policies, list) or not policies:
        _add_finding(
            findings,
            "error",
            "method_registry_empty",
            "Method registry must contain a non-empty policies list.",
            artifact,
        )
        return set()

    policy_names: set[str] = set()
    for index, policy in enumerate(policies):
        if not isinstance(policy, dict):
            _add_finding(
                findings,
                "error",
                "method_registry_policy_invalid",
                f"Registry policy {index} must be a mapping.",
                artifact,
            )
            continue
        name = policy.get("name")
        if not isinstance(name, str) or not name.strip():
            _add_finding(
                findings,
                "error",
                "method_registry_policy_missing_name",
                f"Registry policy {index} is missing name.",
                artifact,
            )
            continue
        policy_names.add(name)

    missing = [name for name in REQUIRED_POLICY_NAMES if name not in policy_names]
    if missing:
        _add_finding(
            findings,
            "error",
            "method_registry_missing_required_policies",
            f"Method registry is missing required policies: {', '.join(missing)}.",
            artifact,
        )
    default_policy = data.get("default_policy")
    if default_policy and default_policy not in policy_names:
        _add_finding(
            findings,
            "error",
            "method_registry_default_missing",
            f"Default policy {default_policy!r} is not present in policies.",
            artifact,
        )
    return policy_names


def _validate_benchmark_rows(
    rows: list[dict[str, str]] | None,
    registry_policy_names: set[str],
    artifact: str,
    findings: list[dict[str, Any]],
    generated_policy_names: set[str] | None = None,
) -> None:
    if rows is None:
        return
    if not rows:
        _add_finding(
            findings,
            "error",
            "benchmark_results_empty",
            "Benchmark results CSV must contain at least one result row.",
            artifact,
        )
        return

    required_columns = (
        "method",
        "best_feasible_utility",
        "hit_rate",
        "regret_proxy",
        "false_claim_rate",
        "evidence_coverage",
    )
    missing_columns = [column for column in required_columns if column not in rows[0]]
    if missing_columns:
        _add_finding(
            findings,
            "error",
            "benchmark_results_missing_columns",
            f"Benchmark results are missing columns: {', '.join(missing_columns)}.",
            artifact,
        )
    methods = {row.get("method", "") for row in rows}
    if "mechanism_aware" not in methods:
        _add_finding(
            findings,
            "error",
            "benchmark_missing_default_policy",
            "Benchmark results must include mechanism_aware.",
            artifact,
        )
    missing_baselines = [baseline for baseline in REQUIRED_BENCHMARK_BASELINES if baseline not in methods]
    if missing_baselines:
        _add_finding(
            findings,
            "error",
            "benchmark_missing_required_baselines",
            "Benchmark results must include required baselines: "
            + ", ".join(REQUIRED_BENCHMARK_BASELINES)
            + f". Missing: {', '.join(missing_baselines)}.",
            artifact,
        )
    if registry_policy_names:
        known_methods = set(registry_policy_names)
        known_methods.update(generated_policy_names or set())
        unknown = sorted(method for method in methods if method and method not in known_methods)
        if unknown:
            _add_finding(
                findings,
                "warning",
                "benchmark_method_not_registered",
                f"Benchmark contains methods not present in registry: {', '.join(unknown)}.",
                artifact,
            )


def _journal_generated_benchmark_methods(root: Path, journal: Any) -> set[str]:
    if not isinstance(journal, dict):
        return set()
    nodes = journal.get("nodes")
    if not isinstance(nodes, list):
        return set()

    methods: set[str] = set()
    for node in nodes:
        if not isinstance(node, dict):
            continue
        policy_name = node.get("benchmark_policy_name")
        if not _has_text(policy_name):
            continue
        if _journal_node_has_valid_generated_artifacts(root, node):
            methods.add(str(policy_name))
    return methods


def _journal_node_has_valid_generated_artifacts(root: Path, node: dict[str, Any]) -> bool:
    if node.get("status") != "completed":
        return False
    contract = node.get("contract")
    if isinstance(contract, dict) and contract.get("valid") is not True:
        return False

    artifact_map = node.get("artifacts") if isinstance(node.get("artifacts"), dict) else {}
    workspace_value = node.get("workspace")
    workspace = _coerce_path(root, workspace_value) if workspace_value else None
    required = {
        **SELECTED_NODE_ARTIFACTS,
        **SELECTED_NODE_V2_ARTIFACTS,
    }
    paths: dict[str, Path] = {}
    for key, filename in required.items():
        path_value = artifact_map.get(key)
        path = _coerce_path(root, path_value) if path_value else (workspace / filename if workspace else None)
        if path is None or not path.is_file() or path.stat().st_size == 0:
            return False
        paths[key] = path

    proposal = _load_json_mapping_silent(paths["proposal"])
    if proposal is None or _missing_proposal_metadata(proposal):
        return False

    novelty = _load_json_mapping_silent(paths["novelty_report"])
    if novelty is None or "baseline_overlap" not in novelty or "selection_overlap_vs_baselines" not in novelty:
        return False
    if _indicates_baseline_clone(novelty):
        return False

    validation_report = _load_json_mapping_silent(paths["validation_report"])
    return validation_report is not None and validation_report.get("valid") is True


def _load_json_mapping_silent(path: Path) -> dict[str, Any] | None:
    try:
        data = read_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _validate_benchmark_summary_rows(
    rows: list[dict[str, str]] | None,
    artifact: str,
    findings: list[dict[str, Any]],
) -> None:
    if rows is None:
        return
    if not rows:
        _add_finding(
            findings,
            "error",
            "benchmark_summary_empty",
            "benchmark_summary.csv exists but contains no result rows.",
            artifact,
        )
        return
    if "method" not in rows[0]:
        _add_finding(
            findings,
            "error",
            "benchmark_summary_missing_method",
            "benchmark_summary.csv must contain a method column.",
            artifact,
        )
    methods = {row.get("method", "") for row in rows}
    missing_baselines = [baseline for baseline in REQUIRED_BENCHMARK_BASELINES if baseline not in methods]
    if missing_baselines:
        _add_finding(
            findings,
            "error",
            "benchmark_summary_missing_required_baselines",
            "benchmark_summary.csv must include required baselines: "
            + ", ".join(REQUIRED_BENCHMARK_BASELINES)
            + f". Missing: {', '.join(missing_baselines)}.",
            artifact,
        )
    required_majority_columns = (
        "world_count",
        "beats_random_feasible_worlds",
        "beats_random_feasible_majority",
        "beats_fixed_mix_worlds",
        "beats_fixed_mix_majority",
    )
    missing_majority_columns = [
        column for column in required_majority_columns if column not in rows[0]
    ]
    if missing_majority_columns:
        _add_finding(
            findings,
            "error",
            "benchmark_summary_missing_majority_columns",
            "benchmark_summary.csv is missing majority comparison columns: "
            + ", ".join(missing_majority_columns)
            + ".",
            artifact,
        )
    if "world_count" in rows[0]:
        for row in rows:
            if _to_float(row.get("world_count")) is None:
                _add_finding(
                    findings,
                    "error",
                    "benchmark_summary_invalid_world_count",
                    f"Benchmark summary row for {row.get('method', 'unknown')!r} has invalid world_count.",
                    artifact,
                )


def _validate_selected_method_benchmark_evidence(
    benchmark_rows: list[dict[str, str]] | None,
    summary_rows: list[dict[str, str]] | None,
    selected_record: dict[str, Any] | None,
    benchmark_artifact: str,
    summary_artifact: str,
    findings: list[dict[str, Any]],
) -> None:
    if not selected_record:
        return
    selected_method = _selected_benchmark_policy_name(selected_record)
    if not selected_method:
        return

    benchmark_methods = _csv_methods(benchmark_rows)
    if selected_method not in benchmark_methods:
        _add_finding(
            findings,
            "error",
            "selected_method_missing_benchmark_results",
            f"Selected benchmark method {selected_method!r} is missing from benchmark_results.csv.",
            benchmark_artifact,
        )

    summary_methods = _csv_methods(summary_rows)
    if selected_method not in summary_methods:
        _add_finding(
            findings,
            "error",
            "selected_method_missing_benchmark_summary",
            f"Selected benchmark method {selected_method!r} is missing from benchmark_summary.csv.",
            summary_artifact,
        )


def _csv_methods(rows: list[dict[str, str]] | None) -> set[str]:
    if not rows:
        return set()
    return {row.get("method", "") for row in rows if row.get("method")}


def _validate_selected_method_false_claim_rate(
    benchmark_rows: list[dict[str, str]] | None,
    summary_rows: list[dict[str, str]] | None,
    selected_record: dict[str, Any] | None,
    benchmark_artifact: str,
    summary_artifact: str,
    findings: list[dict[str, Any]],
) -> None:
    if not selected_record:
        return
    selected_method = _selected_benchmark_policy_name(selected_record)
    if not selected_method:
        return

    comparable, violations = _false_claim_rate_violations_by_world(benchmark_rows, selected_method)
    if comparable:
        if violations:
            _add_finding(
                findings,
                "error",
                "selected_method_false_claim_rate_exceeds_baselines",
                "Selected method false_claim_rate exceeds the worse required baseline in comparable worlds: "
                + ", ".join(violations[:8])
                + ".",
                benchmark_artifact,
            )
        return

    summary_violation = _false_claim_rate_summary_violation(summary_rows, selected_method)
    if summary_violation is not None:
        _add_finding(
            findings,
            "error",
            "selected_method_false_claim_rate_exceeds_baselines",
            summary_violation,
            summary_artifact,
        )


def _false_claim_rate_violations_by_world(
    rows: list[dict[str, str]] | None,
    selected_method: str,
) -> tuple[int, list[str]]:
    if not rows or "world_id" not in rows[0] or "false_claim_rate" not in rows[0]:
        return 0, []

    by_world_method: dict[tuple[str, str], dict[str, str]] = {}
    worlds: set[str] = set()
    for row in rows:
        method = row.get("method")
        world_id = row.get("world_id")
        if not method or not world_id:
            continue
        worlds.add(world_id)
        by_world_method[(world_id, method)] = row

    comparable = 0
    violations: list[str] = []
    for world_id in sorted(worlds):
        selected = by_world_method.get((world_id, selected_method))
        baselines = [
            by_world_method.get((world_id, baseline))
            for baseline in REQUIRED_BENCHMARK_BASELINES
        ]
        if selected is None or any(row is None for row in baselines):
            continue
        selected_rate = _to_float(selected.get("false_claim_rate"))
        baseline_rates = [
            _to_float(row.get("false_claim_rate")) for row in baselines if row is not None
        ]
        if selected_rate is None or any(rate is None for rate in baseline_rates):
            continue
        comparable += 1
        worst_baseline = max(rate for rate in baseline_rates if rate is not None)
        if selected_rate > worst_baseline + FALSE_CLAIM_RATE_TOLERANCE:
            violations.append(f"{world_id} ({selected_rate:g} > {worst_baseline:g})")
    return comparable, violations


def _false_claim_rate_summary_violation(
    rows: list[dict[str, str]] | None,
    selected_method: str,
) -> str | None:
    if not rows:
        return None
    by_method = {row.get("method", ""): row for row in rows if row.get("method")}
    selected = by_method.get(selected_method)
    baselines = [by_method.get(baseline) for baseline in REQUIRED_BENCHMARK_BASELINES]
    if selected is None or any(row is None for row in baselines):
        return None

    selected_rate = _first_float(selected, ("mean_false_claim_rate", "false_claim_rate"))
    baseline_rates = [
        _first_float(row, ("mean_false_claim_rate", "false_claim_rate"))
        for row in baselines
        if row is not None
    ]
    if selected_rate is None or any(rate is None for rate in baseline_rates):
        return None
    worst_baseline = max(rate for rate in baseline_rates if rate is not None)
    if selected_rate <= worst_baseline + FALSE_CLAIM_RATE_TOLERANCE:
        return None
    return (
        "Selected method summary false_claim_rate exceeds the worse required baseline: "
        f"{selected_rate:g} > {worst_baseline:g}."
    )


def _validate_selected_method_majority_wins(
    benchmark_rows: list[dict[str, str]] | None,
    rows: list[dict[str, str]] | None,
    selected_record: dict[str, Any] | None,
    summary_artifact: str,
    benchmark_artifact: str,
    findings: list[dict[str, Any]],
) -> None:
    if not selected_record:
        return
    selected_method = _selected_benchmark_policy_name(selected_record)
    if not isinstance(selected_method, str) or not selected_method:
        return

    benchmark_signal = _per_world_benchmark_win_signal(benchmark_rows, selected_method)
    comparable_signal_found = benchmark_signal is not None
    if benchmark_signal is not None:
        _add_majority_failure_findings(
            benchmark_signal,
            findings,
            benchmark_artifact,
        )

    aggregate = _aggregate_summary_win_signal(rows or [], selected_method)
    if aggregate is not None:
        comparable_signal_found = True
        _add_majority_failure_findings(
            aggregate,
            findings,
            summary_artifact,
        )
        if benchmark_signal is not None and not _majority_signals_agree(aggregate, benchmark_signal):
            _add_finding(
                findings,
                "error",
                "benchmark_summary_majority_disagreement",
                "benchmark_summary.csv majority counts do not agree with benchmark_results.csv for "
                f"selected method {selected_method!r}.",
                summary_artifact,
            )
        return

    per_world = _per_world_summary_win_signal(rows or [], selected_method)
    if per_world is None:
        if not comparable_signal_found:
            _add_finding(
                findings,
                "error",
                "selected_method_missing_comparable_majority_signal",
                "Selected method has no comparable majority-world signal against random_feasible and fixed_mix.",
                summary_artifact,
            )
        return
    comparable_signal_found = True
    _add_majority_failure_findings(
        per_world,
        findings,
        summary_artifact,
    )
    if benchmark_signal is not None and not _majority_signals_agree(per_world, benchmark_signal):
        _add_finding(
            findings,
            "error",
            "benchmark_summary_majority_disagreement",
            "benchmark_summary.csv per-world values do not agree with benchmark_results.csv for "
            f"selected method {selected_method!r}.",
            summary_artifact,
        )


def _selected_benchmark_policy_name(selected_record: dict[str, Any]) -> str | None:
    for key in ("benchmark_policy_name", "method", "benchmark_method"):
        value = selected_record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _add_majority_failure_findings(
    signal: tuple[float, dict[str, float]],
    findings: list[dict[str, Any]],
    artifact: str,
) -> None:
    world_count, wins_by_baseline = signal
    failures = [
        f"{baseline} ({wins:g}/{world_count:g})"
        for baseline, wins in wins_by_baseline.items()
        if wins <= world_count / 2
    ]
    if failures:
        _add_finding(
            findings,
            "error",
            "selected_method_missing_majority_world_wins",
            "Selected method does not beat required baselines in a majority of worlds: "
            + ", ".join(failures)
            + ".",
            artifact,
        )


def _majority_signals_agree(
    left: tuple[float, dict[str, float]],
    right: tuple[float, dict[str, float]],
) -> bool:
    left_world_count, left_wins = left
    right_world_count, right_wins = right
    if left_world_count != right_world_count:
        return False
    for baseline in ("random_feasible", "fixed_mix"):
        if left_wins.get(baseline) != right_wins.get(baseline):
            return False
    return True


def _aggregate_summary_win_signal(
    rows: list[dict[str, str]],
    selected_method: str,
) -> tuple[float, dict[str, float]] | None:
    selected_rows = [row for row in rows if row.get("method") == selected_method]
    if not selected_rows:
        return None
    row = selected_rows[0]
    world_count = _to_float(row.get("world_count"))
    if world_count is None or world_count <= 0:
        return None

    wins_by_baseline: dict[str, float] = {}
    for baseline in ("random_feasible", "fixed_mix"):
        wins = _first_float(
            row,
            (
                f"beats_{baseline}_worlds",
                f"worlds_beating_{baseline}",
                f"wins_vs_{baseline}",
                f"{baseline}_worlds_beaten",
            ),
        )
        if wins is not None:
            wins_by_baseline[baseline] = wins
    if set(wins_by_baseline) != {"random_feasible", "fixed_mix"}:
        return None
    return world_count, wins_by_baseline


def _per_world_summary_win_signal(
    rows: list[dict[str, str]],
    selected_method: str,
) -> tuple[float, dict[str, float]] | None:
    if not rows:
        return None
    if "world_id" not in rows[0] or "best_feasible_utility" not in rows[0]:
        return None
    by_world_method: dict[tuple[str, str], dict[str, str]] = {}
    worlds: set[str] = set()
    for row in rows:
        method = row.get("method")
        world_id = row.get("world_id")
        if not method or not world_id:
            continue
        worlds.add(world_id)
        by_world_method[(world_id, method)] = row
    if not worlds:
        return None

    wins_by_baseline = {"random_feasible": 0.0, "fixed_mix": 0.0}
    comparable_worlds = 0
    for world_id in worlds:
        selected = by_world_method.get((world_id, selected_method))
        random_row = by_world_method.get((world_id, "random_feasible"))
        fixed_row = by_world_method.get((world_id, "fixed_mix"))
        if selected is None or random_row is None or fixed_row is None:
            continue
        selected_best = _to_float(selected.get("best_feasible_utility"))
        random_best = _to_float(random_row.get("best_feasible_utility"))
        fixed_best = _to_float(fixed_row.get("best_feasible_utility"))
        if selected_best is None or random_best is None or fixed_best is None:
            continue
        comparable_worlds += 1
        if selected_best > random_best:
            wins_by_baseline["random_feasible"] += 1
        if selected_best > fixed_best:
            wins_by_baseline["fixed_mix"] += 1
    if comparable_worlds == 0:
        return None
    return float(comparable_worlds), wins_by_baseline


def _per_world_benchmark_win_signal(
    rows: list[dict[str, str]] | None,
    selected_method: str,
) -> tuple[float, dict[str, float]] | None:
    if not rows:
        return None
    if "world_id" not in rows[0] or "best_feasible_utility" not in rows[0]:
        return None
    return _per_world_summary_win_signal(rows, selected_method)


def _validate_ablation_rows(
    rows: list[dict[str, str]] | None,
    artifact: str,
    findings: list[dict[str, Any]],
) -> None:
    if rows is None:
        return
    if not rows:
        _add_finding(
            findings,
            "error",
            "ablation_results_empty",
            "Ablation results CSV must contain at least one result row.",
            artifact,
        )
        return
    if "removed_component" not in rows[0]:
        _add_finding(
            findings,
            "error",
            "ablation_results_missing_component",
            "Ablation results must contain removed_component.",
            artifact,
        )
        return
    components = {row.get("removed_component") for row in rows}
    if "none" not in components:
        _add_finding(
            findings,
            "error",
            "ablation_missing_full_policy",
            "Ablation results must include the full policy row removed_component=none.",
            artifact,
        )
    if len(components) < 2:
        _add_finding(
            findings,
            "error",
            "ablation_missing_component_removal",
            "Ablation results must include at least one component-removal row.",
            artifact,
        )


def _validate_scientist_journal(
    data: Any,
    expected_run_id: str | None,
    artifact: str,
    findings: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        _add_finding(
            findings,
            "error",
            "scientist_journal_invalid",
            "Scientist journal must be a JSON object.",
            artifact,
        )
        return None

    if expected_run_id and data.get("run_id") != expected_run_id:
        _add_finding(
            findings,
            "warning",
            "scientist_journal_run_id_mismatch",
            f"Scientist journal run_id {data.get('run_id')!r} does not match directory {expected_run_id!r}.",
            artifact,
        )

    nodes = data.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        _add_finding(
            findings,
            "error",
            "scientist_journal_missing_nodes",
            "Scientist journal must contain a non-empty nodes list.",
            artifact,
        )
        return None

    selected_node_id = data.get("selected_node_id")
    selected_node = data.get("selected_node")
    if not isinstance(selected_node_id, str) or not selected_node_id:
        _add_finding(
            findings,
            "error",
            "scientist_journal_missing_selected_node",
            "Scientist journal must record selected_node_id.",
            artifact,
        )
        return None

    selected_record = next(
        (node for node in nodes if isinstance(node, dict) and node.get("node_id") == selected_node_id),
        None,
    )
    if selected_record is None:
        _add_finding(
            findings,
            "error",
            "scientist_journal_selected_node_not_found",
            f"Selected node {selected_node_id!r} is not present in nodes.",
            artifact,
        )
        return None

    if isinstance(selected_node, dict) and selected_node.get("node_id") != selected_node_id:
        _add_finding(
            findings,
            "error",
            "scientist_journal_selected_node_mismatch",
            "selected_node does not match selected_node_id.",
            artifact,
        )
    if selected_record.get("status") != "completed":
        _add_finding(
            findings,
            "error",
            "selected_node_not_completed",
            f"Selected node {selected_node_id!r} status is {selected_record.get('status')!r}.",
            artifact,
        )
    contract = selected_record.get("contract")
    if contract is None:
        _add_finding(
            findings,
            "error",
            "selected_node_contract_missing",
            f"Selected node {selected_node_id!r} is missing its method-node contract.",
            artifact,
        )
    elif not isinstance(contract, dict):
        _add_finding(
            findings,
            "error",
            "selected_node_contract_invalid",
            f"Selected node {selected_node_id!r} contract must be a mapping with valid: true.",
            artifact,
        )
    elif contract.get("valid") is not True:
        _add_finding(
            findings,
            "error",
            "selected_node_contract_invalid",
            f"Selected node {selected_node_id!r} failed its method-node contract.",
            artifact,
        )
    return selected_record


def _validate_v3_scientist_journal(
    data: Any,
    expected_run_id: str | None,
    artifact: str,
    findings: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        _add_finding(
            findings,
            "error",
            "scientist_journal_invalid",
            "V3 scientist journal must be a JSON object.",
            artifact,
        )
        return None
    if data.get("version") != "v3":
        _add_finding(
            findings,
            "error",
            "scientist_journal_version_not_v3",
            "V3 validation requires scientist_journal.json version='v3'.",
            artifact,
        )
    if expected_run_id and data.get("run_id") != expected_run_id:
        _add_finding(
            findings,
            "warning",
            "scientist_journal_run_id_mismatch",
            f"Scientist journal run_id {data.get('run_id')!r} does not match directory {expected_run_id!r}.",
            artifact,
        )

    nodes = data.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        _add_finding(
            findings,
            "error",
            "scientist_journal_missing_nodes",
            "V3 scientist journal must contain a non-empty nodes list.",
            artifact,
        )
        return None

    selected_node_id = data.get("selected_node_id")
    selected_node = data.get("selected_node")
    if not isinstance(selected_node_id, str) or not selected_node_id:
        _add_finding(
            findings,
            "error",
            "scientist_journal_missing_selected_node",
            "V3 scientist journal must record selected_node_id.",
            artifact,
        )
        return None

    selected_record = next(
        (node for node in nodes if isinstance(node, dict) and node.get("node_id") == selected_node_id),
        None,
    )
    if selected_record is None:
        _add_finding(
            findings,
            "error",
            "scientist_journal_selected_node_not_found",
            f"Selected node {selected_node_id!r} is not present in V3 journal nodes.",
            artifact,
        )
        return None
    if isinstance(selected_node, dict) and selected_node.get("node_id") != selected_node_id:
        _add_finding(
            findings,
            "error",
            "scientist_journal_selected_node_mismatch",
            "selected_node does not match selected_node_id.",
            artifact,
        )
    if not _v3_selected_node_valid(selected_record):
        _add_finding(
            findings,
            "error",
            "selected_node_contract_invalid",
            f"Selected V3 node {selected_node_id!r} must have contract.valid true or an equivalent node valid flag.",
            artifact,
        )
    return selected_record


def _v3_selected_node_valid(node: dict[str, Any]) -> bool:
    contract = node.get("contract")
    if isinstance(contract, dict) and _truthy(contract.get("valid")):
        return True
    for key in ("valid", "validation_valid", "contract_valid"):
        if _truthy(node.get(key)):
            return True
    return False


def _validate_v3_selected_node_artifacts(
    root: Path,
    run_dir: Path,
    selected_record: dict[str, Any] | None,
    findings: list[dict[str, Any]],
    artifacts: dict[str, str],
) -> dict[str, Any]:
    if not selected_record:
        return {}

    workspace_value = selected_record.get("workspace")
    workspace = _coerce_path(root, workspace_value) if workspace_value else None
    allowed_roots = _selected_artifact_allowed_roots(workspace, run_dir)
    artifact_map = selected_record.get("artifacts") if isinstance(selected_record.get("artifacts"), dict) else {}
    loaded_json: dict[str, Any] = {}

    for filename in artifact_constants.V3_MECHANISM_NODE_ARTIFACTS:
        key = Path(filename).stem
        path = _resolve_v3_selected_node_artifact(root, workspace, artifact_map, filename)
        if path is None:
            _add_finding(
                findings,
                "error",
                "selected_node_artifact_unresolvable",
                f"Cannot resolve selected V3 node artifact {filename}.",
                None,
            )
            continue
        artifacts[f"selected_node_{key}"] = str(path)
        if not _validate_selected_artifact_file(root, path, key, filename, allowed_roots, findings):
            continue
        if filename.endswith(".json"):
            data = _load_json(path, f"selected_node_{key}", findings)
            loaded_json[key] = data
            if key == "mechanism_spec":
                _validate_v3_mechanism_spec(data, _rel(path, root), findings)
            elif key == "validation_report" and isinstance(data, dict) and data.get("valid") is not True:
                _add_finding(
                    findings,
                    "error",
                    "selected_node_validation_invalid",
                    "Selected V3 node validation_report.json must contain valid: true.",
                    _rel(path, root),
                )

    return loaded_json


def _resolve_v3_selected_node_artifact(
    root: Path,
    workspace: Path | None,
    artifact_map: dict[str, Any],
    filename: str,
) -> Path | None:
    stem = Path(filename).stem
    candidates = (
        filename,
        stem,
        stem.replace("_", "-"),
        f"selected_node_{stem}",
    )
    for key in candidates:
        value = artifact_map.get(key)
        if value:
            return _coerce_path(root, value)
    if workspace is not None:
        return workspace / filename
    return None


def _validate_v3_mechanism_spec(
    data: Any,
    artifact: str,
    findings: list[dict[str, Any]],
) -> None:
    if not isinstance(data, dict):
        _add_finding(
            findings,
            "error",
            "selected_node_mechanism_spec_invalid",
            "Selected V3 mechanism_spec.json must be a JSON object.",
            artifact,
        )
        return
    components = data.get("components")
    if not isinstance(components, list) or not components:
        _add_finding(
            findings,
            "error",
            "selected_node_mechanism_spec_missing_components",
            "Selected V3 mechanism_spec.json must include at least one component.",
            artifact,
        )
    if not (
        _has_nonempty_text_or_list(data.get("claims"))
        or _has_nonempty_text_or_list(data.get("claim"))
        or _has_nonempty_text_or_list(data.get("literature_basis"))
        or _has_nonempty_text_or_list(data.get("stress_test_requirement"))
        or _has_nonempty_text_or_list(data.get("stress_test_requirements"))
    ):
        _add_finding(
            findings,
            "error",
            "selected_node_mechanism_spec_missing_evidence_basis",
            "Selected V3 mechanism_spec.json must include a claim, literature_basis, or stress_test_requirement.",
            artifact,
        )


def _selected_v3_mechanism_name(
    selected_record: dict[str, Any] | None,
    mechanism_spec: Any,
    mechanism_metrics: Any,
) -> str | None:
    for source in (selected_record, mechanism_metrics, mechanism_spec):
        if not isinstance(source, dict):
            continue
        for key in (
            "mechanism",
            "mechanism_name",
            "benchmark_mechanism",
            "benchmark_method",
            "benchmark_policy_name",
            "method",
            "mechanism_id",
            "name",
        ):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


def _validate_v3_mechanism_benchmark_results(
    rows: list[dict[str, str]] | None,
    artifact: str,
    findings: list[dict[str, Any]],
) -> None:
    if rows is None:
        return
    if not rows:
        _add_finding(
            findings,
            "error",
            "mechanism_benchmark_results_empty",
            "mechanism_benchmark_results.csv must contain at least one result row.",
            artifact,
        )
        return
    if _v3_name_column(rows) is None:
        _add_finding(
            findings,
            "error",
            "mechanism_benchmark_results_missing_mechanism",
            "mechanism_benchmark_results.csv must contain mechanism or method.",
            artifact,
        )
    missing_columns = [column for column in ("best_feasible_utility", "false_claim_rate") if column not in rows[0]]
    if missing_columns:
        _add_finding(
            findings,
            "error",
            "mechanism_benchmark_results_missing_columns",
            f"mechanism_benchmark_results.csv is missing columns: {', '.join(missing_columns)}.",
            artifact,
        )


def _validate_v3_mechanism_benchmark_summary(
    rows: list[dict[str, str]] | None,
    selected_mechanism: str | None,
    selected_record: dict[str, Any] | None,
    mechanism_metrics: Any,
    artifact: str,
    findings: list[dict[str, Any]],
) -> None:
    if rows is None:
        return
    if not rows:
        _add_finding(
            findings,
            "error",
            "mechanism_benchmark_summary_empty",
            "mechanism_benchmark_summary.csv must contain at least one summary row.",
            artifact,
        )
        return
    name_column = _v3_name_column(rows)
    if name_column is None:
        _add_finding(
            findings,
            "error",
            "mechanism_benchmark_summary_missing_mechanism",
            "mechanism_benchmark_summary.csv must contain mechanism or method.",
            artifact,
        )
        return
    by_mechanism = {row.get(name_column, ""): row for row in rows if row.get(name_column)}
    missing_baselines = [baseline for baseline in REQUIRED_BENCHMARK_BASELINES if baseline not in by_mechanism]
    if missing_baselines:
        _add_finding(
            findings,
            "error",
            "mechanism_benchmark_summary_missing_required_baselines",
            "mechanism_benchmark_summary.csv must include random_feasible and fixed_mix baseline rows. "
            f"Missing: {', '.join(missing_baselines)}.",
            artifact,
        )
    if not selected_mechanism:
        _add_finding(
            findings,
            "error",
            "selected_mechanism_missing_name",
            "Selected V3 node does not expose a mechanism name for benchmark lookup.",
            artifact,
        )
        return
    selected_row = by_mechanism.get(selected_mechanism)
    if selected_row is None:
        _add_finding(
            findings,
            "error",
            "selected_mechanism_missing_benchmark_summary",
            f"Selected mechanism {selected_mechanism!r} is missing from mechanism_benchmark_summary.csv.",
            artifact,
        )
        return

    if (
        _truthy(selected_row.get("architecture_clone"))
        or _truthy(_dict_get(mechanism_metrics, "architecture_clone"))
        or _truthy(_dict_get(selected_record, "architecture_clone"))
    ):
        _add_finding(
            findings,
            "error",
            "selected_node_architecture_clone",
            f"Selected V3 mechanism {selected_mechanism!r} is marked architecture_clone true.",
            artifact,
        )

    summary_eligible = _truthy(selected_row.get("selected_eligible"))
    metrics_eligible = _truthy(_dict_get(mechanism_metrics, "selected_eligible"))
    if not (summary_eligible or metrics_eligible):
        _add_finding(
            findings,
            "error",
            "selected_mechanism_not_eligible",
            f"Selected V3 mechanism {selected_mechanism!r} is not selected_eligible in metrics or benchmark summary.",
            artifact,
        )

    missing_majorities: list[str] = []
    if not _v3_summary_bool(
        selected_row,
        ("majority_win_vs_random_feasible", "beats_random_feasible_majority"),
    ):
        missing_majorities.append("random_feasible")
    if not _v3_summary_bool(
        selected_row,
        ("majority_win_vs_fixed_mix", "beats_fixed_mix_majority"),
    ):
        missing_majorities.append("fixed_mix")
    if missing_majorities:
        _add_finding(
            findings,
            "error",
            "selected_mechanism_missing_majority_wins",
            "Selected V3 mechanism must have majority wins vs required baselines. Missing: "
            + ", ".join(missing_majorities)
            + ".",
            artifact,
        )

    key_delta = _first_float(selected_row, ("key_ablation_delta", "mean_key_ablation_delta"))
    if key_delta is None:
        key_delta = _first_float(_dict_get_dict(mechanism_metrics, "metrics"), ("key_ablation_delta",))
    if key_delta is None or key_delta <= 0:
        _add_finding(
            findings,
            "error",
            "selected_mechanism_key_ablation_delta_nonpositive",
            "Selected V3 mechanism must have key_ablation_delta > 0.",
            artifact,
        )

    selected_false = _first_float(selected_row, ("mean_false_claim_rate", "false_claim_rate"))
    if selected_false is None:
        selected_false = _first_float(
            _dict_get_dict(mechanism_metrics, "metrics"),
            ("mean_false_claim_rate", "false_claim_rate"),
        )
    baseline_false = [
        _first_float(by_mechanism.get(baseline, {}), ("mean_false_claim_rate", "false_claim_rate"))
        for baseline in REQUIRED_BENCHMARK_BASELINES
    ]
    if selected_false is None or any(value is None for value in baseline_false):
        _add_finding(
            findings,
            "error",
            "selected_mechanism_false_claim_rate_missing",
            "Selected V3 mechanism and required baselines must expose false_claim_rate.",
            artifact,
        )
    elif all(selected_false > value + FALSE_CLAIM_RATE_TOLERANCE for value in baseline_false if value is not None):
        _add_finding(
            findings,
            "error",
            "selected_mechanism_false_claim_rate_worse_than_baselines",
            "Selected V3 mechanism false_claim_rate is worse than both required baselines.",
            artifact,
        )


def _validate_v3_mechanism_ablation_results(
    rows: list[dict[str, str]] | None,
    selected_mechanism: str | None,
    artifact: str,
    findings: list[dict[str, Any]],
) -> None:
    if rows is None:
        return
    if not rows:
        _add_finding(
            findings,
            "error",
            "mechanism_ablation_results_empty",
            "mechanism_ablation_results.csv must contain at least one ablation row.",
            artifact,
        )
        return
    ablation_column = "ablation" if "ablation" in rows[0] else "removed_component" if "removed_component" in rows[0] else None
    if ablation_column is None:
        _add_finding(
            findings,
            "error",
            "mechanism_ablation_results_missing_ablation",
            "mechanism_ablation_results.csv must contain ablation or removed_component.",
            artifact,
        )
        return
    selected_rows = rows
    if selected_mechanism and _v3_name_column(rows):
        name_column = _v3_name_column(rows)
        selected_rows = [row for row in rows if row.get(name_column or "") == selected_mechanism]
    labels = {row.get(ablation_column) for row in selected_rows}
    if "none" not in labels:
        _add_finding(
            findings,
            "error",
            "mechanism_ablation_missing_full_mechanism",
            "Selected V3 mechanism ablations must include ablation=none.",
            artifact,
        )
    if len(labels) < 2:
        _add_finding(
            findings,
            "error",
            "mechanism_ablation_missing_component_removal",
            "Selected V3 mechanism ablations must include at least one component-removal row.",
            artifact,
        )


def _v3_name_column(rows: list[dict[str, str]]) -> str | None:
    if not rows:
        return None
    if "mechanism" in rows[0]:
        return "mechanism"
    if "method" in rows[0]:
        return "method"
    return None


def _v3_summary_bool(row: dict[str, str], keys: tuple[str, ...]) -> bool:
    for key in keys:
        if _truthy(row.get(key)):
            return True
    return False


def _truthy(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    if isinstance(value, (int, float)):
        return value == 1
    return False


def _dict_get(data: Any, key: str) -> Any:
    return data.get(key) if isinstance(data, dict) else None


def _dict_get_dict(data: Any, key: str) -> dict[str, Any]:
    value = data.get(key) if isinstance(data, dict) else None
    return value if isinstance(value, dict) else {}


def _validate_selected_node_artifacts(
    root: Path,
    run_dir: Path,
    selected_record: dict[str, Any] | None,
    findings: list[dict[str, Any]],
    artifacts: dict[str, str],
) -> None:
    if not selected_record:
        return

    workspace_value = selected_record.get("workspace")
    workspace = _coerce_path(root, workspace_value) if workspace_value else None
    allowed_roots = _selected_artifact_allowed_roots(workspace, run_dir)
    artifact_map = selected_record.get("artifacts") if isinstance(selected_record.get("artifacts"), dict) else {}
    loaded_json: dict[str, Any] = {}

    for key, filename in SELECTED_NODE_ARTIFACTS.items():
        path_value = artifact_map.get(key)
        if path_value:
            path = _coerce_path(root, path_value)
        elif workspace is not None:
            path = workspace / filename
        else:
            _add_finding(
                findings,
                "error",
                "selected_node_artifact_unresolvable",
                f"Cannot resolve selected node artifact {key}.",
                None,
            )
            continue

        artifacts[f"selected_node_{key}"] = str(path)
        if not _validate_selected_artifact_file(root, path, key, filename, allowed_roots, findings):
            continue
        if filename.endswith(".json"):
            data = _load_json(path, f"selected_node_{key}", findings)
            loaded_json[key] = data
            if key == "validation_report" and isinstance(data, dict) and data.get("valid") is not True:
                _add_finding(
                    findings,
                    "error",
                    "selected_node_validation_invalid",
                    "Selected node validation_report.json must contain valid: true.",
                    _rel(path, root),
                )
            if key == "benchmark_metrics" and isinstance(data, dict):
                if "synthetic_replay" not in data and "method" not in data:
                    _add_finding(
                        findings,
                        "warning",
                        "selected_node_benchmark_metrics_sparse",
                        "Selected node benchmark_metrics.json does not include synthetic replay metrics.",
                        _rel(path, root),
                    )

    manifest = loaded_json.get("manifest")
    selected_is_v2 = _selected_node_is_v2(selected_record, manifest, artifact_map)
    loaded_v2_json: dict[str, Any] = {}
    v2_artifacts: dict[str, str] = {}
    for key, filename in SELECTED_NODE_V2_ARTIFACTS.items():
        path_value = artifact_map.get(key)
        path = _coerce_path(root, path_value) if path_value else (workspace / filename if workspace is not None else None)
        required = selected_is_v2 or bool(path_value)
        if path is None:
            if required:
                _add_finding(
                    findings,
                    "error",
                    "selected_node_artifact_unresolvable",
                    f"Cannot resolve selected node artifact {key}.",
                    None,
                )
            continue

        exists = path.exists()
        if not required and not exists:
            continue
        artifacts[f"selected_node_{key}"] = str(path)
        if not _validate_selected_artifact_file(root, path, key, filename, allowed_roots, findings):
            continue

        data = _load_json(path, f"selected_node_{key}", findings)
        loaded_v2_json[key] = data
        v2_artifacts[key] = _rel(path, root)
        if key == "proposal":
            _validate_selected_proposal(data, _rel(path, root), findings)
        elif key == "novelty_report":
            _validate_selected_novelty_report(data, _rel(path, root), findings)

    if _selected_node_requires_full_chain(selected_record, manifest, artifact_map, loaded_v2_json):
        _validate_selected_full_chain_proposal(
            loaded_v2_json.get("proposal"),
            v2_artifacts.get("proposal", "proposal.json"),
            findings,
        )

    _validate_optional_selected_node_artifacts(
        root,
        selected_record,
        workspace,
        run_dir,
        artifact_map,
        findings,
        artifacts,
    )


def _selected_artifact_allowed_roots(workspace: Path | None, run_dir: Path) -> tuple[Path, ...]:
    roots: list[Path] = []
    if workspace is not None:
        roots.append(_safe_resolve(workspace))
    roots.append(_safe_resolve(run_dir))
    return tuple(roots)


def _validate_selected_artifact_file(
    root: Path,
    path: Path,
    key: str,
    filename: str,
    allowed_roots: tuple[Path, ...],
    findings: list[dict[str, Any]],
) -> bool:
    if not _is_under_any(path, allowed_roots):
        _add_finding(
            findings,
            "error",
            "selected_node_artifact_outside_allowed_roots",
            f"Selected node artifact {key} must resolve under the selected workspace or run directory.",
            _rel(path, root),
        )
        return False
    if not path.exists():
        _add_finding(
            findings,
            "error",
            f"missing_selected_node_{key}",
            f"Missing selected node artifact {filename}.",
            _rel(path, root),
        )
        return False
    if not path.is_file():
        _add_finding(
            findings,
            "error",
            "selected_node_artifact_not_file",
            f"Selected node artifact {key} must be a non-empty file.",
            _rel(path, root),
        )
        return False
    if path.stat().st_size == 0:
        _add_finding(
            findings,
            "error",
            f"empty_selected_node_{key}",
            f"Selected node artifact {filename} is empty.",
            _rel(path, root),
        )
        return False
    return True


def _is_under_any(path: Path, roots: tuple[Path, ...]) -> bool:
    candidate = _safe_resolve(path)
    for root in roots:
        try:
            candidate.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _safe_resolve(path: Path) -> Path:
    try:
        return path.expanduser().resolve()
    except OSError:
        return path.expanduser()


def _selected_node_is_v2(
    selected_record: dict[str, Any],
    manifest: Any,
    artifact_map: dict[str, Any],
) -> bool:
    version_values = [
        selected_record.get("schema_version"),
        selected_record.get("method_node_version"),
        selected_record.get("contract_version"),
    ]
    if isinstance(manifest, dict):
        version_values.extend(
            [
                manifest.get("schema_version"),
                manifest.get("method_node_version"),
                manifest.get("contract_version"),
            ]
        )
    for value in version_values:
        if value in ("v2", "2"):
            return True
        if isinstance(value, (int, float)) and value >= 2:
            return True
    return any(key in artifact_map for key in SELECTED_NODE_V2_ARTIFACTS)


def _validate_selected_proposal(
    data: Any,
    artifact: str,
    findings: list[dict[str, Any]],
) -> None:
    if not isinstance(data, dict):
        _add_finding(
            findings,
            "error",
            "selected_node_proposal_invalid",
            "Selected node proposal.json must be a JSON object.",
            artifact,
        )
        return
    missing = _missing_proposal_metadata(data)
    if missing:
        _add_finding(
            findings,
            "warning",
            "selected_node_proposal_sparse",
            f"Selected node proposal.json is missing required proposal metadata: {', '.join(missing)}.",
            artifact,
        )


def _missing_proposal_metadata(data: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    if not _has_text(data.get("proposal_id")):
        missing.append("proposal_id")
    if not _has_text(data.get("title")):
        missing.append("title")
    if not any(_has_text(data.get(key)) for key in ("method_claim", "summary")):
        missing.append("method_claim_or_summary")
    return missing


def _validate_selected_novelty_report(
    data: Any,
    artifact: str,
    findings: list[dict[str, Any]],
) -> None:
    if not isinstance(data, dict):
        _add_finding(
            findings,
            "error",
            "selected_node_novelty_report_invalid",
            "Selected node novelty_report.json must be a JSON object.",
            artifact,
        )
        return
    if _indicates_baseline_clone(data):
        nearest = data.get("nearest_baseline") or data.get("baseline") or data.get("clone_of") or "baseline"
        _add_finding(
            findings,
            "error",
            "selected_node_baseline_clone",
            f"Selected node novelty_report.json indicates a baseline clone of {nearest!r}.",
            artifact,
        )
    if "novelty_score" not in data:
        _add_finding(
            findings,
            "warning",
            "selected_node_novelty_score_missing",
            "Selected node novelty_report.json does not include novelty_score.",
            artifact,
        )
    if "baseline_overlap" not in data:
        _add_finding(
            findings,
            "warning",
            "selected_node_baseline_overlap_missing",
            "Selected node novelty_report.json does not include baseline_overlap.",
            artifact,
        )
    overlap = _to_float(data.get("selection_overlap_vs_baselines"))
    if overlap is None:
        overlap = _to_float(data.get("baseline_overlap"))
    nearest = data.get("nearest_baseline") or data.get("baseline") or data.get("clone_of")
    if (
        overlap is not None
        and overlap > BASELINE_OVERLAP_ERROR_THRESHOLD
        and isinstance(nearest, str)
        and nearest in SIMPLE_BASELINE_METHODS
    ):
        _add_finding(
            findings,
            "error",
            "selected_node_baseline_overlap_high",
            "Selected node novelty_report.json has high overlap with simple baseline "
            f"{nearest!r}: {overlap:.3f} exceeds {BASELINE_OVERLAP_ERROR_THRESHOLD:.3f}.",
            artifact,
        )


def _selected_node_requires_full_chain(
    selected_record: dict[str, Any],
    manifest: Any,
    artifact_map: dict[str, Any],
    loaded_v2_json: dict[str, Any],
) -> bool:
    version_values = [
        selected_record.get("schema_version"),
        selected_record.get("method_node_version"),
        selected_record.get("contract_version"),
        selected_record.get("productization_schema"),
    ]
    if isinstance(manifest, dict):
        version_values.extend(
            [
                manifest.get("schema_version"),
                manifest.get("method_node_version"),
                manifest.get("contract_version"),
                manifest.get("productization_schema"),
            ]
        )
    for data in loaded_v2_json.values():
        if isinstance(data, dict):
            version_values.extend(
                [
                    data.get("schema_version"),
                    data.get("method_node_version"),
                    data.get("contract_version"),
                    data.get("productization_schema"),
                ]
            )
            if data.get("full_chain_productization") is True:
                return True
    if any(key in artifact_map for key in ("design_space", "candidate_artifacts")):
        return True
    for value in version_values:
        if value in ("v3", "3", "full_chain", "productized", "productization"):
            return True
        if isinstance(value, (int, float)) and value >= 3:
            return True
    return False


def _validate_selected_full_chain_proposal(
    data: Any,
    artifact: str,
    findings: list[dict[str, Any]],
) -> None:
    if not isinstance(data, dict):
        return
    literature_basis = data.get("literature_basis") or data.get("literature_references")
    if not _has_nonempty_text_or_list(literature_basis):
        _add_finding(
            findings,
            "error",
            "selected_node_literature_basis_missing",
            "Full-chain selected proposal must include a literature_basis or literature_references field.",
            artifact,
        )
    if not _has_any_text(
        data,
        (
            "literature_gap",
            "research_gap",
            "selected_gap",
            "identified_gap",
            "gap_for_design_scientist",
            "method_gap",
        ),
    ):
        _add_finding(
            findings,
            "error",
            "selected_node_literature_gap_missing",
            "Full-chain selected proposal must include the literature or research gap it addresses.",
            artifact,
        )
    if not _has_architecture_delta(data):
        _add_finding(
            findings,
            "error",
            "selected_node_architecture_delta_missing",
            "Full-chain selected proposal must include architecture_delta or equivalent architecture novelty.",
            artifact,
        )


def _has_architecture_delta(data: dict[str, Any]) -> bool:
    if _has_any_text(
        data,
        (
            "architecture_delta",
            "architecture_novelty",
            "architectural_delta",
            "method_architecture_delta",
        ),
    ):
        return True
    architecture = data.get("architecture")
    if isinstance(architecture, dict):
        return _has_any_text(architecture, ("delta", "novelty", "architecture_delta"))
    return False


def _validate_optional_selected_node_artifacts(
    root: Path,
    selected_record: dict[str, Any],
    workspace: Path | None,
    run_dir: Path,
    artifact_map: dict[str, Any],
    findings: list[dict[str, Any]],
    artifacts: dict[str, str],
) -> None:
    allowed_roots = _selected_artifact_allowed_roots(workspace, run_dir)
    optional_keys = {
        key
        for key in artifact_map
        if key not in SELECTED_NODE_ARTIFACTS and key not in SELECTED_NODE_V2_ARTIFACTS
    }
    if workspace is not None:
        if "design_space" not in artifact_map and (workspace / "design_space.json").exists():
            optional_keys.add("design_space")
        if "candidate_artifacts" not in artifact_map and (workspace / "candidate_artifacts.json").exists():
            optional_keys.add("candidate_artifacts")

    for key in sorted(optional_keys):
        path_value = artifact_map.get(key)
        path = _coerce_path(root, path_value) if path_value else (workspace / f"{key}.json" if workspace else None)
        if path is None:
            continue
        artifacts[f"selected_node_{key}"] = str(path)
        if not _validate_selected_artifact_file(root, path, key, path.name, allowed_roots, findings):
            continue
        if _is_design_space_key(key):
            _validate_design_space_artifact(path, root, findings)
        elif _is_candidate_artifact_key(key):
            _validate_candidate_artifact(path, root, findings)
        elif path.suffix.lower() == ".json":
            _load_json(path, f"selected_node_{key}", findings)


def _is_design_space_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return "design_space" in normalized


def _is_candidate_artifact_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return "candidate" in normalized and normalized != "candidate_policy"


def _validate_design_space_artifact(
    path: Path,
    root: Path,
    findings: list[dict[str, Any]],
) -> None:
    artifact = _rel(path, root)
    data = _load_json(path, "selected_node_design_space", findings)
    if not isinstance(data, dict):
        _add_finding(
            findings,
            "error",
            "selected_node_design_space_invalid",
            "Selected design-space artifact must be a JSON object.",
            artifact,
        )
        return
    structure_keys = (
        "objectives",
        "constraints",
        "operators",
        "design_operators",
        "candidate_schema",
        "dimensions",
    )
    if not any(data.get(key) not in (None, "", [], {}) for key in structure_keys):
        _add_finding(
            findings,
            "error",
            "selected_node_design_space_missing_structure",
            "Selected design-space artifact must describe objectives, constraints, operators, dimensions, or candidate schema.",
            artifact,
        )


def _validate_candidate_artifact(
    path: Path,
    root: Path,
    findings: list[dict[str, Any]],
) -> None:
    artifact = _rel(path, root)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        rows = _load_csv(path, "selected_node_candidate_artifacts", findings)
        if rows is not None and not rows:
            _add_finding(
                findings,
                "error",
                "selected_node_candidate_artifacts_empty",
                "Selected candidate artifact CSV contains no rows.",
                artifact,
            )
        return
    if suffix == ".json":
        data = _load_json(path, "selected_node_candidate_artifacts", findings)
        if data in (None, "", [], {}):
            _add_finding(
                findings,
                "error",
                "selected_node_candidate_artifacts_empty",
                "Selected candidate artifact JSON is empty.",
                artifact,
            )
        elif not isinstance(data, (dict, list)):
            _add_finding(
                findings,
                "error",
                "selected_node_candidate_artifacts_invalid",
                "Selected candidate artifact JSON must be an object or list.",
                artifact,
            )
        return


def _validate_selected_trace(
    root: Path,
    selected_record: dict[str, Any] | None,
    findings: list[dict[str, Any]],
) -> None:
    if not selected_record:
        return
    trace = selected_record.get("trace")
    if not isinstance(trace, dict):
        _add_finding(
            findings,
            "error",
            "selected_node_trace_missing",
            "Selected node must include a trace mapping.",
            None,
        )
        return
    for key in ("proposal", "benchmark_metrics", "ablation_results", "novelty_report"):
        value = trace.get(key)
        if not _has_text(value):
            _add_finding(
                findings,
                "error",
                f"selected_node_trace_missing_{key}",
                f"Selected node trace is missing {key}.",
                None,
            )
            continue
        path = _coerce_path(root, value)
        if not path.exists():
            _add_finding(
                findings,
                "error",
                f"selected_node_trace_missing_{key}_artifact",
                f"Selected node trace points to missing {key} artifact.",
                _rel(path, root),
            )


def _indicates_baseline_clone(data: dict[str, Any]) -> bool:
    if data.get("baseline_clone") is True or data.get("is_baseline_clone") is True:
        return True
    for key in ("verdict", "classification", "novelty_class", "status"):
        value = data.get(key)
        if isinstance(value, str) and value.strip().lower() in {
            "baseline_clone",
            "baseline clone",
            "clone",
            "not_novel_baseline_clone",
        }:
            return True
    novelty_score = _to_float(data.get("novelty_score"))
    baseline_overlap = _to_float(data.get("baseline_overlap"))
    if novelty_score is not None and baseline_overlap is not None:
        return novelty_score <= 0.1 and baseline_overlap >= 0.95
    return False


def _validate_method_report(
    text: str,
    artifact: str,
    findings: list[dict[str, Any]],
) -> None:
    for heading in METHOD_REPORT_REQUIRED_SECTIONS:
        if heading not in text:
            _add_finding(
                findings,
                "error",
                "method_report_missing_section",
                f"Method report is missing required section {heading}.",
                artifact,
            )


def _validate_v3_method_report(
    text: str,
    artifact: str,
    findings: list[dict[str, Any]],
) -> None:
    for heading in V3_METHOD_REPORT_REQUIRED_SECTIONS:
        if heading not in text:
            _add_finding(
                findings,
                "error",
                "method_report_missing_section",
                f"V3 method report is missing required section {heading}.",
                artifact,
            )


def _validate_state_unsupported_claims(
    root: Path,
    findings: list[dict[str, Any]],
    artifacts: dict[str, str],
) -> None:
    state_path = root / "state" / "design_state.json"
    if not state_path.exists():
        return
    artifacts["design_state"] = str(state_path)
    state = _load_json(state_path, "design_state", findings)
    if isinstance(state, dict):
        unsupported = state.get("unsupported_claims")
        if not isinstance(unsupported, list) or not unsupported:
            _add_finding(
                findings,
                "warning",
                "unsupported_claims_not_recorded",
                "Design state has no unsupported_claims list; method report must not imply these claims are resolved.",
                _rel(state_path, root),
            )


def _require_run_file(
    root: Path,
    run_dir: Path,
    key: str,
    code: str,
    findings: list[dict[str, Any]],
    artifacts: dict[str, str],
) -> Path:
    rel = CRITICAL_RUN_ARTIFACTS[key]
    path = run_dir / rel
    artifacts[key] = str(path)
    if not path.exists():
        _add_finding(
            findings,
            "error",
            code,
            f"Missing critical run artifact: {rel}.",
            _rel(path, root),
        )
    elif path.is_file() and path.stat().st_size == 0:
        _add_finding(
            findings,
            "error",
            f"empty_{key}",
            f"Critical run artifact is empty: {rel}.",
            _rel(path, root),
        )
    return path


def _require_v3_run_file(
    root: Path,
    run_dir: Path,
    key: str,
    code: str,
    findings: list[dict[str, Any]],
    artifacts: dict[str, str],
) -> Path:
    rel = V3_CRITICAL_RUN_ARTIFACTS[key]
    path = run_dir / rel
    artifacts[key] = str(path)
    if not path.exists():
        _add_finding(
            findings,
            "error",
            code,
            f"Missing critical V3 run artifact: {rel}.",
            _rel(path, root),
        )
    elif path.is_file() and path.stat().st_size == 0:
        _add_finding(
            findings,
            "error",
            f"empty_{key}",
            f"Critical V3 run artifact is empty: {rel}.",
            _rel(path, root),
        )
    return path


def _validate_required_v3_run_json(
    root: Path,
    run_dir: Path,
    key: str,
    findings: list[dict[str, Any]],
    artifacts: dict[str, str],
) -> None:
    path = _require_v3_run_file(
        root,
        run_dir,
        key,
        f"missing_{key}",
        findings,
        artifacts,
    )
    data = _load_json(path, key, findings)
    if data is None:
        return
    if not isinstance(data, (dict, list)):
        _add_finding(
            findings,
            "error",
            f"invalid_{key}",
            f"{V3_CRITICAL_RUN_ARTIFACTS[key]} must be a JSON object or list.",
            _rel(path, root),
        )
    elif data in ({}, []):
        _add_finding(
            findings,
            "error",
            f"empty_{key}",
            f"{V3_CRITICAL_RUN_ARTIFACTS[key]} must not be empty.",
            _rel(path, root),
        )


def _optional_run_file(
    root: Path,
    run_dir: Path,
    key: str,
    findings: list[dict[str, Any]],
    artifacts: dict[str, str],
) -> Path:
    rel = OPTIONAL_RUN_ARTIFACTS[key]
    path = run_dir / rel
    if path.exists():
        artifacts[key] = str(path)
        if path.is_file() and path.stat().st_size == 0:
            _add_finding(
                findings,
                "error",
                f"empty_{key}",
                f"Optional run artifact exists but is empty: {rel}.",
                _rel(path, root),
            )
    return path


def _validate_optional_run_json_artifact(
    root: Path,
    run_dir: Path,
    key: str,
    findings: list[dict[str, Any]],
    artifacts: dict[str, str],
) -> None:
    path = _optional_run_file(root, run_dir, key, findings, artifacts)
    if not path.exists():
        return
    data = _load_json(path, key, findings)
    if data is None:
        return
    if not isinstance(data, (dict, list)):
        _add_finding(
            findings,
            "error",
            f"invalid_{key}",
            f"{OPTIONAL_RUN_ARTIFACTS[key]} must be a JSON object or list.",
            _rel(path, root),
        )


def _require_file(
    root: Path,
    rel: str,
    code: str,
    findings: list[dict[str, Any]],
    artifacts: dict[str, str],
) -> Path:
    path = root / rel
    key = rel.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    artifacts[key] = str(path)
    if not path.exists():
        _add_finding(
            findings,
            "error",
            code,
            f"Missing critical framework artifact: {rel}.",
            rel,
        )
    elif path.is_file() and path.stat().st_size == 0:
        _add_finding(
            findings,
            "error",
            f"empty_{key}",
            f"Critical framework artifact is empty: {rel}.",
            rel,
        )
    return path


def _validate_jsonl_memory(
    path: Path,
    label: str,
    artifact: str,
    findings: list[dict[str, Any]],
) -> None:
    if not path.exists() or not path.is_file():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        _add_finding(
            findings,
            "error",
            f"invalid_{label}",
            f"Cannot read memory log {path.name}: {exc}",
            artifact,
        )
        return
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            _add_finding(
                findings,
                "error",
                f"invalid_{label}",
                f"Memory log line {index} is not valid JSON: {exc}",
                artifact,
            )
            continue
        if not isinstance(record, dict):
            _add_finding(
                findings,
                "error",
                f"invalid_{label}",
                f"Memory log line {index} must be a JSON object.",
                artifact,
            )
            continue
        if not _has_text(record.get("summary")):
            _add_finding(
                findings,
                "warning",
                f"{label}_record_missing_summary",
                f"Memory log line {index} does not include a summary.",
                artifact,
            )


def _load_yaml_mapping(
    path: Path,
    label: str,
    findings: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        data = read_yaml(path)
    except Exception as exc:
        _add_finding(
            findings,
            "error",
            f"invalid_{label}",
            f"Cannot read YAML mapping {path.name}: {exc}",
            str(path),
        )
        return None
    if not isinstance(data, dict):
        _add_finding(
            findings,
            "error",
            f"invalid_{label}",
            f"{path.name} must be a YAML mapping.",
            str(path),
        )
        return None
    return data


def _load_json(
    path: Path,
    label: str,
    findings: list[dict[str, Any]],
) -> Any:
    if not path.exists() or not path.is_file():
        return None
    try:
        return read_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        _add_finding(
            findings,
            "error",
            f"invalid_{label}",
            f"Cannot read JSON {path.name}: {exc}",
            str(path),
        )
        return None


def _load_csv(
    path: Path,
    label: str,
    findings: list[dict[str, Any]],
) -> list[dict[str, str]] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except csv.Error as exc:
        _add_finding(
            findings,
            "error",
            f"invalid_{label}",
            f"Cannot read CSV {path.name}: {exc}",
            str(path),
        )
        return None


def _load_text(
    path: Path,
    label: str,
    findings: list[dict[str, Any]],
) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        _add_finding(
            findings,
            "error",
            f"invalid_{label}",
            f"Cannot read text artifact {path.name}: {exc}",
            str(path),
        )
        return None


def _require_mapping_keys(
    data: dict[str, Any],
    keys: tuple[str, ...],
    label: str,
    artifact: str,
    findings: list[dict[str, Any]],
) -> None:
    missing = [key for key in keys if data.get(key) in (None, "", [], {})]
    if missing:
        _add_finding(
            findings,
            "error",
            f"{label}_missing_required_fields",
            f"{label} is missing required fields: {', '.join(missing)}.",
            artifact,
        )


def _add_finding(
    findings: list[dict[str, Any]],
    severity: str,
    code: str,
    message: str,
    artifact: str | None,
) -> None:
    findings.append(
        {
            "severity": severity,
            "code": code,
            "message": message,
            "artifact": artifact,
        }
    )


def _build_report(
    root: Path,
    run_id: str | None,
    run_dir: Path | None,
    findings: list[dict[str, Any]],
    artifacts: dict[str, str],
) -> dict[str, Any]:
    errors = sum(1 for finding in findings if finding["severity"] == "error")
    warnings = sum(1 for finding in findings if finding["severity"] == "warning")
    infos = sum(1 for finding in findings if finding["severity"] == "info")
    return {
        "project_dir": str(root),
        "run_id": run_id,
        "run_dir": str(run_dir) if run_dir is not None else None,
        "valid": errors == 0,
        "status": "passed" if errors == 0 else "failed",
        "summary": {
            "errors": errors,
            "warnings": warnings,
            "infos": infos,
            "checked_artifacts": len(artifacts),
        },
        "findings": findings,
        "artifacts": artifacts,
    }


def _coerce_path(root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except (OSError, ValueError):
        return str(path)


def _has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _has_nonempty_text_or_list(value: Any) -> bool:
    if _has_text(value):
        return True
    if isinstance(value, list):
        return any(_has_text(item) or isinstance(item, dict) and bool(item) for item in value)
    return False


def _has_any_text(data: dict[str, Any], keys: tuple[str, ...]) -> bool:
    for key in keys:
        value = data.get(key)
        if _has_text(value):
            return True
        if isinstance(value, dict) and any(_has_text(item) for item in value.values()):
            return True
        if isinstance(value, list) and any(_has_text(item) for item in value):
            return True
    return False


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_float(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _to_float(row.get(key))
        if value is not None:
            return value
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m design_scientist.framework_validation")
    parser.add_argument("project_dir")
    parser.add_argument("--run-id")
    parser.add_argument("--json", action="store_true", help="Print the full validation report as JSON")
    args = parser.parse_args(argv)

    report = review_framework_run(args.project_dir, run_id=args.run_id)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=False))
    else:
        summary = report["summary"]
        print(
            f"Framework validation {report['status']}: "
            f"{summary['errors']} errors, {summary['warnings']} warnings"
        )
        for finding in report["findings"]:
            artifact = f" [{finding['artifact']}]" if finding.get("artifact") else ""
            print(f"{finding['severity'].upper()} {finding['code']}: {finding['message']}{artifact}")
    return 0 if report["valid"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ALGORITHM_REQUIRED_SECTIONS",
    "CRITICAL_FRAMEWORK_ARTIFACTS",
    "CRITICAL_RUN_ARTIFACTS",
    "METHOD_REPORT_REQUIRED_SECTIONS",
    "SELECTED_NODE_ARTIFACTS",
    "resolve_framework_run",
    "review_framework_run",
]
