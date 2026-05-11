"""V3 scientist manager for literature-grounded mechanism tree search."""

from __future__ import annotations

import csv
import shutil
import time
from pathlib import Path
from textwrap import dedent
from typing import Any

from design_scientist.artifacts import (
    V3_ABLATION_PLAN,
    V3_LITERATURE_CORPUS,
    V3_LITERATURE_READING_TRACE,
    V3_MECHANISM_CARDS,
    V3_MECHANISM_GAP_MATRIX,
    V3_MECHANISM_IMPLEMENTATION,
    V3_MECHANISM_LIBRARY,
    V3_MECHANISM_METRICS,
    V3_MECHANISM_NODE_ARTIFACTS,
    V3_MECHANISM_PROPOSAL,
    V3_MECHANISM_SPEC,
    V3_MECHANISM_VALIDATION_REPORT,
    V3_STRESS_TEST_PLAN,
)
from design_scientist.backends.base import WorkspaceAgentBackend
from design_scientist.io import ensure_dir, read_json, write_json
from design_scientist.mechanism_nodes import (
    MechanismNodeExecutionResult,
    execute_mechanism_node,
)
from design_scientist.mechanism_replay import (
    BASELINE_MECHANISM_NAMES,
    run_mechanism_benchmark,
)
from design_scientist.schemas import WorkspaceAgentTask


SCIENTIST_V3_STAGES = (
    "literature_retrieval",
    "fulltext_reading",
    "mechanism_extraction",
    "mechanism_ideation",
    "mechanism_implementation",
    "debug_repair",
    "ablation_stress",
    "selection_report",
)

NODE_BASELINE_ORDER = (
    "mechanism_aware",
    "fixed_mix",
    "random_feasible",
    "pure_lattice_repair",
    "greedy_utility",
    "pure_uncertainty",
    "top_observed",
)
NONSELECTABLE_BASELINES = set(BASELINE_MECHANISM_NAMES) | {"mechanism_aware"}

WORKSPACE_AGENT_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "mechanism_name": {"type": "string"},
        "files_written": {"type": "array", "items": {"type": "string"}},
        "contract_notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "mechanism_name", "files_written", "contract_notes"],
    "additionalProperties": False,
}


def run_scientist_v3(
    project_dir: str | Path,
    max_papers: int = 60,
    nodes: int = 3,
    rounds: int = 3,
    use_codex: bool = False,
    offline_fixtures: bool = False,
    backend: WorkspaceAgentBackend | None = None,
    strict_literature: bool = True,
) -> dict[str, Any]:
    """Run the V3 literature-to-mechanism scientist loop."""

    if max_papers <= 0:
        raise ValueError("max_papers must be positive")
    if nodes <= 0:
        raise ValueError("nodes must be positive")
    if rounds <= 0:
        raise ValueError("rounds must be positive")

    root = Path(project_dir).expanduser().resolve()
    run_id = _new_run_id(root)
    run_dir = ensure_dir(root / "runs" / run_id)
    node_root = ensure_dir(run_dir / "nodes")
    journal_path = run_dir / "scientist_journal.json"
    stage_progress_path = run_dir / "stage_progress.json"
    route_tree_path = run_dir / "route_tree.json"

    stage_states = _initial_stage_states()
    literature_snapshot = _run_literature_chain(
        root,
        max_papers=max_papers,
        offline_fixtures=offline_fixtures,
        strict=strict_literature,
        stage_states=stage_states,
    )

    node_records: list[dict[str, Any]] = []
    valid_lifecycles: list[dict[str, Any]] = []
    node_by_mechanism: dict[str, dict[str, Any]] = {}
    agent_backend = _resolve_backend(use_codex, backend)

    planned_nodes = _planned_nodes(nodes)
    stage_states["mechanism_ideation"]["status"] = "completed"
    stage_states["mechanism_ideation"]["artifacts"] = [str(node_root)]

    for index, baseline_family in enumerate(planned_nodes, start=1):
        node_id = f"node_{index:02d}_{baseline_family}"
        workspace = _fresh_workspace(node_root / node_id)
        agent_result: dict[str, Any] | None = None
        pre_execution_reasons: list[str] = []

        if use_codex:
            try:
                agent_result = _run_codex_mechanism_node(
                    backend=agent_backend,
                    workspace=workspace,
                    node_id=node_id,
                    baseline_family=baseline_family,
                    literature_snapshot=literature_snapshot,
                )
                pre_execution_reasons = _agent_rejection_reasons(agent_result)
            except Exception as exc:  # pragma: no cover - defensive backend isolation path.
                pre_execution_reasons = [f"workspace agent failed: {type(exc).__name__}: {exc}"]
                agent_result = {
                    "summary": "workspace agent failed before node execution",
                    "returncode": None,
                    "structured": None,
                    "files_touched": [],
                    "commands_run": [],
                    "error": pre_execution_reasons[0],
                }
        else:
            _write_local_mechanism_node(
                workspace,
                node_id=node_id,
                baseline_family=baseline_family,
                literature_snapshot=literature_snapshot,
            )

        if pre_execution_reasons:
            execution = _skipped_execution(workspace, pre_execution_reasons)
        else:
            execution = execute_mechanism_node(workspace, guard_roots=[root])

        reasons = _failure_reasons(execution)
        status = "completed" if execution.valid and not reasons else "failed"
        record = _node_record(
            node_id=node_id,
            baseline_family=baseline_family,
            workspace=workspace,
            status=status,
            execution=execution,
            reasons=reasons,
            agent_result=agent_result,
        )
        node_records.append(record)
        if status == "completed":
            lifecycle = _lifecycle_from_execution(record, execution)
            valid_lifecycles.append(lifecycle)
            node_by_mechanism[lifecycle["name"]] = record

    stage_states["mechanism_implementation"]["status"] = (
        "completed" if node_records else "failed"
    )
    stage_states["mechanism_implementation"]["artifacts"] = [
        record["workspace"] for record in node_records
    ]

    mechanisms: list[Any] = [*valid_lifecycles, "mechanism_aware"]
    benchmark = run_mechanism_benchmark(
        root,
        run_id=run_id,
        mechanisms=mechanisms,
        rounds=rounds,
    )
    summary_rows = _read_csv_rows(benchmark["mechanism_benchmark_summary_path"])
    benchmark["ranking"] = summary_rows
    _attach_benchmark_metrics(node_records, node_by_mechanism, summary_rows)

    failed_nodes = [
        {
            "node_id": record["node_id"],
            "reasons": list(record.get("failure_reasons", [])),
            "failure_kinds": list(record.get("contract", {}).get("failure_kinds", [])),
        }
        for record in node_records
        if record.get("status") != "completed"
    ]
    stage_states["debug_repair"]["status"] = "completed"
    stage_states["debug_repair"]["artifacts"] = [
        record["workspace"] for record in node_records if record.get("status") != "completed"
    ]

    selected_node = _select_node(summary_rows, node_by_mechanism)
    selected_node_id = selected_node["node_id"] if selected_node else None
    for record in node_records:
        record["selected"] = record["node_id"] == selected_node_id

    stage_states["ablation_stress"]["status"] = "completed"
    stage_states["ablation_stress"]["artifacts"] = [
        benchmark.get("mechanism_benchmark_results_path"),
        benchmark.get("mechanism_benchmark_summary_path"),
        benchmark.get("mechanism_ablation_results_path"),
    ]
    stage_states["selection_report"]["status"] = (
        "completed" if selected_node is not None else "failed"
    )
    stage_states["selection_report"]["artifacts"] = [
        str(journal_path),
        str(route_tree_path),
    ]

    memory_append_errors = _append_failure_memory(
        root=root,
        run_id=run_id,
        failed_nodes=failed_nodes,
    )

    stage_progress = _stage_progress_payload(
        run_id=run_id,
        run_dir=run_dir,
        stage_states=stage_states,
        node_records=node_records,
        selected_node_id=selected_node_id,
    )
    route_tree = _route_tree_payload(
        run_id=run_id,
        run_dir=run_dir,
        node_records=node_records,
        selected_node_id=selected_node_id,
    )

    journal = {
        "run_id": run_id,
        "version": "v3",
        "project_dir": str(root),
        "stage_sequence": list(SCIENTIST_V3_STAGES),
        "literature_snapshot": literature_snapshot,
        "mechanism_library_path": str(root / V3_MECHANISM_LIBRARY),
        "nodes": node_records,
        "benchmark": benchmark,
        "benchmark_paths": {
            "results": benchmark.get("mechanism_benchmark_results_path"),
            "summary": benchmark.get("mechanism_benchmark_summary_path"),
            "ablation": benchmark.get("mechanism_ablation_results_path"),
        },
        "selected_node_id": selected_node_id,
        "selected_node": selected_node,
        "failed_nodes": failed_nodes,
        "research_os": {
            "failure_memory_path": str(root / "framework" / "failure_memory.jsonl"),
            "append_errors": memory_append_errors,
        },
        "stage_progress_path": str(stage_progress_path),
        "route_tree_path": str(route_tree_path),
    }
    write_json(stage_progress_path, stage_progress)
    write_json(route_tree_path, route_tree)
    write_json(journal_path, journal)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "journal_path": str(journal_path),
        "stage_progress_path": str(stage_progress_path),
        "route_tree_path": str(route_tree_path),
        "selected_node": selected_node,
        "selected_node_id": selected_node_id,
        "nodes": node_records,
        "benchmark": benchmark,
        "literature_snapshot": literature_snapshot,
        "failed_nodes": failed_nodes,
    }


def _run_literature_chain(
    root: Path,
    *,
    max_papers: int,
    offline_fixtures: bool,
    strict: bool,
    stage_states: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "paper_cards_path": str(root / "framework" / "paper_cards.json"),
        "corpus_path": str(root / V3_LITERATURE_CORPUS),
        "reading_trace_path": str(root / V3_LITERATURE_READING_TRACE),
        "mechanism_cards_path": str(root / V3_MECHANISM_CARDS),
        "mechanism_library_path": str(root / V3_MECHANISM_LIBRARY),
        "mechanism_gap_matrix_path": str(root / V3_MECHANISM_GAP_MATRIX),
        "offline_fixtures": bool(offline_fixtures),
        "max_papers": max_papers,
        "stages": {},
    }

    try:
        from design_scientist.literature_pipeline import run_literature_search

        paper_cards_path = run_literature_search(
            root,
            max_papers=max_papers,
            offline_fixtures=offline_fixtures,
        )
        cards = _read_json_list(Path(paper_cards_path))
        snapshot["paper_cards_path"] = str(Path(paper_cards_path))
        snapshot["paper_count"] = len(cards)
        snapshot["papers"] = cards[:10]
        _mark_stage(
            stage_states,
            "literature_retrieval",
            status="completed",
            artifacts=[str(paper_cards_path)],
        )
        snapshot["stages"]["literature_retrieval"] = {"status": "completed"}
    except Exception as exc:
        _handle_literature_error(
            root,
            strict=strict,
            stage="literature_retrieval",
            exc=exc,
            snapshot=snapshot,
            stage_states=stage_states,
        )

    try:
        from design_scientist.literature_fulltext import build_literature_corpus

        corpus_path = build_literature_corpus(root, offline_fixtures=offline_fixtures)
        snapshot["corpus_path"] = str(corpus_path)
        _mark_stage(
            stage_states,
            "fulltext_reading",
            status="completed",
            artifacts=[str(corpus_path), str(root / V3_LITERATURE_READING_TRACE)],
        )
        snapshot["stages"]["fulltext_reading"] = {"status": "completed"}
    except Exception as exc:
        _handle_literature_error(
            root,
            strict=strict,
            stage="fulltext_reading",
            exc=exc,
            snapshot=snapshot,
            stage_states=stage_states,
        )

    try:
        from design_scientist.mechanism_extraction import extract_mechanisms

        extraction = extract_mechanisms(root)
        status = str(extraction.get("status", "unknown")) if isinstance(extraction, dict) else "unknown"
        if not status.startswith("ok"):
            raise RuntimeError(str(extraction.get("reason") or status))
        snapshot["mechanism_extraction"] = extraction
        _mark_stage(
            stage_states,
            "mechanism_extraction",
            status="completed",
            artifacts=[
                str(root / V3_MECHANISM_CARDS),
                str(root / V3_MECHANISM_LIBRARY),
                str(root / V3_MECHANISM_GAP_MATRIX),
            ],
        )
        snapshot["stages"]["mechanism_extraction"] = {"status": "completed"}
    except Exception as exc:
        _handle_literature_error(
            root,
            strict=strict,
            stage="mechanism_extraction",
            exc=exc,
            snapshot=snapshot,
            stage_states=stage_states,
        )

    if "paper_count" not in snapshot:
        cards = _read_json_list(root / "framework" / "paper_cards.json")
        snapshot["paper_count"] = len(cards)
        snapshot["papers"] = cards[:10]
    return snapshot


def _handle_literature_error(
    root: Path,
    *,
    strict: bool,
    stage: str,
    exc: Exception,
    snapshot: dict[str, Any],
    stage_states: dict[str, dict[str, Any]],
) -> None:
    message = f"{type(exc).__name__}: {exc}"
    snapshot["stages"][stage] = {"status": "failed", "error": message}
    _mark_stage(stage_states, stage, status="failed", artifacts=[], errors=[message])
    if strict:
        raise RuntimeError(f"{stage} failed: {message}") from exc
    if stage == "literature_retrieval":
        framework = ensure_dir(root / "framework")
        paper_cards_path = framework / "paper_cards.json"
        if not paper_cards_path.exists():
            write_json(paper_cards_path, [])


def _write_local_mechanism_node(
    workspace: Path,
    *,
    node_id: str,
    baseline_family: str,
    literature_snapshot: dict[str, Any],
) -> None:
    top_papers = _top_papers(literature_snapshot)
    mechanism_ids = _mechanism_ids(literature_snapshot)
    source = f"""
        from __future__ import annotations

        import json
        from pathlib import Path

        from design_scientist import policies as policy_api


        NODE_ID = {node_id!r}
        SOURCE_BASELINE_FAMILY = {baseline_family!r}
        TOP_PAPERS = {top_papers!r}
        MECHANISM_IDS = {mechanism_ids!r}


        def _write_json(path: Path, payload: dict) -> None:
            path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\\n", encoding="utf-8")


        def _baseline_policy_name() -> str:
            if SOURCE_BASELINE_FAMILY in policy_api.POLICY_REGISTRY:
                if SOURCE_BASELINE_FAMILY == "mechanism_aware":
                    return "random_feasible"
                return SOURCE_BASELINE_FAMILY
            return "random_feasible"


        def fit_state(context):
            state = dict(context)
            state["node_id"] = NODE_ID
            state["source_baseline_family"] = SOURCE_BASELINE_FAMILY
            return state


        def generate_candidates(state):
            return list(state["candidate_records"])


        def score_candidates(state, candidates):
            scored = []
            for candidate in candidates:
                item = dict(candidate)
                modules = item.get("modules") or ()
                item["v3_mechanism_score"] = len(modules) + (1 if item.get("background") == item.get("target_background") else 0)
                scored.append(item)
            return scored


        def select_panel(state, candidates, budget, rng):
            if state.get("policy_ablation") == "key_component_removed":
                baseline_policy = policy_api.get_policy(_baseline_policy_name())
                return baseline_policy(
                    state["observed_records"],
                    candidates,
                    budget,
                    state["round_index"],
                    rng,
                )
            return policy_api.mechanism_aware(
                state["observed_records"],
                candidates,
                budget,
                state["round_index"],
                rng,
            )


        def plan_ablations(state):
            del state
            return [
                {{"name": "none"}},
                {{
                    "name": "key_component_removed",
                    "policy_ablation": "key_component_removed",
                    "removed_component": "mechanism_guardrail_delta",
                }},
            ]


        def run(workspace):
            root = Path(workspace)
            _write_json(
                root / "mechanism_spec.json",
                {{
                    "mechanism_id": NODE_ID,
                    "name": "V3 mechanism node " + NODE_ID,
                    "version": "v3",
                    "source_baseline_family": SOURCE_BASELINE_FAMILY,
                    "mechanism_cards": MECHANISM_IDS,
                    "components": [
                        {{"component_id": "state_passthrough", "component_type": "state_model"}},
                        {{"component_id": "mechanism_guardrail_delta", "component_type": "selection_policy"}},
                        {{"component_id": "key_component_removed", "component_type": "ablation"}},
                    ],
                    "claims": [
                        "Lifecycle mechanism with an explicit ablation should outperform required replay baselines.",
                        "Stress-test mapping should expose transfer, sparse-data, interaction, and noisy-assay failure modes.",
                    ],
                    "literature_basis": TOP_PAPERS,
                    "stress_test_requirements": [
                        "confounded_transfer",
                        "sparse_early_round",
                        "epistatic",
                        "noisy_endpoint",
                    ],
                    "architecture_delta": (
                        "Runs an explicit lifecycle and tests a key-component ablation "
                        "against the baseline family instead of exposing a select_batch-only wrapper."
                    ),
                }},
            )
            _write_json(
                root / "proposal.json",
                {{
                    "mechanism_id": NODE_ID,
                    "title": "Lifecycle mechanism delta over " + SOURCE_BASELINE_FAMILY,
                    "hypothesis": (
                        "A mechanism-aware lifecycle with an explicit ablation should beat "
                        "simple replay baselines while exposing its failure mode."
                    ),
                    "literature_basis": TOP_PAPERS,
                    "literature_gap_ids": MECHANISM_IDS or ["mechanism_gap_matrix"],
                    "reused_components": [SOURCE_BASELINE_FAMILY],
                    "architecture_delta": (
                        "Adds fit/generate/score/select/ablate lifecycle stages plus "
                        "a stress-testable key-component removal path."
                    ),
                    "failure_modes": [
                        "The policy can collapse to its source baseline when the mechanism delta is removed.",
                        "Mechanism-card support may be weak when full text is metadata-only.",
                    ],
                }},
            )
            _write_json(
                root / "ablation_plan.json",
                {{
                    "ablations": [
                        {{
                            "name": "key_component_removed",
                            "policy_ablation": "key_component_removed",
                            "removed_component": "mechanism_guardrail_delta",
                        }}
                    ],
                }},
            )
            _write_json(
                root / "stress_test_plan.json",
                {{
                    "stress_tests": [
                        {{"world_id": "confounded_transfer"}},
                        {{"world_id": "sparse_early_round"}},
                        {{"world_id": "epistatic"}},
                        {{"world_id": "noisy_endpoint"}},
                    ],
                    "claims": [
                        "transfer",
                        "sparse early-round interaction",
                        "noisy assay guardrail",
                    ],
                }},
            )
            _write_json(
                root / "mechanism_metrics.json",
                {{
                    "node_id": NODE_ID,
                    "status": "generated",
                    "source_baseline_family": SOURCE_BASELINE_FAMILY,
                }},
            )
            _write_json(
                root / "validation_report.json",
                {{
                    "node_id": NODE_ID,
                    "valid": True,
                    "findings": [],
                }},
            )
    """
    (workspace / V3_MECHANISM_IMPLEMENTATION).write_text(
        dedent(source).lstrip(),
        encoding="utf-8",
    )


def _run_codex_mechanism_node(
    *,
    backend: WorkspaceAgentBackend | None,
    workspace: Path,
    node_id: str,
    baseline_family: str,
    literature_snapshot: dict[str, Any],
) -> dict[str, Any]:
    if backend is None:
        raise RuntimeError("A WorkspaceAgentBackend is required when use_codex=True")
    task = WorkspaceAgentTask(
        objective=_codex_objective(
            node_id=node_id,
            baseline_family=baseline_family,
            literature_snapshot=literature_snapshot,
        ),
        workspace=str(workspace),
        mode="patch",
        allowed_paths=[str(workspace.resolve())],
        write_policy=(
            "Write only V3 mechanism-node artifacts inside this node workspace: "
            + ", ".join(V3_MECHANISM_NODE_ARTIFACTS)
            + ". Do not read or write outside the workspace."
        ),
        commands_allowed=["python"],
        output_schema=WORKSPACE_AGENT_OUTPUT_SCHEMA,
    )
    result = backend.run_task(task)
    return {
        "summary": result.summary,
        "structured": result.structured,
        "returncode": result.returncode,
        "files_touched": list(result.files_touched),
        "commands_run": list(result.commands_run),
        "transcript_id": result.transcript_id,
    }


def _codex_objective(
    *,
    node_id: str,
    baseline_family: str,
    literature_snapshot: dict[str, Any],
) -> str:
    paper_context = "; ".join(_top_papers(literature_snapshot)[:5]) or "no paper cards available"
    mechanism_context = "; ".join(_mechanism_ids(literature_snapshot)[:8]) or "no mechanism cards available"
    return (
        f"Create V3 mechanism node {node_id} inside the current workspace. "
        f"Use {baseline_family} as the comparison baseline family, but expose a real "
        "fit_state/generate_candidates/score_candidates/select_panel/plan_ablations lifecycle "
        "in mechanism.py. The run(workspace) function must write mechanism_spec.json, "
        "proposal.json, ablation_plan.json, stress_test_plan.json, mechanism_metrics.json, "
        "and validation_report.json with validation_report.valid true. "
        "The key ablation should remove a named mechanism component. "
        f"Paper context: {paper_context}. Mechanism context: {mechanism_context}. "
        "Return JSON matching the output schema."
    )


def _agent_rejection_reasons(agent_result: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if agent_result.get("returncode") not in (0, None):
        reasons.append(f"workspace agent failed with returncode {agent_result.get('returncode')}")
    structured = agent_result.get("structured")
    if structured is not None and not isinstance(structured, dict):
        reasons.append("workspace agent structured output must be a JSON object")
    return reasons


def _resolve_backend(
    use_codex: bool,
    backend: WorkspaceAgentBackend | None,
) -> WorkspaceAgentBackend | None:
    if not use_codex:
        return None
    if backend is not None:
        return backend
    from design_scientist.backends.codex_cli import CodexCliBackend

    return CodexCliBackend()


def _skipped_execution(
    workspace: Path,
    reasons: list[str],
) -> MechanismNodeExecutionResult:
    return MechanismNodeExecutionResult(
        workspace=workspace,
        valid=False,
        executed=False,
        failure_kinds=["workspace_agent"],
        missing_artifacts=list(V3_MECHANISM_NODE_ARTIFACTS),
        errors=list(reasons),
        snapshot_roots=[workspace],
    )


def _node_record(
    *,
    node_id: str,
    baseline_family: str,
    workspace: Path,
    status: str,
    execution: MechanismNodeExecutionResult,
    reasons: list[str],
    agent_result: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "baseline_family": baseline_family,
        "mechanism": node_id,
        "workspace": str(workspace),
        "status": status,
        "failure_reasons": reasons,
        "artifacts": _artifact_paths(workspace),
        "contract": {
            "valid": execution.valid,
            "executed": execution.executed,
            "failure_kinds": list(execution.failure_kinds),
            "missing_artifacts": list(execution.missing_artifacts),
            "malformed_json": dict(execution.malformed_json),
            "out_of_bounds_writes": list(execution.out_of_bounds_writes),
            "escaping_symlinks": list(execution.escaping_symlinks),
            "errors": list(execution.errors),
            "exception": execution.exception,
            "subprocess_returncode": execution.subprocess_returncode,
            "timed_out": execution.timed_out,
        },
        "agent": agent_result,
    }


def _lifecycle_from_execution(
    record: dict[str, Any],
    execution: MechanismNodeExecutionResult,
) -> dict[str, Any]:
    return {
        "name": record["mechanism"],
        "fit_state": execution.callables["fit_state"],
        "generate_candidates": execution.callables["generate_candidates"],
        "score_candidates": execution.callables["score_candidates"],
        "select_panel": execution.callables["select_panel"],
        "plan_ablations": execution.callables["plan_ablations"],
        "claims": (
            "transfer",
            "sparse early-round interaction",
            "noisy assay guardrail",
        ),
        "architecture_clone": False,
        "claim_guardrail": True,
        "confounding_correction": True,
    }


def _failure_reasons(execution: MechanismNodeExecutionResult) -> list[str]:
    reasons: list[str] = []
    reasons.extend(execution.errors)
    reasons.extend(f"missing artifact: {artifact}" for artifact in execution.missing_artifacts)
    reasons.extend(
        f"malformed artifact {artifact}: {message}"
        for artifact, message in execution.malformed_json.items()
    )
    reasons.extend(f"out-of-bounds write: {path}" for path in execution.out_of_bounds_writes)
    if execution.exception:
        reasons.append(execution.exception.strip().splitlines()[-1])
    if not execution.valid and not reasons:
        reasons.extend(execution.failure_kinds or ["mechanism node failed validation"])
    return _unique(reasons)


def _artifact_paths(workspace: Path) -> dict[str, str]:
    names = {
        "mechanism_spec": V3_MECHANISM_SPEC,
        "mechanism": V3_MECHANISM_IMPLEMENTATION,
        "proposal": V3_MECHANISM_PROPOSAL,
        "ablation_plan": V3_ABLATION_PLAN,
        "stress_test_plan": V3_STRESS_TEST_PLAN,
        "mechanism_metrics": V3_MECHANISM_METRICS,
        "validation_report": V3_MECHANISM_VALIDATION_REPORT,
    }
    return {
        key: str(workspace / filename)
        for key, filename in names.items()
        if (workspace / filename).exists()
    }


def _attach_benchmark_metrics(
    node_records: list[dict[str, Any]],
    node_by_mechanism: dict[str, dict[str, Any]],
    summary_rows: list[dict[str, str]],
) -> None:
    for row in summary_rows:
        record = node_by_mechanism.get(row.get("mechanism", ""))
        if record is None:
            continue
        record["benchmark_metrics"] = dict(row)
        record["ranking_score"] = _ranking_score(row)
        record["selected_eligible"] = row.get("selected_eligible") == "true"
        metrics_path = Path(record["artifacts"].get("mechanism_metrics", ""))
        if metrics_path:
            try:
                payload = read_json(metrics_path)
                if not isinstance(payload, dict):
                    payload = {}
            except Exception:
                payload = {}
            payload["mechanism_benchmark_summary"] = dict(row)
            payload["ranking_score"] = record["ranking_score"]
            payload["selected_eligible"] = record["selected_eligible"]
            write_json(metrics_path, payload)
    for record in node_records:
        if record.get("status") == "completed" and "benchmark_metrics" not in record:
            record["status"] = "failed"
            record.setdefault("failure_reasons", []).append("mechanism benchmark produced no summary row")


def _select_node(
    summary_rows: list[dict[str, str]],
    node_by_mechanism: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    for row in sorted(summary_rows, key=_summary_rank):
        mechanism = row.get("mechanism", "")
        if mechanism in NONSELECTABLE_BASELINES:
            continue
        if row.get("selected_eligible") != "true":
            continue
        record = node_by_mechanism.get(mechanism)
        if record is None:
            continue
        return {
            "node_id": record["node_id"],
            "mechanism": mechanism,
            "workspace": record["workspace"],
            "artifacts": record.get("artifacts", {}),
            "benchmark_metrics": dict(row),
            "ranking_score": _ranking_score(row),
            "selected_eligible": True,
        }
    return None


def _summary_rank(row: dict[str, str]) -> int:
    try:
        return int(row.get("rank", "999999"))
    except (TypeError, ValueError):
        return 999999


def _append_failure_memory(
    *,
    root: Path,
    run_id: str,
    failed_nodes: list[dict[str, Any]],
) -> list[str]:
    if not failed_nodes:
        return []
    try:
        from design_scientist.research_os import append_failure
    except Exception as exc:  # pragma: no cover - defensive import path.
        return [f"append_failure import failed: {type(exc).__name__}: {exc}"]

    errors: list[str] = []
    for failed in failed_nodes:
        try:
            append_failure(
                root,
                {
                    "summary": f"V3 mechanism node {failed['node_id']} failed during run {run_id}.",
                    "component": "mechanism_node",
                    "run_id": run_id,
                    "node_id": failed["node_id"],
                    "stage": "debug_repair",
                    "failure_kinds": failed.get("failure_kinds", []),
                    "reasons": failed.get("reasons", []),
                },
            )
        except Exception as exc:  # pragma: no cover - append failure defensive path.
            errors.append(f"append_failure failed for {failed['node_id']}: {type(exc).__name__}: {exc}")
    return errors


def _stage_progress_payload(
    *,
    run_id: str,
    run_dir: Path,
    stage_states: dict[str, dict[str, Any]],
    node_records: list[dict[str, Any]],
    selected_node_id: str | None,
) -> dict[str, Any]:
    stages = []
    for stage in SCIENTIST_V3_STAGES:
        state = stage_states[stage]
        stages.append(
            {
                "stage": stage,
                "status": state["status"],
                "artifacts": [str(path) for path in state.get("artifacts", []) if path],
                "errors": list(state.get("errors", [])),
            }
        )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "version": "v3",
        "stage_sequence": list(SCIENTIST_V3_STAGES),
        "stages": stages,
        "node_count": len(node_records),
        "failed_node_count": sum(1 for record in node_records if record.get("status") != "completed"),
        "selected_node_id": selected_node_id,
    }


def _route_tree_payload(
    *,
    run_id: str,
    run_dir: Path,
    node_records: list[dict[str, Any]],
    selected_node_id: str | None,
) -> dict[str, Any]:
    nodes = []
    edges = []
    for record in node_records:
        nodes.append(
            {
                "node_id": record["node_id"],
                "parent_id": "root",
                "stage": "mechanism_implementation",
                "status": record["status"],
                "workspace": record["workspace"],
                "selected": record["node_id"] == selected_node_id,
                "baseline_family": record["baseline_family"],
                "failure_reasons": record.get("failure_reasons", []),
                "artifacts": record.get("artifacts", {}),
                "ranking_score": record.get("ranking_score"),
            }
        )
        edges.append({"source": "root", "target": record["node_id"]})
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "version": "v3",
        "stage_sequence": list(SCIENTIST_V3_STAGES),
        "root": {
            "node_id": "root",
            "stage": "mechanism_ideation",
            "status": "completed",
        },
        "nodes": nodes,
        "edges": edges,
        "selected_node_id": selected_node_id,
    }


def _initial_stage_states() -> dict[str, dict[str, Any]]:
    return {
        stage: {"status": "pending", "artifacts": [], "errors": []}
        for stage in SCIENTIST_V3_STAGES
    }


def _mark_stage(
    stage_states: dict[str, dict[str, Any]],
    stage: str,
    *,
    status: str,
    artifacts: list[str],
    errors: list[str] | None = None,
) -> None:
    stage_states[stage]["status"] = status
    stage_states[stage]["artifacts"] = list(artifacts)
    if errors:
        stage_states[stage]["errors"] = list(errors)


def _fresh_workspace(workspace: Path) -> Path:
    if workspace.exists() or workspace.is_symlink():
        if workspace.is_symlink() or workspace.is_file():
            workspace.unlink()
        else:
            shutil.rmtree(workspace)
    return ensure_dir(workspace)


def _planned_nodes(nodes: int) -> list[str]:
    baselines = list(NODE_BASELINE_ORDER)
    return [baselines[index % len(baselines)] for index in range(nodes)]


def _new_run_id(root: Path) -> str:
    base = time.strftime("scientist_v3_%Y%m%d_%H%M%S")
    run_id = base
    index = 2
    while (root / "runs" / run_id).exists():
        run_id = f"{base}_{index}"
        index += 1
    return run_id


def _top_papers(literature_snapshot: dict[str, Any]) -> list[str]:
    papers = literature_snapshot.get("papers")
    if not isinstance(papers, list):
        return []
    out: list[str] = []
    for paper in papers[:6]:
        if not isinstance(paper, dict):
            continue
        paper_id = str(paper.get("paper_id") or paper.get("source_id") or "paper")
        title = str(paper.get("title") or paper.get("citation") or "untitled")
        out.append(f"{paper_id}: {title}")
    return out


def _mechanism_ids(literature_snapshot: dict[str, Any]) -> list[str]:
    extraction = literature_snapshot.get("mechanism_extraction")
    if not isinstance(extraction, dict):
        return []
    cards = extraction.get("mechanism_cards")
    if not isinstance(cards, list):
        cards = _read_json_list(Path(literature_snapshot.get("mechanism_cards_path", "")))
    ids: list[str] = []
    for card in cards:
        if isinstance(card, dict) and card.get("mechanism_id"):
            ids.append(str(card["mechanism_id"]))
    return _unique(ids)


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    try:
        data = read_json(path)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _ranking_score(row: dict[str, Any]) -> float:
    try:
        return float(row.get("mean_best_feasible_utility", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        out.append(value)
        seen.add(value)
    return out


__all__ = [
    "SCIENTIST_V3_STAGES",
    "run_scientist_v3",
]
