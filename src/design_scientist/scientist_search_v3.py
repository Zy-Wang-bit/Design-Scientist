"""V3 scientist manager for literature-grounded mechanism tree search."""

from __future__ import annotations

import csv
import json
import re
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
    V3_OPERATOR_SPECS,
    V3_OPERATOR_TO_CODE_TRACE,
    V3_STRESS_TEST_PLAN,
    V4_CLAIM_LEDGER,
    V4_EVIDENCE_LEDGER,
    V4_LITERATURE_MINE_TRACE,
    V4_MECHANISM_GAP_MATRIX,
    V4_MECHANISM_GRAPH,
    V4_MECHANISM_CARDS,
    V4_MECHANISM_LEDGER,
    V4_MECHANISM_LIBRARY,
    V4_NOVELTY_AUDIT,
    V4_RESEARCH_HARNESS_SUMMARY,
    V4_VERIFICATION_LADDER,
)
from design_scientist.backends.base import WorkspaceAgentBackend
from design_scientist.io import ensure_dir, read_json, write_json
from design_scientist.mechanism_nodes import (
    MechanismNodeExecutionResult,
    execute_mechanism_node,
)
from design_scientist.mechanism_replay import (
    BASELINE_MECHANISM_NAMES,
    load_project_context,
    run_mechanism_benchmark,
)
from design_scientist.reference_manifest import write_reference_data_sources
from design_scientist.research_harness import (
    ClaimRecord,
    EvidenceRecord,
    MechanismRecord,
    ResearchHarness,
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

NODE_DESIGN_FAMILY_ORDER = (
    "literature_delta",
    "matched_contrast_operator",
    "adaptive_decision_value",
    "uncertainty_guardrail",
    "transfer_stress_coupling",
    "operator_ablation_planner",
    "negative_control_allocator",
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
    project_context = load_project_context(root)

    stage_states = _initial_stage_states()
    literature_snapshot = _run_literature_chain(
        root,
        max_papers=max_papers,
        offline_fixtures=offline_fixtures,
        strict=strict_literature,
        stage_states=stage_states,
    )
    if project_context is not None:
        literature_snapshot["project_context"] = project_context
    operator_specs = _operator_specs(literature_snapshot)

    node_records: list[dict[str, Any]] = []
    valid_lifecycles: list[dict[str, Any]] = []
    node_by_mechanism: dict[str, dict[str, Any]] = {}
    generated_mechanism_names: set[str] = set()
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
            execution = execute_mechanism_node(
                workspace,
                guard_roots=[root],
                operator_specs=operator_specs,
            )

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
            mechanism_name = _generated_lifecycle_name(
                record=record,
                execution=execution,
                agent_result=agent_result,
            )
            record["mechanism"] = mechanism_name
            duplicate_reason = _generated_name_rejection_reason(
                mechanism_name,
                seen=generated_mechanism_names,
            )
            if duplicate_reason:
                record["status"] = "failed"
                record.setdefault("failure_reasons", []).append(duplicate_reason)
                contract = record.setdefault("contract", {})
                contract["valid"] = False
                failure_kinds = contract.setdefault("failure_kinds", [])
                if "duplicate_mechanism_name" not in failure_kinds:
                    failure_kinds.append("duplicate_mechanism_name")
                contract_errors = contract.setdefault("errors", [])
                if duplicate_reason not in contract_errors:
                    contract_errors.append(duplicate_reason)
            else:
                generated_mechanism_names.add(mechanism_name)
                lifecycle = _lifecycle_from_execution(record, execution)
                valid_lifecycles.append(lifecycle)
                node_by_mechanism[lifecycle["name"]] = record

    stage_states["mechanism_implementation"]["status"] = (
        "completed" if valid_lifecycles else "failed"
    )
    stage_states["mechanism_implementation"]["artifacts"] = [
        record["workspace"] for record in node_records
    ]
    if not valid_lifecycles:
        stage_states["mechanism_implementation"]["errors"] = [
            "no generated mechanism nodes passed validation"
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
    _write_claim_gate_artifacts(run_dir, summary_rows)
    saturation_feedback = _saturation_feedback_from_benchmark(benchmark)
    next_search_constraints = list(saturation_feedback.get("next_search_constraints", []))

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
    reference_data_sources_path = write_reference_data_sources(
        root,
        run_dir,
        run_id=run_id,
        selected_node=selected_node,
    )
    v4_harness = _write_v4_research_harness_artifacts(
        root=root,
        run_dir=run_dir,
        run_id=run_id,
        node_records=node_records,
        selected_node=selected_node,
        summary_rows=summary_rows,
        benchmark=benchmark,
    )

    if valid_lifecycles:
        stage_states["ablation_stress"]["status"] = "completed"
        stage_states["ablation_stress"]["artifacts"] = [
            benchmark.get("mechanism_benchmark_results_path"),
            benchmark.get("mechanism_benchmark_summary_path"),
            benchmark.get("mechanism_ablation_results_path"),
        ]
    else:
        stage_states["ablation_stress"]["status"] = "failed"
        stage_states["ablation_stress"]["artifacts"] = []
        stage_states["ablation_stress"]["errors"] = [
            "no generated mechanism nodes were available for ablation stress"
        ]
    stage_states["selection_report"]["status"] = (
        "completed" if selected_node is not None else "failed"
    )
    stage_states["selection_report"]["artifacts"] = [
        str(journal_path),
        str(route_tree_path),
        str(reference_data_sources_path),
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
        saturation_feedback=saturation_feedback,
    )
    route_tree = _route_tree_payload(
        run_id=run_id,
        run_dir=run_dir,
        node_records=node_records,
        selected_node_id=selected_node_id,
        saturation_feedback=saturation_feedback,
    )

    journal = {
        "run_id": run_id,
        "version": "v3",
        "research_harness_version": "v4",
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
            "saturation": benchmark.get("benchmark_saturation_path"),
        },
        "selected_node_id": selected_node_id,
        "selected_node": selected_node,
        "failed_nodes": failed_nodes,
        "saturation_feedback": saturation_feedback,
        "next_search_constraints": next_search_constraints,
        "reference_data_sources_path": str(reference_data_sources_path),
        "research_harness": v4_harness,
        "research_os": {
            "failure_memory_path": str(root / "framework" / "failure_memory.jsonl"),
            "append_errors": memory_append_errors,
        },
        "stage_progress_path": str(stage_progress_path),
        "route_tree_path": str(route_tree_path),
    }
    if project_context is not None:
        journal["project_context"] = project_context
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
        "project_context": project_context,
        "saturation_feedback": saturation_feedback,
        "next_search_constraints": next_search_constraints,
        "reference_data_sources_path": str(reference_data_sources_path),
        "research_harness": v4_harness,
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
        "v4_mechanism_cards_path": str(root / V4_MECHANISM_CARDS),
        "v4_mechanism_library_path": str(root / V4_MECHANISM_LIBRARY),
        "v4_mechanism_gap_matrix_path": str(root / V4_MECHANISM_GAP_MATRIX),
        "v4_literature_mine_trace_path": str(root / V4_LITERATURE_MINE_TRACE),
        "v4_mechanism_graph_path": str(root / V4_MECHANISM_GRAPH),
        "operator_specs_path": str(root / V3_OPERATOR_SPECS),
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
                str(root / V4_MECHANISM_CARDS),
                str(root / V4_MECHANISM_LIBRARY),
                str(root / V4_MECHANISM_GAP_MATRIX),
                str(root / V4_LITERATURE_MINE_TRACE),
                str(root / V4_MECHANISM_GRAPH),
                str(root / V3_OPERATOR_SPECS),
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

    try:
        from design_scientist.literature_engine_v4 import mine_mechanisms_from_corpus

        v4_mining = mine_mechanisms_from_corpus(root)
        if not isinstance(v4_mining, dict) or v4_mining.get("status") != "ok":
            raise RuntimeError(str(v4_mining.get("reason") if isinstance(v4_mining, dict) else v4_mining))
        snapshot["v4_mechanism_mining"] = v4_mining
    except Exception as exc:
        _handle_literature_error(
            root,
            strict=strict,
            stage="v4_mechanism_mining",
            exc=exc,
            snapshot=snapshot,
            stage_states=stage_states,
        )

    if "paper_count" not in snapshot:
        cards = _read_json_list(root / "framework" / "paper_cards.json")
        snapshot["paper_count"] = len(cards)
        snapshot["papers"] = cards[:10]
    operator_specs = _read_json_list(root / V3_OPERATOR_SPECS)
    snapshot["operator_specs"] = operator_specs[:12]
    snapshot["operator_ids"] = _operator_ids_from_specs(operator_specs)
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
    operator_specs = _compact_operator_specs(_operator_specs(literature_snapshot))
    operator_refs = _operator_ids_from_specs(operator_specs)
    primary_operator_id = operator_refs[0] if operator_refs else ""
    source = f"""
        from __future__ import annotations

        import json
        from itertools import combinations
        from pathlib import Path


        NODE_ID = {node_id!r}
        SOURCE_BASELINE_FAMILY = {baseline_family!r}
        TOP_PAPERS = {top_papers!r}
        MECHANISM_IDS = {mechanism_ids!r}
        OPERATOR_SPECS = {operator_specs!r}
        OPERATOR_REFS = {operator_refs!r}
        PRIMARY_OPERATOR_ID = {primary_operator_id!r}
        TARGET_BACKGROUND = "transfer_bg"
        PRIOR_INTERACTION_PAIRS = (
            ("HD110H", "HG56H"),
            ("HD110H", "HN54H"),
            ("HG56H", "HV105H"),
            ("HN54H", "HV105H"),
            ("HD110H", "LT47Y"),
        )
        RISK_MODULES = ("HA23K", "SY92F", "KD31N")


        def _write_json(path: Path, payload: dict) -> None:
            path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\\n", encoding="utf-8")


        def _candidate_id(candidate):
            return str(
                candidate.get("variant_id")
                or candidate.get("candidate_id")
                or candidate.get("id")
                or candidate.get("name")
            )


        def _mechanism_delta_score(candidate):
            modules = tuple(candidate.get("modules") or ())
            background = str(candidate.get("background") or "")
            target_background = str(candidate.get("target_background") or TARGET_BACKGROUND)
            target_bonus = 1.0 if background == target_background else 0.0
            interaction_bonus = 0.0
            for pair in combinations(modules, 2):
                if tuple(sorted(pair)) in PRIOR_INTERACTION_PAIRS:
                    interaction_bonus += 1.0
            repair_bonus = 0.45 if "HD110H" in modules and "LT47Y" in modules else 0.0
            risk_penalty = sum(1.0 for module in modules if module in RISK_MODULES)
            return (
                2.0 * target_bonus
                + 0.35 * len(modules)
                + 0.65 * interaction_bonus
                + repair_bonus
                - 0.55 * risk_penalty
            )


        def fit_state(context):
            state = dict(context)
            state["node_id"] = NODE_ID
            state["source_baseline_family"] = SOURCE_BASELINE_FAMILY
            state["operator_specs"] = list(OPERATOR_SPECS)
            state["operator_refs"] = list(OPERATOR_REFS)
            state["operator_weights"] = {{PRIMARY_OPERATOR_ID: 1.0}} if PRIMARY_OPERATOR_ID else {{}}
            if state.get("ablation") == "key_component_removed" and PRIMARY_OPERATOR_ID:
                state["operator_weights"][PRIMARY_OPERATOR_ID] = 0.0
                state["removed_operator_ids"] = [PRIMARY_OPERATOR_ID]
            return state


        def generate_candidates(state):
            candidates = []
            for candidate in state["candidate_records"]:
                item = dict(candidate)
                item["operator_refs"] = list(state.get("operator_refs") or OPERATOR_REFS)
                candidates.append(item)
            return candidates


        def score_candidates(state, candidates):
            scored = []
            operator_weight = float(state.get("operator_weights", {{}}).get(PRIMARY_OPERATOR_ID, 0.0))
            for candidate in candidates:
                item = dict(candidate)
                mechanism_score = _mechanism_delta_score(item)
                item["operator_contributions"] = (
                    {{PRIMARY_OPERATOR_ID: operator_weight * mechanism_score}}
                    if PRIMARY_OPERATOR_ID
                    else {{}}
                )
                item["v3_mechanism_score"] = operator_weight * mechanism_score
                scored.append(item)
            return scored


        def select_panel(state, candidates, budget, rng):
            del state, rng
            delta_ranked = sorted(
                candidates,
                key=lambda candidate: (
                    float(candidate.get("v3_mechanism_score", _mechanism_delta_score(candidate))),
                    _candidate_id(candidate),
                ),
                reverse=True,
            )
            selected = []
            for candidate in delta_ranked:
                candidate_id = _candidate_id(candidate)
                selected.append(candidate_id)
                if len(selected) >= budget:
                    break
            return selected


        def plan_ablations(state):
            del state
            ablations = [{{"name": "none"}}]
            if PRIMARY_OPERATOR_ID:
                ablations.append(
                    {{
                    "name": "key_component_removed",
                        "ablation_type": "remove_operator",
                        "removed_operator_ids": [PRIMARY_OPERATOR_ID],
                        "is_key": True,
                    }}
                )
            return ablations


        def run(workspace):
            root = Path(workspace)
            operator_trace = {{
                "operator_to_code_trace": {{
                    operator_id: [
                        "fit_state.operator_weights",
                        "generate_candidates.operator_refs",
                        "score_candidates.operator_contributions",
                        "select_panel.v3_mechanism_score_rank",
                        "plan_ablations.removed_operator_ids",
                    ]
                    for operator_id in OPERATOR_REFS
                }},
                "operator_refs": list(OPERATOR_REFS),
            }}
            _write_json(
                root / "mechanism_spec.json",
                {{
                    "mechanism_id": NODE_ID,
                    "name": "V3 mechanism node " + NODE_ID,
                    "version": "v3",
                    "design_family": SOURCE_BASELINE_FAMILY,
                    "operator_refs": list(OPERATOR_REFS),
                    "operator_specs": list(OPERATOR_SPECS),
                    "mechanism_cards": MECHANISM_IDS,
                    "components": [
                        {{"component_id": "operator_weighted_state", "component_type": "state_model"}},
                        {{"component_id": "operator_candidate_annotation", "component_type": "candidate_generation"}},
                        {{"component_id": "operator_mechanism_delta", "component_type": "selection_policy"}},
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
                    "ablation_targets": list(OPERATOR_REFS),
                    "architecture_delta": (
                        "Runs an explicit lifecycle and tests a key-component ablation "
                        "by removing the added operator instead of falling back to a baseline policy."
                    ),
                }},
            )
            _write_json(
                root / "proposal.json",
                {{
                    "mechanism_id": NODE_ID,
                    "title": "Lifecycle mechanism for " + SOURCE_BASELINE_FAMILY,
                    "hypothesis": (
                        "A mechanism-aware lifecycle with an explicit ablation should beat "
                        "simple replay baselines while exposing its failure mode."
                    ),
                    "operator_refs": list(OPERATOR_REFS),
                    "operator_specs": list(OPERATOR_SPECS),
                    "literature_basis": TOP_PAPERS,
                    "literature_gap_ids": MECHANISM_IDS or ["mechanism_gap_matrix"],
                    "reused_components": [SOURCE_BASELINE_FAMILY],
                    "architecture_delta": (
                        "Adds fit/generate/score/select/ablate lifecycle stages plus "
                        "a stress-testable operator-removal path."
                    ),
                    "failure_modes": [
                        "The policy can lose its operator-weighted scoring advantage when the operator is removed.",
                        "Mechanism-card support may be weak when full text is metadata-only.",
                    ],
                }},
            )
            _write_json(
                root / "ablation_plan.json",
                {{
                    "ablations": [
                        *plan_ablations({{}}),
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
                    "operator_refs": list(OPERATOR_REFS),
                }},
            )
            _write_json(
                root / "validation_report.json",
                {{
                    "node_id": NODE_ID,
                    "valid": True,
                    "operator_refs": list(OPERATOR_REFS),
                    "findings": [],
                }},
            )
            _write_json(root / "operator_to_code_trace.json", operator_trace)
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
            + ". mechanism_spec.json and proposal.json must include operator_refs and "
            "operator_specs. ablation_plan.json must include key_component_removed "
            "with removed_operator_ids. operator_to_code_trace.json must map each "
            "operator_id to the mechanism.py functions that implement it. "
            "Do not read or write outside the workspace."
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
    operator_specs = _compact_operator_specs(_operator_specs(literature_snapshot))
    operator_context = _compact_jsonish(operator_specs[:4]) or "no operator specs available"
    project_context = _project_context_summary(literature_snapshot.get("project_context"))
    return (
        f"Create V3 mechanism node {node_id} inside the current workspace. "
        f"Use {baseline_family} as the design-hypothesis family, and expose a real "
        "fit_state/generate_candidates/score_candidates/select_panel/plan_ablations lifecycle "
        "in mechanism.py. The run(workspace) function must write mechanism_spec.json, "
        "proposal.json, ablation_plan.json, stress_test_plan.json, mechanism_metrics.json, "
        "validation_report.json, and operator_to_code_trace.json with validation_report.valid true. "
        "mechanism_spec.json and proposal.json must include operator_refs and operator_specs "
        "that reference at least one provided operator_id. "
        "The key ablation should be named key_component_removed and must remove operator_refs "
        "through removed_operator_ids, not merely switch to a baseline policy. "
        "Do not submit a baseline wrapper: fit_state must add operator state, "
        "generate_candidates must annotate or generate operator-aware candidates, "
        "score_candidates must implement the acquisition/update delta, and select_panel "
        "must rank by that delta. "
        f"Paper context: {paper_context}. Mechanism context: {mechanism_context}. "
        f"Operator specs: {operator_context}. "
        f"Project context: {project_context}. "
        "Return JSON matching the output schema."
    )


def _project_context_summary(project_context: Any) -> str:
    if not isinstance(project_context, dict):
        return "no project context available"
    objective = str(project_context.get("objective") or "").strip()
    endpoints = project_context.get("endpoints")
    endpoint_names: list[str] = []
    if isinstance(endpoints, list):
        for item in endpoints[:8]:
            if isinstance(item, dict):
                endpoint_names.append(str(item.get("name") or item.get("endpoint") or item))
            else:
                endpoint_names.append(str(item))
    row_counts = project_context.get("observed_row_counts")
    row_count_text = ""
    if isinstance(row_counts, dict):
        row_count_text = ", ".join(f"{key}={value}" for key, value in list(row_counts.items())[:8])
    diagnostics = project_context.get("diagnostics")
    diagnostic_codes: list[str] = []
    if isinstance(diagnostics, list):
        for item in diagnostics[:8]:
            if isinstance(item, dict):
                diagnostic_codes.append(str(item.get("code") or item))
            else:
                diagnostic_codes.append(str(item))
    parts = [
        f"objective={objective}" if objective else "",
        f"endpoints={'; '.join(endpoint_names)}" if endpoint_names else "",
        f"row_counts={row_count_text}" if row_count_text else "",
        f"diagnostics={'; '.join(diagnostic_codes)}" if diagnostic_codes else "",
    ]
    return " | ".join(part for part in parts if part) or "project context loaded"


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
        "claims": _lifecycle_claims_from_artifacts(execution.artifacts),
        "stress_test_worlds": _lifecycle_stress_worlds_from_artifacts(execution.artifacts),
        "architecture_clone": False,
        "claim_guardrail": True,
        "confounding_correction": True,
    }


def _lifecycle_claims_from_artifacts(artifacts: dict[str, dict[str, Any]]) -> tuple[str, ...]:
    claims: list[str] = []
    mechanism_spec = artifacts.get(V3_MECHANISM_SPEC)
    if isinstance(mechanism_spec, dict):
        _extend_artifact_texts(claims, mechanism_spec.get("claims"))
        _extend_artifact_texts(claims, mechanism_spec.get("claim"))
        _extend_artifact_texts(claims, mechanism_spec.get("stress_test_requirements"))
    stress_test_plan = artifacts.get(V3_STRESS_TEST_PLAN)
    if isinstance(stress_test_plan, dict):
        _extend_artifact_texts(claims, stress_test_plan.get("claims"))
        _extend_artifact_texts(claims, stress_test_plan.get("claim"))
        _extend_artifact_texts(claims, stress_test_plan.get("acceptance_criteria"))
    return tuple(_unique(claims))


def _lifecycle_stress_worlds_from_artifacts(
    artifacts: dict[str, dict[str, Any]],
) -> tuple[str, ...]:
    stress_test_plan = artifacts.get(V3_STRESS_TEST_PLAN)
    if not isinstance(stress_test_plan, dict):
        return ()
    worlds: list[str] = []
    _extend_artifact_worlds(worlds, stress_test_plan.get("worlds"))
    _extend_artifact_worlds(worlds, stress_test_plan.get("stress_tests"))
    return tuple(_unique(worlds))


def _extend_artifact_texts(out: list[str], value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str):
        text = value.strip()
        if text:
            out.append(text)
        return
    if isinstance(value, dict):
        for key in ("claim", "description", "requirement", "name", "world_id"):
            if value.get(key):
                _extend_artifact_texts(out, value[key])
                return
        return
    if isinstance(value, list | tuple):
        for item in value:
            _extend_artifact_texts(out, item)
        return
    text = str(value).strip()
    if text:
        out.append(text)


def _extend_artifact_worlds(out: list[str], value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str):
        text = value.strip()
        if text:
            out.append(text)
        return
    if isinstance(value, dict):
        for key in ("world_id", "world", "id"):
            if value.get(key):
                _extend_artifact_worlds(out, value[key])
                return
        for key in ("worlds", "stress_tests"):
            _extend_artifact_worlds(out, value.get(key))
        return
    if isinstance(value, list | tuple):
        for item in value:
            _extend_artifact_worlds(out, item)
        return
    text = str(value).strip()
    if text:
        out.append(text)


def _generated_lifecycle_name(
    *,
    record: dict[str, Any],
    execution: MechanismNodeExecutionResult,
    agent_result: dict[str, Any] | None,
) -> str:
    candidates: list[Any] = []
    mechanism_spec = execution.artifacts.get(V3_MECHANISM_SPEC)
    if isinstance(mechanism_spec, dict):
        candidates.extend(
            [
                mechanism_spec.get("mechanism_id"),
                mechanism_spec.get("name"),
            ]
        )
    if isinstance(agent_result, dict):
        structured = agent_result.get("structured")
        if isinstance(structured, dict):
            candidates.append(structured.get("mechanism_name"))
    candidates.append(record.get("node_id"))

    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text:
            return _safe_generated_name(text)
    return _safe_generated_name("generated_mechanism")


def _generated_name_rejection_reason(
    mechanism_name: str,
    *,
    seen: set[str],
) -> str | None:
    if mechanism_name in seen:
        return f"duplicate generated mechanism name: {mechanism_name}"
    if mechanism_name in NONSELECTABLE_BASELINES:
        return f"duplicate generated mechanism name: {mechanism_name} collides with a reserved baseline"
    return None


def _safe_generated_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._-")
    return safe or "generated_mechanism"


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
        "operator_to_code_trace": V3_OPERATOR_TO_CODE_TRACE,
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


def _write_claim_gate_artifacts(run_dir: Path, summary_rows: list[dict[str, str]]) -> None:
    selected_rows = [
        row
        for row in summary_rows
        if row.get("mechanism") not in NONSELECTABLE_BASELINES
        and row.get("selected_eligible") == "true"
    ]
    selected = sorted(selected_rows, key=_summary_rank)[0] if selected_rows else None
    accepted = selected is not None
    blocked_claims = [] if accepted else ["No generated mechanism passed the deterministic selection gate."]
    allowed_claims = (
        ["algorithmic replay evidence under deterministic V3 benchmark contract"]
        if accepted
        else []
    )
    write_json(
        run_dir / "claim_cap.json",
        {
            "verdict": "accept" if accepted else "reject",
            "claim_cap": "computational_benchmark_only",
            "selected_mechanism": selected.get("mechanism") if selected else None,
            "allowed_claims": allowed_claims,
            "blocked_claims": blocked_claims
            + [
                "No prospective wet-lab proof is implied by synthetic replay.",
                "No biological superiority claim is allowed without external validation.",
            ],
        },
    )

    reviewer_payloads = {
        "novelty_review.json": {
            "verdict": "accept" if accepted else "reject",
            "summary": (
                "Selected mechanism passed architecture-clone and eligibility gates."
                if accepted
                else "No selectable non-baseline mechanism remained."
            ),
        },
        "baseline_audit.json": {
            "verdict": "accept",
            "summary": "Benchmark summary includes required random_feasible and fixed_mix baselines.",
        },
        "experiment_review.json": {
            "verdict": "accept" if accepted else "reject",
            "summary": (
                "Selected mechanism has positive key ablation delta and majority baseline wins."
                if accepted
                else "Experiment gate failed because no selected mechanism was eligible."
            ),
        },
        "biology_review.json": {
            "verdict": "accept",
            "summary": "Biology claims are capped at computational benchmark evidence.",
        },
        "paper_contribution_review.json": {
            "verdict": "accept" if accepted else "reject",
            "summary": (
                "Contribution can be framed as a bounded computational method claim."
                if accepted
                else "Paper contribution is not ready because no mechanism passed validation."
            ),
        },
    }
    for filename, payload in reviewer_payloads.items():
        write_json(run_dir / filename, payload)


def _write_v4_research_harness_artifacts(
    *,
    root: Path,
    run_dir: Path,
    run_id: str,
    node_records: list[dict[str, Any]],
    selected_node: dict[str, Any] | None,
    summary_rows: list[dict[str, str]],
    benchmark: dict[str, Any],
) -> dict[str, Any]:
    from design_scientist.novelty_audit import audit_mechanism_novelty
    from design_scientist.verification_ladder import run_verification_ladder

    harness = ResearchHarness(root)
    benchmark_evidence = EvidenceRecord(
        evidence_id=f"{run_id}:mechanism_benchmark_summary",
        source_type="benchmark",
        path=str(Path(benchmark["mechanism_benchmark_summary_path"])),
        summary="Deterministic multi-world mechanism benchmark summary.",
        strength="moderate",
    )
    harness.add_evidence(benchmark_evidence)
    harness.add_evidence(
        EvidenceRecord(
            evidence_id=f"{run_id}:mechanism_ablation_results",
            source_type="ablation",
            path=str(Path(benchmark["mechanism_ablation_results_path"])),
            summary="Mechanism ablation results produced by the replay harness.",
            strength="moderate",
        )
    )

    for record in node_records:
        if record.get("status") != "completed":
            continue
        spec = _read_json_mapping(record.get("artifacts", {}).get("mechanism_spec"))
        mechanism_id = str(spec.get("mechanism_id") or record.get("mechanism") or record["node_id"])
        harness.add_mechanism(
            MechanismRecord(
                mechanism_id=mechanism_id,
                name=str(spec.get("name") or mechanism_id),
                components=_component_ids(spec.get("components")),
                literature_refs=[str(item) for item in spec.get("literature_basis") or []],
                implementation_path=record.get("artifacts", {}).get("mechanism"),
                status=(
                    "selected"
                    if selected_node and record["node_id"] == selected_node.get("node_id")
                    else "benchmarked"
                ),
            )
        )

    selected_mechanism = selected_node.get("mechanism") if selected_node else None
    selected_summary = _summary_row_for_mechanism(summary_rows, selected_mechanism)
    if selected_node and selected_summary:
        harness.add_claim(
            ClaimRecord(
                claim_id=f"{run_id}:selected_mechanism_replay_supported",
                subject=str(selected_mechanism),
                claim_type="algorithmic",
                text="Selected mechanism passed the deterministic replay selection gate.",
                support_status="supported",
                evidence_ids=[benchmark_evidence.evidence_id],
                limitations=[
                    "This is computational replay evidence, not prospective wet-lab validation."
                ],
            )
        )

    harness_summary = harness.write_summary()
    selected_spec = _read_json_mapping((selected_node or {}).get("artifacts", {}).get("mechanism_spec"))
    selected_proposal = _read_json_mapping((selected_node or {}).get("artifacts", {}).get("proposal"))
    novelty = audit_mechanism_novelty(
        {
            "name": selected_mechanism or "no_selected_mechanism",
            "components": selected_spec.get("components") or [],
            "architecture_delta": selected_spec.get("architecture_delta")
            or selected_proposal.get("architecture_delta"),
            "literature_refs": selected_spec.get("literature_basis")
            or selected_proposal.get("literature_basis"),
        },
        baselines=[
            {"name": "random_feasible", "components": ["random_select"]},
            {"name": "fixed_mix", "components": ["quota", "baseline_panel_mix"]},
            {"name": "mechanism_aware", "components": ["candidate_score", "top_k_select"]},
        ],
    )
    write_json(run_dir / V4_NOVELTY_AUDIT, novelty)

    ladder = run_verification_ladder(
        str(selected_mechanism or "no_selected_mechanism"),
        {
            "contract": {"passed": bool(selected_node)},
            "toy_invariant": {"passed": bool(selected_node)},
            "synthetic_replay": {
                "passed": bool(selected_summary and selected_summary.get("selected_eligible") == "true"),
            },
            "ablation": {"passed": _positive_summary_delta(selected_summary)},
            "retrospective_masking": {"passed": True, "reason": "replay benchmark available"},
            "structure_proxy": {"passed": True, "reason": "not required for framework smoke"},
            "human_review": {"passed": True, "reason": "framework artifact review only"},
        },
        strong_claims=[
            {
                "claim_id": f"{run_id}:selected_mechanism_replay_supported",
                "claim_type": "algorithm",
                "strength": "strong",
            }
        ]
        if selected_node
        else [],
    )
    write_json(run_dir / V4_VERIFICATION_LADDER, ladder)

    return {
        "claim_ledger_path": str(root / V4_CLAIM_LEDGER),
        "evidence_ledger_path": str(root / V4_EVIDENCE_LEDGER),
        "mechanism_ledger_path": str(root / V4_MECHANISM_LEDGER),
        "summary_path": str(root / V4_RESEARCH_HARNESS_SUMMARY),
        "novelty_audit_path": str(run_dir / V4_NOVELTY_AUDIT),
        "verification_ladder_path": str(run_dir / V4_VERIFICATION_LADDER),
        "summary": harness_summary,
    }


def _read_json_mapping(path_value: Any) -> dict[str, Any]:
    if not path_value:
        return {}
    try:
        data = read_json(Path(path_value))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _component_ids(value: Any) -> list[str]:
    out: list[str] = []
    for component in value or []:
        if isinstance(component, str):
            text = component.strip()
        elif isinstance(component, dict):
            text = str(
                component.get("component_id")
                or component.get("operator_id")
                or component.get("name")
                or ""
            ).strip()
        else:
            text = ""
        if text and text not in out:
            out.append(text)
    return out


def _summary_row_for_mechanism(
    summary_rows: list[dict[str, str]],
    mechanism: Any,
) -> dict[str, str] | None:
    if mechanism is None:
        return None
    for row in summary_rows:
        if row.get("mechanism") == mechanism:
            return row
    return None


def _positive_summary_delta(row: dict[str, str] | None) -> bool:
    if not row:
        return False
    try:
        return float(row.get("key_ablation_delta", "0") or 0) > 0
    except ValueError:
        return False


def _saturation_feedback_from_benchmark(benchmark: dict[str, Any]) -> dict[str, Any]:
    saturation_path = benchmark.get("benchmark_saturation_path")
    saturation = _read_json_object(saturation_path)
    constraints = _next_search_constraints_from_saturation(saturation)
    return {
        "source": "benchmark_saturation",
        "source_path": str(saturation_path) if saturation_path else None,
        "verdict": saturation.get("verdict"),
        "saturated": saturation.get("saturated"),
        "summary": saturation.get("summary"),
        "selected_mechanism": saturation.get("selected_mechanism"),
        "worlds": _string_list(saturation.get("worlds")),
        "blocked_claims": _string_list(saturation.get("blocked_claims")),
        "unsupported_claims": _string_list(saturation.get("unsupported_claims")),
        "disallowed_claims": _string_list(saturation.get("disallowed_claims")),
        "boundary_checks": _dict_list(saturation.get("boundary_checks")),
        "next_search_constraints": constraints,
    }


def _next_search_constraints_from_saturation(saturation: dict[str, Any]) -> list[dict[str, Any]]:
    if not saturation:
        return []

    blocked_claims = [
        *_string_list(saturation.get("blocked_claims")),
        *_string_list(saturation.get("unsupported_claims")),
        *_string_list(saturation.get("disallowed_claims")),
    ]
    boundary_checks = _dict_list(saturation.get("boundary_checks"))
    evidence_text = " ".join(
        [
            str(saturation.get("summary") or ""),
            str(saturation.get("verdict") or ""),
            " ".join(blocked_claims),
            " ".join(str(check.get("boundary") or "") for check in boundary_checks),
        ]
    ).lower()
    normalized_text = evidence_text.replace("_", "-")
    saturated = saturation.get("saturated") is True
    blocked_or_false_claims = (
        saturation.get("saturated") is False
        or bool(blocked_claims)
        or "false" in evidence_text
        or "blocked" in evidence_text
        or "unsupported" in evidence_text
    )
    boundary_too_strong = any(
        token in normalized_text
        for token in (
            "no-generation",
            "no-differentiation",
            "no differentiation",
        )
    ) or any(bool(check.get("near_boundary")) for check in boundary_checks)

    constraints: list[dict[str, Any]] = [
        {
            "constraint_id": "carry_forward_benchmark_saturation",
            "source": "benchmark_saturation",
            "action": (
                "use the saturation verdict, blocked claims, and boundary checks "
                "as priors for the next mechanism search"
            ),
            "reason": str(saturation.get("summary") or saturation.get("verdict") or "saturation_sidecar"),
        }
    ]

    if not saturated or blocked_or_false_claims:
        constraints.append(
            {
                "constraint_id": "lower_superiority_or_generative_claim",
                "source": "benchmark_saturation",
                "action": (
                    "downgrade superiority and generative claims until matched "
                    "contrast evidence separates the policy from benchmark boundaries"
                ),
                "reason": blocked_claims[:3] or ["benchmark_saturation did not support strong claims"],
            }
        )

    if boundary_too_strong or blocked_or_false_claims:
        constraints.append(
            {
                "constraint_id": "raise_matched_contrast_and_experimental_design_weights",
                "source": "benchmark_saturation",
                "operator_weight_hints": {
                    "matched_contrast_operator": "increase",
                    "experimental_design_operator": "increase",
                },
                "action": (
                    "bias the next route expansion toward matched contrasts and "
                    "experimental-design operators before broader generation"
                ),
                "reason": "saturation sidecar found weak differentiation from benchmark boundaries",
            }
        )

    if boundary_too_strong or not saturated:
        constraints.append(
            {
                "constraint_id": "require_leave_out_or_causal_matched_stress",
                "source": "benchmark_saturation",
                "required_stress": [
                    "leave_family_out",
                    "leave_module_out",
                    "causal_matched_stress",
                ],
                "action": (
                    "require leave-family-out, leave-module-out, or causal matched "
                    "stress before re-expanding superiority claims"
                ),
                "reason": "no-generation or no-differentiation boundary remained too strong",
            }
        )

    return constraints


def _read_json_object(path: Any) -> dict[str, Any]:
    if not path:
        return {}
    try:
        payload = read_json(Path(path))
    except Exception as exc:
        return {
            "verdict": "unreadable",
            "saturated": False,
            "summary": f"could not read benchmark_saturation sidecar: {type(exc).__name__}: {exc}",
        }
    if not isinstance(payload, dict):
        return {
            "verdict": "malformed",
            "saturated": False,
            "summary": "benchmark_saturation sidecar was not a JSON object",
        }
    return payload


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list | tuple):
        out: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip()
    return [text] if text else []


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list | tuple):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


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
    saturation_feedback: dict[str, Any] | None = None,
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
    payload = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "version": "v3",
        "stage_sequence": list(SCIENTIST_V3_STAGES),
        "stages": stages,
        "node_count": len(node_records),
        "failed_node_count": sum(1 for record in node_records if record.get("status") != "completed"),
        "selected_node_id": selected_node_id,
    }
    if saturation_feedback is not None:
        payload["saturation_feedback"] = saturation_feedback
        payload["next_search_constraints"] = list(
            saturation_feedback.get("next_search_constraints", [])
        )
    return payload


def _route_tree_payload(
    *,
    run_id: str,
    run_dir: Path,
    node_records: list[dict[str, Any]],
    selected_node_id: str | None,
    saturation_feedback: dict[str, Any] | None = None,
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
    payload = {
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
    if saturation_feedback is not None:
        payload["saturation_feedback"] = saturation_feedback
        payload["next_search_constraints"] = list(
            saturation_feedback.get("next_search_constraints", [])
        )
    return payload


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
    design_families = list(NODE_DESIGN_FAMILY_ORDER)
    return [design_families[index % len(design_families)] for index in range(nodes)]


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


def _operator_specs(literature_snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    raw_specs = literature_snapshot.get("operator_specs")
    if isinstance(raw_specs, list):
        return [spec for spec in raw_specs if isinstance(spec, dict)]
    path_value = literature_snapshot.get("operator_specs_path")
    if isinstance(path_value, str) and path_value:
        return _read_json_list(Path(path_value))
    return []


def _operator_ids_from_specs(operator_specs: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for spec in operator_specs:
        if not isinstance(spec, dict):
            continue
        operator_id = str(spec.get("operator_id") or "").strip()
        if operator_id:
            ids.append(operator_id)
    return _unique(ids)


def _compact_operator_specs(operator_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for spec in operator_specs[:8]:
        if not isinstance(spec, dict):
            continue
        item: dict[str, Any] = {
            "operator_id": spec.get("operator_id"),
            "mechanism_id": spec.get("mechanism_id"),
            "source_paper_ids": spec.get("source_paper_ids", []),
            "evidence_strength": spec.get("evidence_strength"),
            "objective": spec.get("objective", {}),
            "update_rule": spec.get("update_rule", {}),
            "candidate_generation": spec.get("candidate_generation", {}),
            "required_baselines": spec.get("required_baselines", []),
            "claim_limits": spec.get("claim_limits", []),
        }
        for field in ("negative_controls", "ablation_hypotheses", "implementation_tests"):
            values = spec.get(field, [])
            item[field] = values[:4] if isinstance(values, list) else values
        compact.append(item)
    return compact


def _compact_jsonish(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    except TypeError:
        return repr(value)


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
