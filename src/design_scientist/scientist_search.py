"""Controlled scientist tree search over method-node workspaces."""

from __future__ import annotations

import csv
import json
import random
import shutil
import time
from pathlib import Path
from textwrap import dedent
from typing import Any

from design_scientist.backends.base import WorkspaceAgentBackend
from design_scientist.io import ensure_dir, read_json, write_json
from design_scientist.method_nodes import (
    NOVELTY_SELECTION_OVERLAP_THRESHOLD,
    MethodNodeExecutionResult,
    execute_method_node,
)
from design_scientist import policies as policy_api
from design_scientist.schemas import WorkspaceAgentTask, to_plain_data
from design_scientist.synthetic_replay import DEFAULT_METHODS, run_synthetic_benchmark


SCIENTIST_STAGES = (
    "baseline_reproduction",
    "literature_grounded_ideation",
    "policy_implementation",
    "debug_and_repair",
    "creative_research",
    "ablation_and_stress",
    "selection_and_report",
)

NODE_METHOD_ORDER = (
    "mechanism_aware",
    "fixed_mix",
    "pure_lattice_repair",
    "greedy_utility",
    "pure_uncertainty",
    "top_observed",
    "random_feasible",
    "literature_gap",
)
NO_BASELINE_METHODS = ("literature_gap",)

METHOD_NODE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "policy_name": {"type": "string"},
        "policy_entrypoint": {"type": "string"},
        "files_written": {"type": "array", "items": {"type": "string"}},
        "contract_notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "policy_name", "policy_entrypoint", "files_written", "contract_notes"],
    "additionalProperties": False,
}
METHOD_NODE_AGENT_FILES = (
    "proposal.json",
    "method.py",
    "manifest.json",
    "candidate_policy.json",
    "novelty_report.json",
    "benchmark_metrics.json",
    "validation_report.json",
)


def develop_method(
    project_dir: str | Path,
    nodes: int = 3,
    use_codex: bool = False,
    backend: WorkspaceAgentBackend | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Develop and rank method nodes with the deterministic synthetic replay."""

    return _develop_method(
        project_dir=project_dir,
        nodes=nodes,
        rounds=3,
        use_codex=use_codex,
        backend=backend,
        run_id=run_id,
        literature_snapshot=None,
    )


def run_scientist_search(
    project_dir: str | Path,
    max_papers: int = 60,
    nodes: int = 3,
    rounds: int = 3,
    use_codex: bool = False,
    offline_fixtures: bool = False,
    backend: WorkspaceAgentBackend | None = None,
    strict_literature: bool = True,
) -> dict[str, Any]:
    """Run literature discovery, method-node development, and benchmark ranking."""

    if max_papers <= 0:
        raise ValueError("max_papers must be positive")
    root = Path(project_dir).expanduser().resolve()

    literature_error: str | None = None
    extraction_result: dict[str, Any] | None = None
    extraction_error: str | None = None
    try:
        from design_scientist.literature_pipeline import run_literature_search

        run_literature_search(
            root,
            max_papers=max_papers,
            offline_fixtures=offline_fixtures,
        )
    except Exception as exc:  # pragma: no cover - exercised when external services fail.
        literature_error = f"{type(exc).__name__}: {exc}"

    try:
        from design_scientist.method_extraction import extract_methods

        extraction_result = extract_methods(root)
        if extraction_result.get("status") != "ok":
            extraction_error = str(extraction_result.get("reason") or extraction_result.get("status"))
    except Exception as exc:  # pragma: no cover - defensive artifact path.
        extraction_error = f"{type(exc).__name__}: {exc}"

    literature_snapshot = _literature_snapshot(
        root,
        max_papers=max_papers,
        status="failed" if literature_error else "completed",
        error=literature_error,
    )
    literature_snapshot["method_extraction"] = {
        "status": "failed" if extraction_error else "completed",
        "error": extraction_error,
        "method_module_count": (
            extraction_result.get("method_module_count")
            if isinstance(extraction_result, dict)
            else None
        ),
    }
    if strict_literature and (literature_error or extraction_error):
        failures: list[str] = []
        if literature_error:
            failures.append(f"literature search failed: {literature_error}")
        if extraction_error:
            failures.append(f"method extraction failed: {extraction_error}")
        raise RuntimeError("; ".join(failures))
    return _develop_method(
        project_dir=root,
        nodes=nodes,
        rounds=rounds,
        use_codex=use_codex,
        backend=backend,
        run_id=None,
        literature_snapshot=literature_snapshot,
    )


def _develop_method(
    *,
    project_dir: str | Path,
    nodes: int,
    rounds: int,
    use_codex: bool,
    backend: WorkspaceAgentBackend | None,
    run_id: str | None,
    literature_snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    if nodes <= 0:
        raise ValueError("nodes must be positive")
    if rounds <= 0:
        raise ValueError("rounds must be positive")

    root = Path(project_dir).expanduser().resolve()
    safe_run_id = _safe_run_id(run_id) if run_id is not None else _timestamp_run_id()
    run_dir = ensure_dir(root / "runs" / safe_run_id)
    node_root = ensure_dir(run_dir / "nodes")
    journal_path = run_dir / "scientist_journal.json"
    stage_progress_path = run_dir / "stage_progress.json"
    route_tree_path = run_dir / "route_tree.json"
    snapshot = literature_snapshot or _literature_snapshot(root)
    agent_backend = _resolve_backend(use_codex, backend)

    planned_methods = _planned_methods(nodes)
    node_records: list[dict[str, Any]] = []
    valid_nodes: list[dict[str, Any]] = []
    failed_nodes: list[dict[str, Any]] = []
    policy_callables: dict[str, policy_api.PolicyCallable] = {}

    for index, method_name in enumerate(planned_methods, start=1):
        node_id = f"node_{index:02d}_{method_name}"
        workspace = _fresh_node_workspace(node_root / node_id)
        agent_result: dict[str, Any] | None = None
        agent_rejection_reasons: list[str] = []

        if use_codex:
            agent_result = _run_codex_node_agent(
                backend=agent_backend,
                root=root,
                workspace=workspace,
                node_id=node_id,
                method_name=method_name,
                literature_snapshot=snapshot,
            )
            agent_rejection_reasons = _codex_agent_rejection_reasons(agent_result)
        else:
            _write_local_method_node(workspace, node_id=node_id, method_name=method_name)

        if agent_rejection_reasons:
            execution = _skipped_method_node_result(workspace, errors=agent_rejection_reasons)
            benchmark_method = method_name if method_name in DEFAULT_METHODS else None
            policy_callable = None
            policy_error = None
        else:
            execution = execute_method_node(workspace, guard_roots=[root])
            benchmark_method = _benchmark_method_from_execution(execution, fallback=method_name)
            policy_callable, policy_error = _policy_callable_from_execution(
                execution,
                node_id=node_id,
                fallback_policy_name=f"{node_id}_policy",
            )
        reasons = _failure_reasons(
            execution,
            benchmark_method=benchmark_method,
            policy_error=policy_error,
            allow_no_baseline=(
                method_name in NO_BASELINE_METHODS and policy_callable is not None
            ),
        )
        status = "completed" if not reasons else "failed"
        record = _node_record(
            node_id=node_id,
            workspace=workspace,
            planned_method=method_name,
            benchmark_method=benchmark_method,
            benchmark_policy_name=_callable_policy_name(policy_callable) if policy_callable else None,
            status=status,
            execution=execution,
            agent_result=agent_result,
            reasons=reasons,
        )
        node_records.append(record)
        if status == "completed" and policy_callable is not None:
            valid_nodes.append(record)
            policy_callables[node_id] = policy_callable
        else:
            failed_nodes.append({"node_id": node_id, "reasons": reasons})

    benchmark = _run_and_attach_benchmark(
        root=root,
        run_id=safe_run_id,
        rounds=rounds,
        node_records=valid_nodes,
        policy_callables=policy_callables,
    )
    ranking = benchmark.get("ranking", [])
    failed_nodes = [
        {
            "node_id": record["node_id"],
            "reasons": list(record.get("failure_reasons", [])),
        }
        for record in node_records
        if record.get("status") != "completed"
    ]

    for record in node_records:
        ranked = next(
            (row for row in ranking if row["node_id"] == record["node_id"]),
            None,
        )
        if ranked is not None:
            record["benchmark_metrics"] = ranked["benchmark_metrics"]
            record["ranking_score"] = ranked["ranking_score"]
        record["trace"] = _node_trace(
            record=record,
            benchmark=benchmark,
            journal_path=journal_path,
            stage_progress_path=stage_progress_path,
            route_tree_path=route_tree_path,
        )

    records_by_id = {record["node_id"]: record for record in node_records}
    for ranked in ranking:
        record = records_by_id.get(ranked["node_id"])
        if record is not None:
            ranked["trace"] = dict(record["trace"])

    selected_node = ranking[0] if ranking else None
    selected_node_id = selected_node["node_id"] if selected_node else None
    for record in node_records:
        record["selected"] = record["node_id"] == selected_node_id

    selected_record = records_by_id.get(selected_node_id) if selected_node_id else None
    if selected_record is not None:
        _attach_selected_full_chain_artifacts(root, selected_record)
        selected_record["trace"] = _node_trace(
            record=selected_record,
            benchmark=benchmark,
            journal_path=journal_path,
            stage_progress_path=stage_progress_path,
            route_tree_path=route_tree_path,
        )
        if selected_node is not None:
            selected_node["trace"] = dict(selected_record["trace"])

    memory_append_errors = _append_research_os_memory(
        root=root,
        run_id=safe_run_id,
        selected_node=selected_node,
        failed_nodes=failed_nodes,
    )

    stage_progress = _stage_progress_payload(
        run_id=safe_run_id,
        run_dir=run_dir,
        node_records=node_records,
        selected_node_id=selected_node_id,
        benchmark=benchmark,
        journal_path=journal_path,
        route_tree_path=route_tree_path,
    )
    route_tree = _route_tree_payload(
        run_id=safe_run_id,
        run_dir=run_dir,
        node_records=node_records,
        selected_node_id=selected_node_id,
    )

    journal = {
        "run_id": safe_run_id,
        "project_dir": str(root),
        "literature_snapshot": snapshot,
        "stage_progress_path": str(stage_progress_path),
        "route_tree_path": str(route_tree_path),
        "nodes": node_records,
        "benchmark": benchmark,
        "selected_node_id": selected_node_id,
        "selected_node": selected_node,
        "failed_nodes": failed_nodes,
        "research_os": {
            "append_errors": memory_append_errors,
        },
    }
    write_json(stage_progress_path, stage_progress)
    write_json(route_tree_path, route_tree)
    write_json(journal_path, journal)

    return {
        "run_id": safe_run_id,
        "run_dir": str(run_dir),
        "journal_path": str(journal_path),
        "stage_progress_path": str(stage_progress_path),
        "route_tree_path": str(route_tree_path),
        "literature_snapshot": snapshot,
        "nodes": node_records,
        "benchmark": benchmark,
        "selected_node": selected_node,
        "failed_nodes": failed_nodes,
    }


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


def _fresh_node_workspace(workspace: Path) -> Path:
    if workspace.exists() or workspace.is_symlink():
        if workspace.is_symlink() or workspace.is_file():
            workspace.unlink()
        else:
            shutil.rmtree(workspace)
    return ensure_dir(workspace)


def _run_codex_node_agent(
    *,
    backend: WorkspaceAgentBackend | None,
    root: Path,
    workspace: Path,
    node_id: str,
    method_name: str,
    literature_snapshot: dict[str, Any],
) -> dict[str, Any]:
    if backend is None:
        raise RuntimeError("A WorkspaceAgentBackend is required when use_codex=True")

    task = WorkspaceAgentTask(
        objective=_codex_objective(
            node_id=node_id,
            method_name=method_name,
            context=_codex_prompt_context(root, literature_snapshot),
        ),
        workspace=str(workspace),
        mode="patch",
        allowed_paths=[str(workspace)],
        write_policy=(
            "Write only proposal.json, method.py, manifest.json, candidate_policy.json, "
            "novelty_report.json, benchmark_metrics.json, and validation_report.json inside "
            "the node workspace."
        ),
        commands_allowed=["python"],
        output_schema=METHOD_NODE_OUTPUT_SCHEMA,
    )
    result = backend.run_task(task)
    return {
        "summary": result.summary,
        "structured": result.structured,
        "transcript_id": result.transcript_id,
        "returncode": result.returncode,
        "files_touched": list(result.files_touched),
        "commands_run": list(result.commands_run),
    }


def _codex_agent_rejection_reasons(agent_result: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    returncode = agent_result.get("returncode")
    if returncode != 0:
        reasons.append(f"Codex backend failed with returncode {returncode}")

    structured = agent_result.get("structured")
    if not isinstance(structured, dict):
        reasons.append("invalid Codex structured output: expected JSON object")
        return reasons

    required_fields = METHOD_NODE_OUTPUT_SCHEMA["required"]
    missing_fields = [
        field
        for field in required_fields
        if field not in structured
    ]
    if missing_fields:
        reasons.append(
            "invalid Codex structured output: missing required fields "
            + ", ".join(missing_fields)
        )

    for field in ("summary", "policy_name", "policy_entrypoint"):
        value = structured.get(field)
        if field in structured and (not isinstance(value, str) or not value.strip()):
            reasons.append(f"invalid Codex structured output: {field} must be a non-empty string")

    files_written = structured.get("files_written")
    if not isinstance(files_written, list) or not all(
        isinstance(item, str) and item.strip()
        for item in files_written
    ):
        reasons.append("invalid Codex structured output: files_written must be a string list")
    else:
        allowed_files = set(METHOD_NODE_AGENT_FILES)
        normalized_files = {Path(item).name for item in files_written}
        unsafe_files = [
            item
            for item in files_written
            if Path(item).name != item or "/" in item or "\\" in item
        ]
        missing_files = sorted(allowed_files - normalized_files)
        unexpected_files = sorted(normalized_files - allowed_files)
        if unsafe_files:
            reasons.append(
                "invalid Codex files_written entries: "
                + ", ".join(str(item) for item in unsafe_files)
            )
        if missing_files:
            reasons.append(
                "invalid Codex files_written entries: missing "
                + ", ".join(missing_files)
            )
        if unexpected_files:
            reasons.append(
                "invalid Codex files_written entries: unexpected "
                + ", ".join(unexpected_files)
            )

    contract_notes = structured.get("contract_notes")
    if "contract_notes" in structured and not isinstance(contract_notes, list):
        reasons.append("invalid Codex structured output: contract_notes must be a list")

    extra_fields = sorted(set(structured) - set(METHOD_NODE_OUTPUT_SCHEMA["properties"]))
    if extra_fields:
        reasons.append(
            "invalid Codex structured output: unexpected fields "
            + ", ".join(extra_fields)
        )
    return _unique_strings(reasons)


def _skipped_method_node_result(
    workspace: Path,
    *,
    errors: list[str],
) -> MethodNodeExecutionResult:
    return MethodNodeExecutionResult(
        workspace=workspace,
        valid=False,
        executed=False,
        missing_artifacts=list(METHOD_NODE_AGENT_FILES),
        errors=list(errors),
        snapshot_roots=[workspace],
    )


def _codex_objective(
    *,
    node_id: str,
    method_name: str,
    context: str,
) -> str:
    required = [
        "proposal.json",
        "method.py",
        "manifest.json",
        "candidate_policy.json",
        "novelty_report.json",
        "benchmark_metrics.json",
        "validation_report.json",
    ]
    if method_name in NO_BASELINE_METHODS:
        invention_instruction = (
            "Use literature_gap/no_baseline invention mode: invent from the cited "
            "literature gaps without declaring a DEFAULT_METHODS baseline family as "
            "source_baseline_family. Do not create a baseline wrapper or tail-swap "
            "variant. Reuse helper code only when proposal.json declares the reused "
            "components and a real architecture_delta."
        )
    else:
        invention_instruction = (
            f"Use {method_name} only as the comparison baseline family, then implement "
            "a non-enum policy callable with a real architecture_delta instead of a "
            "tail-swap baseline wrapper."
        )
    return (
        f"Create a Design Scientist method node named {node_id}. "
        f"{invention_instruction} "
        "The policy callable in method.py must be named select_batch(observed, candidates, "
        "budget, round_index, rng). "
        "The node must satisfy the method_nodes v2 contract: proposal.json includes "
        "proposal_id, title, method_claim or summary, method_hypothesis, "
        "literature_basis, literature_gap or research_gap, literature_gap_ids, "
        "reused_components, architecture_delta, new_mechanism_claim, "
        "algorithm_mechanism, expected_advantage, failure_modes, and planned_ablation; "
        "manifest.json may declare run as a legacy entrypoint; "
        "method.py defines run and the policy callable; running run writes or refreshes "
        "candidate_policy.json, novelty_report.json, benchmark_metrics.json, and "
        "validation_report.json; candidate_policy.json records policy_entrypoint; "
        "novelty_report.json must not mark a baseline clone, must include "
        "baseline_overlap, and must keep selection_overlap_vs_baselines at or below "
        f"{NOVELTY_SELECTION_OVERLAP_THRESHOLD:.2f}; validation_report.json must contain valid: true. "
        f"{context} "
        f"Only write these files: {', '.join(required)}. "
        "Do not read or write outside the node workspace. "
        "Return JSON matching the provided output schema."
    )


def _codex_prompt_context(root: Path, literature_snapshot: dict[str, Any]) -> str:
    sections = [
        ("Top papers", _top_paper_context(literature_snapshot)),
        ("Method modules", _method_module_context(root)),
        ("Research gap matrix", _research_gap_context(root)),
        ("Failure memory summary", _failure_memory_context(root)),
    ]
    lines: list[str] = ["Use this literature-grounded invention context:"]
    for title, entries in sections:
        lines.append(f"{title}:")
        lines.extend(f"- {entry}" for entry in entries)
    return " ".join(lines)


def _top_paper_context(literature_snapshot: dict[str, Any]) -> list[str]:
    cards = literature_snapshot.get("cards") if isinstance(literature_snapshot, dict) else None
    if not isinstance(cards, list) or not cards:
        return ["none available; still declare literature_gap_ids from method modules or gap matrix if present"]
    entries: list[str] = []
    for card in cards[:5]:
        if not isinstance(card, dict):
            continue
        paper_id = card.get("paper_id") or "unknown_paper"
        title = card.get("title") or "untitled"
        year = card.get("year")
        relevance = card.get("relevance")
        details = [str(part) for part in (year, relevance) if part]
        suffix = f" ({'; '.join(details)})" if details else ""
        entries.append(f"{paper_id}: {title}{suffix}")
    return entries or ["none available"]


def _method_module_context(root: Path) -> list[str]:
    data = _read_optional_json(root / "framework" / "method_modules.json")
    modules = []
    if isinstance(data, dict):
        raw_modules = data.get("method_modules") or data.get("modules")
        if isinstance(raw_modules, list):
            modules = [module for module in raw_modules if isinstance(module, dict)]
    entries: list[str] = []
    for module in modules[:6]:
        module_id = module.get("module_id") or module.get("id") or "unknown_module"
        name = module.get("name") or module_id
        reusable = _compact_list(module.get("reusable_ideas"), limit=2)
        gap = module.get("gap_for_design_scientist") or module.get("gap") or ""
        parts = [f"{module_id}: {name}"]
        if reusable:
            parts.append(f"reusable={reusable}")
        if gap:
            parts.append(f"gap={gap}")
        entries.append("; ".join(parts))
    return entries or ["none available"]


def _research_gap_context(root: Path) -> list[str]:
    path = root / "framework" / "research_gap_matrix.csv"
    if not path.exists():
        return ["none available"]
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return ["unreadable"]
    entries: list[str] = []
    for row in rows[:8]:
        gap_id = (
            row.get("literature_gap_id")
            or row.get("method_module_id")
            or row.get("gap_id")
            or "unknown_gap"
        )
        gap = row.get("gap_for_design_scientist") or row.get("gap") or ""
        reusable = row.get("reusable_ideas") or row.get("reused_components") or ""
        parts = [str(gap_id)]
        if gap:
            parts.append(str(gap))
        if reusable:
            parts.append(f"reuse={reusable}")
        entries.append(": ".join(parts))
    return entries or ["none available"]


def _failure_memory_context(root: Path) -> list[str]:
    path = root / "framework" / "failure_memory.jsonl"
    if not path.exists():
        return ["none recorded"]
    entries: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ["unreadable"]
    for line in lines[-5:]:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        summary = record.get("summary")
        if summary:
            entries.append(str(summary))
            continue
        reasons = record.get("reasons")
        if isinstance(reasons, list) and reasons:
            entries.append("; ".join(str(reason) for reason in reasons[:3]))
    return entries or ["none recorded"]


def _read_optional_json(path: Path) -> Any:
    try:
        return read_json(path)
    except Exception:
        return None


def _compact_list(value: Any, *, limit: int) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    items = [str(item) for item in value[:limit] if item]
    return "; ".join(items)


def _write_local_method_node(workspace: Path, *, node_id: str, method_name: str) -> None:
    write_json(
        workspace / "manifest.json",
        {
            "schema_version": 2,
            "name": node_id,
            "entrypoint": "run",
            "policy_entrypoint": "select_batch",
            "source_baseline_family": method_name,
            "description": f"Deterministic local v2 method node for {method_name}.",
        },
    )
    method_source = f"""
        from __future__ import annotations

        import json
        from pathlib import Path

        from design_scientist import policies as policy_api


        NODE_ID = {node_id!r}
        SOURCE_BASELINE_FAMILY = {method_name!r}
        POLICY_NAME = {f"{node_id}_invented_policy"!r}


        def _write_json(path: Path, data: dict) -> None:
            path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\\n", encoding="utf-8")


        def _candidate_id(record) -> str:
            return str(record.get("variant_id") or record.get("candidate_id") or record.get("id") or "")


        def _module_count(record) -> int:
            modules = record.get("modules") or ()
            return len(modules)


        def select_batch(observed, candidates, budget, round_index, rng):
            budget = max(0, budget)
            if budget == 0:
                return []
            repair_pool = sorted(
                candidates,
                key=lambda record: (
                    record.get("background") != record.get("target_background"),
                    _module_count(record),
                    _candidate_id(record),
                ),
            )
            selected = []
            selected_ids = set()

            gap_slots = min(budget, max(1, budget // 2))
            for candidate in repair_pool:
                candidate_id = _candidate_id(candidate)
                if candidate_id and candidate_id not in selected_ids:
                    selected.append(candidate_id)
                    selected_ids.add(candidate_id)
                if len(selected) >= gap_slots:
                    break

            if SOURCE_BASELINE_FAMILY in policy_api.POLICY_REGISTRY:
                base_policy = policy_api.get_policy(SOURCE_BASELINE_FAMILY)
                fill_ids = list(base_policy(observed, candidates, budget, round_index, rng))
            else:
                fill_ids = [_candidate_id(candidate) for candidate in repair_pool]

            for candidate_id in fill_ids:
                if candidate_id and candidate_id not in selected_ids:
                    selected.append(candidate_id)
                    selected_ids.add(candidate_id)
                if len(selected) >= budget:
                    break
            return selected[:budget]


        select_batch.policy_name = POLICY_NAME


        def run(workspace):
            root = Path(workspace)
            _write_json(
                root / "proposal.json",
                {{
                    "schema_version": 2,
                    "proposal_id": NODE_ID,
                    "title": "Deterministic repair policy for " + SOURCE_BASELINE_FAMILY,
                    "summary": (
                        "Local method node that uses a literature-gap allocation layer "
                        "before filling the remaining batch."
                    ),
                    "method_claim": (
                        "A gap-first allocation layer can reduce baseline cloning while "
                        "preserving high-confidence choices for the remaining batch."
                    ),
                    "method_hypothesis": (
                        "A policy can become less clone-like by committing an early budget "
                        "slice to literature-backed sparse evidence gaps before utility fill."
                    ),
                    "literature_basis": [
                        "active learning benefits from balancing exploitation with coverage repair",
                        "transfer settings need explicit checks against baseline policy cloning",
                    ],
                    "literature_gap": (
                        "Tiny-data transfer design lacks an artifact-linked gap allocation "
                        "step that is checked against baseline policy cloning."
                    ),
                    "literature_gap_ids": [
                        "gap_constraints_tiny_data",
                        "gap_baseline_clone_guard",
                    ],
                    "reused_components": [
                        SOURCE_BASELINE_FAMILY,
                        "candidate_id_helper",
                    ] if SOURCE_BASELINE_FAMILY in policy_api.POLICY_REGISTRY else [
                        "candidate_id_helper",
                    ],
                    "architecture_delta": (
                        "Adds a gap-first allocation stage before any baseline or deterministic "
                        "fill, changing the selected prefix rather than swapping only the tail."
                    ),
                    "new_mechanism_claim": (
                        "Literature-gap allocation changes which candidates are eligible for "
                        "the first batch slots under tiny-data transfer constraints."
                    ),
                    "algorithm_mechanism": (
                        "Allocate the first budget slice to low-order unobserved gap-repair "
                        "candidates, then fill remaining slots from the comparison family or "
                        "the same deterministic gap ranking in no-baseline mode."
                    ),
                    "expected_advantage": (
                        "Maintains the baseline utility signal while lowering overlap with simple "
                        "baselines and exposing sparse lattice regions."
                    ),
                    "failure_modes": [
                        "the repair slot can displace a high-value candidate",
                        "baseline-family signal can dominate when the lattice is dense",
                    ],
                    "planned_ablation": "disable_gap_first_allocation",
                    "planned_ablations": [
                        "disable_gap_first_allocation",
                        "vary_gap_budget_fraction",
                    ],
                }},
            )
            candidate_policy = {{
                "node_id": NODE_ID,
                "policy_id": POLICY_NAME,
                "policy_entrypoint": "select_batch",
                "contract": "design_scientist.method_nodes",
                "deterministic": True,
            }}
            if SOURCE_BASELINE_FAMILY in policy_api.POLICY_REGISTRY:
                candidate_policy["source_baseline_family"] = SOURCE_BASELINE_FAMILY
            else:
                candidate_policy["invention_mode"] = "literature_gap_no_baseline"
            _write_json(root / "candidate_policy.json", candidate_policy)
            _write_json(
                root / "novelty_report.json",
                {{
                    "schema_version": 2,
                    "node_id": NODE_ID,
                    "baseline_clone": False,
                    "baseline_overlap": None,
                    "selection_overlap_vs_baselines": None,
                    "novelty_score": None,
                    "status": "pending_synthetic_replay",
                    "rationale": (
                        "Policy uses a callable with a deterministic repair slot; quantitative "
                        "novelty metrics are filled from synthetic replay."
                    ),
                }},
            )
            benchmark_metrics = {{
                "node_id": NODE_ID,
                "method": POLICY_NAME,
                "status": "pending_synthetic_replay",
                "score": 0.0,
            }}
            if SOURCE_BASELINE_FAMILY in policy_api.POLICY_REGISTRY:
                benchmark_metrics["source_baseline_family"] = SOURCE_BASELINE_FAMILY
            else:
                benchmark_metrics["invention_mode"] = "literature_gap_no_baseline"
            _write_json(root / "benchmark_metrics.json", benchmark_metrics)
            _write_json(
                root / "validation_report.json",
                {{
                    "node_id": NODE_ID,
                    "valid": True,
                    "findings": [],
                }},
            )
    """
    (workspace / "method.py").write_text(dedent(method_source).lstrip(), encoding="utf-8")


def _run_and_attach_benchmark(
    *,
    root: Path,
    run_id: str,
    rounds: int,
    node_records: list[dict[str, Any]],
    policy_callables: dict[str, policy_api.PolicyCallable],
) -> dict[str, Any]:
    if not node_records:
        return {
            "run_id": run_id,
            "rounds": rounds,
            "results_path": None,
            "ablation_results_path": None,
            "summary_results_path": None,
            "summary": {"method_rankings": []},
            "ranking": [],
        }

    node_policies = [
        policy_callables[record["node_id"]]
        for record in node_records
        if record["node_id"] in policy_callables
    ]
    methods: list[str | policy_api.PolicyCallable] = [
        *node_policies,
        "mechanism_aware",
        "random_feasible",
        "fixed_mix",
    ]
    benchmark_result = run_synthetic_benchmark(
        root,
        run_id=run_id,
        rounds=rounds,
        methods=methods,
    )
    summary_by_method = {
        str(row["method"]): row
        for row in benchmark_result.get("summary", {}).get("method_rankings", [])
        if isinstance(row, dict) and "method" in row
    }
    rows_by_method: dict[str, list[dict[str, Any]]] = {}
    for row in benchmark_result["benchmark_results"]:
        rows_by_method.setdefault(str(row["method"]), []).append(row)

    ranking: list[dict[str, Any]] = []
    for record in node_records:
        benchmark_policy_name = record.get("benchmark_policy_name")
        method_rows = rows_by_method.get(str(benchmark_policy_name), [])
        if not method_rows:
            continue
        summary_metrics = dict(summary_by_method.get(benchmark_policy_name, {}))
        ranking_metrics = summary_metrics or dict(method_rows[-1])
        replay_failure_reasons = _synthetic_replay_failure_reasons(method_rows)
        if replay_failure_reasons:
            _mark_record_failed(record, replay_failure_reasons)
            _write_node_benchmark_metrics(
                record=record,
                benchmark_policy_name=benchmark_policy_name,
                benchmark_result=benchmark_result,
                summary_metrics=summary_metrics,
                method_rows=method_rows,
                ranking_score=None,
                status="failed",
                failure_reasons=replay_failure_reasons,
            )
            continue

        _refresh_node_novelty_report(record, method_rows)
        clone_reasons = [
            *_post_refresh_novelty_reasons(record),
            *_behavioral_clone_reasons(
                method_rows,
                source_baseline=record.get("benchmark_method"),
            ),
        ]
        if clone_reasons:
            _mark_record_failed(record, clone_reasons)
            _write_node_benchmark_metrics(
                record=record,
                benchmark_policy_name=benchmark_policy_name,
                benchmark_result=benchmark_result,
                summary_metrics=summary_metrics,
                method_rows=method_rows,
                ranking_score=None,
                status="failed",
                failure_reasons=clone_reasons,
            )
            continue
        ranking_score = _ranking_score(ranking_metrics)
        _write_node_benchmark_metrics(
            record=record,
            benchmark_policy_name=benchmark_policy_name,
            benchmark_result=benchmark_result,
            summary_metrics=summary_metrics,
            method_rows=method_rows,
            ranking_score=ranking_score,
            status="completed",
            failure_reasons=[],
        )
        ranking.append(
            {
                "node_id": record["node_id"],
                "workspace": record["workspace"],
                "benchmark_method": record.get("benchmark_method"),
                "benchmark_policy_name": benchmark_policy_name,
                "ranking_score": ranking_score,
                "summary_rank": _optional_float(ranking_metrics.get("rank")),
                "benchmark_metrics": ranking_metrics,
            }
        )

    ranking.sort(
        key=lambda row: (
            row.get("summary_rank") is not None,
            -float(row["summary_rank"]) if row.get("summary_rank") is not None else row["ranking_score"],
            row["ranking_score"],
            -_method_priority(row["benchmark_method"]),
            row["node_id"],
        ),
        reverse=True,
    )
    return {
        "run_id": benchmark_result["run_id"],
        "rounds": rounds,
        "results_path": benchmark_result["benchmark_results_path"],
        "ablation_results_path": benchmark_result["ablation_results_path"],
        "summary_results_path": benchmark_result["summary_results_path"],
        "summary": benchmark_result.get("summary", {}),
        "ranking": ranking,
    }


def _attach_selected_full_chain_artifacts(root: Path, record: dict[str, Any]) -> None:
    artifacts = record.get("artifacts") if isinstance(record.get("artifacts"), dict) else None
    workspace_value = record.get("workspace")
    if artifacts is None or not workspace_value:
        return

    workspace = Path(str(workspace_value))
    state = _read_optional_json(root / "state" / "design_state.json")
    if not isinstance(state, dict):
        state = {"project_id": root.name}
    else:
        state = dict(state)
        state.setdefault("project_id", root.name)
    evidence = _read_optional_json(root / "state" / "evidence_cards.json")
    if not isinstance(evidence, list):
        evidence = []

    from design_scientist.candidates import build_design_space, generate_candidate_pool

    strategy = str(record.get("benchmark_method") or record.get("planned_method") or "mechanism_aware")
    design_space = build_design_space(state, evidence)
    pool = generate_candidate_pool(design_space, state, evidence, strategy=strategy)
    if not pool.candidates:
        design_space, pool = _synthetic_replay_candidate_pool(root, strategy=strategy)
    diagnostics = {
        **dict(pool.diagnostics),
        "candidate_pool_id": pool.candidate_pool_id,
        "design_space_id": pool.design_space_id,
        "strategy": pool.strategy,
    }

    design_space_path = workspace / "design_space.json"
    candidate_pool_csv_path = workspace / "candidate_pool.csv"
    candidate_pool_jsonl_path = workspace / "candidate_pool.jsonl"
    diagnostics_path = workspace / "candidate_pool_diagnostics.json"

    write_json(design_space_path, design_space)
    _write_candidate_pool_csv(candidate_pool_csv_path, pool.candidates)
    _write_jsonl(candidate_pool_jsonl_path, [to_plain_data(candidate) for candidate in pool.candidates])
    write_json(diagnostics_path, diagnostics)

    artifacts.update(
        {
            "design_space": str(design_space_path),
            "candidate_pool_csv": str(candidate_pool_csv_path),
            "candidate_pool_jsonl": str(candidate_pool_jsonl_path),
            "candidate_pool_diagnostics": str(diagnostics_path),
        }
    )
    _merge_json_artifact(
        Path(artifacts.get("candidate_policy", workspace / "candidate_policy.json")),
        {
            "design_space_id": pool.design_space_id,
            "candidate_pool_id": pool.candidate_pool_id,
            "generator_limited_diagnostics": diagnostics,
        },
    )


def _write_candidate_pool_csv(path: Path, candidates: list[Any]) -> None:
    rows = [_candidate_artifact_row(candidate) for candidate in candidates]
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["candidate_id"]

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _synthetic_replay_candidate_pool(root: Path, *, strategy: str) -> tuple[Any, Any]:
    """Build explicit benchmark-scope candidates when no real project pool exists."""

    from itertools import combinations

    from design_scientist.candidates import candidate_pool_diagnostics
    from design_scientist.schemas import (
        Candidate,
        CandidateLineage,
        CandidatePool,
        DesignOperatorSpec,
        DesignSpace,
    )
    from design_scientist.synthetic_replay import (
        BACKGROUND_ORDER,
        ENDPOINTS,
        FORBIDDEN_DESIGN_PAIRS,
        MAX_MODULES,
        MODULES,
        TARGET_BACKGROUND,
    )

    design_space_id = f"{root.name}:synthetic_replay:{strategy}"
    design_space = DesignSpace(
        design_space_id=design_space_id,
        project_id=root.name,
        target_systems=["synthetic_replay"],
        modules=list(MODULES),
        backgrounds=list(BACKGROUND_ORDER),
        operators=[
            DesignOperatorSpec(
                operator_id="synthetic_replay_variant",
                category="lattice_repair",
                description=(
                    "Benchmark-scope synthetic replay variant. These records document "
                    "the replay design space and are not wet-lab executable candidates."
                ),
                required_endpoints=list(ENDPOINTS),
                default_cost=1.0,
            )
        ],
        constraints={
            "artifact_scope": "synthetic_replay_benchmark",
            "wet_lab_executable": False,
            "forbidden_design_pairs": [
                "+".join(pair) for pair in sorted(FORBIDDEN_DESIGN_PAIRS)
            ],
        },
        evidence_refs=["synthetic_replay_config"],
        config={
            "source": "synthetic_replay_benchmark",
            "candidate_artifact_note": (
                "Generated because this framework run has no real project design state."
            ),
        },
    )

    candidates: list[Candidate] = []
    forbidden_pairs = [set(pair) for pair in FORBIDDEN_DESIGN_PAIRS]
    for background in BACKGROUND_ORDER:
        for module_count in range(MAX_MODULES + 1):
            for modules in combinations(MODULES, module_count):
                module_set = set(modules)
                if any(pair.issubset(module_set) for pair in forbidden_pairs):
                    continue
                suffix = "WT" if not modules else "_".join(modules)
                candidate_id = f"{background}__{suffix}"
                category = (
                    "control"
                    if not modules
                    else "lattice_repair"
                    if len(modules) == 1
                    else "interaction_square"
                )
                operator_args = {
                    "background": background,
                    "modules": list(modules),
                    "benchmark": "synthetic_multi_round_wet_lab_replay",
                }
                candidates.append(
                    Candidate(
                        candidate_id=candidate_id,
                        variant_id=candidate_id,
                        design_space_id=design_space_id,
                        operator="synthetic_replay_variant",
                        category=category,
                        target_system="synthetic_replay",
                        background=background,
                        target_background=TARGET_BACKGROUND,
                        design_context={
                            "artifact_scope": "synthetic_replay_benchmark",
                            "wet_lab_executable": False,
                            "benchmark_background": background,
                            "target_background": TARGET_BACKGROUND,
                        },
                        modules=list(modules),
                        rationale=(
                            "Synthetic replay candidate used to audit policy behavior "
                            "when no real project candidate pool has been initialized."
                        ),
                        evidence_refs=["synthetic_replay_config"],
                        required_measurements=list(ENDPOINTS),
                        required_endpoints=list(ENDPOINTS),
                        operator_args=operator_args,
                        lineage=CandidateLineage(
                            operator="synthetic_replay_variant",
                            operator_args=operator_args,
                            parent_ids=[background],
                            source_refs=["synthetic_replay_config"],
                        ),
                        parent_ids=[background],
                        source_refs=["synthetic_replay_config"],
                        cost=1.0,
                    )
                )

    pool = CandidatePool(
        candidate_pool_id=f"{design_space_id}:candidate_pool",
        design_space_id=design_space_id,
        strategy=strategy,
        candidates=candidates,
    )
    pool.diagnostics = {
        **candidate_pool_diagnostics(pool),
        "source": "synthetic_replay_benchmark",
        "real_project_candidate_pool_missing": True,
        "wet_lab_executable": False,
    }
    return design_space, pool


def _candidate_artifact_row(candidate: Any) -> dict[str, Any]:
    row = to_plain_data(candidate)
    if not isinstance(row, dict):
        row = {"candidate_id": str(candidate)}
    for key, value in list(row.items()):
        if isinstance(value, (dict, list)):
            row[key] = json.dumps(value, sort_keys=True)
    return row


def _write_jsonl(path: Path, rows: list[Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=False) + "\n")


def _merge_json_artifact(path: Path, updates: dict[str, Any]) -> None:
    try:
        data = read_json(path)
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    data.update(updates)
    write_json(path, data)


def _synthetic_replay_failure_reasons(method_rows: list[dict[str, Any]]) -> list[str]:
    failed_rows = [
        row
        for row in method_rows
        if str(row.get("status") or "completed") != "completed"
    ]
    if not failed_rows:
        return []
    details = _unique_strings(
        str(row.get("error") or row.get("failure_reason") or "unknown replay error")
        for row in failed_rows
    )
    detail_text = "; ".join(details[:3])
    suffix = f": {detail_text}" if detail_text else ""
    return [
        "synthetic replay failed "
        f"in {len(failed_rows)}/{len(method_rows)} worlds{suffix}"
    ]


def _write_node_benchmark_metrics(
    *,
    record: dict[str, Any],
    benchmark_policy_name: Any,
    benchmark_result: dict[str, Any],
    summary_metrics: dict[str, Any],
    method_rows: list[dict[str, Any]],
    ranking_score: float | None,
    status: str,
    failure_reasons: list[str],
) -> None:
    synthetic_replay = summary_metrics or (dict(method_rows[-1]) if method_rows else {})
    node_metrics = {
        "node_id": record["node_id"],
        "method": benchmark_policy_name,
        "source_baseline_family": record.get("benchmark_method"),
        "status": status,
        "synthetic_replay": synthetic_replay,
        "synthetic_replay_rows": method_rows,
        "synthetic_replay_summary": summary_metrics,
        "ranking_score": ranking_score,
        "benchmark_results_path": benchmark_result["benchmark_results_path"],
        "ablation_results_path": benchmark_result["ablation_results_path"],
        "summary_results_path": benchmark_result["summary_results_path"],
    }
    if failure_reasons:
        node_metrics["failure_reasons"] = list(failure_reasons)
    write_json(Path(record["artifacts"]["benchmark_metrics"]), node_metrics)


def _refresh_node_novelty_report(record: dict[str, Any], method_rows: list[dict[str, Any]]) -> None:
    artifacts = record.get("artifacts") if isinstance(record.get("artifacts"), dict) else {}
    path_value = artifacts.get("novelty_report")
    if not path_value or not method_rows:
        return
    path = Path(path_value)
    try:
        report = read_json(path)
    except Exception:
        report = {}
    if not isinstance(report, dict):
        report = {}

    worst_overlap = _worst_default_baseline_overlap(method_rows)
    measured = worst_overlap["max_by_baseline"]
    nearest_baseline = worst_overlap["nearest_baseline"]
    max_overlap = worst_overlap["max_overlap"]
    clone = bool(
        nearest_baseline
        and max_overlap is not None
        and max_overlap > NOVELTY_SELECTION_OVERLAP_THRESHOLD
    )
    report.update(
        {
            "baseline_clone": clone,
            "baseline_overlap": max_overlap,
            "selection_overlap_vs_baselines": max_overlap,
            "nearest_baseline": nearest_baseline,
            "nearest_baseline_world_id": worst_overlap["world_id"],
            "selection_overlap_max_by_baseline": measured,
            "novelty_score": _mean_metric(method_rows, "novelty_score"),
            "synthetic_replay_worlds": len({str(row.get("world_id")) for row in method_rows}),
            "status": "measured_synthetic_replay",
        }
    )
    write_json(path, report)


def _worst_default_baseline_overlap(method_rows: list[dict[str, Any]]) -> dict[str, Any]:
    default_baselines = set(DEFAULT_METHODS)
    max_by_baseline: dict[str, float] = {}
    nearest_baseline: str | None = None
    nearest_world_id: str | None = None
    max_overlap: float | None = None
    for row in method_rows:
        if str(row.get("status") or "completed") != "completed":
            continue
        for key, raw_value in row.items():
            if not key.startswith("selection_overlap_"):
                continue
            baseline = key.removeprefix("selection_overlap_")
            if baseline not in default_baselines:
                continue
            value = _optional_float(raw_value)
            if value is None:
                continue
            previous = max_by_baseline.get(baseline)
            if previous is None or value > previous:
                max_by_baseline[baseline] = value
            if max_overlap is None or value > max_overlap:
                max_overlap = value
                nearest_baseline = baseline
                nearest_world_id = str(row.get("world_id")) if row.get("world_id") is not None else None
    return {
        "max_by_baseline": max_by_baseline,
        "nearest_baseline": nearest_baseline,
        "world_id": nearest_world_id,
        "max_overlap": max_overlap,
    }


def _post_refresh_novelty_reasons(record: dict[str, Any]) -> list[str]:
    artifacts = record.get("artifacts") if isinstance(record.get("artifacts"), dict) else {}
    path_value = artifacts.get("novelty_report")
    if not path_value:
        return []
    try:
        report = read_json(Path(path_value))
    except Exception:
        return []
    if not isinstance(report, dict):
        return []

    reasons: list[str] = []
    if report.get("baseline_clone") is True:
        reasons.append("post-refresh novelty guard failed: novelty_report marks baseline_clone true")
    overlap = _optional_float(report.get("selection_overlap_vs_baselines"))
    if overlap is not None and overlap > NOVELTY_SELECTION_OVERLAP_THRESHOLD:
        reasons.append(
            "post-refresh novelty guard failed: selection_overlap_vs_baselines "
            f"{overlap:.3f} exceeds {NOVELTY_SELECTION_OVERLAP_THRESHOLD:.3f}"
        )
    wrapper_pattern = str(
        report.get("wrapper_pattern")
        or report.get("clone_pattern")
        or report.get("policy_pattern")
        or ""
    ).lower()
    if report.get("tail_swap_baseline_wrapper") is True or (
        "tail" in wrapper_pattern and "swap" in wrapper_pattern
    ):
        reasons.append("post-refresh novelty guard failed: tail-swap baseline wrapper")
    return reasons


def _behavioral_clone_reasons(
    method_rows: list[dict[str, Any]],
    *,
    source_baseline: Any,
) -> list[str]:
    if not isinstance(source_baseline, str) or not source_baseline:
        return []
    key = f"selection_overlap_{source_baseline}"
    overlaps = [
        value
        for row in method_rows
        if (value := _optional_float(row.get(key))) is not None
    ]
    if overlaps and min(overlaps) >= 0.999:
        return [
            "behavioral baseline clone detected: "
            f"selected policy has complete selection overlap with {source_baseline}"
        ]
    return []


def _mark_record_failed(record: dict[str, Any], reasons: list[str]) -> None:
    record["status"] = "failed"
    existing = list(record.get("failure_reasons", []))
    record["failure_reasons"] = _unique_strings([*existing, *reasons])
    contract = record.get("contract")
    if isinstance(contract, dict):
        contract["valid"] = False
        contract["errors"] = _unique_strings([*contract.get("errors", []), *reasons])


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        value
        for row in rows
        if (value := _optional_float(row.get(key))) is not None
    ]
    if not values:
        return None
    return round(sum(values) / len(values), 9)


def _ranking_score(metrics: dict[str, Any]) -> float:
    best = float(metrics.get("best_feasible_utility", metrics.get("mean_best_feasible_utility", 0.0)))
    hit_rate = float(metrics.get("hit_rate", metrics.get("mean_hit_rate", 0.0)))
    coverage = float(metrics.get("evidence_coverage", metrics.get("mean_evidence_coverage", 0.0)))
    efficiency = float(metrics.get("round_efficiency", metrics.get("mean_round_efficiency", 0.0)))
    regret = float(metrics.get("regret_proxy", metrics.get("mean_regret_proxy", 0.0)))
    false_claim_rate = float(metrics.get("false_claim_rate", metrics.get("mean_false_claim_rate", 0.0)))
    return round(
        best
        + 0.20 * hit_rate
        + 0.10 * coverage
        + 0.10 * efficiency
        - 0.30 * regret
        - 0.15 * false_claim_rate,
        9,
    )


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _method_priority(method_name: Any) -> int:
    if not isinstance(method_name, str):
        return len(NODE_METHOD_ORDER)
    try:
        return NODE_METHOD_ORDER.index(method_name)
    except ValueError:
        return len(NODE_METHOD_ORDER)


def _policy_callable_from_execution(
    execution: MethodNodeExecutionResult,
    *,
    node_id: str,
    fallback_policy_name: str,
) -> tuple[policy_api.PolicyCallable | None, str | None]:
    if not execution.valid:
        return None, None

    candidate_policy = execution.artifacts.get("candidate_policy.json") or {}
    entrypoint = candidate_policy.get("policy_entrypoint") or candidate_policy.get("entrypoint") or "select_batch"
    if not isinstance(entrypoint, str) or not entrypoint.strip():
        return None, "candidate_policy.json policy_entrypoint must be a non-empty string"

    raw_policy_name = (
        candidate_policy.get("policy_id")
        or candidate_policy.get("policy_name")
        or fallback_policy_name
    )
    policy_name = _safe_policy_name(raw_policy_name, node_id=node_id, fallback=fallback_policy_name)
    try:
        attribute = _entrypoint_attribute(entrypoint)
    except RuntimeError as exc:
        return None, str(exc)
    policy_callable = execution.callables.get(attribute)
    if not callable(policy_callable):
        return None, f"method.py does not define callable policy {entrypoint!r}"
    return _named_policy_callable(policy_callable, policy_name=policy_name), None


def _entrypoint_attribute(entrypoint: str) -> str:
    if ":" not in entrypoint and "." not in entrypoint:
        return entrypoint
    separator = ":" if ":" in entrypoint else "."
    module_name, function_name = entrypoint.rsplit(separator, 1)
    if module_name not in {"method", "method.py", ""}:
        raise RuntimeError(f"Unsupported policy entrypoint module {module_name!r}; expected 'method'.")
    if not function_name:
        raise RuntimeError("policy entrypoint function must be non-empty")
    return function_name


def _named_policy_callable(
    policy_callable: policy_api.PolicyCallable,
    *,
    policy_name: str,
) -> policy_api.PolicyCallable:
    def node_policy(
        observed: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
        budget: int,
        round_index: int,
        rng: random.Random,
    ) -> list[str]:
        return list(policy_callable(observed, candidates, budget, round_index, rng))

    node_policy.__name__ = policy_name
    node_policy.policy_name = policy_name  # type: ignore[attr-defined]
    return node_policy


def _safe_policy_name(raw_name: Any, *, node_id: str, fallback: str) -> str:
    text = str(raw_name or fallback).strip()
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in text).strip("._-")
    if not safe:
        safe = fallback
    if safe != node_id and not safe.startswith(f"{node_id}_"):
        safe = f"{node_id}_{safe}"
    return safe


def _callable_policy_name(policy_callable: policy_api.PolicyCallable | None) -> str | None:
    if policy_callable is None:
        return None
    value = getattr(policy_callable, "policy_name", None) or getattr(policy_callable, "__name__", None)
    return str(value) if value else None


def _node_trace(
    *,
    record: dict[str, Any],
    benchmark: dict[str, Any],
    journal_path: Path,
    stage_progress_path: Path,
    route_tree_path: Path,
) -> dict[str, Any]:
    artifacts = record.get("artifacts") if isinstance(record.get("artifacts"), dict) else {}
    trace = {
        "proposal": artifacts.get("proposal"),
        "benchmark_metrics": artifacts.get("benchmark_metrics"),
        "ablation_results": benchmark.get("ablation_results_path"),
        "journal": str(journal_path),
        "stage_progress": str(stage_progress_path),
        "route_tree": str(route_tree_path),
        "novelty_report": artifacts.get("novelty_report"),
        "validation_report": artifacts.get("validation_report"),
    }
    for key in (
        "design_space",
        "candidate_pool_csv",
        "candidate_pool_jsonl",
        "candidate_pool_diagnostics",
    ):
        if artifacts.get(key):
            trace[key] = artifacts[key]
    return trace


def _stage_progress_payload(
    *,
    run_id: str,
    run_dir: Path,
    node_records: list[dict[str, Any]],
    selected_node_id: str | None,
    benchmark: dict[str, Any],
    journal_path: Path,
    route_tree_path: Path,
) -> dict[str, Any]:
    failed_count = sum(1 for record in node_records if record.get("status") != "completed")
    stage_artifacts = {
        "baseline_reproduction": [benchmark.get("results_path")],
        "literature_grounded_ideation": [str(run_dir / "nodes")],
        "policy_implementation": [record.get("workspace") for record in node_records],
        "debug_and_repair": [record.get("workspace") for record in node_records if record.get("status") != "completed"],
        "creative_research": [
            record.get("artifacts", {}).get("proposal")
            for record in node_records
            if isinstance(record.get("artifacts"), dict)
        ],
        "ablation_and_stress": [benchmark.get("ablation_results_path")],
        "selection_and_report": [str(journal_path), str(route_tree_path)],
    }
    stages: list[dict[str, Any]] = []
    for stage in SCIENTIST_STAGES:
        status = "completed"
        if stage == "selection_and_report" and selected_node_id is None:
            status = "failed"
        stages.append(
            {
                "stage": stage,
                "status": status,
                "artifacts": [
                    str(path)
                    for path in stage_artifacts.get(stage, [])
                    if path
                ],
            }
        )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "stage_sequence": list(SCIENTIST_STAGES),
        "stages": stages,
        "node_count": len(node_records),
        "failed_node_count": failed_count,
        "selected_node_id": selected_node_id,
    }


def _route_tree_payload(
    *,
    run_id: str,
    run_dir: Path,
    node_records: list[dict[str, Any]],
    selected_node_id: str | None,
) -> dict[str, Any]:
    route_nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    previous_id = "root"
    for record in node_records:
        node_id = str(record["node_id"])
        route_nodes.append(
            {
                "node_id": node_id,
                "parent_id": previous_id,
                "stage": "creative_research",
                "workspace": record.get("workspace"),
                "status": record.get("status"),
                "ranking_score": record.get("ranking_score"),
                "selected": node_id == selected_node_id,
                "artifacts": record.get("artifacts", {}),
                "failure_reasons": record.get("failure_reasons", []),
            }
        )
        edges.append({"source": previous_id, "target": node_id})
        previous_id = node_id

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "stage_sequence": list(SCIENTIST_STAGES),
        "root": {
            "node_id": "root",
            "stage": "baseline_reproduction",
            "status": "completed",
        },
        "nodes": route_nodes,
        "edges": edges,
        "selected_node_id": selected_node_id,
    }


def _append_research_os_memory(
    *,
    root: Path,
    run_id: str,
    selected_node: dict[str, Any] | None,
    failed_nodes: list[dict[str, Any]],
) -> list[str]:
    try:
        from design_scientist.research_os import append_failure, append_finding
    except Exception as exc:  # pragma: no cover - import failure defensive path.
        return [f"research_os import failed: {type(exc).__name__}: {exc}"]

    errors: list[str] = []
    if selected_node is not None:
        try:
            append_finding(
                root,
                {
                    "summary": f"Selected method node {selected_node.get('node_id')} for run {run_id}.",
                    "run_id": run_id,
                    "node_id": selected_node.get("node_id"),
                    "benchmark_method": selected_node.get("benchmark_method"),
                    "ranking_score": selected_node.get("ranking_score"),
                    "trace": selected_node.get("trace"),
                },
            )
        except Exception as exc:  # pragma: no cover - append failure defensive path.
            errors.append(f"append_finding failed: {type(exc).__name__}: {exc}")

    for failed in failed_nodes:
        try:
            append_failure(
                root,
                {
                    "summary": f"Method node {failed.get('node_id')} failed during run {run_id}.",
                    "run_id": run_id,
                    "node_id": failed.get("node_id"),
                    "reasons": failed.get("reasons", []),
                },
            )
        except Exception as exc:  # pragma: no cover - append failure defensive path.
            errors.append(f"append_failure failed: {type(exc).__name__}: {exc}")
    return errors


def _benchmark_method_from_execution(
    execution: MethodNodeExecutionResult,
    *,
    fallback: str,
) -> str | None:
    artifacts = execution.artifacts
    for artifact_name in (
        "candidate_policy.json",
        "proposal.json",
        "manifest.json",
        "benchmark_metrics.json",
    ):
        data = artifacts.get(artifact_name)
        if not isinstance(data, dict):
            continue
        for key in (
            "source_baseline_family",
            "baseline_family",
            "planned_method",
            "method",
            "synthetic_replay_method",
            "policy_id",
        ):
            value = data.get(key)
            if value in DEFAULT_METHODS:
                return str(value)
    if fallback in DEFAULT_METHODS:
        return fallback
    return None


def _failure_reasons(
    execution: MethodNodeExecutionResult,
    *,
    benchmark_method: str | None,
    policy_error: str | None,
    allow_no_baseline: bool = False,
) -> list[str]:
    reasons: list[str] = []
    if not execution.valid:
        if not execution.executed:
            reasons.append("method.py did not execute")
        if execution.missing_artifacts:
            reasons.append(f"missing artifacts: {', '.join(execution.missing_artifacts)}")
        if execution.malformed_json:
            malformed = ", ".join(sorted(execution.malformed_json))
            reasons.append(f"malformed JSON artifacts: {malformed}")
        if execution.out_of_bounds_writes:
            reasons.append("out-of-bounds writes detected")
        if execution.escaping_symlinks:
            reasons.append("escaping symlinks detected")
        if execution.exception:
            reasons.append("runtime exception during method node execution")
        reasons.extend(execution.errors)
    if benchmark_method is None and not allow_no_baseline:
        reasons.append("no supported baseline family declared")
    if policy_error:
        reasons.append(policy_error)
    return _unique_strings(reason for reason in reasons if reason)


def _node_record(
    *,
    node_id: str,
    workspace: Path,
    planned_method: str,
    benchmark_method: str | None,
    benchmark_policy_name: str | None,
    status: str,
    execution: MethodNodeExecutionResult,
    agent_result: dict[str, Any] | None,
    reasons: list[str],
) -> dict[str, Any]:
    artifacts = {
        "proposal": str(workspace / "proposal.json"),
        "method": str(workspace / "method.py"),
        "manifest": str(workspace / "manifest.json"),
        "candidate_policy": str(workspace / "candidate_policy.json"),
        "novelty_report": str(workspace / "novelty_report.json"),
        "benchmark_metrics": str(workspace / "benchmark_metrics.json"),
        "validation_report": str(workspace / "validation_report.json"),
    }
    return {
        "node_id": node_id,
        "workspace": str(workspace),
        "planned_method": planned_method,
        "benchmark_method": benchmark_method,
        "benchmark_policy_name": benchmark_policy_name,
        "status": status,
        "selected": False,
        "artifacts": artifacts,
        "trace": {},
        "contract": {
            "valid": execution.valid,
            "executed": execution.executed,
            "missing_artifacts": list(execution.missing_artifacts),
            "malformed_json": dict(execution.malformed_json),
            "out_of_bounds_writes": list(execution.out_of_bounds_writes),
            "escaping_symlinks": list(execution.escaping_symlinks),
            "errors": list(execution.errors),
            "exception": execution.exception,
        },
        "agent": agent_result,
        "failure_reasons": reasons,
        "benchmark_metrics": None,
        "ranking_score": None,
    }


def _literature_snapshot(
    project_dir: Path,
    *,
    max_papers: int | None = None,
    status: str = "not_run",
    error: str | None = None,
) -> dict[str, Any]:
    path = project_dir / "framework" / "paper_cards.json"
    cards: list[dict[str, Any]] = []
    if path.exists():
        loaded = read_json(path)
        if isinstance(loaded, list):
            cards = [card for card in loaded if isinstance(card, dict)]
            status = "available" if status == "not_run" else status
    limit = max_papers if max_papers is not None else len(cards)
    limited_cards = cards[: max(0, limit)]
    return {
        "status": status,
        "path": str(path),
        "paper_count": len(cards),
        "max_papers": max_papers,
        "cards": [_paper_snapshot(card) for card in limited_cards],
        "error": error,
    }


def _paper_snapshot(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": card.get("paper_id") or card.get("id") or card.get("source_id"),
        "title": card.get("title"),
        "year": card.get("year"),
        "venue": card.get("venue"),
        "relevance": card.get("relevance"),
        "tags": card.get("tags", []),
    }


def _planned_methods(nodes: int) -> list[str]:
    methods = list(NODE_METHOD_ORDER)
    return [methods[index % len(methods)] for index in range(nodes)]


def _safe_run_id(run_id: str) -> str:
    if not run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a non-empty relative name")
    if "/" in run_id or "\\" in run_id or ".." in Path(run_id).parts:
        raise ValueError("run_id must not contain path separators or '..'")
    return run_id


def _timestamp_run_id() -> str:
    return time.strftime("scientist_%Y%m%d_%H%M%S")


def _unique_strings(values: Any) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


__all__ = [
    "METHOD_NODE_OUTPUT_SCHEMA",
    "SCIENTIST_STAGES",
    "develop_method",
    "run_scientist_search",
]
