"""Markdown method report generation for framework R&D runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from design_scientist.framework_validation import resolve_framework_run
from design_scientist.io import ensure_dir, read_json, read_yaml


def write_method_report(project_dir: str | Path, run_id: str | None = None) -> Path:
    """Write ``runs/<run_id>/method_report.md`` for a framework R&D run."""

    root, selected_run_id, run_dir = resolve_framework_run(project_dir, run_id=run_id)
    if run_dir is None or selected_run_id is None or not run_dir.is_dir():
        requested = run_id or "<latest>"
        raise FileNotFoundError(f"No framework run directory found for run_id={requested!r}")

    framework_dir = root / "framework"
    spec = _load_yaml(framework_dir / "framework_spec.yaml")
    reference_audit_text = _read_text(framework_dir / "reference_audit.md")
    reference_components = _load_json(framework_dir / "reference_components.json")
    quest = _load_yaml(framework_dir / "quest.yaml")
    research_map = _load_json(framework_dir / "research_map.json")
    findings_memory = _read_jsonl(framework_dir / "findings_memory.jsonl")
    failure_memory = _read_jsonl(framework_dir / "failure_memory.jsonl")
    literature_queries = _load_yaml(framework_dir / "literature_queries.yaml").get("queries", [])
    literature_plan = _load_json(framework_dir / "literature_search_plan.json")
    literature_trace = _load_json(framework_dir / "literature_search_trace.json")
    citation_graph = _load_json(framework_dir / "citation_graph.json")
    paper_scores = _read_csv(framework_dir / "paper_scores.csv")
    paper_cards = _as_list(_load_json(framework_dir / "paper_cards.json"))
    method_modules_data = _load_json(framework_dir / "method_modules.json")
    method_modules = _as_list(
        method_modules_data.get("method_modules") or method_modules_data.get("modules")
        if isinstance(method_modules_data, dict)
        else []
    )
    registry = _load_yaml(framework_dir / "method_registry.yaml")
    method_hypotheses_text = _read_text(framework_dir / "method_hypotheses.md")
    algorithm_spec_text = _read_text(framework_dir / "algorithm_spec.md")
    benchmark_rows = _read_csv(run_dir / "benchmark_results.csv")
    benchmark_summary_rows = _read_csv(run_dir / "benchmark_summary.csv")
    ablation_rows = _read_csv(run_dir / "ablation_results.csv")
    stage_progress = _load_json(run_dir / "stage_progress.json")
    route_tree = _load_json(run_dir / "route_tree.json")
    journal = _load_json(run_dir / "scientist_journal.json")
    state = _load_json(root / "state" / "design_state.json")
    policy_metrics = _load_json(run_dir / "policy_metrics.json")

    artifacts = _artifact_paths(root, run_dir, journal)
    selected_node = _selected_node_record(journal)
    selected_summary = _selected_node_summary(journal, selected_node)
    selected_benchmark_summary = _selected_benchmark_summary(benchmark_summary_rows, selected_summary)
    selected_proposal = _selected_json_artifact(root, selected_node, "proposal", "proposal.json")
    selected_novelty_report = _selected_json_artifact(root, selected_node, "novelty_report", "novelty_report.json")
    selected_candidate_policy = _selected_json_artifact(
        root,
        selected_node,
        "candidate_policy",
        "candidate_policy.json",
    )
    failed_nodes = _failed_nodes(journal)
    unsupported_claims = _unsupported_claims(
        state=state,
        policy_metrics=policy_metrics,
        paper_cards=paper_cards,
        journal=journal,
    )
    next_questions = _next_development_questions(
        method_modules=method_modules,
        failed_nodes=failed_nodes,
        ablation_rows=ablation_rows,
        benchmark_rows=benchmark_rows,
    )

    lines: list[str] = [
        f"# Method Report: {selected_run_id}",
        "",
        f"Project directory: `{root}`",
        f"Run directory: `{_rel(run_dir, root)}`",
        f"Framework: `{spec.get('framework_id', root.name)}`",
        f"Domain: {spec.get('domain', 'not recorded')}",
        "",
        "This report summarizes framework-method evidence. It does not convert synthetic replay results into wet-lab claims.",
        "",
    ]

    lines.extend(_reference_basis_section(root, reference_audit_text, reference_components))
    lines.extend(_research_os_section(root, quest, research_map, findings_memory, failure_memory))
    lines.extend(
        _literature_basis_section(
            root,
            literature_queries,
            literature_plan,
            literature_trace,
            citation_graph,
            paper_scores,
            paper_cards,
            method_modules,
        )
    )
    lines.extend(_method_hypotheses_section(root, method_hypotheses_text, method_modules))
    lines.extend(_algorithm_mechanism_section(root, algorithm_spec_text, registry))
    lines.extend(_stage_progress_section(root, run_dir, stage_progress))
    lines.extend(_route_tree_section(root, run_dir, route_tree))
    lines.extend(_baseline_comparison_section(root, run_dir, benchmark_rows, selected_summary))
    lines.extend(
        _no_baseline_rationale_section(selected_proposal, selected_novelty_report, selected_candidate_policy)
    )
    lines.extend(_architecture_novelty_section(selected_proposal, selected_novelty_report))
    lines.extend(_benchmark_summary_section(root, run_dir, benchmark_summary_rows))
    lines.extend(_ablation_section(root, run_dir, ablation_rows))
    lines.extend(
        _selected_method_section(
            root,
            selected_summary,
            selected_benchmark_summary,
            selected_node,
            selected_proposal,
            selected_novelty_report,
        )
    )
    lines.extend(
        _generator_limited_diagnostics_section(
            selected_proposal,
            selected_novelty_report,
            selected_candidate_policy,
        )
    )
    lines.extend(_failed_methods_section(failed_nodes))
    lines.extend(_unsupported_claims_section(unsupported_claims))
    lines.extend(_next_questions_section(next_questions))
    lines.extend(_artifact_paths_section(artifacts))

    out = run_dir / "method_report.md"
    ensure_dir(out.parent)
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return out


def _reference_basis_section(
    root: Path,
    reference_audit_text: str,
    reference_components: Any,
) -> list[str]:
    lines = [
        "## Reference Basis",
        "",
        f"Artifacts: `{_rel(root / 'framework' / 'reference_audit.md', root)}`, `{_rel(root / 'framework' / 'reference_components.json', root)}`",
        "",
    ]
    if isinstance(reference_components, dict):
        summary = reference_components.get("summary") if isinstance(reference_components.get("summary"), dict) else {}
        components = _as_list(reference_components.get("components"))
        if summary:
            lines.append(
                "Reference Audit summary: "
                f"{summary.get('present', 'n/a')}/{summary.get('total', 'n/a')} components present; "
                f"{summary.get('missing', 'n/a')} missing."
            )
        if components:
            lines.extend(["", "Audited components:"])
            for component in components[:8]:
                lines.append(
                    f"- `{component.get('component_id', 'component')}`: "
                    f"{component.get('status', 'unknown')} - {_one_line(component.get('design_scientist_use'))}"
                )
            if len(components) > 8:
                lines.append(f"- Additional audited components omitted from this summary: {len(components) - 8}")
    elif reference_audit_text:
        lines.append("Reference Audit Markdown was present, but component JSON was not loaded.")
    else:
        lines.append("Reference Audit has not been run for this project.")
    lines.append("")
    return lines


def _research_os_section(
    root: Path,
    quest: dict[str, Any],
    research_map: Any,
    findings_memory: list[dict[str, Any]],
    failure_memory: list[dict[str, Any]],
) -> list[str]:
    lines = [
        "## Research OS",
        "",
        f"Artifacts: `{_rel(root / 'framework' / 'quest.yaml', root)}`, `{_rel(root / 'framework' / 'research_map.json', root)}`, `{_rel(root / 'framework' / 'findings_memory.jsonl', root)}`, `{_rel(root / 'framework' / 'failure_memory.jsonl', root)}`",
        "",
    ]
    if quest:
        lines.append(
            f"Quest `{quest.get('quest_id', 'unknown')}` is {quest.get('status', 'unknown')} "
            f"with active loop `{quest.get('active_loop', 'unknown')}`."
        )
    else:
        lines.append("Quest metadata was not loaded.")

    loops = research_map.get("loops") if isinstance(research_map, dict) else []
    if isinstance(loops, list) and loops:
        lines.extend(["", "Research map loops:"])
        for loop in loops[:8]:
            if isinstance(loop, dict):
                lines.append(
                    f"- `{loop.get('loop_id', 'loop')}`: {loop.get('status', 'unknown')} - {_one_line(loop.get('goal'))}"
                )
    else:
        lines.extend(["", "Research map loops were not loaded."])

    lines.extend(["", "Memory summary:"])
    lines.append(f"- Findings memory records: {len(findings_memory)}")
    for record in findings_memory[:3]:
        lines.append(f"  - {_one_line(record.get('summary'))}")
    lines.append(f"- Failure memory records: {len(failure_memory)}")
    for record in failure_memory[:3]:
        lines.append(f"  - {_one_line(record.get('summary'))}")
    lines.append("")
    return lines


def _literature_basis_section(
    root: Path,
    literature_queries: Any,
    literature_plan: Any,
    literature_trace: Any,
    citation_graph: Any,
    paper_scores: list[dict[str, str]],
    paper_cards: list[dict[str, Any]],
    method_modules: list[dict[str, Any]],
) -> list[str]:
    lines = [
        "## Literature Basis",
        "",
        f"Artifacts: `{_rel(root / 'framework' / 'literature_queries.yaml', root)}`, `{_rel(root / 'framework' / 'literature_search_plan.json', root)}`, `{_rel(root / 'framework' / 'literature_search_trace.json', root)}`, `{_rel(root / 'framework' / 'citation_graph.json', root)}`, `{_rel(root / 'framework' / 'paper_scores.csv', root)}`, `{_rel(root / 'framework' / 'paper_cards.json', root)}`, `{_rel(root / 'framework' / 'literature_map.md', root)}`",
        "",
        f"Paper cards reviewed: {len(paper_cards)}",
        f"Paper score rows: {len(paper_scores)}",
        "",
        "Queries:",
    ]
    queries = literature_queries if isinstance(literature_queries, list) else []
    plan_queries = literature_plan.get("queries") if isinstance(literature_plan, dict) else []
    if isinstance(plan_queries, list) and plan_queries:
        queries = plan_queries
    if queries:
        for query in queries[:8]:
            if not isinstance(query, dict):
                continue
            lines.append(
                f"- `{query.get('query_id', 'query')}`: {_one_line(query.get('query'))} "
                f"({query.get('purpose', 'method discovery')})"
            )
    else:
        lines.append("- No literature queries were recorded.")

    trace_events = literature_trace.get("events") if isinstance(literature_trace, dict) else []
    graph_nodes = citation_graph.get("nodes") if isinstance(citation_graph, dict) else []
    graph_edges = citation_graph.get("edges") if isinstance(citation_graph, dict) else []
    lines.extend(
        [
            "",
            "Literature search trace:",
            f"- Trace events: {len(trace_events) if isinstance(trace_events, list) else 0}",
            f"- Citation graph nodes: {len(graph_nodes) if isinstance(graph_nodes, list) else 0}",
            f"- Citation graph edges: {len(graph_edges) if isinstance(graph_edges, list) else 0}",
        ]
    )

    lines.extend(["", "Paper cards:"])
    if paper_cards:
        for card in paper_cards[:12]:
            title = card.get("title") or card.get("citation") or card.get("paper_id", "untitled")
            bits = [str(item) for item in (card.get("source"), card.get("year"), card.get("venue")) if item]
            suffix = f" ({', '.join(bits)})" if bits else ""
            lines.append(f"- `{card.get('paper_id', 'paper')}`: {_one_line(title)}{suffix}")
        if len(paper_cards) > 12:
            lines.append(f"- Additional paper cards omitted from this summary: {len(paper_cards) - 12}")
    else:
        lines.append("- No paper cards were loaded.")

    lines.extend(["", "Extracted method modules:"])
    if method_modules:
        for module in method_modules:
            papers = ", ".join(f"`{paper_id}`" for paper_id in module.get("source_papers", [])) or "none"
            lines.append(
                f"- `{module.get('module_id', 'module')}`: {_one_line(module.get('name'))}; "
                f"sources: {papers}"
            )
    else:
        lines.append("- No method modules were loaded.")
    lines.append("")
    return lines


def _method_hypotheses_section(
    root: Path,
    method_hypotheses_text: str,
    method_modules: list[dict[str, Any]],
) -> list[str]:
    lines = [
        "## Method Hypotheses",
        "",
        f"Artifact: `{_rel(root / 'framework' / 'method_hypotheses.md', root)}`",
        "",
    ]
    headings = _markdown_h2(method_hypotheses_text)
    if headings:
        lines.append("Recorded hypotheses:")
        lines.extend(f"- {heading}" for heading in headings)
    else:
        excerpt = _first_nonempty_lines(method_hypotheses_text, limit=5)
        if excerpt:
            lines.append("Recorded hypothesis excerpt:")
            lines.extend(f"- {_one_line(line)}" for line in excerpt)
        else:
            lines.append("No method hypotheses text was loaded.")

    module_gaps = [
        f"`{module.get('module_id')}`: {_one_line(module.get('gap_for_design_scientist'))}"
        for module in method_modules
        if module.get("gap_for_design_scientist")
    ]
    if module_gaps:
        lines.extend(["", "Literature-to-framework gaps:"])
        lines.extend(f"- {gap}" for gap in module_gaps[:8])
    lines.append("")
    return lines


def _algorithm_mechanism_section(
    root: Path,
    algorithm_spec_text: str,
    registry: dict[str, Any],
) -> list[str]:
    lines = [
        "## Algorithm Mechanism",
        "",
        f"Artifacts: `{_rel(root / 'framework' / 'algorithm_spec.md', root)}`, `{_rel(root / 'framework' / 'method_registry.yaml', root)}`",
        "",
    ]
    algorithm_id = _extract_line_value(algorithm_spec_text, "Algorithm id:")
    if algorithm_id:
        lines.append(f"Algorithm id: {algorithm_id}")
    sections = _markdown_h2(algorithm_spec_text)
    if sections:
        lines.append(f"Algorithm sections present: {', '.join(sections)}")

    policies = registry.get("policies") if isinstance(registry, dict) else []
    default_policy = registry.get("default_policy", "mechanism_aware") if isinstance(registry, dict) else "mechanism_aware"
    lines.extend(["", f"Default candidate policy: `{default_policy}`"])
    if isinstance(policies, list):
        selected = next(
            (policy for policy in policies if isinstance(policy, dict) and policy.get("name") == default_policy),
            None,
        )
        if selected:
            lines.append(f"Selection rule: {_one_line(selected.get('selection_rule'))}")
            if selected.get("weights"):
                weights = ", ".join(f"{key}={value}" for key, value in selected["weights"].items())
                lines.append(f"Weights: {weights}")
        lines.extend(["", "Registry policies:"])
        for policy in policies:
            if isinstance(policy, dict):
                lines.append(
                    f"- `{policy.get('name', 'policy')}` ({policy.get('role') or policy.get('stage') or 'policy'}): "
                    f"{_one_line(policy.get('evaluation') or policy.get('selection_rule'))}"
                )
    lines.append("")
    return lines


def _stage_progress_section(root: Path, run_dir: Path, stage_progress: Any) -> list[str]:
    lines = [
        "## Stage Progress",
        "",
        f"Artifact: `{_rel(run_dir / 'stage_progress.json', root)}`",
        "",
    ]
    stages = stage_progress.get("stages") if isinstance(stage_progress, dict) else stage_progress
    if isinstance(stages, list) and stages:
        for stage in stages[:12]:
            if isinstance(stage, dict):
                lines.append(
                    f"- `{stage.get('stage_id') or stage.get('name') or stage.get('stage') or 'stage'}`: "
                    f"{stage.get('status', 'unknown')}"
                )
            else:
                lines.append(f"- {_one_line(stage)}")
    elif isinstance(stage_progress, dict) and stage_progress:
        for key, value in list(stage_progress.items())[:12]:
            lines.append(f"- `{key}`: {_one_line(value)}")
    else:
        lines.append("No stage_progress.json artifact was loaded for this run.")
    lines.append("")
    return lines


def _route_tree_section(root: Path, run_dir: Path, route_tree: Any) -> list[str]:
    lines = [
        "## Route Tree",
        "",
        f"Artifact: `{_rel(run_dir / 'route_tree.json', root)}`",
        "",
    ]
    if isinstance(route_tree, dict) and route_tree:
        selected_path = route_tree.get("selected_path")
        if isinstance(selected_path, list) and selected_path:
            lines.append("Selected route: " + " -> ".join(f"`{item}`" for item in selected_path))
        routes = route_tree.get("routes") or route_tree.get("nodes")
        if isinstance(routes, list) and routes:
            lines.extend(["", "Route nodes:"])
            for route in routes[:12]:
                if isinstance(route, dict):
                    lines.append(
                        f"- `{route.get('node_id') or route.get('id') or 'node'}`: {route.get('status', 'unknown')}"
                    )
    else:
        lines.append("No route_tree.json artifact was loaded for this run.")
    lines.append("")
    return lines


def _baseline_comparison_section(
    root: Path,
    run_dir: Path,
    benchmark_rows: list[dict[str, str]],
    selected_summary: dict[str, Any],
) -> list[str]:
    lines = [
        "## Baseline Comparison",
        "",
        f"Artifact: `{_rel(run_dir / 'benchmark_results.csv', root)}`",
        "",
    ]
    if not benchmark_rows:
        lines.extend(["No benchmark rows were loaded.", ""])
        return lines

    preferred = [
        "world_id",
        "method",
        "best_feasible_utility",
        "hit_rate",
        "regret_proxy",
        "false_claim_rate",
        "evidence_coverage",
        "round_efficiency",
        "novelty_score",
        "baseline_overlap",
    ]
    lines.extend(_markdown_table(preferred, benchmark_rows))
    selected_method = _selected_report_method(selected_summary)
    if selected_method:
        selected_row = next((row for row in benchmark_rows if row.get("method") == selected_method), None)
        if selected_row:
            lines.extend(
                [
                    "",
                    (
                        f"Selected replay method `{selected_method}` achieved best_feasible_utility="
                        f"{selected_row.get('best_feasible_utility', 'n/a')} and regret_proxy="
                        f"{selected_row.get('regret_proxy', 'n/a')} in this synthetic replay."
                    ),
                ]
            )
    lines.append("")
    return lines


def _benchmark_summary_section(
    root: Path,
    run_dir: Path,
    benchmark_summary_rows: list[dict[str, str]],
) -> list[str]:
    lines = [
        "## Multi-World Benchmark Summary",
        "",
        f"Artifact: `{_rel(run_dir / 'benchmark_summary.csv', root)}`",
        "",
    ]
    if not benchmark_summary_rows:
        lines.extend(["No benchmark_summary.csv rows were loaded.", ""])
        return lines
    preferred = [
        "rank",
        "method",
        "world_count",
        "beats_random_feasible_worlds",
        "beats_random_feasible_majority",
        "beats_fixed_mix_worlds",
        "beats_fixed_mix_majority",
        "mean_best_feasible_utility",
        "mean_regret_proxy",
        "mean_hit_rate",
        "mean_false_claim_rate",
        "mean_novelty_score",
        "mean_baseline_overlap",
    ]
    lines.extend(_markdown_table(preferred, benchmark_summary_rows))
    lines.append("")
    return lines


def _no_baseline_rationale_section(
    selected_proposal: Any,
    selected_novelty_report: Any,
    selected_candidate_policy: Any,
) -> list[str]:
    lines = ["## No-Baseline Rationale", ""]
    if isinstance(selected_novelty_report, dict) and selected_novelty_report:
        lines.append(f"- baseline_clone: {_one_line(selected_novelty_report.get('baseline_clone', 'n/a'))}")
        if "nearest_baseline" in selected_novelty_report:
            lines.append(f"- nearest_baseline: {_one_line(selected_novelty_report.get('nearest_baseline'))}")
        if "selection_overlap_vs_baselines" in selected_novelty_report:
            lines.append(
                "- selection_overlap_vs_baselines: "
                f"{_one_line(selected_novelty_report.get('selection_overlap_vs_baselines'))}"
            )
        elif "baseline_overlap" in selected_novelty_report:
            lines.append(f"- baseline_overlap: {_one_line(selected_novelty_report.get('baseline_overlap'))}")
    else:
        lines.append("- No novelty_report.json artifact was loaded for the selected node.")

    rationale = _first_present_field(
        selected_proposal,
        selected_novelty_report,
        selected_candidate_policy,
        keys=("no_baseline_rationale", "non_baseline_rationale", "baseline_clone_rationale", "rationale"),
    )
    if rationale is not None:
        lines.append(f"- no_baseline_rationale: {_one_line(rationale)}")
    else:
        lines.append(
            "- no_baseline_rationale: selected method requires a novelty report with baseline_clone=false "
            "and acceptable measured overlap before prospective use."
        )
    lines.append("")
    return lines


def _architecture_novelty_section(
    selected_proposal: Any,
    selected_novelty_report: Any,
) -> list[str]:
    lines = ["## Architecture Novelty", ""]
    fields = (
        "architecture_delta",
        "architecture_novelty",
        "architectural_delta",
        "method_architecture_delta",
        "algorithm_mechanism",
        "expected_advantage",
    )
    emitted = False
    for label, data in (("proposal", selected_proposal), ("novelty_report", selected_novelty_report)):
        if not isinstance(data, dict):
            continue
        for key in fields:
            if key in data and data.get(key) not in (None, "", [], {}):
                lines.append(f"- {label}.{key}: {_one_line(data.get(key))}")
                emitted = True
        architecture = data.get("architecture")
        if isinstance(architecture, dict):
            for key in ("delta", "novelty", "architecture_delta"):
                if architecture.get(key) not in (None, "", [], {}):
                    lines.append(f"- {label}.architecture.{key}: {_one_line(architecture.get(key))}")
                    emitted = True
    if not emitted:
        lines.append("- No explicit architecture_delta or architecture novelty field was loaded.")
    lines.append("")
    return lines


def _ablation_section(root: Path, run_dir: Path, ablation_rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "## Ablation",
        "",
        f"Artifact: `{_rel(run_dir / 'ablation_results.csv', root)}`",
        "",
    ]
    if not ablation_rows:
        lines.extend(["No ablation rows were loaded.", ""])
        return lines

    preferred = [
        "removed_component",
        "best_feasible_utility",
        "regret_proxy",
        "delta_from_full_best_feasible_utility",
        "delta_from_full_regret_proxy",
        "false_claim_rate",
        "evidence_coverage",
    ]
    lines.extend(_markdown_table(preferred, ablation_rows))
    strongest = _largest_ablation_delta(ablation_rows)
    if strongest:
        lines.extend(
            [
                "",
                f"Largest best-feasible utility change from the full policy: `{strongest[0]}` ({strongest[1]}).",
            ]
        )
    lines.append("")
    return lines


def _selected_method_section(
    root: Path,
    selected_summary: dict[str, Any],
    selected_benchmark_summary: dict[str, str],
    selected_node: dict[str, Any] | None,
    selected_proposal: Any,
    selected_novelty_report: Any,
) -> list[str]:
    lines = ["## Selected Method", ""]
    if not selected_summary:
        lines.extend(["No selected method was recorded in scientist_journal.json.", ""])
        return lines

    lines.extend(
        [
            f"Selected node: `{selected_summary.get('node_id', 'unknown')}`",
            f"Benchmark method: `{selected_summary.get('benchmark_method', 'unknown')}`",
            "Benchmark policy: "
            f"`{selected_summary.get('benchmark_policy_name', selected_summary.get('benchmark_method', 'unknown'))}`",
            f"Ranking score: {selected_summary.get('ranking_score', 'n/a')}",
            f"Workspace: `{selected_summary.get('workspace', 'unknown')}`",
            "",
            "Selection basis: synthetic replay ranking plus method-node contract validation.",
        ]
    )
    metrics = selected_summary.get("benchmark_metrics")
    if isinstance(metrics, dict):
        lines.extend(
            [
                "",
                "Selected metrics:",
                f"- best_feasible_utility: {metrics.get('best_feasible_utility', 'n/a')}",
                f"- regret_proxy: {metrics.get('regret_proxy', 'n/a')}",
                f"- hit_rate: {metrics.get('hit_rate', 'n/a')}",
                f"- false_claim_rate: {metrics.get('false_claim_rate', 'n/a')}",
            ]
        )
    if selected_benchmark_summary:
        summary_keys = (
            "rank",
            "world_count",
            "beats_random_feasible_worlds",
            "beats_random_feasible_majority",
            "beats_fixed_mix_worlds",
            "beats_fixed_mix_majority",
            "mean_best_feasible_utility",
            "mean_regret_proxy",
            "mean_hit_rate",
            "mean_false_claim_rate",
            "mean_novelty_score",
            "mean_baseline_overlap",
        )
        lines.extend(["", "Selected multi-world summary:"])
        for key in summary_keys:
            if selected_benchmark_summary.get(key) not in (None, ""):
                lines.append(f"- {key}: {selected_benchmark_summary.get(key)}")
    lines.extend(["", "Selected proposal:"])
    if isinstance(selected_proposal, dict) and selected_proposal:
        for key in ("proposal_id", "title", "method_claim", "expected_differentiator", "summary"):
            if selected_proposal.get(key):
                lines.append(f"- {key}: {_one_line(selected_proposal.get(key))}")
    else:
        lines.append("- No proposal.json artifact was loaded for the selected node.")

    lines.extend(["", "Novelty report:"])
    if isinstance(selected_novelty_report, dict) and selected_novelty_report:
        for key in ("novelty_score", "baseline_overlap", "baseline_clone", "nearest_baseline", "verdict"):
            if key in selected_novelty_report:
                lines.append(f"- {key}: {_one_line(selected_novelty_report.get(key))}")
    else:
        lines.append("- No novelty_report.json artifact was loaded for the selected node.")

    if selected_node and isinstance(selected_node.get("artifacts"), dict):
        lines.extend(["", "Selected node artifacts:"])
        for key, value in selected_node["artifacts"].items():
            lines.append(f"- {key}: `{_rel(Path(str(value)), root)}`")
    lines.append("")
    return lines


def _generator_limited_diagnostics_section(
    selected_proposal: Any,
    selected_novelty_report: Any,
    selected_candidate_policy: Any,
) -> list[str]:
    lines = ["## Generator-Limited Diagnostics", ""]
    keys = (
        "generator_limited_diagnostics",
        "generator_diagnostics",
        "generator_limitations",
        "generation_diagnostics",
        "diagnostics",
    )
    emitted = False
    for label, data in (
        ("proposal", selected_proposal),
        ("novelty_report", selected_novelty_report),
        ("candidate_policy", selected_candidate_policy),
    ):
        if not isinstance(data, dict):
            continue
        for key in keys:
            if key in data and data.get(key) not in (None, "", [], {}):
                lines.append(f"- {label}.{key}: {_compact_jsonish(data.get(key))}")
                emitted = True
    if not emitted:
        lines.append("- No generator-limited diagnostics were present in selected node artifacts.")
    lines.append("")
    return lines


def _failed_methods_section(failed_nodes: list[dict[str, Any]]) -> list[str]:
    lines = ["## Failed Methods", ""]
    if not failed_nodes:
        lines.extend(["No failed method nodes were recorded.", ""])
        return lines
    for node in failed_nodes:
        reasons = node.get("reasons") or node.get("failure_reasons") or ["reason not recorded"]
        lines.append(f"- `{node.get('node_id', 'unknown')}`: {'; '.join(map(str, reasons))}")
    lines.append("")
    return lines


def _unsupported_claims_section(unsupported_claims: list[str]) -> list[str]:
    lines = ["## Unsupported Claims", ""]
    if unsupported_claims:
        lines.extend(f"- {claim}" for claim in unsupported_claims)
    else:
        lines.append("- No unsupported claims were recorded in the available state or run artifacts.")
    lines.append("")
    return lines


def _next_questions_section(questions: list[str]) -> list[str]:
    lines = ["## Next Development Questions", ""]
    lines.extend(f"- {question}" for question in questions)
    lines.append("")
    return lines


def _artifact_paths_section(artifacts: dict[str, str]) -> list[str]:
    lines = ["## Artifact Paths", ""]
    for key in sorted(artifacts):
        lines.append(f"- {key}: `{artifacts[key]}`")
    lines.append("")
    return lines


def _artifact_paths(root: Path, run_dir: Path, journal: Any) -> dict[str, str]:
    paths = {
        "framework_spec": _rel(root / "framework" / "framework_spec.yaml", root),
        "reference_audit": _rel(root / "framework" / "reference_audit.md", root),
        "reference_components": _rel(root / "framework" / "reference_components.json", root),
        "quest": _rel(root / "framework" / "quest.yaml", root),
        "research_map": _rel(root / "framework" / "research_map.json", root),
        "findings_memory": _rel(root / "framework" / "findings_memory.jsonl", root),
        "failure_memory": _rel(root / "framework" / "failure_memory.jsonl", root),
        "literature_queries": _rel(root / "framework" / "literature_queries.yaml", root),
        "literature_search_plan": _rel(root / "framework" / "literature_search_plan.json", root),
        "literature_search_trace": _rel(root / "framework" / "literature_search_trace.json", root),
        "citation_graph": _rel(root / "framework" / "citation_graph.json", root),
        "paper_scores": _rel(root / "framework" / "paper_scores.csv", root),
        "paper_cards": _rel(root / "framework" / "paper_cards.json", root),
        "literature_map": _rel(root / "framework" / "literature_map.md", root),
        "research_gap_matrix": _rel(root / "framework" / "research_gap_matrix.csv", root),
        "method_hypotheses": _rel(root / "framework" / "method_hypotheses.md", root),
        "method_modules": _rel(root / "framework" / "method_modules.json", root),
        "algorithm_spec": _rel(root / "framework" / "algorithm_spec.md", root),
        "method_registry": _rel(root / "framework" / "method_registry.yaml", root),
        "benchmark_results": _rel(run_dir / "benchmark_results.csv", root),
        "benchmark_summary": _rel(run_dir / "benchmark_summary.csv", root),
        "ablation_results": _rel(run_dir / "ablation_results.csv", root),
        "stage_progress": _rel(run_dir / "stage_progress.json", root),
        "route_tree": _rel(run_dir / "route_tree.json", root),
        "scientist_journal": _rel(run_dir / "scientist_journal.json", root),
        "method_report": _rel(run_dir / "method_report.md", root),
    }
    selected = _selected_node_record(journal)
    if selected and isinstance(selected.get("artifacts"), dict):
        for key, value in selected["artifacts"].items():
            paths[f"selected_node_{key}"] = _rel(Path(str(value)), root)
    return paths


def _selected_node_record(journal: Any) -> dict[str, Any] | None:
    if not isinstance(journal, dict):
        return None
    selected_node_id = journal.get("selected_node_id")
    nodes = journal.get("nodes")
    if not isinstance(nodes, list):
        return None
    return next(
        (node for node in nodes if isinstance(node, dict) and node.get("node_id") == selected_node_id),
        None,
    )


def _selected_node_summary(journal: Any, selected_node: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(journal, dict):
        return {}
    selected = journal.get("selected_node")
    summary = dict(selected) if isinstance(selected, dict) else {}
    if selected_node:
        summary.setdefault("node_id", selected_node.get("node_id"))
        summary.setdefault("workspace", selected_node.get("workspace"))
        summary.setdefault("benchmark_method", selected_node.get("benchmark_method"))
        summary.setdefault("benchmark_policy_name", selected_node.get("benchmark_policy_name"))
        summary.setdefault("ranking_score", selected_node.get("ranking_score"))
        summary.setdefault("benchmark_metrics", selected_node.get("benchmark_metrics"))
    return summary


def _selected_benchmark_summary(
    benchmark_summary_rows: list[dict[str, str]],
    selected_summary: dict[str, Any],
) -> dict[str, str]:
    selected_method = _selected_report_method(selected_summary)
    if not selected_method:
        return {}
    row = next((row for row in benchmark_summary_rows if row.get("method") == selected_method), None)
    return dict(row) if row is not None else {}


def _selected_report_method(selected_summary: dict[str, Any]) -> str | None:
    for key in ("benchmark_policy_name", "method", "benchmark_method"):
        value = selected_summary.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _selected_json_artifact(
    root: Path,
    selected_node: dict[str, Any] | None,
    key: str,
    filename: str,
) -> Any:
    if not selected_node:
        return {}
    path: Path | None = None
    artifacts = selected_node.get("artifacts")
    if isinstance(artifacts, dict) and artifacts.get(key):
        path = _coerce_artifact_path(root, artifacts[key])
    elif selected_node.get(key):
        path = _coerce_artifact_path(root, selected_node[key])
    elif selected_node.get("workspace"):
        path = _coerce_artifact_path(root, selected_node["workspace"]) / filename
    if path is None:
        return {}
    return _load_json(path)


def _failed_nodes(journal: Any) -> list[dict[str, Any]]:
    if not isinstance(journal, dict):
        return []
    failed = journal.get("failed_nodes")
    if isinstance(failed, list):
        return [node for node in failed if isinstance(node, dict)]
    nodes = journal.get("nodes")
    if not isinstance(nodes, list):
        return []
    return [
        node
        for node in nodes
        if isinstance(node, dict) and node.get("status") not in {None, "completed"}
    ]


def _unsupported_claims(
    *,
    state: Any,
    policy_metrics: Any,
    paper_cards: list[dict[str, Any]],
    journal: Any,
) -> list[str]:
    claims: list[str] = []
    if isinstance(state, dict):
        claims.extend(str(item) for item in state.get("unsupported_claims", []) if item)
    if isinstance(policy_metrics, dict):
        claims.extend(str(item) for item in policy_metrics.get("unsupported_claims", []) if item)
    if any(card.get("review_status") == "seed_unverified" for card in paper_cards):
        claims.append("Seed literature cards are method scaffolds, not verified literature support.")
    if isinstance(journal, dict):
        claims.append("Synthetic replay ranking does not establish prospective wet-lab superiority.")
        if journal.get("literature_snapshot", {}).get("status") == "failed":
            claims.append("The literature search stage failed for this run; method provenance is incomplete.")
    return _unique(claims)


def _next_development_questions(
    *,
    method_modules: list[dict[str, Any]],
    failed_nodes: list[dict[str, Any]],
    ablation_rows: list[dict[str, str]],
    benchmark_rows: list[dict[str, str]],
) -> list[str]:
    questions = [
        "Which real historical rounds can replace or complement the fixed synthetic replay?",
        "What benchmark gate should a generated policy clear before it is eligible for prospective panel selection?",
        "Which unsupported method claim needs a matched contrast, ablation, or external benchmark next?",
    ]
    gaps = [
        _one_line(module.get("gap_for_design_scientist"))
        for module in method_modules
        if module.get("gap_for_design_scientist")
    ]
    for gap in gaps[:4]:
        questions.append(f"How should the framework address this extracted gap: {gap}")
    if failed_nodes:
        questions.append("Can failed method-node contracts be converted into smaller executable policy variants?")
    strongest = _largest_ablation_delta(ablation_rows)
    if strongest:
        questions.append(f"Why does removing `{strongest[0]}` shift best feasible utility by {strongest[1]}?")
    if benchmark_rows and not any(row.get("method") == "random_feasible" for row in benchmark_rows):
        questions.append("Add the random_feasible baseline to the next benchmark run for a sanity floor.")
    return _unique(questions)


def _largest_ablation_delta(rows: list[dict[str, str]]) -> tuple[str, str] | None:
    best: tuple[str, str, float] | None = None
    for row in rows:
        component = row.get("removed_component")
        raw_delta = row.get("delta_from_full_best_feasible_utility")
        if not component or raw_delta in (None, ""):
            continue
        try:
            magnitude = abs(float(raw_delta))
        except ValueError:
            continue
        if best is None or magnitude > best[2]:
            best = (component, raw_delta, magnitude)
    if best is None:
        return None
    return best[0], best[1]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _load_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return read_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return read_yaml(path)
    except (OSError, ValueError):
        return {}


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _as_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _markdown_h2(text: str) -> list[str]:
    headings = []
    for line in text.splitlines():
        if line.startswith("## ") and not line.startswith("### "):
            heading = line.removeprefix("## ").strip()
            if heading:
                headings.append(heading)
    return headings


def _first_nonempty_lines(text: str, *, limit: int) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        out.append(stripped)
        if len(out) >= limit:
            break
    return out


def _extract_line_value(text: str, prefix: str) -> str | None:
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return None


def _markdown_table(columns: list[str], rows: list[dict[str, str]]) -> list[str]:
    present = [column for column in columns if any(column in row for row in rows)]
    if not present:
        present = list(rows[0].keys()) if rows else []
    lines = [
        "| " + " | ".join(present) + " |",
        "| " + " | ".join("---" for _ in present) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_cell(row.get(column, "")) for column in present) + " |")
    return lines


def _cell(value: Any) -> str:
    text = _one_line(value)
    text = text.replace("|", "\\|")
    return text or " "


def _one_line(value: Any) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).split())
    return text


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = _one_line(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _first_present_field(*items: Any, keys: tuple[str, ...]) -> Any:
    for item in items:
        if not isinstance(item, dict):
            continue
        for key in keys:
            value = item.get(key)
            if value not in (None, "", [], {}):
                return value
    return None


def _compact_jsonish(value: Any) -> str:
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, sort_keys=True)
        except (TypeError, ValueError):
            return _one_line(value)
    return _one_line(value)


def _coerce_artifact_path(root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _rel(path: Path, root: Path) -> str:
    try:
        candidate = path.expanduser().resolve()
    except OSError:
        candidate = path
    try:
        return str(candidate.relative_to(root))
    except ValueError:
        text = str(path)
        match = re.search(r"/runs/[^/]+/nodes/.*", text)
        if match:
            return text[match.start() + 1 :]
        return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m design_scientist.method_report")
    parser.add_argument("project_dir")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)

    path = write_method_report(args.project_dir, run_id=args.run_id)
    print(f"Wrote method report: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["write_method_report"]
