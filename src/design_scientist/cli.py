"""Command line interface for the local Design Scientist framework."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="design-scientist")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize a project skeleton")
    init_parser.add_argument("project_dir")
    init_parser.add_argument("--project-id", default="anti_hbsag")

    init_framework = subparsers.add_parser("init-framework", help="Initialize framework-first artifacts")
    init_framework.add_argument("root")
    init_framework.add_argument("--domain", required=True)
    init_framework.add_argument(
        "--force-domain-update",
        action="store_true",
        help="Rewrite existing Research OS domain metadata when it differs from --domain",
    )

    audit_references = subparsers.add_parser(
        "audit-references", help="Audit local reference repositories"
    )
    audit_references.add_argument("root")

    literature_search = subparsers.add_parser(
        "literature-search", help="debug/development: run framework literature search only"
    )
    literature_search.add_argument("root")
    literature_search.add_argument("--max-papers", type=int, default=60)
    literature_search.add_argument("--offline-fixtures", action="store_true")
    literature_search.add_argument("--sources", nargs="*")

    extract_methods = subparsers.add_parser(
        "extract-methods", help="debug/development: extract method modules only"
    )
    extract_methods.add_argument("root")

    develop_method = subparsers.add_parser(
        "develop-method", help="debug/development: develop and rank method nodes only"
    )
    develop_method.add_argument("root")
    develop_method.add_argument("--nodes", type=int, default=3)
    develop_method.add_argument("--use-codex", action="store_true")
    develop_method.add_argument("--run-id")

    benchmark_methods = subparsers.add_parser(
        "benchmark-methods", help="debug/development: run baseline-only synthetic replay benchmarks"
    )
    benchmark_methods.add_argument("root")
    benchmark_methods.add_argument("--rounds", type=int, default=3)
    benchmark_methods.add_argument("--budget", type=int, default=24)
    benchmark_methods.add_argument("--run-id")

    run_scientist = subparsers.add_parser(
        "run-scientist", help="recommended full-chain framework scientist loop"
    )
    run_scientist.add_argument("root")
    run_scientist.add_argument("--max-papers", type=int, default=60)
    run_scientist.add_argument("--nodes", type=int, default=3)
    run_scientist.add_argument("--rounds", type=int, default=3)
    run_scientist.add_argument("--use-codex", action="store_true")
    run_scientist.add_argument("--offline-fixtures", action="store_true")
    run_scientist.add_argument(
        "--allow-degraded-literature",
        action="store_true",
        help="Continue method development when literature search or extraction fails",
    )

    review_framework = subparsers.add_parser(
        "review-framework", help="Validate framework run artifacts"
    )
    review_framework.add_argument("root")
    review_framework.add_argument("--run-id")
    review_framework.add_argument("--json", action="store_true")

    build_state = subparsers.add_parser("build-state", help="Build project design state")
    build_state.add_argument("project_dir")

    literature = subparsers.add_parser(
        "literature", help="Legacy project execution: create seed cards and gap matrix"
    )
    literature.add_argument("project_dir")

    hypotheses = subparsers.add_parser("hypotheses", help="Build or update hypothesis board")
    hypotheses.add_argument("project_dir")

    retrospective = subparsers.add_parser("retrospective", help="Run retrospective policy evaluation")
    retrospective.add_argument("project_dir")
    retrospective.add_argument("--budget", type=int, default=24)

    merge_data = subparsers.add_parser("merge-data", help="Intake new data as pending schema audit")
    merge_data.add_argument("project_dir")
    merge_data.add_argument("new_data_dir")
    merge_data.add_argument("--round-label")

    dry_panel = subparsers.add_parser("run-dry-panel", help="Run dry panel selection")
    dry_panel.add_argument("project_dir")
    dry_panel.add_argument("--budget", type=int, default=24)
    dry_panel.add_argument("--use-codex", action="store_true")

    full = subparsers.add_parser("run-full", help="Run the complete local Design Scientist workflow")
    full.add_argument("project_dir")
    full.add_argument("--budget", type=int, default=24)
    full.add_argument("--use-codex", action="store_true")

    review = subparsers.add_parser("review", help="Validate a project or run")
    review.add_argument("project_dir")
    review.add_argument("--run-id")

    args_parser = subparsers.add_parser("codex-task", help="Run a controlled Codex task")
    args_parser.add_argument("workspace")
    args_parser.add_argument("objective")
    args_parser.add_argument("--mode", choices=["read_only_plan", "patch", "verify"], default="read_only_plan")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "init":
        from design_scientist.projects import initialize_project

        initialize_project(args.project_dir, project_id=args.project_id)
        return 0
    if args.command == "init-framework":
        from design_scientist.framework import init_framework

        try:
            init_framework(
                args.root,
                domain=args.domain,
                force_domain_update=args.force_domain_update,
            )
        except ValueError as exc:
            parser.error(str(exc))
        return 0
    if args.command == "audit-references":
        from design_scientist.reference_audit import audit_references

        result = audit_references(args.root)
        print(f"Wrote reference audit: {result['artifacts']['reference_audit']}")
        print(f"Wrote reference components: {result['artifacts']['reference_components']}")
        return 0
    if args.command == "literature-search":
        from design_scientist.literature_pipeline import run_literature_search

        out = run_literature_search(
            args.root,
            max_papers=args.max_papers,
            offline_fixtures=args.offline_fixtures,
            sources=args.sources,
        )
        print(f"Wrote paper cards: {out}")
        return 0
    if args.command == "extract-methods":
        from design_scientist.method_extraction import extract_methods

        result = extract_methods(args.root)
        print(f"Method extraction {result['status']}: {result.get('method_module_count', 0)} modules")
        return 0 if result["status"] == "ok" else 1
    if args.command == "develop-method":
        from design_scientist.scientist_search import develop_method

        result = develop_method(
            args.root,
            nodes=args.nodes,
            use_codex=args.use_codex,
            run_id=args.run_id,
        )
        print(f"Wrote scientist journal: {result['journal_path']}")
        if not result.get("selected_node"):
            print("No valid selected method node was produced.")
            return 1
        return 0
    if args.command == "benchmark-methods":
        from design_scientist.synthetic_replay import DEFAULT_RUN_ID, run_synthetic_benchmark

        run_id = args.run_id or f"baseline_{DEFAULT_RUN_ID}"
        run_dir = Path(args.root).expanduser().resolve() / "runs" / run_id
        if (run_dir / "scientist_journal.json").exists():
            parser.error(
                "benchmark-methods refuses to write into an existing scientist run; "
                "choose a baseline-only --run-id or omit --run-id to use the baseline default"
            )
        result = run_synthetic_benchmark(
            args.root,
            run_id=run_id,
            rounds=args.rounds,
            budget=args.budget,
        )
        print(f"Wrote benchmark results: {result['benchmark_results_path']}")
        return 0
    if args.command == "run-scientist":
        from design_scientist.method_report import write_method_report
        from design_scientist.scientist_search import run_scientist_search

        result = run_scientist_search(
            args.root,
            max_papers=args.max_papers,
            nodes=args.nodes,
            rounds=args.rounds,
            use_codex=args.use_codex,
            offline_fixtures=args.offline_fixtures,
            strict_literature=not args.allow_degraded_literature,
        )
        report_path = write_method_report(args.root, run_id=result["run_id"])
        print(f"Wrote scientist journal: {result['journal_path']}")
        print(f"Wrote method report: {report_path}")
        if not result.get("selected_node"):
            print("No valid selected method node was produced.")
            return 1
        return 0
    if args.command == "review-framework":
        import json

        from design_scientist.framework_validation import review_framework_run

        report = review_framework_run(args.root, run_id=args.run_id)
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
    if args.command == "build-state":
        from design_scientist.state import build_state

        build_state(args.project_dir)
        return 0
    if args.command == "literature":
        from design_scientist.literature import build_research_gap_matrix, create_seed_paper_cards
        from design_scientist.methods import propose_method_hypotheses

        create_seed_paper_cards(args.project_dir)
        build_research_gap_matrix(args.project_dir)
        propose_method_hypotheses(args.project_dir)
        return 0
    if args.command == "hypotheses":
        from design_scientist.hypotheses import build_hypothesis_board

        build_hypothesis_board(args.project_dir)
        return 0
    if args.command == "retrospective":
        from design_scientist.retrospective import run_retrospective_eval

        run_retrospective_eval(args.project_dir, budget=args.budget)
        return 0
    if args.command == "merge-data":
        from design_scientist.data_merge import merge_new_data

        merge_new_data(args.project_dir, args.new_data_dir, round_label=args.round_label)
        return 0
    if args.command == "run-dry-panel":
        from design_scientist.pipeline import run_dry_panel

        run_dry_panel(args.project_dir, budget=args.budget, use_codex=args.use_codex)
        return 0
    if args.command == "run-full":
        from design_scientist.pipeline import run_full_workflow

        run_full_workflow(args.project_dir, budget=args.budget, use_codex=args.use_codex)
        return 0
    if args.command == "review":
        from design_scientist.validators import validate_project

        report = validate_project(args.project_dir, run_id=args.run_id)
        return 0 if report.valid else 1
    if args.command == "codex-task":
        from design_scientist.backends.codex_cli import CodexCliBackend
        from design_scientist.schemas import WorkspaceAgentTask

        task = WorkspaceAgentTask(
            objective=args.objective,
            workspace=args.workspace,
            mode=args.mode,
            allowed_paths=[args.workspace],
        )
        result = CodexCliBackend().run_task(task)
        print(result.summary)
        return result.returncode

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
