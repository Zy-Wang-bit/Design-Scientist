"""Command line interface for the local Design Scientist framework."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path


LITERATURE_SOURCES_ENV = "DESIGN_SCIENTIST_LITERATURE_SOURCES"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="design-scientist")
    visible_commands = (
        "init",
        "start-1e62",
        "literature-1e62",
        "advance-1e62",
        "review-1e62-panel",
        "finalize-1e62-result",
        "init-framework",
        "audit-references",
        "literature-search",
        "read-literature",
        "extract-mechanisms",
        "mine-literature",
        "compile-operators",
        "run-scientist",
        "run-algorithm-benchmark",
        "run-generative-benchmark",
        "run-external-antibody-benchmark",
        "run-external-ph-switch-benchmark",
        "run-project-benchmark",
        "run-project-ph-switch-design",
        "review-framework",
        "generate-short-paper",
        "generate-algorithm-paper",
        "audit-schema",
        "standardize-startup-data",
        "build-state",
        "literature",
        "hypotheses",
        "retrospective",
        "merge-data",
        "run-dry-panel",
        "run-full",
        "review",
        "codex-task",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        metavar="{" + ",".join(visible_commands) + "}",
    )

    init_parser = subparsers.add_parser("init", help="Initialize a project skeleton")
    init_parser.add_argument("project_dir")
    init_parser.add_argument("--project-id", default="anti_hbsag")

    start_onee62 = subparsers.add_parser(
        "start-1e62",
        help="Start a 1E62 project from a startup data package",
    )
    start_onee62.add_argument("startup_dir")
    start_onee62.add_argument("project_dir")

    literature_onee62 = subparsers.add_parser(
        "literature-1e62",
        help="Run the explicit 1E62 literature and mechanism stage",
    )
    literature_onee62.add_argument("project_dir")
    literature_onee62.add_argument("--max-papers", type=int, default=30)
    literature_onee62.add_argument("--offline-fixtures", action="store_true")
    literature_onee62.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of writing a degraded literature report when a source stage fails",
    )

    advance_onee62 = subparsers.add_parser(
        "advance-1e62",
        help="Advance a started 1E62 project into dry-panel draft stage",
    )
    advance_onee62.add_argument("project_dir")
    advance_onee62.add_argument("--budget", type=int, default=24)
    advance_onee62.add_argument("--use-codex", action="store_true")
    advance_onee62.add_argument("--max-papers", type=int, default=30)
    advance_onee62.add_argument("--offline-fixtures", action="store_true")

    review_onee62_panel = subparsers.add_parser(
        "review-1e62-panel",
        help="Write the 1E62 dry-panel human review gate report",
    )
    review_onee62_panel.add_argument("project_dir")
    review_onee62_panel.add_argument("--run-id")

    finalize_onee62_result = subparsers.add_parser(
        "finalize-1e62-result",
        help="Write the final 1E62 computational result package",
    )
    finalize_onee62_result.add_argument("project_dir")
    finalize_onee62_result.add_argument("--run-id")

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

    read_literature = subparsers.add_parser(
        "read-literature", help="debug/development: build V3 literature corpus only"
    )
    read_literature.add_argument("root")

    extract_mechanisms = subparsers.add_parser(
        "extract-mechanisms", help="debug/development: extract V3 mechanism cards only"
    )
    extract_mechanisms.add_argument("root")

    mine_literature = subparsers.add_parser(
        "mine-literature", help="debug/development: mine V4 mechanism cards from the corpus"
    )
    mine_literature.add_argument("root")

    compile_operators = subparsers.add_parser(
        "compile-operators",
        help="debug/development: compile operator artifacts from mechanism cards only",
    )
    compile_operators.add_argument("root")

    run_scientist = subparsers.add_parser(
        "run-scientist",
        help="recommended full-chain V3 mechanism scientist loop with V4 research harness",
    )
    run_scientist.add_argument("root")
    run_scientist.add_argument("--max-papers", type=int, default=60)
    run_scientist.add_argument("--nodes", type=int, default=3)
    run_scientist.add_argument("--rounds", type=int, default=3)
    run_scientist.add_argument("--use-codex", action="store_true")
    run_scientist.add_argument("--offline-fixtures", action="store_true")
    run_scientist.add_argument("--sources", nargs="*")
    run_scientist.add_argument(
        "--skip-paper",
        action="store_true",
        help="Skip automatic paper bundle generation for debug runs",
    )
    run_scientist.add_argument(
        "--allow-degraded-literature",
        action="store_true",
        help="Continue method development when literature search or extraction fails",
    )

    algorithm_benchmark = subparsers.add_parser(
        "run-algorithm-benchmark",
        help="Run generic MCCBD algorithm benchmark over controlled mechanism worlds",
    )
    algorithm_benchmark.add_argument("root")
    algorithm_benchmark.add_argument("--run-id", default="mccbd_algorithm")
    algorithm_benchmark.add_argument("--seeds", nargs="*", type=int)
    algorithm_benchmark.add_argument("--rounds", type=int, default=3)
    algorithm_benchmark.add_argument("--budget", type=int, default=6)
    algorithm_benchmark.add_argument("--mechanisms", nargs="*")
    algorithm_benchmark.add_argument("--worlds", nargs="*")

    generative_benchmark = subparsers.add_parser(
        "run-generative-benchmark",
        help="Run generative sequence-design benchmark",
    )
    generative_benchmark.add_argument("root")
    generative_benchmark.add_argument("--run-id", default="ph_switch_graph_generative")
    generative_benchmark.add_argument("--seeds", nargs="*", type=int)
    generative_benchmark.add_argument("--budget", type=int, default=5)
    generative_benchmark.add_argument("--generation-budget", type=int, default=32)
    generative_benchmark.add_argument("--max-edits-per-candidate", type=int)
    generative_benchmark.add_argument("--mechanisms", nargs="*")
    generative_benchmark.add_argument("--worlds", nargs="*")

    external_antibody_benchmark = subparsers.add_parser(
        "run-external-antibody-benchmark",
        help="Run external FLAb-style antibody held-out replay benchmark",
    )
    external_antibody_benchmark.add_argument("root")
    external_antibody_benchmark.add_argument("--run-id", default="external_antibody_replay")
    external_antibody_benchmark.add_argument("--seeds", nargs="*", type=int)
    external_antibody_benchmark.add_argument("--budget", type=int, default=5)
    external_antibody_benchmark.add_argument("--offline-fixtures", action="store_true")
    external_antibody_benchmark.add_argument("--refresh", action="store_true")

    external_ph_switch_benchmark = subparsers.add_parser(
        "run-external-ph-switch-benchmark",
        help="Run curated public pH-switch antibody literature-table sanity benchmark",
    )
    external_ph_switch_benchmark.add_argument("root")
    external_ph_switch_benchmark.add_argument("--run-id", default="external_ph_switch_replay")
    external_ph_switch_benchmark.add_argument("--budget", type=int, default=2)

    project_benchmark = subparsers.add_parser(
        "run-project-benchmark",
        help="Run real project-data retrospective masking benchmark",
    )
    project_benchmark.add_argument("root")
    project_benchmark.add_argument("--run-id", default="project_masking")
    project_benchmark.add_argument("--budget", type=int, default=4)
    project_benchmark.add_argument("--folds", default="kfold_5")
    project_benchmark.add_argument("--mechanisms", nargs="*")

    project_cmdgd_design = subparsers.add_parser(
        "run-project-cmdgd-design",
        help="Legacy/debug only: run rejected CMD-GD candidate generation",
    )
    project_cmdgd_design.add_argument("root")
    project_cmdgd_design.add_argument("--run-id", default="cmdgd_project_design")
    project_cmdgd_design.add_argument("--budget", type=int, default=5)
    project_cmdgd_design.add_argument("--generation-budget", type=int, default=160)
    project_cmdgd_design.add_argument("--max-edits-per-candidate", type=int)
    project_cmdgd_design.add_argument("--seed", type=int, default=1729)
    project_cmdgd_design.add_argument(
        "--allow-legacy-cmdgd",
        action="store_true",
        help="Required to run the legacy CMD-GD debug path; never used by V3 defaults",
    )

    project_ph_switch_design = subparsers.add_parser(
        "run-project-ph-switch-design",
        help="Run prospective pH-switch graph candidate generation from visible project data",
    )
    project_ph_switch_design.add_argument("root")
    project_ph_switch_design.add_argument("--run-id", default="ph_switch_graph_project_design")
    project_ph_switch_design.add_argument("--budget", type=int, default=5)
    project_ph_switch_design.add_argument("--generation-budget", type=int, default=160)
    project_ph_switch_design.add_argument("--max-edits-per-candidate", type=int)
    project_ph_switch_design.add_argument("--seed", type=int, default=1729)

    review_framework = subparsers.add_parser(
        "review-framework", help="Validate framework run artifacts"
    )
    review_framework.add_argument("root")
    review_framework.add_argument("--run-id")
    review_framework.add_argument(
        "--scientific-only",
        action="store_true",
        help="Ignore submission-metadata-only paper readiness blockers while keeping scientific artifact validation strict.",
    )
    review_framework.add_argument("--json", action="store_true")

    short_paper = subparsers.add_parser(
        "generate-short-paper",
        help="Generate a conservative short-paper bundle from framework V3 artifacts",
    )
    short_paper.add_argument("root")
    short_paper.add_argument("--run-id")

    algorithm_paper = subparsers.add_parser(
        "generate-algorithm-paper",
        help="Generate a manuscript centered on a project masking algorithm run",
    )
    algorithm_paper.add_argument("root")
    algorithm_paper.add_argument("--run-id", required=True)
    algorithm_paper.add_argument("--algorithm", default="evidence_calibrated_ucb")

    audit_schema = subparsers.add_parser(
        "audit-schema", help="Audit raw startup data declared by data_contract.allowed_sources"
    )
    audit_schema.add_argument("project_dir")

    standardize_startup = subparsers.add_parser(
        "standardize-startup-data",
        help="Standardize raw startup data into generic project context artifacts",
    )
    standardize_startup.add_argument("project_dir")

    build_state = subparsers.add_parser("build-state", help="Build project design state")
    build_state.add_argument("project_dir")

    literature = subparsers.add_parser(
        "literature", help="Project execution: create seed cards and gap matrix"
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
    if args.command == "start-1e62":
        from design_scientist.onee62_startup import start_1e62_project

        start_1e62_project(args.startup_dir, args.project_dir)
        return 0
    if args.command == "literature-1e62":
        from design_scientist.onee62_startup import run_1e62_literature_stage

        run_1e62_literature_stage(
            args.project_dir,
            max_papers=args.max_papers,
            offline_fixtures=args.offline_fixtures,
            allow_degraded=not args.strict,
        )
        return 0
    if args.command == "advance-1e62":
        from design_scientist.onee62_startup import advance_1e62_project

        advance_1e62_project(
            args.project_dir,
            budget=args.budget,
            use_codex=args.use_codex,
            max_papers=args.max_papers,
            offline_fixtures=args.offline_fixtures,
        )
        return 0
    if args.command == "review-1e62-panel":
        from design_scientist.onee62_startup import review_1e62_panel

        review_1e62_panel(args.project_dir, run_id=args.run_id)
        return 0
    if args.command == "finalize-1e62-result":
        from design_scientist.onee62_startup import finalize_1e62_result

        finalize_1e62_result(args.project_dir, run_id=args.run_id)
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
    if args.command == "read-literature":
        try:
            from design_scientist.literature_fulltext import build_literature_corpus
        except ModuleNotFoundError as exc:
            if exc.name == "design_scientist.literature_fulltext":
                parser.error(
                    "read-literature is not available yet: "
                    "missing design_scientist.literature_fulltext"
                )
            raise

        result = build_literature_corpus(args.root)
        if result is not None:
            print(f"Built literature corpus: {result}")
        return 0
    if args.command == "extract-mechanisms":
        try:
            from design_scientist.mechanism_extraction import extract_mechanisms
        except ModuleNotFoundError as exc:
            if exc.name == "design_scientist.mechanism_extraction":
                parser.error(
                    "extract-mechanisms is not available yet: "
                    "missing design_scientist.mechanism_extraction"
                )
            raise

        result = extract_mechanisms(args.root)
        if isinstance(result, dict) and "status" in result:
            print(f"Mechanism extraction {result['status']}")
            return 0 if result["status"] == "ok" else 1
        if result is not None:
            print(f"Extracted mechanisms: {result}")
        return 0
    if args.command == "mine-literature":
        from design_scientist.literature_engine_v4 import mine_mechanisms_from_corpus

        result = mine_mechanisms_from_corpus(args.root)
        print(f"V4 literature mining {result.get('status', 'unknown')}")
        if result.get("artifacts"):
            print(f"Wrote mechanism graph: {result['artifacts'].get('mechanism_graph')}")
        return 0 if result.get("status") == "ok" else 1
    if args.command == "compile-operators":
        from design_scientist.operator_specs import write_operator_spec_artifacts

        root = Path(args.root).expanduser().resolve()
        framework_dir = root / "framework"
        cards_path = framework_dir / "mechanism_cards.json"
        if not cards_path.is_file():
            parser.error(
                "compile-operators requires framework/mechanism_cards.json; "
                "run extract-mechanisms first"
            )
        try:
            mechanism_cards = json.loads(cards_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            parser.error(f"framework/mechanism_cards.json is invalid JSON: {exc.msg}")
        if not isinstance(mechanism_cards, list) or not all(
            isinstance(card, dict) for card in mechanism_cards
        ):
            parser.error("framework/mechanism_cards.json must contain a JSON list of objects")

        paths = write_operator_spec_artifacts(framework_dir, mechanism_cards)
        print(f"Wrote operator specs: {paths['operator_specs']}")
        print(f"Wrote operator gap matrix: {paths['operator_gap_matrix']}")
        print(f"Wrote operator evidence map: {paths['operator_evidence_map']}")
        print(f"Wrote operator negative controls: {paths['operator_negative_controls']}")
        return 0
    if args.command == "run-scientist":
        from design_scientist.method_report import write_method_report

        from design_scientist.scientist_search_v3 import run_scientist_v3

        with _scoped_literature_sources(args.sources):
            result = run_scientist_v3(
                args.root,
                max_papers=args.max_papers,
                nodes=args.nodes,
                rounds=args.rounds,
                use_codex=args.use_codex,
                offline_fixtures=args.offline_fixtures,
                strict_literature=not args.allow_degraded_literature,
            )
        _normalize_v3_cli_mechanism_metrics(result)
        report_path = write_method_report(args.root, run_id=result["run_id"])
        print(f"Wrote scientist journal: {result['journal_path']}")
        print(f"Wrote method report: {report_path}")
        if not result.get("selected_node"):
            print("No valid selected mechanism node was produced.")
            return 1
        if args.skip_paper:
            return 0
        from design_scientist.manuscript import generate_short_paper

        try:
            paper_result = generate_short_paper(args.root, run_id=result["run_id"])
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        readiness = paper_result["readiness"]
        summary = readiness["summary"]
        print(f"Wrote paper bundle: {paper_result['paper_dir']}")
        print(f"Wrote short paper: {paper_result['artifacts']['short_paper.md']}")
        print(
            f"Wrote paper readiness report: "
            f"{paper_result['artifacts']['paper_readiness_report.json']}"
        )
        print(
            f"Paper readiness {readiness['status']}: "
            f"{summary['errors']} errors, {summary['warnings']} warnings"
        )
        return 0 if readiness["valid"] else 1
    if args.command == "run-algorithm-benchmark":
        from design_scientist.algorithm_benchmark import run_mccbd_benchmark

        try:
            result = run_mccbd_benchmark(
                args.root,
                run_id=args.run_id,
                seeds=args.seeds if args.seeds is not None else range(10),
                rounds=args.rounds,
                budget=args.budget,
                mechanisms=args.mechanisms,
                worlds=args.worlds,
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(f"Wrote MCCBD benchmark: {result['benchmark_results_path']}")
        print(f"Wrote MCCBD summary: {result['summary_results_path']}")
        print(f"Wrote MCCBD ablations: {result['ablation_results_path']}")
        print(
            "MCCBD benchmark gate: "
            f"selected={result['selected_mechanism']} "
            f"passes={result['selected_passes_gate']}"
        )
        return 0 if result.get("status") == "completed" else 1
    if args.command == "run-generative-benchmark":
        from design_scientist.generative_benchmark import run_generative_benchmark

        try:
            result = run_generative_benchmark(
                args.root,
                run_id=args.run_id,
                seeds=args.seeds if args.seeds is not None else range(10),
                budget=args.budget,
                generation_budget=args.generation_budget,
                max_edits_per_candidate=args.max_edits_per_candidate,
                mechanisms=args.mechanisms,
                worlds=args.worlds,
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(f"Wrote generative benchmark: {result['benchmark_results_path']}")
        print(f"Wrote generative summary: {result['summary_results_path']}")
        print(f"Wrote generative ablations: {result['ablation_results_path']}")
        print(f"Wrote generative design examples: {result['design_examples_path']}")
        print(
            "Generative benchmark gate: "
            f"selected={result['selected_mechanism']} "
            f"passes={result['selected_passes_gate']}"
        )
        return 0 if result.get("status") == "completed" else 1
    if args.command == "run-external-antibody-benchmark":
        from design_scientist.external_antibody_benchmark import run_external_antibody_benchmark

        try:
            result = run_external_antibody_benchmark(
                args.root,
                run_id=args.run_id,
                seeds=args.seeds if args.seeds is not None else range(5),
                budget=args.budget,
                offline_fixtures=args.offline_fixtures,
                refresh=args.refresh,
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(f"Wrote external antibody benchmark: {result['benchmark_results_path']}")
        print(f"Wrote external antibody summary: {result['summary_results_path']}")
        print(f"Wrote external antibody source trace: {result['source_trace_path']}")
        return 0 if result.get("status") == "completed" else 1
    if args.command == "run-external-ph-switch-benchmark":
        from design_scientist.external_antibody_benchmark import run_external_ph_switch_benchmark

        try:
            result = run_external_ph_switch_benchmark(
                args.root,
                run_id=args.run_id,
                budget=args.budget,
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(f"Wrote external pH-switch benchmark: {result['benchmark_results_path']}")
        print(f"Wrote external pH-switch summary: {result['summary_results_path']}")
        print(f"Wrote external pH-switch source trace: {result['source_trace_path']}")
        return 0 if result.get("status") == "completed" else 1
    if args.command == "run-project-benchmark":
        from design_scientist.project_replay import run_project_masking_benchmark

        try:
            result = run_project_masking_benchmark(
                args.root,
                run_id=args.run_id,
                budget=args.budget,
                folds=args.folds,
                mechanisms=args.mechanisms,
            )
        except ValueError as exc:
            parser.error(str(exc))
        if result.get("status") != "completed":
            print(f"Project masking benchmark unavailable: {result.get('error', 'unknown error')}")
            return 1
        print(f"Wrote project masking benchmark: {result['benchmark_results_path']}")
        print(f"Wrote project masking summary: {result['summary_results_path']}")
        print(f"Wrote project masking ablation: {result['ablation_results_path']}")
        print(f"Wrote project masking config: {result['config_path']}")
        return 0
    if args.command == "run-project-cmdgd-design":
        from design_scientist.project_replay import run_project_cmdgd_design

        if not args.allow_legacy_cmdgd:
            parser.error(
                "run-project-cmdgd-design is a legacy/debug command. "
                "Use run-project-ph-switch-design for the V3 route, or pass "
                "--allow-legacy-cmdgd only when intentionally reproducing the rejected legacy path."
            )
        try:
            result = run_project_cmdgd_design(
                args.root,
                run_id=args.run_id,
                budget=args.budget,
                generation_budget=args.generation_budget,
                cmdgd_config=(
                    {"max_edits_per_candidate": args.max_edits_per_candidate}
                    if args.max_edits_per_candidate is not None
                    else None
                ),
                seed=args.seed,
            )
        except ValueError as exc:
            parser.error(str(exc))
        if result.get("status") != "completed":
            print(f"Project CMD-GD design unavailable: {result.get('error', 'unknown error')}")
            return 1
        print(f"Wrote CMD-GD project candidates: {result['generated_candidates_path']}")
        print(f"Wrote CMD-GD project summary: {result['design_summary_path']}")
        return 0
    if args.command == "run-project-ph-switch-design":
        from design_scientist.project_replay import run_project_ph_switch_graph_design

        try:
            result = run_project_ph_switch_graph_design(
                args.root,
                run_id=args.run_id,
                budget=args.budget,
                generation_budget=args.generation_budget,
                config=(
                    {"max_edits_per_candidate": args.max_edits_per_candidate}
                    if args.max_edits_per_candidate is not None
                    else None
                ),
                seed=args.seed,
            )
        except ValueError as exc:
            parser.error(str(exc))
        if result.get("status") != "completed":
            print(f"pH-switch graph project design unavailable: {result.get('error', 'unknown error')}")
            return 1
        print(f"Wrote pH-switch graph project candidates: {result['generated_candidates_path']}")
        print(f"Wrote pH-switch graph project summary: {result['design_summary_path']}")
        return 0
    if args.command == "review-framework":
        from design_scientist.framework_validation import review_framework_run

        try:
            report = review_framework_run(
                args.root,
                run_id=args.run_id,
                scientific_only=args.scientific_only,
            )
        except ValueError as exc:
            parser.error(str(exc))
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
    if args.command == "generate-short-paper":
        from design_scientist.manuscript import generate_short_paper

        try:
            result = generate_short_paper(args.root, run_id=args.run_id)
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        readiness = result["readiness"]
        summary = readiness["summary"]
        print(f"Wrote short paper: {result['artifacts']['short_paper.md']}")
        print(f"Wrote paper readiness report: {result['artifacts']['paper_readiness_report.json']}")
        print(
            f"Paper readiness {readiness['status']}: "
            f"{summary['errors']} errors, {summary['warnings']} warnings"
        )
        return 0 if readiness["valid"] else 1
    if args.command == "generate-algorithm-paper":
        from design_scientist.manuscript import generate_algorithm_manuscript

        try:
            result = generate_algorithm_manuscript(
                args.root,
                run_id=args.run_id,
                algorithm=args.algorithm,
            )
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        readiness = result["readiness"]
        summary = readiness["summary"]
        print(f"Wrote algorithm manuscript: {result['artifacts']['manuscript.md']}")
        print(f"Wrote paper readiness report: {result['artifacts']['paper_readiness_report.json']}")
        print(
            f"Paper readiness {readiness['status']}: "
            f"{summary['errors']} errors, {summary['warnings']} warnings"
        )
        return 0 if readiness["valid"] else 1
    if args.command == "audit-schema":
        from design_scientist.startup_standardization import audit_schema

        try:
            report = audit_schema(args.project_dir)
        except ValueError as exc:
            parser.error(str(exc))
        print(
            "Schema audit "
            f"{report['status']}: {report['source_count']} sources, "
            f"{len(report['diagnostics'])} diagnostics"
        )
        print(f"Wrote schema audit: {Path(args.project_dir).resolve() / 'standardized' / 'schema_audit_report.json'}")
        return 0 if report["status"] == "ok" else 1
    if args.command == "standardize-startup-data":
        from design_scientist.startup_standardization import standardize_startup_data

        try:
            result = standardize_startup_data(args.project_dir)
        except ValueError as exc:
            parser.error(str(exc))
        summary = result["summary"]
        print(
            "Startup standardization "
            f"{result['status']}: {summary['sources']} sources, "
            f"{summary['observations']} observations, {summary['sequences']} sequences"
        )
        print(f"Wrote project context: {result['artifacts']['project_context']}")
        return 0 if result["status"] == "ok" else 1
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


def _normalize_v3_cli_mechanism_metrics(result: dict[str, object]) -> None:
    selected_node = result.get("selected_node")
    if not isinstance(selected_node, dict):
        return
    artifacts = selected_node.get("artifacts")
    if not isinstance(artifacts, dict):
        return
    metrics_value = artifacts.get("mechanism_metrics")
    if not isinstance(metrics_value, str):
        return

    metrics_path = Path(metrics_value).expanduser()
    if not metrics_path.is_file():
        return
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(metrics, dict):
        return

    changed = False
    mechanism = selected_node.get("mechanism") or selected_node.get("node_id")
    if isinstance(mechanism, str) and mechanism and metrics.get("mechanism") != mechanism:
        metrics["mechanism"] = mechanism
        changed = True
    summary = metrics.get("mechanism_benchmark_summary")
    if isinstance(summary, dict) and not isinstance(metrics.get("benchmark_summary"), dict):
        metrics["benchmark_summary"] = dict(summary)
        changed = True

    if changed:
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=False) + "\n", encoding="utf-8")


@contextmanager
def _scoped_literature_sources(sources: list[str] | None):
    if sources is None:
        yield
        return

    previous = os.environ.get(LITERATURE_SOURCES_ENV)
    if sources:
        os.environ[LITERATURE_SOURCES_ENV] = ",".join(sources)
    else:
        os.environ.pop(LITERATURE_SOURCES_ENV, None)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(LITERATURE_SOURCES_ENV, None)
        else:
            os.environ[LITERATURE_SOURCES_ENV] = previous


if __name__ == "__main__":
    raise SystemExit(main())
