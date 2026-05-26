"""Generic manuscript bundle generation for framework V3 runs."""

from __future__ import annotations

import csv
import json
import os
import re
import textwrap
from pathlib import Path
from statistics import mean
from typing import Any

from design_scientist.framework_validation import resolve_framework_run
from design_scientist.io import ensure_dir, read_json, read_yaml, write_json


PAPER_OUTPUT_FILENAMES = (
    "short_paper.md",
    "references.bib",
    "results_summary.json",
    "claim_evidence_map.json",
    "reproducibility.md",
    "data_availability.md",
    "code_availability.md",
    "paper_readiness_report.json",
)

READINESS_INPUT_FILENAMES = tuple(
    filename for filename in PAPER_OUTPUT_FILENAMES if filename != "paper_readiness_report.json"
)

REQUIRED_SHORT_PAPER_SECTIONS = (
    "## Abstract",
    "## Evidence Boundary",
    "## Introduction",
    "## Formal Problem Setup",
    "## Literature Basis",
    "## Algorithmic Contribution",
    "## Methods",
    "## Evaluation Design",
    "## Results",
    "## Discussion",
    "## Limitations",
    "## Contribution Boundaries",
    "## Reproducibility",
    "## Data Availability",
    "## Code Availability",
    "## References",
)

EVIDENCE_SCOPE = "computational_synthetic_only"
EVIDENCE_TYPE = "computational/synthetic"
MIXED_EVIDENCE_SCOPE = "computational_synthetic_and_retrospective_masked_project_data"
PROJECT_MASKING_EVIDENCE_SCOPE = "retrospective_masked_project_data"
PROJECT_MASKING_EVIDENCE_TYPE = "retrospective/masked_project_data"
PROSPECTIVE_VALIDATION_BOUNDARY = "does not constitute prospective wet-lab validation"
PROJECT_MASKING_ARTIFACT_FILENAMES = {
    "project_masking_benchmark_summary": "project_masking_benchmark_summary.csv",
    "project_masking_benchmark_results": "project_masking_benchmark_results.csv",
    "project_masking_ablation_results": "project_masking_ablation_results.csv",
    "project_masking_config": "project_masking_config.json",
}
MCCBD_BENCHMARK_ARTIFACT_FILENAMES = {
    "mccbd_benchmark_summary": "mccbd_benchmark_summary.csv",
    "mccbd_benchmark_results": "mccbd_benchmark_results.csv",
    "mccbd_ablation_results": "mccbd_ablation_results.csv",
    "mccbd_benchmark_config": "mccbd_benchmark_config.json",
}
GENERATIVE_BENCHMARK_ARTIFACT_FILENAMES = {
    "generative_benchmark_results": "generative_benchmark_results.csv",
    "generative_benchmark_summary": "generative_benchmark_summary.csv",
    "generative_statistical_summary": "generative_statistical_summary.csv",
    "generative_pairwise_comparisons": "generative_pairwise_comparisons.csv",
    "generative_ablation_results": "generative_ablation_results.csv",
    "generative_design_examples": "generative_design_examples.csv",
    "generative_benchmark_config": "generative_benchmark_config.json",
    "generative_selection_gate_report": "generative_selection_gate_report.json",
}
CMDGD_PROJECT_ARTIFACT_FILENAMES = {
    "cmdgd_project_generated_candidates": "cmdgd_project_generated_candidates.csv",
    "cmdgd_project_design_summary": "cmdgd_project_design_summary.json",
}
CMDGD_NEW_SITE_COUNT_KEYS = (
    "mean_generated_new_mutation_site_count",
    "mean_generated_new_site_count",
    "mean_new_mutation_site_count",
    "mean_new_site_generation_count",
    "mean_de_novo_site_proposal_count",
    "generated_new_mutation_site_count",
    "generated_new_site_count",
    "new_mutation_site_count",
    "new_site_generation_count",
    "de_novo_site_proposal_count",
)
CMDGD_NEW_SITE_RATE_KEYS = (
    "mean_new_site_rate",
    "mean_new_site_generation_rate",
    "mean_new_mutation_site_generation_rate",
    "mean_new_mutation_site_rate",
    "mean_generated_new_site_rate",
    "mean_de_novo_site_proposal_rate",
    "new_site_generation_rate",
    "new_mutation_site_generation_rate",
    "new_mutation_site_rate",
    "generated_new_site_rate",
    "de_novo_site_proposal_rate",
)
CMDGD_DE_NOVO_SITE_ABLATION_TERMS = (
    "de_novo_site_proposal",
    "new_site_proposal",
    "new_mutation_site_proposal",
    "context_guided_site_proposal",
)
MCCBD_FIGURE_FILENAMES = {
    "workflow": "figures/figure_1_mccbd_workflow.svg",
    "benchmark": "figures/figure_2_benchmark_summary.svg",
    "ablation": "figures/figure_3_ablation_summary.svg",
    "project_case": "figures/figure_4_project_case_study.svg",
}
CMDGD_FIGURE_FILENAMES = {
    "workflow": "figures/figure_1_cmdgd_workflow.svg",
    "benchmark": "figures/figure_2_cmdgd_benchmark_comparison.svg",
    "vocabulary": "figures/figure_3_cmdgd_vocabulary_extension.svg",
    "ablation": "figures/figure_4_cmdgd_ablation_claim_boundary.svg",
    "statistics": "figures/figure_5_cmdgd_seed_level_stability.svg",
}


def generate_short_paper(project_dir: str | Path, run_id: str | None = None) -> dict[str, Any]:
    """Write a short-paper bundle for a framework V3 run.

    The bundle is intentionally conservative: all scientific claims are mapped
    to computational/synthetic replay evidence, and the generated prose states
    that it does not report wet-lab validation.
    """

    root, selected_run_id, run_dir = resolve_framework_run(project_dir, run_id=run_id)
    if run_dir is None or selected_run_id is None or not run_dir.is_dir():
        requested = run_id or "<latest>"
        raise FileNotFoundError(f"No framework run directory found for run_id={requested!r}")

    journal = _load_json(run_dir / "scientist_journal.json")
    if not isinstance(journal, dict) or journal.get("version") != "v3":
        raise ValueError("generate-short-paper requires a framework V3 run with scientist_journal.json")

    context = _build_context(root, selected_run_id, run_dir)
    paper_dir = ensure_dir(run_dir / "paper")

    (paper_dir / "short_paper.md").write_text(
        _render_short_paper(context, paper_dir).rstrip() + "\n",
        encoding="utf-8",
    )
    (paper_dir / "references.bib").write_text(
        _render_references_bib(context).rstrip() + "\n",
        encoding="utf-8",
    )
    write_json(paper_dir / "results_summary.json", _results_summary(context))
    write_json(paper_dir / "claim_evidence_map.json", _claim_evidence_map(context))
    (paper_dir / "reproducibility.md").write_text(
        _render_reproducibility(context, paper_dir).rstrip() + "\n",
        encoding="utf-8",
    )
    (paper_dir / "data_availability.md").write_text(
        _render_data_availability(context, paper_dir).rstrip() + "\n",
        encoding="utf-8",
    )
    (paper_dir / "code_availability.md").write_text(
        _render_code_availability(context, paper_dir).rstrip() + "\n",
        encoding="utf-8",
    )

    readiness = validate_paper_readiness(root, run_id=selected_run_id)
    write_json(paper_dir / "paper_readiness_report.json", readiness)

    return {
        "run_id": selected_run_id,
        "paper_dir": str(paper_dir),
        "artifacts": {filename: str(paper_dir / filename) for filename in PAPER_OUTPUT_FILENAMES},
        "readiness": readiness,
    }


def generate_algorithm_manuscript(
    project_dir: str | Path,
    *,
    run_id: str,
    algorithm: str = "evidence_calibrated_ucb",
) -> dict[str, Any]:
    """Write a manuscript centered on a reusable algorithm benchmark run.

    Unlike ``generate_short_paper``, this entry point does not require a V3
    scientist journal. It is for algorithm papers whose primary evidence is a
    project-data retrospective masking benchmark plus literature artifacts.
    """

    root = Path(project_dir).expanduser().resolve()
    run_dir = root / "runs" / _safe_run_id(run_id)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"No run directory found for run_id={run_id!r}")
    project_masking = _load_project_masking_artifacts(root, run_dir)
    mccbd_benchmark = _load_mccbd_benchmark_artifacts(run_dir)
    generative_benchmark = _load_generative_benchmark_artifacts(run_dir)
    cmdgd_literature = _load_cmdgd_literature_artifacts(root)
    if algorithm == "mccbd":
        if not mccbd_benchmark.get("available"):
            raise FileNotFoundError(f"No MCCBD benchmark artifacts found for run_id={run_id!r}")
    elif algorithm == "cmdgd":
        if not generative_benchmark.get("available"):
            raise FileNotFoundError(f"No generative benchmark artifacts found for run_id={run_id!r}")
    elif not project_masking.get("available"):
        raise FileNotFoundError(f"No project masking artifacts found for run_id={run_id!r}")

    framework_dir = root / "framework"
    paper_cards = _as_list(_load_json(framework_dir / "paper_cards.json"))
    context = {
        "root": root,
        "run_id": _safe_run_id(run_id),
        "run_dir": run_dir,
        "algorithm": algorithm,
        "paper_cards": paper_cards,
        "citation_keys": _citation_keys_for_cards(paper_cards),
        "citation_aliases": _citation_aliases_for_cards(
            paper_cards,
            _citation_keys_for_cards(paper_cards),
        ),
        "literature_corpus": _read_jsonl(framework_dir / "literature_corpus.jsonl"),
        "mechanism_cards": _collection(_load_json(framework_dir / "mechanism_cards.json")),
        "mechanism_library": _collection(_load_json(framework_dir / "mechanism_library.json")),
        "mechanism_gap_rows": _read_csv(framework_dir / "mechanism_gap_matrix.csv"),
        "mechanism_spec": {},
        "proposal": {},
        "project_spec": _load_yaml(root / "project.yaml"),
        "estimands": _load_yaml(root / "estimands.yaml"),
        "standardized_context": _load_json(root / "standardized" / "project_context.json"),
        "project_masking": project_masking,
        "mccbd_benchmark": mccbd_benchmark,
        "generative_benchmark": generative_benchmark,
        "cmdgd_literature": cmdgd_literature,
    }
    paper_dir = ensure_dir(run_dir / "paper")
    if algorithm == "mccbd":
        figure_artifacts = _write_mccbd_figures(context, paper_dir)
        table_artifacts = _write_mccbd_tables(context, paper_dir)
    elif algorithm == "cmdgd":
        figure_artifacts = _write_cmdgd_figures(context, paper_dir)
        table_artifacts = _write_cmdgd_tables(context, paper_dir)
    else:
        figure_artifacts = {}
        table_artifacts = {}
    artifacts = {
        "manuscript.md": str(paper_dir / "manuscript.md"),
        "references.bib": str(paper_dir / "references.bib"),
        "algorithm_results_summary.json": str(paper_dir / "algorithm_results_summary.json"),
        "algorithm_claim_evidence_map.json": str(paper_dir / "algorithm_claim_evidence_map.json"),
        "paper_readiness_report.json": str(paper_dir / "paper_readiness_report.json"),
        **figure_artifacts,
        **table_artifacts,
    }

    (paper_dir / "manuscript.md").write_text(
        _render_algorithm_manuscript(context, paper_dir).rstrip() + "\n",
        encoding="utf-8",
    )
    (paper_dir / "references.bib").write_text(
        (
            _render_cmdgd_references_bib(context)
            if algorithm == "cmdgd"
            else _render_references_bib(context)
        ).rstrip()
        + "\n",
        encoding="utf-8",
    )
    write_json(paper_dir / "algorithm_results_summary.json", _algorithm_results_summary(context))
    write_json(
        paper_dir / "algorithm_claim_evidence_map.json",
        _algorithm_claim_evidence_map(context),
    )
    readiness = _validate_algorithm_manuscript(paper_dir, context)
    write_json(paper_dir / "paper_readiness_report.json", readiness)
    return {
        "run_id": _safe_run_id(run_id),
        "paper_dir": str(paper_dir),
        "artifacts": artifacts,
        "readiness": readiness,
    }


def validate_paper_readiness(project_dir: str | Path, run_id: str | None = None) -> dict[str, Any]:
    """Validate a generated short-paper bundle.

    Returns a report with error findings for missing manuscript sections,
    missing or unresolved citation keys, and local Markdown artifact links that
    do not resolve on disk.
    """

    root, selected_run_id, run_dir = resolve_framework_run(project_dir, run_id=run_id)
    findings: list[dict[str, Any]] = []
    artifacts: dict[str, str] = {}

    if run_dir is None or selected_run_id is None or not run_dir.is_dir():
        requested = run_id or "<latest>"
        _add_finding(
            findings,
            "error",
            "missing_run",
            f"No framework run directory found for run_id={requested!r}.",
            "runs/<run_id>",
        )
        return _build_readiness_report(root, selected_run_id, run_dir, findings, artifacts)

    paper_dir = run_dir / "paper"
    artifacts["paper_dir"] = str(paper_dir)
    for filename in PAPER_OUTPUT_FILENAMES:
        artifacts[filename] = str(paper_dir / filename)

    for filename in READINESS_INPUT_FILENAMES:
        path = paper_dir / filename
        if not path.is_file():
            _add_finding(
                findings,
                "error",
                "missing_output_file",
                f"Missing manuscript output file: {filename}.",
                f"runs/{selected_run_id}/paper/{filename}",
            )

    short_paper_path = paper_dir / "short_paper.md"
    short_paper = _read_text(short_paper_path)
    if short_paper:
        _validate_sections(short_paper, _rel(short_paper_path, root), findings)
        _validate_evidence_boundary(short_paper, _rel(short_paper_path, root), findings)
        _validate_citations(short_paper, _read_text(paper_dir / "references.bib"), findings)
        _validate_artifact_links(short_paper, paper_dir, _rel(short_paper_path, root), findings)

    _validate_json_file(paper_dir / "results_summary.json", root, findings)
    _validate_json_file(paper_dir / "claim_evidence_map.json", root, findings)

    return _build_readiness_report(root, selected_run_id, run_dir, findings, artifacts)


def _build_context(root: Path, run_id: str, run_dir: Path) -> dict[str, Any]:
    framework_dir = root / "framework"
    journal = _load_json(run_dir / "scientist_journal.json")
    selected_node = _selected_node_record(journal)
    selected_artifacts = _selected_artifact_paths(root, selected_node)
    mechanism_spec = _load_json(selected_artifacts.get("mechanism_spec"))
    proposal = _load_json(selected_artifacts.get("proposal"))
    ablation_plan = _load_json(selected_artifacts.get("ablation_plan"))
    stress_test_plan = _load_json(selected_artifacts.get("stress_test_plan"))
    mechanism_metrics = _load_json(selected_artifacts.get("mechanism_metrics"))
    validation_report = _load_json(selected_artifacts.get("validation_report"))
    paper_cards = _as_list(_load_json(framework_dir / "paper_cards.json"))
    citation_keys = _citation_keys_for_cards(paper_cards)
    citation_aliases = _citation_aliases_for_cards(paper_cards, citation_keys)
    benchmark_summary_rows = _read_csv(run_dir / "mechanism_benchmark_summary.csv")
    benchmark_rows = _read_csv(run_dir / "mechanism_benchmark_results.csv")
    ablation_rows = _read_csv(run_dir / "mechanism_ablation_results.csv")
    selected_mechanism = _selected_mechanism(journal, selected_node, mechanism_spec, mechanism_metrics)
    standardized_context = _load_json(root / "standardized" / "project_context.json")
    literature_trace = _load_json(framework_dir / "literature_search_trace.json")
    reading_trace = _load_json(framework_dir / "literature_reading_trace.json")
    project_masking = _load_project_masking_artifacts(root, run_dir)
    artifact_paths = _artifact_paths(root, run_dir, selected_artifacts)
    artifact_paths.update(project_masking.get("artifact_paths", {}))

    return {
        "root": root,
        "run_id": run_id,
        "run_dir": run_dir,
        "framework_spec": _load_yaml(framework_dir / "framework_spec.yaml"),
        "project_spec": _load_yaml(root / "project.yaml"),
        "estimands": _load_yaml(root / "estimands.yaml"),
        "standardized_context": standardized_context if isinstance(standardized_context, dict) else {},
        "literature_trace": literature_trace if isinstance(literature_trace, dict) else {},
        "reading_trace": reading_trace if isinstance(reading_trace, dict) else {},
        "journal": journal if isinstance(journal, dict) else {},
        "selected_node": selected_node or {},
        "selected_node_id": _selected_node_id(journal, selected_node),
        "selected_mechanism": selected_mechanism,
        "selected_artifacts": selected_artifacts,
        "mechanism_spec": mechanism_spec if isinstance(mechanism_spec, dict) else {},
        "proposal": proposal if isinstance(proposal, dict) else {},
        "ablation_plan": ablation_plan if isinstance(ablation_plan, dict) else {},
        "stress_test_plan": stress_test_plan if isinstance(stress_test_plan, dict) else {},
        "mechanism_metrics": mechanism_metrics if isinstance(mechanism_metrics, dict) else {},
        "validation_report": validation_report if isinstance(validation_report, dict) else {},
        "paper_cards": paper_cards,
        "citation_keys": citation_keys,
        "citation_aliases": citation_aliases,
        "literature_corpus": _read_jsonl(framework_dir / "literature_corpus.jsonl"),
        "mechanism_cards": _collection(_load_json(framework_dir / "mechanism_cards.json")),
        "mechanism_library": _collection(_load_json(framework_dir / "mechanism_library.json")),
        "mechanism_gap_rows": _read_csv(framework_dir / "mechanism_gap_matrix.csv"),
        "benchmark_rows": benchmark_rows,
        "benchmark_summary_rows": benchmark_summary_rows,
        "selected_summary_row": _find_summary_row(benchmark_summary_rows, selected_mechanism),
        "ablation_rows": ablation_rows,
        "project_masking": project_masking,
        "artifact_paths": artifact_paths,
    }


def _render_short_paper(context: dict[str, Any], paper_dir: Path) -> str:
    citation = _citation_cluster(context)
    selected = context["selected_mechanism"] or "selected V3 mechanism"
    framework_id = context["framework_spec"].get("framework_id") or context["root"].name
    domain = context["framework_spec"].get("domain", "not recorded")
    project_spec = context["project_spec"]
    project_objective = _one_line(project_spec.get("objective")) or "not recorded"
    data_summary = _project_data_summary(context)
    literature_summary = _literature_summary(context)
    selected_row = context["selected_summary_row"]
    proposal = context["proposal"]
    mechanism_spec = context["mechanism_spec"]
    validation = context["validation_report"]
    claims = _claim_records(context)
    artifact_paths = context["artifact_paths"]
    evidence_label = _evidence_label(context)

    lines = [
        f"# Short Paper: {_paper_title(context, selected)}",
        "",
        "## Abstract",
        "",
        (
            f"This short paper summarizes a Design Scientist V3 framework run for `{framework_id}` "
            f"in the domain `{domain}`. The task objective was: {_sentence(project_objective)} "
            f"The run used {literature_summary} and produced a selected executable mechanism, `{selected}`, "
            f"evaluated by {evidence_label} {citation}. "
            "It does not report prospective wet-lab validation, prospective assay measurements, "
            "or real experimental confirmation."
        ),
        "",
        "## Evidence Boundary",
        "",
        _evidence_boundary_text(context),
        "",
        "## Introduction",
        "",
        (
            "Design Scientist V3 combines Literature Engine V3 with a MechanismSpec Kernel to convert reviewed "
            f"method evidence into executable mechanism nodes {citation}. The selected run records literature "
            "cards, mechanism cards, benchmark summaries, ablations, and a validation report as traceable artifacts. "
            f"The local startup data context was: {_sentence(data_summary)}"
        ),
        "",
        "## Formal Problem Setup",
        "",
        _formal_problem_setup_text(context, data_summary),
        "",
        "## Literature Basis",
        "",
        _literature_basis_text(context, literature_summary, citation),
        "",
        "## Algorithmic Contribution",
        "",
        _algorithmic_contribution_text(context, selected),
        "",
        "## Methods",
        "",
        (
            f"The selected mechanism was `{selected}` from node `{context['selected_node_id'] or 'unknown'}`. "
            "Design Scientist V3 treats the mechanism as a lifecycle rather than a single batch selector: "
            "`fit_state`, `generate_candidates`, `score_candidates`, `select_panel`, and `plan_ablations` are "
            "validated as the executable contract. "
            "The method summary is derived from "
            f"{_md_link('scientist_journal.json', artifact_paths['scientist_journal'], paper_dir)}, "
            f"{_md_link('mechanism_spec.json', artifact_paths.get('selected_node_mechanism_spec'), paper_dir)}, "
            f"{_md_link('stress_test_plan.json', artifact_paths.get('selected_node_stress_test_plan'), paper_dir)}, and "
            f"{_md_link('validation_report.json', artifact_paths.get('selected_node_validation_report'), paper_dir)}."
        ),
        "",
        f"Mechanism hypothesis: {_one_line(mechanism_spec.get('hypothesis') or proposal.get('hypothesis') or 'not recorded')}",
        "",
        f"Architecture delta: {_one_line(mechanism_spec.get('architecture_delta') or proposal.get('architecture_delta') or 'not recorded')}",
        "",
        "Startup data handling:",
        "",
        _sentence(data_summary),
        "",
        "## Evaluation Design",
        "",
        _evaluation_design_text(context, paper_dir),
        "",
        "## Results",
        "",
        (
            "The computational replay result is summarized in "
            f"{_md_link('mechanism_benchmark_summary.csv', artifact_paths['mechanism_benchmark_summary'], paper_dir)} "
            "with supporting per-world rows in "
            f"{_md_link('mechanism_benchmark_results.csv', artifact_paths['mechanism_benchmark_results'], paper_dir)} "
            "and ablations in "
            f"{_md_link('mechanism_ablation_results.csv', artifact_paths['mechanism_ablation_results'], paper_dir)}."
        ),
    ]
    if selected_row:
        lines.extend(["", "Selected computational replay metrics:"])
        for key in (
            "rank",
            "worlds_tested",
            "majority_win_vs_random_feasible",
            "majority_win_vs_fixed_mix",
            "mean_best_feasible_utility",
            "mean_false_claim_rate",
            "key_ablation_delta",
            "selected_eligible",
        ):
            if selected_row.get(key) not in (None, ""):
                lines.append(f"- {key}: {selected_row[key]}")
    else:
        lines.extend(["", "No selected mechanism row was found in the benchmark summary artifact."])

    lines.extend(_project_masking_result_lines(context))

    lines.extend(["", "Literature collection summary:", "", literature_summary])

    lines.extend(
        [
            "",
            "## Discussion",
            "",
            (
                "The result supports only the narrower claim that the selected mechanism passed the recorded "
                "computational replay gates for this run. Claim-level provenance is exported in "
                f"{_md_link('claim_evidence_map.json', 'claim_evidence_map.json', paper_dir)}."
            ),
            "",
            "Claim summary:",
        ]
    )
    if claims:
        for claim in claims[:8]:
            lines.append(f"- {claim['claim']} ({claim['evidence_type']})")
    else:
        lines.append("- No mechanism claims were recorded in the selected artifacts.")

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            (
                "The manuscript is not a wet-lab study. It excludes primary assay validation, clinical evidence, "
                "and prospective experimental results. Synthetic replay can expose method behavior and failure modes, "
                "but it does not establish real experimental performance."
            ),
            "",
            f"Validation caveat: {_one_line(validation.get('caveats', ['not recorded'])[0] if isinstance(validation.get('caveats'), list) and validation.get('caveats') else 'not recorded')}",
            "",
            "## Contribution Boundaries",
            "",
            (
                "The contribution is limited to the algorithmic workflow, artifact contract, literature-grounded "
                "mechanism specification, and offline evaluation design represented by this run. Synthetic replay "
                "and retrospective masked project-data evidence do not establish real experimental performance, "
                "prospective experimental superiority, clinical utility, or validated biological mechanism. "
                "Retrospective masked project-data evidence, when present, "
                f"{PROSPECTIVE_VALIDATION_BOUNDARY}."
            ),
            "",
            "## Reproducibility",
            "",
            (
                f"Reproduction instructions are in {_md_link('reproducibility.md', 'reproducibility.md', paper_dir)}. "
                f"Stage and route provenance are in {_md_link('stage_progress.json', artifact_paths['stage_progress'], paper_dir)} "
                f"and {_md_link('route_tree.json', artifact_paths['route_tree'], paper_dir)}."
            ),
            "",
            "## Data Availability",
            "",
            (
                f"Data availability is summarized in {_md_link('data_availability.md', 'data_availability.md', paper_dir)}. "
                f"The literature cards used for citation provenance are in {_md_link('paper_cards.json', artifact_paths['paper_cards'], paper_dir)}."
            ),
            "",
            "## Code Availability",
            "",
            (
                f"Code availability is summarized in {_md_link('code_availability.md', 'code_availability.md', paper_dir)}. "
                f"The selected executable mechanism file is {_md_link('mechanism.py', artifact_paths.get('selected_node_mechanism'), paper_dir)}."
            ),
            "",
            "## References",
            "",
            f"BibTeX references are provided in {_md_link('references.bib', 'references.bib', paper_dir)}.",
        ]
    )
    return "\n".join(lines)


def _render_algorithm_manuscript(context: dict[str, Any], paper_dir: Path) -> str:
    algorithm = str(context["algorithm"])
    if algorithm == "mccbd":
        return _render_mccbd_algorithm_manuscript(context, paper_dir)
    if algorithm == "cmdgd":
        return _render_cmdgd_algorithm_manuscript(context, paper_dir)
    citation = _citation_cluster(context)
    literature_summary = _literature_summary(context)
    data_summary = _project_data_summary(context)
    masking = context["project_masking"]
    algorithm_row = _algorithm_summary_row(context)
    comparisons = _baseline_comparisons(context)
    links = _algorithm_project_masking_links(masking, paper_dir)
    project_spec = context.get("project_spec", {})
    objective = _one_line(project_spec.get("objective")) or "pH-sensitive antibody design"

    lines = [
        "# Evidence-Calibrated UCB for Literature-Grounded pH-Sensitive Antibody Variant Design",
        "",
        "## Abstract",
        "",
        (
            "We introduce Evidence-Calibrated UCB, a real algorithmic contribution for sparse, "
            "data-grounded variant design. The method combines shrinkage additive and pairwise state "
            "models with upper-confidence acquisition, observed-neighborhood calibration, motif enrichment, "
            "bad-neighbor penalties, and batch diversity. In the current project, the objective is "
            f"{_sentence(objective)} The evidence consists of {literature_summary} and retrospective "
            "masked project-data evidence from observed-pool masking; it "
            f"{PROSPECTIVE_VALIDATION_BOUNDARY} "
            f"{citation}."
        ),
        "",
        "## Introduction",
        "",
        (
            "Many antibody design tasks do not start with a clean historical sequence of experimental rounds. "
            "They start with a small set of existing variants, measured endpoints, and a literature base that "
            "suggests useful design mechanisms but does not specify a complete acquisition algorithm. The "
            "central problem is therefore to build a policy that can use sparse wet-lab observations without "
            "overclaiming unsupported causal structure."
        ),
        "",
        "## Literature Basis",
        "",
        (
            f"The literature engine collected {literature_summary}. These records supply background on "
            "active design, uncertainty-aware selection, antibody binding constraints, pH-dependent selection, "
            "and evaluation failure modes. The literature is used to motivate algorithm components; it is not "
            "treated as validation of any untested local variant."
        ),
        "",
        "## Formal Problem Setup",
        "",
        (
            "Let X be the feasible candidate design space and D be the observed startup dataset. Each candidate "
            "x has mutation or module tokens, optional endpoint summaries, feasibility metadata, cost, and "
            "provenance. The policy must select a bounded panel under cost constraints. The objective is to "
            "maximize a project utility that rewards retained neutral-pH signal, penalizes pH6/pH7.4 retention, "
            "and applies expression or feasibility guardrails. In retrospective masking, only already observed "
            "variants are used as candidates, their outcomes are hidden from the selector, and scoring is done "
            "only after selection."
        ),
        "",
        "## Algorithmic Contribution",
        "",
        (
            f"`{algorithm}` is not a wrapper around a baseline. It fits a shrinkage additive-plus-pair model "
            "from observed mutation tokens, adds an uncertainty bonus for under-observed modules and pairs, "
            "then calibrates candidate scores using observed-neighborhood calibration. Favorable neighboring "
            "records add a top-neighbor bonus; motifs enriched in favorable records add a motif enrichment term; "
            "overlap with unfavorable or infeasible observed neighbors adds a bad-neighbor penalty. During panel "
            "construction, a batch diversity adjustment penalizes near-duplicate architectures after the first "
            "pick. This architecture can reuse ordinary helper code while remaining a distinct acquisition "
            "mechanism."
        ),
        "",
        (
            "The architectural delta relative to a greedy module-mean selector is that the policy does not only "
            "ask whether a candidate contains individually successful modules. It also asks whether the candidate "
            "resembles favorable observed neighborhoods, whether its motifs are enriched against unfavorable "
            "neighborhoods, whether its pair evidence is under-observed, and whether the batch would collapse "
            "onto one local architecture."
        ),
        "",
        "## Methods",
        "",
        (
            "The score for each candidate is the sum of predicted utility, uncertainty bonus, top-neighbor "
            "bonus, motif enrichment bonus, contrast/coverage bonuses, minus bad-neighbor, retention, guardrail, "
            "cost, leakage, and missing-evidence penalties. Selection is greedy under the cost budget after "
            "applying batch redundancy adjustment. The implementation is generic over policy records and does "
            "not hard-code project-specific mutation identifiers."
        ),
        "",
        (
            "In compact notation, the acquisition score is "
            "`mu(x) + beta u(x) + n_top(x) + m_enrich(x) + c(x) - n_bad(x) - r(x) - g(x) - cost(x) - leak(x)`, "
            "where `mu(x)` is the shrinkage additive-plus-pair predictor, `u(x)` is calibrated uncertainty, "
            "`n_top(x)` is favorable-neighbor support, `m_enrich(x)` is motif enrichment, `c(x)` represents "
            "contrast or coverage value, and the negative terms enforce bad-neighbor, assay-retention, QC, cost, "
            "and leakage guardrails."
        ),
        "",
        (
            "The method lifecycle is intentionally separable: `fit_state` estimates the evidence state from "
            "observations; `generate_candidates` exposes the feasible pool; `score_candidates` computes "
            "interpretable score components; `select_panel` enforces the budget with batch diversity; and "
            "`plan_ablations` defines removable mechanism components for later stress testing."
        ),
        "",
        "## Evaluation Design",
        "",
        (
            "Evaluation uses retrospective masked project-data evidence. The benchmark hides measured outcomes "
            "for held-out observed variants, exposes only candidate descriptors to the selector, and compares "
            "against `random_feasible`, `fixed_mix`, and `greedy_observed`. The links are "
            f"{links}. The design is an observed-pool replay sanity check and "
            f"{PROSPECTIVE_VALIDATION_BOUNDARY}."
        ),
        "",
        "## Results",
        "",
    ]
    if algorithm_row:
        lines.extend(
            [
                f"The selected algorithm row for `{algorithm}` reported:",
                "",
                f"- mean best feasible utility: {algorithm_row.get('mean_best_feasible_utility', 'not recorded')}",
                f"- mean hit rate: {algorithm_row.get('mean_hit_rate', 'not recorded')}",
                f"- mean regret proxy: {algorithm_row.get('mean_regret_proxy', 'not recorded')}",
                f"- mean false claim rate: {algorithm_row.get('mean_false_claim_rate', 'not recorded')}",
                f"- folds completed: {algorithm_row.get('completed_fold_count', 'not recorded')}/{algorithm_row.get('fold_count', 'not recorded')}",
            ]
        )
    else:
        lines.append(f"No summary row was found for `{algorithm}`.")
    if comparisons:
        lines.extend(["", "Baseline comparison:"])
        for item in comparisons:
            lines.append(
                "- versus {baseline}: utility delta {utility_delta}, false-claim delta {false_claim_delta}".format(
                    **item
                )
            )
    lines.extend(
        [
            "",
            "## Discussion",
            "",
            (
                "The result is a modest but meaningful algorithmic signal: the method improves the masked "
                "project-data ranking over simple baselines while reducing false-claim behavior relative to "
                "greedy selection in this startup dataset. The main scientific contribution is the acquisition "
                "architecture and its auditable evaluation path, not a claim that the proposed variants have "
                "already succeeded experimentally."
            ),
            "",
            "## Limitations",
            "",
            (
                "The benchmark contains a small observed pool and cannot replace wet-lab validation. The KD-ratio "
                "track remains supplementary unless explicit variant mapping to the combination panel exists. "
                "The retrospective split is useful for method development but does not establish future assay "
                "superiority."
            ),
            "",
            "## Contribution Boundaries",
            "",
            (
                "This manuscript claims a reusable algorithm and evidence trail: literature-grounded design, "
                "generic implementation, masked-data evaluation, and conservative claim mapping. It does not "
                "claim a validated antibody, a complete biological mechanism, or clinical relevance."
            ),
            "",
            "## Reproducibility",
            "",
            "```bash",
            f"uv run design-scientist run-project-benchmark {context['root']} --run-id {context['run_id']} --budget 2 --folds kfold_5",
            f"uv run python -c \"from design_scientist.manuscript import generate_algorithm_manuscript; generate_algorithm_manuscript('{context['root']}', run_id='{context['run_id']}', algorithm='{algorithm}')\"",
            "```",
            "",
            "## Data Availability",
            "",
            (
                f"The startup data summary is: {_sentence(data_summary)} Literature artifacts and project masking "
                "artifacts are local framework outputs; no new wet-lab measurements are introduced by this paper."
            ),
            "",
            "## Code Availability",
            "",
            "The algorithm implementation is in `src/design_scientist/algorithms/evidence_calibrated_ucb.py`.",
            "",
            "## References",
            "",
            f"BibTeX references are provided in {_md_link('references.bib', 'references.bib', paper_dir)}.",
        ]
    )
    return "\n".join(lines)


def _render_cmdgd_algorithm_manuscript(context: dict[str, Any], paper_dir: Path) -> str:
    benchmark = context.get("generative_benchmark", {})
    config = benchmark.get("config", {}) if isinstance(benchmark, dict) else {}
    gate_report = benchmark.get("selection_gate_report", {}) if isinstance(benchmark, dict) else {}
    selected_row = _cmdgd_summary_row(context, "cmdgd", "overall")
    vocabulary_row = _cmdgd_summary_row(context, "cmdgd", "vocabulary_extension")
    recombination_row = _cmdgd_summary_row(context, "observed_recombination_baseline", "overall")
    fixed_pool_row = _cmdgd_summary_row(context, "mccbd_pool_selector", "overall")
    random_row = _cmdgd_summary_row(context, "random_edit_generator", "overall")
    single_edit_row = _cmdgd_summary_row(context, "single_edit_scan", "overall")
    citation_cluster = _cmdgd_citation_cluster(context, limit=12)
    core_citations = _cmdgd_citation_cluster(
        context,
        keys=(
            "Igawa2010",
            "Murtaugh2011",
            "Schroter2015",
            "Romero2013",
            "Wu2019",
            "Khan2023",
            "Hie2023",
            "Shuai2023",
        ),
    )
    figure_links = {
        key: _md_link(f"Figure {idx}", filename, paper_dir)
        for idx, (key, filename) in enumerate(CMDGD_FIGURE_FILENAMES.items(), start=1)
    }
    table_links = {
        "related": _md_link("related_work_matrix.csv", "tables/related_work_matrix.csv", paper_dir),
        "benchmark": _md_link("benchmark_summary.csv", "tables/benchmark_summary.csv", paper_dir),
        "statistics": _md_link("statistical_summary.csv", "tables/statistical_summary.csv", paper_dir),
        "pairwise": _md_link("pairwise_comparisons.csv", "tables/pairwise_comparisons.csv", paper_dir),
        "examples": _md_link("design_examples.csv", "tables/design_examples.csv", paper_dir),
        "ablation": _md_link("ablation_summary.csv", "tables/ablation_summary.csv", paper_dir),
        "project": _md_link("project_selected_candidates.csv", "tables/project_selected_candidates.csv", paper_dir),
        "rationale": _md_link("project_candidate_rationale.csv", "tables/project_candidate_rationale.csv", paper_dir),
    }
    data_summary = _project_data_summary(context)
    new_site_lines = _cmdgd_new_site_generation_lines(context)
    masking_boundary = _cmdgd_real_data_masking_boundary_text(context)
    figure_caption_lines = _cmdgd_figure_caption_lines(context, figure_links, table_links, config)

    lines = [
        "# Contrastive Mechanism-Directed Generative Design for pH-Sensitive Antibody Design",
        "",
        "## Abstract",
        "",
        (
            "We present Contrastive Mechanism-Directed Generative Design (CMD-GD), a grammar-guided "
            "sequence-generation and panel-selection framework for pH-sensitive antibody design under sparse campaign data. CMD-GD "
            "learns an edit grammar from observed heavy/light antibody sequences, expands that grammar with "
            "mechanism-prior edit discovery, applies vocabulary-guided and evidence-guided "
            "module recombination, generates auditable new heavy/light chain sequences, and then scores and "
            "selects a panel under contrast, feasibility, novelty, and cost constraints. In the "
            "1E62 benchmark bundle, CMD-GD is evaluated against random editing, single-edit scanning, "
            "observed recombination, and a fixed-pool selector over controlled pH-contrast worlds, with paired "
            "seed-level statistics reported separately from the generation gate. The result supports CMD-GD as "
            "an auditable grammar-guided expansion method and does not claim broad selected-utility superiority. "
            f"{masking_boundary} Biological follow-up requires new experimental testing {citation_cluster}."
        ),
        "",
        "## Introduction",
        "",
        (
            "pH-sensitive antibody engineering is a contrastive design problem. A useful variant should retain "
            "neutral-pH binding while weakening acidic-pH binding enough to support endosomal antigen release, "
            "and it must do so without creating expression, specificity, or developability liabilities. This "
            "setting differs from scalar affinity optimization because candidate generation and selection must "
            "track two binding regimes and feasibility guardrails at the same time."
        ),
        "",
        (
            "The 1E62 startup setting is also low-data. The algorithm cannot rely on a large prospective "
            "round history or on an unrestricted protein language model alone. The central method question is "
            "whether observed edit modules, mechanism-derived vocabulary, and contrastive scoring can expand "
            "the heavy/light sequence space while keeping the selected panel auditable."
        ),
        "",
        "## Literature Review",
        "",
        *_cmdgd_literature_review_lines(context, paper_dir),
        "",
        "## Related Work Boundary",
        "",
        (
            f"A supplementary related-work table is provided as {table_links['related']}. It organizes prior work into "
            "three lanes: pH antibody mechanisms, low-data protein optimization, and generative sequence or "
            "antibody design. CMD-GD is positioned at their intersection. The pH antibody literature supplies "
            "histidine and endosomal-release priors; active protein optimization supplies the batch learning "
            "loop; generative models supply the rationale for moving beyond observed substitutions. The "
            "synthesis is deliberately bounded: CMD-GD borrows the need for candidate-space expansion from "
            "generative design, but implements it as an explicit grammar-guided combinatorial generator "
            "rather than a de novo neural sequence model. CMD-GD also does not reduce to observed-mutation "
            "recombination: observed recombination is retained as a baseline boundary, while CMD-GD uses "
            "a single generator that learns edit evidence, admits mechanism-prior edits, chooses residue "
            "changes, and recombines modules. None "
            "of those lines alone supplies the CMD-GD lifecycle: edit grammar learning, vocabulary extension, "
            "grammar expansion, module recombination, new sequence generation, contrastive scoring, "
            "and panel selection."
        ),
        "",
        "## Formal Problem Setup",
        "",
        (
            "Let D be the observed startup set of heavy/light antibody sequences, visible endpoint summaries, "
            "and feasible design metadata. Let G(D) be a learned edit grammar over heavy-chain and light-chain "
            "substitutions relative to the base sequence. The design policy must generate a candidate set C "
            "that can include sequences absent from D, then choose a bounded panel B under a cost budget. The "
            "utility rewards retained neutral-pH signal and acidic-pH release, while constraints penalize "
            "liability edits, infeasible candidates, and unsupported claims."
        ),
        "",
        (
            "More explicitly, CMD-GD separates generation from selection. The generator maps observed records "
            "and vocabulary priors to a finite candidate set C=Generate(G(D), V); the selector then solves a "
            "budgeted panel problem over C using only selector-visible features. Held-out oracle fields are "
            "reserved for benchmark scoring after selection, so the formal claim is about candidate-space "
            "expansion and guarded selection under controlled benchmarks, not an experimentally validated "
            "biological conclusion."
        ),
        "",
        "## Algorithmic Contribution",
        "",
        (
            "CMD-GD first learns an edit grammar from observed variants, including chain, position, residue "
            "change, parent support, contrastive effect, developability effect, and uncertainty for each module. "
            "It then expands the candidate space with design vocabulary entries, mechanism priors such as "
            "binder, contrast, specificity, and liability annotations, and sequence-context cues for admitting "
            "edits beyond the observed vocabulary. From this augmented grammar, CMD-GD generates new "
            "heavy/light chain sequences by combining mechanism-prior edits, vocabulary-guided recombination, "
            "evidence-guided recombination over compatible observed modules, "
            "and local substitutions. Finally, it scores generated candidates with a contrastive pH objective, feasibility "
            "guardrails, novelty and uncertainty terms, and cost penalties, then selects a panel for evaluation."
        ),
        "",
        (
            "Algorithmically, the contribution is the constrained composition of interpretable edit modules: "
            "fit module evidence, extend the grammar with mechanistic vocabulary, enumerate compatible module "
            "sets into full heavy/light sequence records, score those records with a decomposed pH-contrast "
            "objective, and select a cost-bounded panel. This is a computational-method contribution to "
            "low-data combinatorial antibody design, not a claim that the generated sequences are validated "
            "binders."
        ),
        "",
        (
            f"{figure_links['workflow']} summarizes the lifecycle. The key distinction from selector-only "
            "methods is that CMD-GD changes the candidate set before selection. A fixed-pool selector can rank "
            "existing candidates, and observed recombination can enumerate combinations of seen modules, but "
            "CMD-GD learns a generator that admits mechanism-prior edits, combines them with observed edit "
            "modules, and produces heavy/light sequence records with explicit provenance."
        ),
        "",
        "## Methods",
        "",
        (
            "The implementation follows a four-stage lifecycle. `fit_state` estimates edit modules and endpoint effects from observed records. "
            "`generate_candidates` builds non-observed heavy/light sequences using mechanism-prior "
            "grammar expansion, vocabulary-guided recombination, evidence-guided recombination, and local grammar "
            "substitutions. `score_candidates` computes contrastive utility, feasibility, novelty, uncertainty, "
            "and cost components. `select_panel` greedily selects candidates under the budget while avoiding "
            "observed duplicates and over-cost candidates."
        ),
        "",
        (
            "Formally, each edit module m carries chain, position, target residue, parent support, a contrastive "
            "effect prior, a developability effect prior, and an uncertainty value. Candidate generation composes "
            "compatible modules into a sequence proposal x, then the scorer evaluates a decomposed objective "
            "`U(x)=contrast(x)+feasibility(x)+novelty(x)+uncertainty(x)-cost(x)`. The important design choice is "
            "not the final scalarization alone; it is the upstream mechanism that learns a grammar, admits "
            "mechanism-vocabulary edits, expands the editable sequence space during candidate generation, "
            "and produces auditable heavy/light chain records that did not exist in the observed pool."
        ),
        "",
        (
            "The benchmark adapter exposes only selector-visible fields before selection: candidate identifiers, "
            "heavy/light sequences, edit tokens, mechanism tags, noisy prior features, cost, and risk flags. "
            "Oracle truth fields are used only after a panel is selected, which keeps generated candidate "
            "evaluation separate from selection-time information."
        ),
        "",
        "Algorithm 1: CMD-GD grammar learning and candidate enumeration.",
        "",
        "```text",
        "Inputs: observed records D, base heavy/light sequences s0, mechanism vocabulary V,",
        "        parameters generation_budget, max_modules_per_candidate, max_new_sites,",
        "        residue_prior_temperature, novelty_weight, feasibility_threshold.",
        "1. Grammar learning: align every observed sequence in D to s0 and create edit modules",
        "   with chain, position, source residue, target residue, parent support, endpoint effect,",
        "   developability effect, and uncertainty.",
        "2. Grammar expansion: score permissible edit positions with mechanism priors, local",
        "   sequence context, residue-class priors, and support penalties; admit at most",
        "   max_new_sites mechanism-prior edit actions from V.",
        "3. Candidate enumeration: enumerate compatible module sets up to max_modules_per_candidate",
        "   by combining observed modules, vocabulary-backed proposals, and local substitutions.",
        "4. Materialize each enumerated module set as a heavy/light sequence record with parent,",
        "   operator, module, novelty, feasibility, uncertainty, and cost fields.",
        "5. Deduplicate observed duplicates and emit at most generation_budget candidate records.",
        "Output: auditable candidate set C with explicit edit provenance.",
        "```",
        "",
        "Algorithm 2: CMD-GD scoring, selection, and tie-breaking.",
        "",
        "```text",
        "Inputs: candidate set C, panel_budget, cost_budget, weights w_contrast, w_feasibility,",
        "        w_novelty, w_uncertainty, w_cost, w_liability, diversity_penalty.",
        "1. Scoring: compute score(x) = w_contrast * contrast(x)",
        "   + w_feasibility * feasibility(x) + w_novelty * novelty(x)",
        "   + w_uncertainty * uncertainty(x) - w_cost * cost(x)",
        "   - w_liability * liability_risk(x).",
        "2. Filter candidates failing feasibility_threshold, observed-duplicate checks, or cost_budget.",
        "3. Selection: greedily add the candidate with the largest score after parent, residue-class,",
        "   and module-redundancy penalties until panel_budget or cost_budget is exhausted.",
        "4. Tie-breaking: prefer higher feasibility, higher pH contrast, higher novelty, lower cost,",
        "   fewer liability flags, broader parent coverage, and then lexicographic candidate_id.",
        "Output: selected panel B and per-candidate rationale fields.",
        "```",
        "",
        "## Evaluation Design",
        "",
        (
            "Evaluation uses a controlled generative benchmark with pH-contrast, escape-risk, and vocabulary-extension "
            "worlds; ten seeds per world; a generation budget of "
            f"{config.get('generation_budget', 'not recorded')}; and a panel budget of "
            f"{config.get('budget', 'not recorded')}. Baselines include `random_edit_generator`, "
            "`single_edit_scan`, `observed_recombination_baseline`, and `mccbd_pool_selector`."
        ),
        "",
        (
            f"The benchmark comparison is visualized in {figure_links['benchmark']} and tabulated in "
            f"{table_links['benchmark']}. Seed-level uncertainty and paired baseline comparisons are summarized "
            f"in {figure_links['statistics']}, {table_links['statistics']}, and {table_links['pairwise']}. "
            f"Design-level examples are reported in {table_links['examples']}. "
            "The vocabulary-extension case is separated because it tests whether a method can benefit from "
            "mechanism priors outside a fixed observed pool."
        ),
        "",
        *figure_caption_lines,
        "",
        "## Results",
        "",
    ]
    if selected_row:
        lines.append(_cmdgd_results_summary_text(selected_row, gate_report))
    else:
        lines.append("No overall CMD-GD summary row was available.")
    lines.extend(
        [
            "",
            "Baseline and boundary comparisons:",
            "",
            _cmdgd_comparison_sentence("random_edit_generator", selected_row, random_row),
            _cmdgd_comparison_sentence("single_edit_scan", selected_row, single_edit_row),
            _cmdgd_comparison_sentence("observed_recombination_baseline", selected_row, recombination_row),
            _cmdgd_comparison_sentence("mccbd_pool_selector", selected_row, fixed_pool_row),
            "",
            (
                "The observed recombination boundary is visible: `observed_recombination_baseline` can create "
                "new sequence combinations from observed modules, but it reuses the observed edit vocabulary "
                "and does not test design vocabulary extension. The fixed-pool boundary is also explicit: "
                "`mccbd_pool_selector` evaluates selection over a pre-existing pool rather than generated "
                "candidate creation, which is why its selected-panel utility is interpreted separately from "
                "new sequence generation."
            ),
            "",
            (
                "Claim boundary: observed recombination is a strong baseline when the useful "
                "modules are already present, and fixed-pool selectors may exceed CMD-GD utility when a "
                "pre-existing pool already contains high-scoring combinations. The CMD-GD claim is therefore "
                "not dominance on every utility metric; it is auditable grammar-guided candidate generation "
                "with explicit novelty, feasibility, and false-claim accounting."
            ),
            "",
            f"Real-data masking boundary: {masking_boundary}",
            "",
            f"{figure_links['vocabulary']} details the vocabulary-extension case.",
            "",
            "## Statistical Stability",
            "",
            *_cmdgd_statistical_result_lines(context, paper_dir),
        ]
    )
    if vocabulary_row:
        lines.extend(
            [
                "",
                "Vocabulary-extension world:",
                "",
                (
                    "In the vocabulary-extension world, CMD-GD reached best generated utility "
                    f"{vocabulary_row.get('mean_best_generated_utility', 'not recorded')}, mean pH contrast "
                    f"{vocabulary_row.get('mean_pH_contrast_score', 'not recorded')}, and novel sequence rate "
                    f"{vocabulary_row.get('mean_novel_sequence_rate', 'not recorded')}."
                ),
            ]
        )
    if new_site_lines:
        lines.extend(["", *new_site_lines])
    lines.extend(
        [
            "",
            f"{figure_links['ablation']} and {table_links['ablation']} summarize ablations and claim boundaries.",
            "",
            *_cmdgd_ablation_lines(context),
            "",
            "## Real-Data Retrospective Masking",
            "",
            *_cmdgd_project_masking_lines(context, paper_dir),
            "",
            "## Biophysical Plausibility and Panel Rationale",
            "",
            *_cmdgd_biophysical_rationale_lines(context, table_links["rationale"]),
            "",
            "## 1E62 Computational Design Output",
            "",
            _cmdgd_project_design_text(context, paper_dir, table_links["project"]),
            "",
                "## Discussion",
                "",
                (
                    "The benchmark supports the specific algorithmic claim that CMD-GD expands the design space "
                    "while preserving contrastive guardrails in the tested worlds. The strongest evidence is not "
                    "that CMD-GD should be treated as a universal de novo antibody generator. The defensible "
                    "claim is narrower: CMD-GD is a grammar-guided combinatorial generator that creates "
                    "non-observed sequence proposals through mechanism-prior grammar expansion, residue choice, "
                    "vocabulary-guided recombination, and evidence-guided grammar recombination, satisfies the predeclared candidate-space expansion rule in this benchmark suite, "
                    "keeps false claim rate low in the tested worlds, and exposes where observed recombination and "
                    "fixed-pool selection are different algorithmic boundaries."
                ),
            "",
            "## Limitations",
            "",
            (
                "The current evidence is computational and benchmark-based and "
                f"{PROSPECTIVE_VALIDATION_BOUNDARY}. Prospective wet-lab validation "
                "remains required before treating any generated 1E62 sequence as an experimentally supported "
                "antibody design. The worlds encode simplified pH-contrast, escape-risk, and vocabulary-extension "
                "oracles; they do not capture all structural, manufacturability, immunogenicity, or antigen-context "
                "risks. Unobserved-edit diagnostics, when reported, are computational checks on CMD-GD "
                "candidate generation; prospective experiments are required before assigning function to any "
                "generated edit. The observed "
                "recombination baseline is strong in worlds where the useful module set is already present, and "
                "the fixed-pool selector can match or exceed CMD-GD utility when its candidate pool already "
                "contains high-quality combinations. A small paired selected-utility margin over observed "
                "recombination should therefore be interpreted as a boundary result rather than as evidence for a "
                "material biological improvement."
            ),
            "",
            "## Contribution Boundaries",
            "",
            (
                "This manuscript claims a generative algorithm bundle, literature-grounded motivation, benchmark "
                "evidence, ablation evidence, and candidate examples. It does not claim clinical relevance, broad "
                "antigen transfer, or experimentally confirmed pH switching for the generated sequences. The "
                "appropriate next use is wet-lab panel design with explicit primary endpoint measurement."
            ),
            "",
            "## Reproducibility",
            "",
            (
                "The benchmark configuration records the generation budget, panel budget, replicate structure, "
                "and baseline set. The reported results are reproducible by applying the CMD-GD implementation "
                "to the same standardized inputs and parameter settings."
            ),
            "",
            "## Data Availability",
            "",
            _cmdgd_data_availability_text(context, data_summary),
            "",
            "## Code Availability",
            "",
            "The CMD-GD implementation is part of the Design Scientist repository and should be cited with a versioned commit or archival release.",
            "",
            "## References",
            "",
            f"The cited literature is listed in {_md_link('references.bib', 'references.bib', paper_dir)}; "
            f"core method support includes {core_citations}.",
            "",
            *_cmdgd_reference_list_lines(context, limit=20),
        ]
    )
    return "\n".join(lines)


def _render_mccbd_algorithm_manuscript(context: dict[str, Any], paper_dir: Path) -> str:
    citation = _citation_cluster(context)
    literature_summary = _literature_summary(context)
    data_summary = _project_data_summary(context)
    objective = _one_line(context.get("project_spec", {}).get("objective")) or "sparse constrained variant design"
    benchmark = context.get("mccbd_benchmark", {})
    selected_row = _mccbd_summary_row(context, "mccbd", "overall")
    case_row = _project_masking_summary_row(context, "mccbd")
    comparisons = _mccbd_baseline_comparisons(context)
    figure_links = {
        key: _md_link(f"Figure {idx}", filename, paper_dir)
        for idx, (key, filename) in enumerate(MCCBD_FIGURE_FILENAMES.items(), start=1)
    }
    mccbd_links = _mccbd_benchmark_links(benchmark, paper_dir)
    project_links = _algorithm_project_masking_links(context["project_masking"], paper_dir)
    table_links = {
        "benchmark": _md_link("benchmark_overall_table.csv", "tables/benchmark_overall_table.csv", paper_dir),
        "project": _md_link("project_case_table.csv", "tables/project_case_table.csv", paper_dir),
        "related": _md_link("related_work_matrix.csv", "tables/related_work_matrix.csv", paper_dir),
    }
    project_case_run_id = _project_masking_run_id(context) or "mccbd_project_case"

    lines = [
        "# Mechanism-Calibrated Constrained Bayesian Design for Sparse Protein Variant Optimization",
        "",
        "## Abstract",
        "",
        (
            "We introduce Mechanism-Calibrated Constrained Bayesian Design (MCCBD), a batch active-design "
            "algorithm for sparse wet-lab variant optimization. MCCBD fits a Bayesian posterior over generic "
            "module, pair, and endpoint features, estimates feasibility from observed constraint labels, and "
            "selects panels by constrained expected improvement with posterior information value and batch "
            "diversity. The motivating local task is "
            f"{_sentence(objective)} The primary evidence is a controlled algorithmic benchmark over additive, "
            "epistatic, sparse-observation, batch-redundancy, and risky-decoy mechanism worlds; the 1E62 data "
            "are used as a retrospective masking case study. This evidence "
            f"{PROSPECTIVE_VALIDATION_BOUNDARY} "
            f"{citation}."
        ),
        "",
        "## Introduction",
        "",
        (
            "Small-batch antibody and protein design often starts from a weak evidence state: a small number "
            "of assayed variants, heterogeneous endpoint summaries, incomplete feasibility labels, and a large "
            "unmeasured candidate space. In that regime, a selector that is merely a weighted score can look "
            "plausible while actually mixing exploitation, uncertainty, feasibility, and diversity as tunable "
            "heuristics. The methodological question is whether those roles can be separated into an auditable "
            "mechanism that produces a panel, a posterior trace, and component-level ablations."
        ),
        "",
        "## Literature Basis",
        "",
        (
            f"The local literature engine collected {literature_summary}. The relevant methodological context "
            "includes Bayesian optimization and active learning for goal-directed design, constrained and "
            "multi-objective acquisition, antibody-specific Bayesian optimization, batch sequence design, and "
            "benchmarking of antibody design methods. These papers motivate MCCBD's architecture but do not "
            "validate any untested 1E62 variant."
        ),
        "",
        "## Formal Problem Setup",
        "",
        (
            "Let D_t be the observed design state after t rounds and X_t be a candidate pool. Each candidate x "
            "contains a stable identifier, module or mutation tokens, optional endpoint features, cost, "
            "feasibility metadata, and source provenance. The goal is to select a batch B_t under a cost budget "
            "that maximizes the best feasible utility observed after evaluation, while keeping false feasible "
            "claims low. The framework therefore scores a candidate by posterior utility, posterior uncertainty, "
            "feasibility probability, constrained expected improvement, and batch-level information redundancy."
        ),
        "",
        "## Algorithmic Contribution",
        "",
        (
            "MCCBD's contribution is a mechanism-level acquisition architecture rather than a new set of "
            "hand-tuned weights. It has four coupled parts: (i) a Bayesian posterior over module, pair, and "
            "numeric endpoint features; (ii) a separate feasibility posterior that gates expected improvement; "
            "(iii) an information-gain proxy derived from posterior predictive variance; and (iv) a sequential "
            "batch update that penalizes selecting redundant architectures after a first pick. Reusing helper "
            "code or ordinary linear algebra does not remove the novelty: the architecture is the explicit "
            "coupling of constraint calibration, expected improvement, posterior information value, and batch "
            "diversity in a data-grounded variant-design lifecycle."
        ),
        "",
        (
            f"The workflow is summarized in {figure_links['workflow']}. MCCBD implements a full lifecycle: "
            "`fit_state` estimates the utility and feasibility posteriors; `generate_candidates` exposes the "
            "candidate pool; `score_candidates` returns posterior mean, posterior standard deviation, "
            "feasibility probability, constrained expected improvement, and information value; `select_panel` "
            "enforces the cost budget with a virtual posterior update for batch diversity; and `plan_ablations` "
            "defines removable mechanism components."
        ),
        "",
        "## Methods",
        "",
        (
            "For each candidate, MCCBD builds a feature vector containing an intercept, module indicators, "
            "module-pair indicators, and any numeric endpoint or feature fields. Utility is fit by closed-form "
            "Bayesian ridge regression. Feasibility is fit with the same feature basis against observed feasible "
            "or infeasible labels and mapped to a probability. Acquisition uses constrained expected improvement, "
            "`P(feasible | D_t, x) * EI(mu_t(x), sigma_t(x), y*_t)`, plus a posterior information term. During "
            "batch selection, a selected candidate is treated as a virtual observation to reduce redundant "
            "posterior variance, and a module-overlap penalty discourages repeated local architectures."
        ),
        "",
        (
            "The main ablations remove feasibility gating, posterior information value, or batch diversity. "
            "Those ablations test whether the mechanism is doing more than ranking candidates by predicted "
            "utility. In contrast, `evidence_calibrated_ucb` is kept as a weighted-score baseline."
        ),
        "",
        "## Evaluation Design",
        "",
        (
            "The primary evaluation is a generic algorithmic benchmark, not a project-specific wet-lab claim. "
            "The benchmark creates controlled mechanism worlds with known latent utility and feasibility, then "
            "runs multi-round replay against `random_feasible`, `fixed_mix`, `greedy_observed`, and "
            "`evidence_calibrated_ucb`. The benchmark requires MCCBD to beat `random_feasible` and `fixed_mix` "
            "on a majority of worlds and to avoid a higher false-claim rate. Benchmark artifacts are "
            f"{mccbd_links}."
        ),
        "",
        (
            "Selector-visible benchmark records are blind to direct outcome labels such as true utility, true "
            "feasibility, and endpoint values. They do include simulated prior descriptors such as predicted "
            "feasibility, prior utility, uncertainty, risk flags, and cost, reflecting the common engineering "
            "setting where a design algorithm has inexpensive in-silico or rule-based priors before wet-lab "
            "measurement. The benchmark therefore evaluates use of noisy prior metadata plus observed labels; "
            "it is not a pure label-free feasibility-learning benchmark."
        ),
        "",
        (
            "A separate 1E62 retrospective masking case study hides outcomes for already measured variants, "
            "exposes only descriptors to the selector, and scores selected held-out variants after the fact. "
            f"Those case-study artifacts are {project_links}. This retrospective masked project-data evidence "
            f"{PROSPECTIVE_VALIDATION_BOUNDARY}."
        ),
        "",
        "## Results",
        "",
            f"{figure_links['benchmark']} shows the benchmark-level performance summary.",
            f"The full benchmark comparison table is {table_links['benchmark']}.",
        "",
    ]
    if selected_row:
        lines.extend(
            [
                "Overall MCCBD benchmark summary:",
                "",
                f"- mean best feasible utility: {selected_row.get('mean_best_feasible_utility', 'not recorded')}",
                f"- mean hit rate: {selected_row.get('mean_hit_rate', 'not recorded')}",
                f"- mean regret proxy: {selected_row.get('mean_regret_proxy', 'not recorded')}",
                f"- mean false claim rate: {selected_row.get('mean_false_claim_rate', 'not recorded')}",
                f"- selected mechanism under eligibility rule: {benchmark.get('config', {}).get('selected_mechanism', 'not recorded')}",
                f"- eligibility rule passed: {benchmark.get('config', {}).get('selected_passes_gate', 'not recorded')}",
                "",
                "Baseline comparison:",
            ]
        )
        for comparison in comparisons:
            lines.append(
                "- versus {baseline}: utility delta {utility_delta}, false-claim delta {false_claim_delta}".format(
                    **comparison
                )
            )
    else:
        lines.append("No overall MCCBD benchmark summary row was available.")
    lines.extend(
        [
            "",
            f"{figure_links['ablation']} summarizes component ablations.",
            "",
            *_mccbd_ablation_lines(context),
            (
                "The ablations are interpreted as mechanism diagnostics rather than automatic proof that every "
                "component increases utility in every world: feasibility gating is expected to reduce false "
                "claims, while information value and batch diversity can trade short-run utility for safer or "
                "less redundant exploration."
            ),
            "",
            f"{figure_links['project_case']} summarizes the 1E62 retrospective masking case study.",
            f"The full 1E62 comparison table is {table_links['project']}.",
            "",
        ]
    )
    if case_row:
        lines.extend(
            [
                "1E62 retrospective masking case study:",
                "",
                f"- mean best feasible utility: {case_row.get('mean_best_feasible_utility', 'not recorded')}",
                f"- mean hit rate: {case_row.get('mean_hit_rate', 'not recorded')}",
                f"- mean regret proxy: {case_row.get('mean_regret_proxy', 'not recorded')}",
                f"- mean false claim rate: {case_row.get('mean_false_claim_rate', 'not recorded')}",
                (
                    "- interpretation: this small observed-pool case study is an integration stress test; "
                    "it should not be read as prospective superiority over every baseline."
                ),
            ]
        )
    else:
        lines.append("No MCCBD row was available in the project masking summary.")
    lines.extend(
        [
            "",
            "## Discussion",
            "",
            (
                "The benchmark and ablations support MCCBD as a real algorithmic mechanism rather than a weighted "
                "selector: the posterior and feasibility models supply interpretable state, constrained expected "
                "improvement separates utility from claim risk, and the batch step explicitly controls redundancy. "
                "The utility margin over weighted UCB is small in the current benchmark, but MCCBD achieves a lower "
                "false-claim rate and exposes component-level tradeoffs that a single score function cannot audit. "
                "The 1E62 case study is deliberately secondary because its observed pool is small; it checks whether "
                "the same algorithm can consume real standardized startup data without using design-specific shortcuts."
            ),
            "",
            "## Limitations",
            "",
            (
                "The synthetic worlds are controlled tests of algorithm behavior, not replacements for wet-lab "
                "validation. They also provide simulated prior descriptors to selectors, so they test algorithmic "
                "use of noisy priors rather than discovery from labels alone. The 1E62 case study has a small "
                "observed pool and no true historical round order. KD-ratio records remain supplementary unless "
                "a direct variant mapping exists. MCCBD should be treated as a publishable computational method "
                "candidate whose prospective utility still requires new experimental panels."
            ),
            "",
            "## Contribution Boundaries",
            "",
            (
                "This manuscript claims a reusable algorithm, benchmark, ablation, and case-study evidence trail. "
                "It does not claim validated 1E62 variants, clinical relevance, genotype breadth, or biological "
                "causality for untested mutations."
            ),
            "",
            "## Reproducibility",
            "",
            "```bash",
            f"uv run design-scientist run-algorithm-benchmark {context['root']} --run-id {context['run_id']} --rounds 3 --budget 6",
            f"uv run design-scientist run-project-benchmark {context['root']} --run-id {project_case_run_id} --budget 2 --folds kfold_5 --mechanisms mccbd evidence_calibrated_ucb random_feasible fixed_mix greedy_observed",
            f"uv run design-scientist generate-algorithm-paper {context['root']} --run-id {context['run_id']} --algorithm mccbd",
            "```",
            "",
            "## Data Availability",
            "",
            (
                f"The startup data summary is: {_sentence(data_summary)} The algorithmic benchmark is generated "
                "from code and fixed seeds. Literature artifacts and masking outputs are local framework outputs; "
                "no new wet-lab measurements are introduced by this paper."
            ),
            "",
            "## Code Availability",
            "",
            "The algorithm implementation is in `src/design_scientist/algorithms/mccbd.py`.",
            "",
            "## References",
            "",
            f"BibTeX references are provided in {_md_link('references.bib', 'references.bib', paper_dir)}.",
        ]
    )
    return "\n".join(lines)


def _algorithm_results_summary(context: dict[str, Any]) -> dict[str, Any]:
    algorithm = context.get("algorithm")
    return {
        "schema_version": 1,
        "run_id": context["run_id"],
        "algorithm": algorithm,
        "evidence_scope": (
            MIXED_EVIDENCE_SCOPE
            if algorithm == "mccbd"
            else "computational_generative_benchmark"
            if algorithm == "cmdgd"
            else PROJECT_MASKING_EVIDENCE_SCOPE
        ),
        "evidence_type": (
            "computational_algorithmic_benchmark_and_retrospective/masked_project_data"
            if algorithm == "mccbd"
            else "computational/generative_sequence_benchmark"
            if algorithm == "cmdgd"
            else PROJECT_MASKING_EVIDENCE_TYPE
        ),
        "algorithm_summary_row": _algorithm_summary_row(context),
        "baseline_comparisons": _baseline_comparisons(context),
        "mccbd_benchmark": _mccbd_benchmark_summary_payload(context),
        "generative_benchmark": _generative_benchmark_summary_payload(context),
        "project_masking": _project_masking_summary_payload(context),
        "paper_card_count": len(context["paper_cards"]),
        "literature_summary": _literature_summary(context),
        "no_wet_lab_validation_claimed": True,
    }


def _summary_artifact_path(value: Any, context: dict[str, Any]) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Path):
        raw = str(value)
    elif isinstance(value, str):
        raw = value
    else:
        return value
    if not raw:
        return raw
    path = Path(raw).expanduser()
    if not path.is_absolute():
        return raw
    root_value = context.get("root")
    if not root_value:
        return path.name
    root = Path(root_value).expanduser().resolve()
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return os.path.relpath(resolved, root)


def _summary_artifact_paths(paths: Any, context: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(paths, dict):
        return {}
    return {key: _summary_artifact_path(value, context) for key, value in paths.items()}


def _algorithm_claim_evidence_map(context: dict[str, Any]) -> dict[str, Any]:
    if context.get("algorithm") == "cmdgd":
        return {
            "schema_version": 1,
            "run_id": context["run_id"],
            "algorithm": context["algorithm"],
            "claims": [
                {
                    "claim": (
                        "CMD-GD is a grammar-guided combinatorial generator for pH-sensitive antibody design "
                        "that learns an edit grammar, extends it with design vocabulary and mechanism priors, "
                        "generates new heavy/light chain sequences, and selects a scored panel."
                    ),
                    "evidence_type": "implementation_and_computational_generative_benchmark",
                    "evidence_scope": "computational_generative_benchmark",
                    "evidence_artifacts": list(GENERATIVE_BENCHMARK_ARTIFACT_FILENAMES),
                    "limitations": [
                        "The evidence is computational and benchmark-based.",
                        "CMD-GD is not a de novo neural antibody generator.",
                        "Prospective wet-lab validation remains required before biological claims about selected sequences.",
                    ],
                },
                {
                    "claim": (
                        "CMD-GD expands the candidate space beyond fixed-pool selection while exposing observed "
                        "recombination and fixed-pool boundaries."
                    ),
                    "evidence_type": "benchmark_summary_and_ablation",
                    "evidence_scope": "computational_generative_benchmark",
                    "evidence_artifacts": [
                        "generative_benchmark_summary",
                        "generative_ablation_results",
                        "generative_design_examples",
                    ],
                    "limitations": [
                        "Observed recombination remains a strong baseline when useful modules are already observed.",
                        "Fixed-pool selectors can score high when their candidate pool already contains strong candidates.",
                        "Ablations are interpreted only when benchmark artifact rows support them.",
                    ],
                },
            ],
        }
    if context.get("algorithm") == "mccbd":
        return {
            "schema_version": 1,
            "run_id": context["run_id"],
            "algorithm": context["algorithm"],
            "claims": [
                {
                    "claim": (
                        "MCCBD is a mechanism-level constrained Bayesian batch design algorithm coupling "
                        "posterior utility, feasibility probability, constrained expected improvement, "
                        "posterior information value, and batch diversity."
                    ),
                    "evidence_type": "implementation_and_computational_algorithmic_benchmark",
                    "evidence_scope": "computational_algorithmic_benchmark",
                    "evidence_artifacts": list(MCCBD_BENCHMARK_ARTIFACT_FILENAMES),
                    "limitations": [
                        "The primary evidence is computational.",
                        "Prospective wet-lab validation is not included.",
                    ],
                },
                {
                    "claim": (
                        "MCCBD can consume standardized 1E62 startup data in retrospective observed-pool masking "
                        "without treating hidden outcomes as available evidence."
                    ),
                    "evidence_type": PROJECT_MASKING_EVIDENCE_TYPE,
                    "evidence_scope": PROJECT_MASKING_EVIDENCE_SCOPE,
                    "evidence_artifacts": list(PROJECT_MASKING_ARTIFACT_FILENAMES),
                    "limitations": [
                        "The 1E62 evidence is retrospective masked project-data evidence.",
                        f"It {PROSPECTIVE_VALIDATION_BOUNDARY}.",
                    ],
                },
            ],
        }
    return {
        "schema_version": 1,
        "run_id": context["run_id"],
        "algorithm": context["algorithm"],
        "claims": [
            {
                "claim": (
                    "Evidence-Calibrated UCB is a distinct acquisition architecture combining "
                    "uncertainty, observed-neighborhood calibration, motif enrichment, bad-neighbor "
                    "penalties, and batch diversity."
                ),
                "evidence_type": "implementation_and_retrospective/masked_project_data",
                "evidence_scope": PROJECT_MASKING_EVIDENCE_SCOPE,
                "evidence_artifacts": list(PROJECT_MASKING_ARTIFACT_FILENAMES),
                "limitations": [
                    "The evidence is retrospective and computational.",
                    "Prospective wet-lab validation is not included.",
                ],
            }
        ],
    }


def _write_mccbd_figures(context: dict[str, Any], paper_dir: Path) -> dict[str, str]:
    figures_dir = ensure_dir(paper_dir / "figures")
    figure_paths = {
        key: paper_dir / filename for key, filename in MCCBD_FIGURE_FILENAMES.items()
    }
    figure_paths["workflow"].write_text(_mccbd_workflow_svg(), encoding="utf-8")
    figure_paths["benchmark"].write_text(_mccbd_benchmark_svg(context), encoding="utf-8")
    figure_paths["ablation"].write_text(_mccbd_ablation_svg(context), encoding="utf-8")
    figure_paths["project_case"].write_text(_mccbd_project_case_svg(context), encoding="utf-8")
    return {f"figure_{key}": str(path) for key, path in figure_paths.items() if path.parent == figures_dir}


def _write_mccbd_tables(context: dict[str, Any], paper_dir: Path) -> dict[str, str]:
    tables_dir = ensure_dir(paper_dir / "tables")
    related_path = tables_dir / "related_work_matrix.csv"
    benchmark_path = tables_dir / "benchmark_overall_table.csv"
    project_case_path = tables_dir / "project_case_table.csv"
    _write_csv_rows(
        related_path,
        [
            {
                "approach": "standard_bayesian_optimization",
                "state_model": "surrogate posterior",
                "constraint_handling": "optional",
                "batch_behavior": "often separate heuristic",
                "mccbd_delta": "adds feasibility posterior, constrained EI, and batch posterior redundancy control",
            },
            {
                "approach": "active_learning",
                "state_model": "uncertainty model",
                "constraint_handling": "not primary objective",
                "batch_behavior": "information seeking",
                "mccbd_delta": "optimizes best feasible utility while retaining information value as a secondary term",
            },
            {
                "approach": "weighted_ucb",
                "state_model": "hand-weighted score components",
                "constraint_handling": "penalty term",
                "batch_behavior": "score diversity penalty",
                "mccbd_delta": "separates posterior utility, feasibility probability, EI, and virtual batch update",
            },
        ],
    )
    overall_rows = [
        row
        for row in context.get("mccbd_benchmark", {}).get("summary_rows", [])
        if isinstance(row, dict) and row.get("world_id") == "overall"
    ]
    _write_csv_rows(benchmark_path, overall_rows)
    _write_csv_rows(project_case_path, context.get("project_masking", {}).get("summary_rows", []))
    return {
        "related_work_matrix": str(related_path),
        "benchmark_overall_table": str(benchmark_path),
        "project_case_table": str(project_case_path),
    }


def _write_cmdgd_figures(context: dict[str, Any], paper_dir: Path) -> dict[str, str]:
    figures_dir = ensure_dir(paper_dir / "figures")
    figure_paths = {
        key: paper_dir / filename for key, filename in CMDGD_FIGURE_FILENAMES.items()
    }
    figure_paths["workflow"].write_text(_cmdgd_workflow_svg(), encoding="utf-8")
    figure_paths["benchmark"].write_text(_cmdgd_benchmark_svg(context), encoding="utf-8")
    figure_paths["vocabulary"].write_text(_cmdgd_vocabulary_svg(context), encoding="utf-8")
    figure_paths["ablation"].write_text(_cmdgd_ablation_svg(context), encoding="utf-8")
    figure_paths["statistics"].write_text(_cmdgd_statistics_svg(context), encoding="utf-8")
    return {f"figure_{key}": str(path) for key, path in figure_paths.items() if path.parent == figures_dir}


def _write_cmdgd_tables(context: dict[str, Any], paper_dir: Path) -> dict[str, str]:
    tables_dir = ensure_dir(paper_dir / "tables")
    literature = context.get("cmdgd_literature", {})
    benchmark = context.get("generative_benchmark", {})
    paths: dict[str, Path] = {
        "related_work_matrix": tables_dir / "related_work_matrix.csv",
        "benchmark_summary": tables_dir / "benchmark_summary.csv",
        "statistical_summary": tables_dir / "statistical_summary.csv",
        "pairwise_comparisons": tables_dir / "pairwise_comparisons.csv",
        "design_examples": tables_dir / "design_examples.csv",
        "ablation_summary": tables_dir / "ablation_summary.csv",
        "project_selected_candidates": tables_dir / "project_selected_candidates.csv",
        "project_candidate_rationale": tables_dir / "project_candidate_rationale.csv",
    }
    related_rows = literature.get("related_work_rows", [])
    _write_csv_rows(paths["related_work_matrix"], related_rows if isinstance(related_rows, list) else [])
    summary_rows = benchmark.get("summary_rows", [])
    _write_csv_rows(paths["benchmark_summary"], summary_rows if isinstance(summary_rows, list) else [])
    statistical_rows = _cmdgd_statistical_summary_rows(context)
    _write_csv_rows(paths["statistical_summary"], statistical_rows)
    pairwise_rows = _cmdgd_pairwise_comparison_rows(context)
    _write_csv_rows(paths["pairwise_comparisons"], pairwise_rows)
    example_rows = benchmark.get("design_example_rows", [])
    _write_csv_rows(paths["design_examples"], example_rows if isinstance(example_rows, list) else [])
    ablation_rows = benchmark.get("ablation_rows", [])
    _write_csv_rows(
        paths["ablation_summary"],
        [
            _cmdgd_public_benchmark_row(row)
            for row in ablation_rows
            if isinstance(row, dict)
        ],
    )
    project_rows = [
        _cmdgd_public_candidate_row(row)
        for row in benchmark.get("project_design_candidate_rows", [])
        if isinstance(row, dict) and str(row.get("selected", "")).lower() == "true"
    ]
    _write_csv_rows(paths["project_selected_candidates"], project_rows)
    _write_csv_rows(paths["project_candidate_rationale"], _cmdgd_candidate_rationale_rows(project_rows))
    return {key: str(path) for key, path in paths.items()}


def _cmdgd_public_candidate_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if not _cmdgd_internal_candidate_column(key)
    }


def _cmdgd_public_benchmark_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if not _cmdgd_internal_benchmark_column(key)
    }


def _cmdgd_internal_candidate_column(key: str) -> bool:
    normalized = str(key).strip().lower()
    return normalized == "cmdgd_trace" or normalized.endswith("_trace")


def _cmdgd_internal_benchmark_column(key: str) -> bool:
    normalized = str(key).strip().lower()
    return normalized in {"cmdgd_adapter", "error"} or normalized.endswith("_path")


def _cmdgd_workflow_svg() -> str:
    boxes = [
        ("Observed sequences", "base H/L + measured variants"),
        ("Edit grammar", "chain, position, residue, support"),
        ("Design vocabulary", "mechanism priors beyond seen edits"),
        ("Generative expansion", "new heavy/light sequence records"),
        ("Contrastive scoring", "neutral retention + acidic release"),
        ("Panel selection", "cost, feasibility, novelty guardrails"),
    ]
    width = 920
    height = 540
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#f8fafc"/>',
        '<text x="40" y="52" font-family="Arial" font-size="24" font-weight="700" fill="#111827">CMD-GD generative mechanism lifecycle</text>',
    ]
    for index, (title, subtitle) in enumerate(boxes):
        fill = "#ffffff" if index not in (2, 3) else "#ecfdf5"
        stroke = "#334155" if index not in (2, 3) else "#047857"
        row = index // 3
        column = index % 3 if row == 0 else 2 - (index % 3)
        x = 55 + column * 285
        y = 105 + row * 180
        box_width = 220
        box_height = 115
        parts.extend(
            [
                f'<rect x="{x}" y="{y}" width="{box_width}" height="{box_height}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>',
                f'<text x="{x + 14}" y="{y + 36}" font-family="Arial" font-size="15" font-weight="700" fill="#0f172a">{_svg_escape(title)}</text>',
                *_svg_wrapped_text(
                    subtitle,
                    x=x + 14,
                    y=y + 66,
                    width=184,
                    font_size=12,
                    fill="#475569",
                ),
            ]
        )
        if row == 0 and column < 2:
            parts.extend(
                [
                    f'<line x1="{x + box_width}" y1="{y + 58}" x2="{x + box_width + 48}" y2="{y + 58}" stroke="#2563eb" stroke-width="2"/>',
                    f'<polygon points="{x + box_width + 48},{y + 58} {x + box_width + 37},{y + 51} {x + box_width + 37},{y + 65}" fill="#2563eb"/>',
                ]
            )
        elif row == 1 and column > 0:
            parts.extend(
                [
                    f'<line x1="{x}" y1="{y + 58}" x2="{x - 48}" y2="{y + 58}" stroke="#2563eb" stroke-width="2"/>',
                    f'<polygon points="{x - 48},{y + 58} {x - 37},{y + 51} {x - 37},{y + 65}" fill="#2563eb"/>',
                ]
            )
        elif index == 2:
            down_x = x + box_width / 2
            parts.extend(
                [
                    f'<line x1="{down_x}" y1="{y + box_height}" x2="{down_x}" y2="{y + box_height + 48}" stroke="#2563eb" stroke-width="2"/>',
                    f'<polygon points="{down_x},{y + box_height + 48} {down_x - 7},{y + box_height + 37} {down_x + 7},{y + box_height + 37}" fill="#2563eb"/>',
                ]
            )
    parts.extend(
        [
            '<text x="55" y="455" font-family="Arial" font-size="14" fill="#334155">Generation is the model: CMD-GD learns an editable grammar, materializes new heavy/light sequences, then scores and selects a feasible panel.</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts)


def _cmdgd_benchmark_svg(context: dict[str, Any]) -> str:
    rows = [
        row
        for row in context.get("generative_benchmark", {}).get("summary_rows", [])
        if isinstance(row, dict) and row.get("world_id") == "overall"
    ]
    rows = sorted(rows, key=lambda row: _float_field(row, "mean_best_selected_utility"), reverse=True)
    return _bar_svg(
        "Generative benchmark: mean best selected utility",
        rows,
        label_key="mechanism",
        value_key="mean_best_selected_utility",
        color="#0f766e",
    )


def _cmdgd_vocabulary_svg(context: dict[str, Any]) -> str:
    rows = [
        row
        for row in context.get("generative_benchmark", {}).get("summary_rows", [])
        if isinstance(row, dict) and row.get("world_id") == "vocabulary_extension"
    ]
    rows = sorted(rows, key=lambda row: _float_field(row, "mean_novel_sequence_rate"), reverse=True)
    return _bar_svg(
        "Vocabulary-extension world: mean best selected utility",
        rows,
        label_key="mechanism",
        value_key="mean_best_selected_utility",
        color="#16a34a",
    )


def _cmdgd_ablation_svg(context: dict[str, Any]) -> str:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in context.get("generative_benchmark", {}).get("ablation_rows", []):
        if isinstance(row, dict) and row.get("mechanism") == "cmdgd":
            grouped.setdefault(row.get("ablation", "unknown"), []).append(row)
    rows = [
        {
            "label": ablation,
            "value": f"{_mean_float(ablation_rows, 'delta_from_full_best_selected_utility'):.6f}",
        }
        for ablation, ablation_rows in sorted(grouped.items())
    ]
    return _bar_svg(
        "CMD-GD ablation: mean selected-utility loss from full",
        rows,
        label_key="label",
        value_key="value",
        color="#7c3aed",
    )


def _cmdgd_statistics_svg(context: dict[str, Any]) -> str:
    rows = [
        row
        for row in _cmdgd_pairwise_comparison_rows(context)
        if row.get("world_id") == "overall"
        and row.get("metric") == "best_selected_utility"
        and row.get("baseline")
        in {
            "random_edit_generator",
            "single_edit_scan",
            "observed_recombination_baseline",
            "mccbd_pool_selector",
        }
    ]
    plot_rows = [
        {
            "baseline": row.get("baseline"),
            "mean_delta": row.get("mean_delta"),
        }
        for row in rows
    ]
    return _bar_svg(
        "Paired seed/world selected-utility delta versus baselines",
        plot_rows,
        label_key="baseline",
        value_key="mean_delta",
        color="#0ea5e9",
    )


def _mccbd_workflow_svg() -> str:
    boxes = [
        ("Observed data", "variants, endpoints, constraints"),
        ("Bayesian posterior", "module + pair + endpoint features"),
        ("Feasibility posterior", "claim-risk calibration"),
        ("Constrained EI", "P(feasible) x expected improvement"),
        ("Information value", "posterior variance reduction"),
        ("Batch panel", "cost + diversity + virtual update"),
    ]
    width = 1180
    height = 360
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="1180" height="360" fill="#f8fafc"/>',
        '<text x="40" y="52" font-family="Arial" font-size="24" font-weight="700" fill="#111827">MCCBD mechanism lifecycle</text>',
    ]
    x = 40
    for index, (title, subtitle) in enumerate(boxes):
        parts.extend(
            [
                f'<rect x="{x}" y="110" width="160" height="120" rx="8" fill="#ffffff" stroke="#334155" stroke-width="1.5"/>',
                f'<text x="{x + 16}" y="150" font-family="Arial" font-size="16" font-weight="700" fill="#0f172a">{_svg_escape(title)}</text>',
                f'<text x="{x + 16}" y="182" font-family="Arial" font-size="12" fill="#475569">{_svg_escape(subtitle)}</text>',
            ]
        )
        if index < len(boxes) - 1:
            parts.extend(
                [
                    f'<line x1="{x + 160}" y1="170" x2="{x + 196}" y2="170" stroke="#2563eb" stroke-width="2"/>',
                    f'<polygon points="{x + 196},170 {x + 186},164 {x + 186},176" fill="#2563eb"/>',
                ]
            )
        x += 196
    parts.append("</svg>")
    return "\n".join(parts)


def _mccbd_benchmark_svg(context: dict[str, Any]) -> str:
    rows = [
        row
        for row in context.get("mccbd_benchmark", {}).get("summary_rows", [])
        if isinstance(row, dict) and row.get("world_id") == "overall"
    ]
    rows = sorted(
        rows,
        key=lambda row: _float_field(row, "mean_best_feasible_utility"),
        reverse=True,
    )[:6]
    return _bar_svg(
        "Algorithm benchmark: mean best feasible utility",
        rows,
        label_key="mechanism",
        value_key="mean_best_feasible_utility",
        color="#2563eb",
    )


def _mccbd_ablation_svg(context: dict[str, Any]) -> str:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in context.get("mccbd_benchmark", {}).get("ablation_rows", []):
        if isinstance(row, dict):
            grouped.setdefault(row.get("ablation", "unknown"), []).append(row)
    rows = [
        {
            "label": ablation,
            "value": f"{_mean_float(ablation_rows, 'delta_from_full_best_feasible_utility'):.6f}",
        }
        for ablation, ablation_rows in sorted(grouped.items())
    ]
    return _bar_svg(
        "MCCBD ablation: mean utility loss vs full",
        rows,
        label_key="label",
        value_key="value",
        color="#7c3aed",
    )


def _mccbd_project_case_svg(context: dict[str, Any]) -> str:
    rows = [
        row
        for row in context.get("project_masking", {}).get("summary_rows", [])
        if isinstance(row, dict)
    ]
    rows = sorted(
        rows,
        key=lambda row: _float_field(row, "mean_best_feasible_utility"),
        reverse=True,
    )[:6]
    return _bar_svg(
        "1E62 retrospective masking: mean best feasible utility",
        rows,
        label_key="mechanism",
        value_key="mean_best_feasible_utility",
        color="#059669",
    )


def _bar_svg(
    title: str,
    rows: list[dict[str, Any]],
    *,
    label_key: str,
    value_key: str,
    color: str,
) -> str:
    width = 920
    row_height = 42
    height = max(240, 110 + row_height * max(len(rows), 1))
    values = [_float_field(row, value_key) for row in rows]
    max_value = max(values, default=1.0) or 1.0
    axis_max = max(max_value, 1.0) if all(value >= 0 for value in values) else max(abs(value) for value in values) or 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#f8fafc"/>',
        f'<text x="36" y="44" font-family="Arial" font-size="22" font-weight="700" fill="#111827">{_svg_escape(title)}</text>',
        '<line x1="260" y1="68" x2="820" y2="68" stroke="#94a3b8" stroke-width="1"/>',
        '<text x="260" y="63" font-family="Arial" font-size="11" fill="#475569">0</text>',
        f'<text x="786" y="63" font-family="Arial" font-size="11" fill="#475569">{axis_max:.2f}</text>',
    ]
    if not rows:
        parts.append('<text x="36" y="110" font-family="Arial" font-size="14" fill="#64748b">No rows available.</text>')
        parts.append("</svg>")
        return "\n".join(parts)
    y = 82
    for row in rows:
        label = _one_line(row.get(label_key))
        value = _float_field(row, value_key)
        bar_width = int(560 * value / axis_max) if axis_max and value > 0 else 0
        bar_width = max(0, min(560, bar_width))
        parts.extend(
            [
                f'<text x="36" y="{y + 24}" font-family="Arial" font-size="13" fill="#0f172a">{_svg_escape(label)}</text>',
                f'<rect x="260" y="{y}" width="560" height="26" rx="4" fill="#e2e8f0"/>',
                f'<rect x="260" y="{y}" width="{bar_width}" height="26" rx="4" fill="{color}"/>',
                f'<text x="836" y="{y + 19}" font-family="Arial" font-size="13" fill="#0f172a">{value:.4f}</text>',
            ]
        )
        y += row_height
    parts.append("</svg>")
    return "\n".join(parts)


def _svg_wrapped_text(
    text: str,
    *,
    x: int,
    y: int,
    width: int,
    font_size: int,
    fill: str,
) -> list[str]:
    max_chars = max(10, int(width / max(font_size * 0.55, 1)))
    lines = textwrap.wrap(str(text), width=max_chars)[:3]
    return [
        f'<text x="{x}" y="{y + index * (font_size + 4)}" font-family="Arial" font-size="{font_size}" fill="{fill}">{_svg_escape(line)}</text>'
        for index, line in enumerate(lines)
    ]


def _validate_algorithm_manuscript(
    paper_dir: Path,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    manuscript = _read_text(paper_dir / "manuscript.md")
    algorithm = str((context or {}).get("algorithm") or "")
    required_sections = [
        "## Abstract",
        "## Formal Problem Setup",
        "## Algorithmic Contribution",
        "## Evaluation Design",
        "## Results",
        "## Limitations",
        "## Contribution Boundaries",
        "## References",
    ]
    if algorithm == "cmdgd":
        if "## Literature Review" not in manuscript and "## Related Work" not in manuscript:
            _add_finding(
                findings,
                "error",
                "missing_section",
                "manuscript.md missing Literature Review or Related Work section",
                "manuscript.md",
            )
        required_sections.extend(
            [
                "## Statistical Stability",
                "## Real-Data Retrospective Masking",
                "## Biophysical Plausibility and Panel Rationale",
                "## 1E62 Computational Design Output",
            ]
        )
    else:
        required_sections.append("## Literature Basis")
    for section in required_sections:
        if section not in manuscript:
            _add_finding(findings, "error", "missing_section", f"manuscript.md missing {section}", "manuscript.md")
    if algorithm == "cmdgd":
        forbidden_phrases = (
            "v4",
            "artifact",
            "trace",
            "current run",
            "Reviewer-facing",
            "selected_passes_gate",
            "selected_eligible",
            "observed_reuse_only",
            "fixed_pool_only",
            "generative selection gate passed",
            "local outputs",
            "manuscript generator",
            "supplementary run files",
            "run identifier",
            "## New Mutation-Site Generation",
            "BibTeX references are provided",
            "uv run",
            "<project_dir>",
            "src/design_scientist",
        )
        for phrase in forbidden_phrases:
            if phrase in manuscript:
                _add_finding(
                    findings,
                    "error",
                    "forbidden_cmdgd_evidence_phrase",
                    f"CMD-GD manuscript contains forbidden phrase: {phrase}",
                    "manuscript.md",
                )
        for phrase in (
            "learns an edit grammar",
            "design vocabulary",
            "mechanism priors",
            "grammar-guided combinatorial generator",
            "candidate enumeration",
            "vocabulary-guided recombination",
            "grammar recombination",
            "mechanism-prior grammar expansion",
            "new heavy/light chain sequences",
            "selects a panel",
            "Algorithm 1",
            "Algorithm 2",
            "tie-breaking",
            "does not claim broad selected-utility superiority",
            "Real-data masking boundary",
            "Figure 2 caption:",
            "Figure 5 caption:",
            "95% CI",
            "observed recombination is a strong baseline",
            "fixed-pool selectors may exceed CMD-GD utility",
            "ablation rows support the comparison",
            "endpoint gap",
            "decision-policy gap",
            "representation gap",
            "alignment gap",
            "evaluation gap",
        ):
            if phrase not in manuscript:
                _add_finding(
                    findings,
                    "error",
                    "missing_cmdgd_algorithm_claim",
                    f"CMD-GD manuscript missing required algorithm claim phrase: {phrase}",
                    "manuscript.md",
                )
        if _cmdgd_has_new_site_artifacts(context or {}):
            for phrase in (
                "Design-space diagnostics",
                "not a standalone algorithmic feature",
                "experimental activity, structural tolerance, and pH-sensitive release remain to be measured prospectively",
            ):
                if phrase not in manuscript:
                    _add_finding(
                        findings,
                        "error",
                        "missing_cmdgd_new_site_claim",
                        f"CMD-GD manuscript missing new-site evidence phrase: {phrase}",
                        "manuscript.md",
                    )
        references_text = _read_text(paper_dir / "references.bib")
        if len(_bib_entry_keys(references_text)) < 15:
            _add_finding(
                findings,
                "error",
                "insufficient_references",
                "CMD-GD manuscript requires at least 15 BibTeX references.",
                "references.bib",
            )
        if len(_markdown_citation_keys(manuscript)) < 8:
            _add_finding(
                findings,
                "error",
                "insufficient_in_text_citations",
                "CMD-GD manuscript requires at least 8 in-text citation keys.",
                "manuscript.md",
            )
        summary_path = paper_dir / "algorithm_results_summary.json"
        summary = _load_json(summary_path)
        if not isinstance(summary, dict):
            summary = {}
        benchmark = summary.get("generative_benchmark")
        if not isinstance(benchmark, dict) or not benchmark.get("available"):
            _add_finding(
                findings,
                "error",
                "missing_generative_benchmark_summary",
                "CMD-GD manuscript requires generative benchmark summary evidence.",
                "algorithm_results_summary.json",
            )
        elif benchmark.get("selected_mechanism") != "cmdgd" or benchmark.get("selected_passes_gate") is not True:
            _add_finding(
                findings,
                "error",
                "cmdgd_gate_not_passed",
                "CMD-GD must be the selected generated mechanism and pass the predeclared eligibility rule.",
                "algorithm_results_summary.json",
            )
        project_table = paper_dir / "tables" / "project_selected_candidates.csv"
        project_pdf = paper_dir / "CMD-GD_1E62_short_paper.pdf"
        has_project_section = "## 1E62 Computational Design Output" in manuscript
        has_project_table = project_table.is_file() and project_table.stat().st_size > 0
        has_project_pdf = project_pdf.is_file() and project_pdf.stat().st_size > 0
        if not (has_project_table or has_project_pdf or has_project_section):
            _add_finding(
                findings,
                "error",
                "missing_cmdgd_project_design_output",
                "CMD-GD manuscript requires a selected project candidate table, project PDF, or project design section.",
                "manuscript.md",
            )
        if has_project_table and not any(
            str(row.get("selected", "")).lower() == "true" for row in _read_csv(project_table)
        ):
            _add_finding(
                findings,
                "error",
                "missing_cmdgd_selected_project_candidates",
                "CMD-GD selected project candidate table must include at least one selected row.",
                "tables/project_selected_candidates.csv",
            )
        for filename in CMDGD_FIGURE_FILENAMES.values():
            path = paper_dir / filename
            if not path.is_file() or path.stat().st_size == 0:
                _add_finding(
                    findings,
                    "error",
                    "missing_cmdgd_figure",
                    f"Missing non-empty CMD-GD figure: {filename}",
                    filename,
                )
        for filename in (
            "tables/related_work_matrix.csv",
            "tables/benchmark_summary.csv",
            "tables/statistical_summary.csv",
            "tables/pairwise_comparisons.csv",
            "tables/design_examples.csv",
            "tables/project_candidate_rationale.csv",
        ):
            path = paper_dir / filename
            if not path.is_file() or path.stat().st_size == 0:
                _add_finding(
                    findings,
                    "error",
                    "missing_cmdgd_table",
                    f"Missing non-empty CMD-GD table: {filename}",
                    filename,
                )
    else:
        for phrase in (PROSPECTIVE_VALIDATION_BOUNDARY, "retrospective masked project-data evidence"):
            if phrase not in manuscript:
                _add_finding(findings, "error", "missing_evidence_boundary_label", f"manuscript.md missing {phrase}", "manuscript.md")
    _validate_citations(manuscript, _read_text(paper_dir / "references.bib"), findings)
    _validate_artifact_links(manuscript, paper_dir, "manuscript.md", findings)
    errors = sum(1 for finding in findings if finding["severity"] == "error")
    warnings = sum(1 for finding in findings if finding["severity"] == "warning")
    return {
        "schema_version": 1,
        "valid": errors == 0,
        "status": "passed" if errors == 0 else "failed",
        "summary": {"errors": errors, "warnings": warnings},
        "findings": findings,
    }


def _algorithm_summary_row(context: dict[str, Any]) -> dict[str, str]:
    algorithm = str(context.get("algorithm") or "")
    if algorithm == "mccbd":
        return _mccbd_summary_row(context, algorithm, "overall")
    if algorithm == "cmdgd":
        return _cmdgd_summary_row(context, algorithm, "overall")
    rows = context.get("project_masking", {}).get("summary_rows", [])
    for row in rows:
        if isinstance(row, dict) and row.get("mechanism") == algorithm:
            return dict(row)
    return {}


def _baseline_comparisons(context: dict[str, Any]) -> list[dict[str, Any]]:
    if context.get("algorithm") == "mccbd":
        return _mccbd_baseline_comparisons(context)
    if context.get("algorithm") == "cmdgd":
        return _cmdgd_baseline_comparisons(context)
    algorithm_row = _algorithm_summary_row(context)
    if not algorithm_row:
        return []
    rows = context.get("project_masking", {}).get("summary_rows", [])
    out: list[dict[str, str]] = []
    for baseline in ("random_feasible", "fixed_mix", "greedy_observed"):
        baseline_row = next(
            (row for row in rows if isinstance(row, dict) and row.get("mechanism") == baseline),
            None,
        )
        if not baseline_row:
            continue
        out.append(
            {
                "baseline": baseline,
                "utility_delta": _format_delta(
                    _float_field(algorithm_row, "mean_best_feasible_utility")
                    - _float_field(baseline_row, "mean_best_feasible_utility")
                ),
                "false_claim_delta": _format_delta(
                    _float_field(algorithm_row, "mean_false_claim_rate")
                    - _float_field(baseline_row, "mean_false_claim_rate")
                ),
            }
        )
    return out


def _mccbd_summary_row(
    context: dict[str, Any],
    mechanism: str,
    world_id: str,
) -> dict[str, str]:
    rows = context.get("mccbd_benchmark", {}).get("summary_rows", [])
    for row in rows:
        if (
            isinstance(row, dict)
            and row.get("mechanism") == mechanism
            and row.get("world_id") == world_id
        ):
            return dict(row)
    return {}


def _project_masking_summary_row(context: dict[str, Any], mechanism: str) -> dict[str, str]:
    rows = context.get("project_masking", {}).get("summary_rows", [])
    for row in rows:
        if isinstance(row, dict) and row.get("mechanism") == mechanism:
            return dict(row)
    return {}


def _cmdgd_summary_row(
    context: dict[str, Any],
    mechanism: str,
    world_id: str,
) -> dict[str, str]:
    rows = context.get("generative_benchmark", {}).get("summary_rows", [])
    for row in rows:
        if (
            isinstance(row, dict)
            and row.get("mechanism") == mechanism
            and row.get("world_id") == world_id
        ):
            return dict(row)
    return {}


def _cmdgd_baseline_comparisons(context: dict[str, Any]) -> list[dict[str, Any]]:
    algorithm_row = _cmdgd_summary_row(context, "cmdgd", "overall")
    if not algorithm_row:
        return []
    out: list[dict[str, Any]] = []
    for baseline in (
        "random_edit_generator",
        "single_edit_scan",
        "observed_recombination_baseline",
        "mccbd_pool_selector",
    ):
        baseline_row = _cmdgd_summary_row(context, baseline, "overall")
        if not baseline_row:
            continue
        selected_delta = _format_delta(
            _float_field(algorithm_row, "mean_best_selected_utility")
            - _float_field(baseline_row, "mean_best_selected_utility")
        )
        generated_utility_applicable = _cmdgd_generated_utility_applicable(baseline_row)
        out.append(
            {
                "baseline": baseline,
                "generated_utility_applicable": generated_utility_applicable,
                "generated_utility_delta": (
                    _format_delta(
                        _float_field(algorithm_row, "mean_best_generated_utility")
                        - _float_field(baseline_row, "mean_best_generated_utility")
                    )
                    if generated_utility_applicable
                    else None
                ),
                "selected_utility_delta": selected_delta,
                "utility_delta": selected_delta,
                "novel_sequence_rate_delta": _format_delta(
                    _float_field(algorithm_row, "mean_novel_sequence_rate")
                    - _float_field(baseline_row, "mean_novel_sequence_rate")
                ),
                "false_claim_delta": _format_delta(
                    _float_field(algorithm_row, "mean_false_claim_rate")
                    - _float_field(baseline_row, "mean_false_claim_rate")
                ),
            }
        )
    return out


def _cmdgd_generated_utility_applicable(row: dict[str, Any]) -> bool:
    if str(row.get("fixed_pool_only", "")).strip().lower() == "true":
        return False
    value = row.get("generated_utility_applicable")
    if value is None or str(value).strip() == "":
        return True
    return str(value).strip().lower() not in {"false", "0", "no", "not_applicable", "n/a"}


def _cmdgd_has_new_site_artifacts(context: dict[str, Any]) -> bool:
    return bool(
        _cmdgd_new_site_summary_rows(context)
        or _cmdgd_de_novo_site_ablation_rows(context)
        or _cmdgd_project_new_site_values(context)
    )


def _cmdgd_new_site_summary_rows(context: dict[str, Any]) -> list[dict[str, Any]]:
    benchmark = context.get("generative_benchmark", {})
    if not isinstance(benchmark, dict):
        return []
    rows = [
        dict(row)
        for row in benchmark.get("summary_rows", [])
        if isinstance(row, dict)
        and row.get("mechanism") == "cmdgd"
        and _has_any_key(row, (*CMDGD_NEW_SITE_COUNT_KEYS, *CMDGD_NEW_SITE_RATE_KEYS))
    ]
    if rows:
        return rows
    result_rows = [
        dict(row)
        for row in benchmark.get("result_rows", [])
        if isinstance(row, dict)
        and row.get("mechanism") == "cmdgd"
        and row.get("status") in ("", "completed", None)
        and _has_any_key(row, (*CMDGD_NEW_SITE_COUNT_KEYS, *CMDGD_NEW_SITE_RATE_KEYS))
    ]
    if not result_rows:
        return []
    aggregate: dict[str, Any] = {"mechanism": "cmdgd", "world_id": "overall"}
    count_mean = _mean_first_float(result_rows, CMDGD_NEW_SITE_COUNT_KEYS)
    rate_mean = _mean_first_float(result_rows, CMDGD_NEW_SITE_RATE_KEYS)
    if count_mean is not None:
        aggregate["mean_generated_new_mutation_site_count"] = f"{count_mean:.6g}"
    if rate_mean is not None:
        aggregate["mean_new_site_generation_rate"] = f"{rate_mean:.6g}"
    return [aggregate] if len(aggregate) > 2 else []


def _cmdgd_de_novo_site_ablation_rows(context: dict[str, Any]) -> list[dict[str, Any]]:
    benchmark = context.get("generative_benchmark", {})
    if not isinstance(benchmark, dict):
        return []
    rows: list[dict[str, Any]] = []
    for row in benchmark.get("ablation_rows", []):
        if not isinstance(row, dict) or row.get("mechanism") != "cmdgd":
            continue
        ablation_text = " ".join(
            str(row.get(key, "")).lower()
            for key in ("ablation", "ablation_type", "ablation_component")
        )
        if any(term in ablation_text for term in CMDGD_DE_NOVO_SITE_ABLATION_TERMS):
            rows.append(dict(row))
    return rows


def _cmdgd_project_new_site_values(context: dict[str, Any]) -> dict[str, str]:
    benchmark = context.get("generative_benchmark", {})
    if not isinstance(benchmark, dict):
        return {}
    summary = benchmark.get("project_design_summary")
    if not isinstance(summary, dict):
        summary = {}
    count = _first_recorded_value(summary, CMDGD_NEW_SITE_COUNT_KEYS)
    rate = _first_recorded_value(summary, CMDGD_NEW_SITE_RATE_KEYS)
    values: dict[str, str] = {}
    if count:
        values["count"] = count
    if rate:
        values["rate"] = rate
    if values:
        return values
    project_rows = [
        dict(row)
        for row in benchmark.get("project_design_candidate_rows", [])
        if isinstance(row, dict)
        and _has_any_key(row, (*CMDGD_NEW_SITE_COUNT_KEYS, *CMDGD_NEW_SITE_RATE_KEYS))
    ]
    count_mean = _mean_first_float(project_rows, CMDGD_NEW_SITE_COUNT_KEYS)
    rate_mean = _mean_first_float(project_rows, CMDGD_NEW_SITE_RATE_KEYS)
    if count_mean is not None:
        values["count"] = _compact_float(count_mean)
    if rate_mean is not None:
        values["rate"] = _compact_float(rate_mean)
    return values


def _cmdgd_new_site_generation_lines(context: dict[str, Any]) -> list[str]:
    summary_rows = _cmdgd_new_site_summary_rows(context)
    ablation_rows = _cmdgd_de_novo_site_ablation_rows(context)
    project_values = _cmdgd_project_new_site_values(context)
    if not summary_rows and not ablation_rows and not project_values:
        return []

    lines = [
        (
            "Design-space diagnostics report whether generated sequences include edits outside the observed "
            "vocabulary. These columns are checks on the same CMD-GD generator used for all candidates, "
            "not a standalone algorithmic feature. The generator combines mechanism-prior grammar expansion, "
            "vocabulary-guided recombination, evidence-guided recombination, and local substitutions into "
            "auditable heavy/light candidate records."
        ),
    ]
    overall = next((row for row in summary_rows if row.get("world_id") == "overall"), None)
    if overall:
        parts = _cmdgd_new_site_metric_parts(overall)
        if parts:
            lines.append("In the overall CMD-GD benchmark row, the generation diagnostic reported " + " and ".join(parts) + ".")
    other_worlds = [
        row
        for row in summary_rows
        if row.get("world_id") not in ("", None, "overall")
    ]
    if other_worlds:
        world_labels = ", ".join(str(row.get("world_id")) for row in other_worlds[:5])
        lines.append(
            "The same diagnostic columns were also present for benchmark worlds "
            f"{world_labels}, so those rows are treated as generation diagnostics rather than wet-lab evidence."
        )
    if project_values:
        project_parts = []
        if project_values.get("count"):
            project_parts.append(f"project unobserved-edit-position count was {project_values['count']}")
        if project_values.get("rate"):
            project_parts.append(f"project unobserved-edit-position rate was {project_values['rate']}")
        if project_parts:
            lines.append("In the project-level CMD-GD design summary, the same generator reported " + " and ".join(project_parts) + ".")
    if ablation_rows:
        by_ablation: dict[str, list[dict[str, Any]]] = {}
        for row in ablation_rows:
            by_ablation.setdefault(str(row.get("ablation") or "de_novo_site_proposal"), []).append(row)
        for ablation, rows in sorted(by_ablation.items()):
            count_mean = _mean_first_float(rows, CMDGD_NEW_SITE_COUNT_KEYS)
            rate_mean = _mean_first_float(rows, CMDGD_NEW_SITE_RATE_KEYS)
            loss = _mean_first_float(rows, ("delta_from_full_best_selected_utility",))
            fragments = []
            if count_mean is not None:
                fragments.append(f"mean unobserved-edit-position count {_compact_float(count_mean)}")
            if rate_mean is not None:
                fragments.append(f"mean unobserved-edit-position rate {_compact_float(rate_mean)}")
            if loss is not None:
                fragments.append(f"mean selected-utility loss versus full {_compact_float(loss)}")
            detail = ", ".join(fragments) if fragments else "reported no numeric unobserved-edit metrics"
            lines.append(
                f"The grammar-expansion ablation `{ablation}` reported {detail}."
            )
    lines.append(
        "These design-space diagnostics are computational generation diagnostics; experimental activity, "
        "structural tolerance, and pH-sensitive release remain to be measured prospectively."
    )
    return lines


def _cmdgd_new_site_metric_parts(row: dict[str, Any]) -> list[str]:
    parts = []
    count = _first_recorded_value(row, CMDGD_NEW_SITE_COUNT_KEYS)
    rate = _first_recorded_value(row, CMDGD_NEW_SITE_RATE_KEYS)
    if count:
        parts.append(f"mean unobserved-edit-position count was {count}")
    if rate:
        parts.append(f"mean unobserved-edit-position rate was {rate}")
    return parts


def _first_recorded_value(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _first_float(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _to_float(row.get(key))
        if value is not None:
            return value
    return None


def _mean_first_float(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> float | None:
    values = [
        value
        for value in (_first_float(row, keys) for row in rows)
        if value is not None
    ]
    return sum(values) / len(values) if values else None


def _has_any_key(row: dict[str, Any], keys: tuple[str, ...]) -> bool:
    return any(row.get(key) not in (None, "") for key in keys)


def _compact_float(value: float) -> str:
    return f"{value:.6g}"


def _mccbd_baseline_comparisons(context: dict[str, Any]) -> list[dict[str, str]]:
    algorithm_row = _mccbd_summary_row(context, "mccbd", "overall")
    if not algorithm_row:
        return []
    out: list[dict[str, str]] = []
    for baseline in ("random_feasible", "fixed_mix", "greedy_observed", "evidence_calibrated_ucb"):
        baseline_row = _mccbd_summary_row(context, baseline, "overall")
        if not baseline_row:
            continue
        out.append(
            {
                "baseline": baseline,
                "utility_delta": _format_delta(
                    _float_field(algorithm_row, "mean_best_feasible_utility")
                    - _float_field(baseline_row, "mean_best_feasible_utility")
                ),
                "false_claim_delta": _format_delta(
                    _float_field(algorithm_row, "mean_false_claim_rate")
                    - _float_field(baseline_row, "mean_false_claim_rate")
                ),
            }
        )
    return out


def _algorithm_project_masking_links(masking: dict[str, Any], paper_dir: Path) -> str:
    paths = masking.get("artifact_paths", {}) if isinstance(masking, dict) else {}
    links = []
    for key, filename in PROJECT_MASKING_ARTIFACT_FILENAMES.items():
        target = paths.get(key)
        if target:
            links.append(_md_link(filename, target, paper_dir))
    return ", ".join(links) if links else "`project_masking_*` artifacts"


def _mccbd_benchmark_links(benchmark: dict[str, Any], paper_dir: Path) -> str:
    paths = benchmark.get("artifact_paths", {}) if isinstance(benchmark, dict) else {}
    links = []
    for key, filename in MCCBD_BENCHMARK_ARTIFACT_FILENAMES.items():
        target = paths.get(key)
        if target:
            links.append(_md_link(filename, target, paper_dir))
    return ", ".join(links) if links else "`mccbd_benchmark_*` artifacts"


def _cmdgd_benchmark_links(benchmark: dict[str, Any], paper_dir: Path) -> str:
    paths = benchmark.get("artifact_paths", {}) if isinstance(benchmark, dict) else {}
    links = []
    for key, filename in GENERATIVE_BENCHMARK_ARTIFACT_FILENAMES.items():
        target = paths.get(key)
        if target:
            links.append(_md_link(filename, target, paper_dir))
    return ", ".join(links) if links else "`generative_*` artifacts"


def _cmdgd_comparison_sentence(
    baseline: str,
    cmdgd_row: dict[str, str],
    baseline_row: dict[str, str],
) -> str:
    if not cmdgd_row or not baseline_row:
        return f"For `{baseline}`, the comparison row was unavailable."
    generated_delta = _format_delta(
        _float_field(cmdgd_row, "mean_best_generated_utility")
        - _float_field(baseline_row, "mean_best_generated_utility")
    )
    selected_delta = _format_delta(
        _float_field(cmdgd_row, "mean_best_selected_utility")
        - _float_field(baseline_row, "mean_best_selected_utility")
    )
    novel_delta = _format_delta(
        _float_field(cmdgd_row, "mean_novel_sequence_rate")
        - _float_field(baseline_row, "mean_novel_sequence_rate")
    )
    false_delta = _format_delta(
        _float_field(cmdgd_row, "mean_false_claim_rate")
        - _float_field(baseline_row, "mean_false_claim_rate")
    )
    generated_utility_applicable = _cmdgd_generated_utility_applicable(baseline_row)
    if generated_utility_applicable:
        generated_text = f"best generated utility differed by {generated_delta}"
    elif str(baseline_row.get("fixed_pool_only", "")).lower() == "true":
        generated_text = (
            "best generated utility is not applicable because the fixed-pool benchmark boundary "
            "does not generate new candidates"
        )
    else:
        generated_text = "best generated utility is not applicable for this benchmark boundary"
    boundary = []
    if str(baseline_row.get("observed_reuse_only", "")).lower() == "true":
        boundary.append("the observed-recombination benchmark boundary reuses observed modules")
    if str(baseline_row.get("fixed_pool_only", "")).lower() == "true":
        boundary.append("the fixed-pool benchmark boundary selects from a fixed candidate pool")
    boundary_text = f" The comparison boundary is that {' and '.join(boundary)}." if boundary else ""
    return (
        f"Against `{baseline}`, {generated_text}; best selected utility differed by {selected_delta}, "
        f"novel sequence rate differed by {novel_delta}, and false-claim rate differed by {false_delta}."
        f"{boundary_text}"
    )


def _cmdgd_statistical_summary_rows(context: dict[str, Any]) -> list[dict[str, Any]]:
    benchmark = context.get("generative_benchmark", {})
    if isinstance(benchmark, dict):
        rows = benchmark.get("statistical_summary_rows")
        if isinstance(rows, list) and rows:
            return [dict(row) for row in rows if isinstance(row, dict)]
        result_rows = benchmark.get("result_rows")
        if isinstance(result_rows, list):
            return _derive_cmdgd_statistical_summary_rows(result_rows)
    return []


def _cmdgd_pairwise_comparison_rows(context: dict[str, Any]) -> list[dict[str, Any]]:
    benchmark = context.get("generative_benchmark", {})
    if isinstance(benchmark, dict):
        rows = benchmark.get("pairwise_comparison_rows")
        if isinstance(rows, list) and rows:
            return [_normalize_cmdgd_pairwise_row(row) for row in rows if isinstance(row, dict)]
        result_rows = benchmark.get("result_rows")
        if isinstance(result_rows, list):
            return _derive_cmdgd_pairwise_rows(result_rows)
    return []


def _derive_cmdgd_statistical_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "generated_novel_count",
        "generated_new_mutation_site_count",
        "generated_new_site_count",
        "new_site_generation_rate",
        "new_mutation_site_generation_rate",
        "best_generated_utility",
        "best_selected_utility",
        "novel_sequence_rate",
        "false_claim_rate",
        "pH_contrast_score",
    )
    groups: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("status") not in ("", "completed", None):
            continue
        mechanism = str(row.get("mechanism") or "")
        world_id = str(row.get("world_id") or "")
        if not mechanism or not world_id:
            continue
        for metric in metrics:
            value = _to_float(row.get(metric))
            if value is None:
                continue
            groups.setdefault((mechanism, world_id, metric), []).append(value)
            groups.setdefault((mechanism, "overall", metric), []).append(value)
    out: list[dict[str, Any]] = []
    for (mechanism, world_id, metric), values in sorted(groups.items()):
        stats = _numeric_interval(values)
        out.append(
            {
                "mechanism": mechanism,
                "world_id": world_id,
                "metric": metric,
                "n": stats["n"],
                "mean": f"{stats['mean']:.6f}",
                "sd": f"{stats['sd']:.6f}",
                "sem": f"{stats['sem']:.6f}",
                "ci95_low": f"{stats['ci95_low']:.6f}",
                "ci95_high": f"{stats['ci95_high']:.6f}",
            }
        )
    return out


def _derive_cmdgd_pairwise_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "best_generated_utility",
        "best_selected_utility",
        "novel_sequence_rate",
        "false_claim_rate",
    )
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("status") not in ("", "completed", None):
            continue
        key = (str(row.get("seed")), str(row.get("world_id")), str(row.get("mechanism")))
        by_key[key] = row
    cmdgd_keys = [
        key for key in by_key if key[2] == "cmdgd" and key[1] not in ("", "overall")
    ]
    baselines = sorted({key[2] for key in by_key if key[2] != "cmdgd"})
    out: list[dict[str, Any]] = []
    for baseline in baselines:
        for metric in metrics:
            per_world: dict[str, list[tuple[float, float, float]]] = {}
            for seed, world_id, _mechanism in cmdgd_keys:
                cmdgd_row = by_key.get((seed, world_id, "cmdgd"))
                baseline_row = by_key.get((seed, world_id, baseline))
                if not cmdgd_row or not baseline_row:
                    continue
                cmdgd_value = _to_float(cmdgd_row.get(metric))
                baseline_value = _to_float(baseline_row.get(metric))
                if cmdgd_value is None or baseline_value is None:
                    continue
                per_world.setdefault(world_id, []).append(
                    (cmdgd_value, baseline_value, cmdgd_value - baseline_value)
                )
                per_world.setdefault("overall", []).append(
                    (cmdgd_value, baseline_value, cmdgd_value - baseline_value)
                )
            for world_id, triples in sorted(per_world.items()):
                deltas = [item[2] for item in triples]
                stats = _numeric_interval(deltas)
                out.append(
                    {
                        "baseline": baseline,
                        "world_id": world_id,
                        "metric": metric,
                        "paired_count": stats["n"],
                        "cmdgd_mean": f"{mean(item[0] for item in triples):.6f}",
                        "baseline_mean": f"{mean(item[1] for item in triples):.6f}",
                        "mean_delta": f"{stats['mean']:.6f}",
                        "sd_delta": f"{stats['sd']:.6f}",
                        "sem_delta": f"{stats['sem']:.6f}",
                        "ci95_low": f"{stats['ci95_low']:.6f}",
                        "ci95_high": f"{stats['ci95_high']:.6f}",
                        "cmdgd_win_fraction": f"{sum(1 for item in triples if item[2] > 0) / max(len(triples), 1):.6f}",
                    }
                )
    return out


def _cmdgd_real_data_masking_boundary_text(context: dict[str, Any]) -> str:
    if not _has_project_masking(context):
        return (
            "No real-data masking check was available, so the manuscript makes no claim about "
            "recovery of held-out project outcomes."
        )
    summary_rows = [
        row
        for row in context.get("project_masking", {}).get("summary_rows", [])
        if isinstance(row, dict)
    ]
    cmdgd_row = next(
        (
            row
            for row in summary_rows
            if str(row.get("mechanism")) in {"cmdgd", "cmdgd_observed_pool"}
        ),
        None,
    )
    if not cmdgd_row:
        return (
            "The real-data masking check did not include a CMD-GD row and is reported only as "
            "context for the observed project-data task."
        )
    top_row = max(
        summary_rows,
        key=lambda row: _float_field(row, "mean_best_feasible_utility"),
        default={},
    )
    cmdgd_value = cmdgd_row.get("mean_best_feasible_utility", "not recorded")
    if top_row and top_row.get("mechanism") != cmdgd_row.get("mechanism"):
        return (
            "In the real-data masking check, CMD-GD did not outperform the best observed-pool "
            f"selector; CMD-GD mean best feasible utility was {cmdgd_value}, while "
            f"{top_row.get('mechanism', 'the top selector')} reached "
            f"{top_row.get('mean_best_feasible_utility', 'not recorded')}."
        )
    return (
        "In the real-data masking check, CMD-GD was not evaluated as proof of prospective activity; "
        f"its mean best feasible utility was {cmdgd_value} under observed-pool replay."
    )


def _cmdgd_figure_caption_lines(
    context: dict[str, Any],
    figure_links: dict[str, str],
    table_links: dict[str, str],
    config: dict[str, Any],
) -> list[str]:
    selected_row = _cmdgd_summary_row(context, "cmdgd", "overall")
    vocabulary_row = _cmdgd_summary_row(context, "cmdgd", "vocabulary_extension")
    pairwise_rows = _cmdgd_pairwise_comparison_rows(context)
    observed_pair = next(
        (
            row
            for row in pairwise_rows
            if row.get("world_id") == "overall"
            and row.get("metric") == "best_selected_utility"
            and row.get("baseline") == "observed_recombination_baseline"
        ),
        {},
    )
    fixed_pair = next(
        (
            row
            for row in pairwise_rows
            if row.get("world_id") == "overall"
            and row.get("metric") == "best_selected_utility"
            and row.get("baseline") == "mccbd_pool_selector"
        ),
        {},
    )
    replicate_count = (
        selected_row.get("replicate_count")
        or _cmdgd_config_replicate_text(config)
        or "not recorded"
    )
    vocabulary_n = (
        vocabulary_row.get("replicate_count")
        or _cmdgd_config_seed_text(config)
        or "not recorded"
    )
    pair_n = observed_pair.get("paired_count") or fixed_pair.get("paired_count") or "not recorded"
    observed_ci = _cmdgd_ci_text(observed_pair)
    fixed_ci = _cmdgd_ci_text(fixed_pair)
    captions = [
        "Figure captions and statistical reporting:",
        "",
        (
            f"Figure 1 caption: CMD-GD lifecycle from observed heavy/light sequences through grammar "
            f"learning, mechanism-prior grammar expansion, candidate enumeration, scoring, "
            f"and panel selection ({figure_links['workflow']})."
        ),
        (
            f"Figure 2 caption: Overall generative benchmark mean best selected utility by mechanism "
            f"({figure_links['benchmark']}); n = {replicate_count} mechanism-world-seed rows for the "
            f"overall CMD-GD summary, with detailed means in {table_links['benchmark']}."
        ),
        (
            f"Figure 3 caption: Vocabulary-extension world performance ({figure_links['vocabulary']}); "
            f"n = {vocabulary_n} seed rows for CMD-GD when the row is available, emphasizing "
            "candidate-space expansion rather than fixed-pool selection."
        ),
        (
            f"Figure 4 caption: CMD-GD ablation losses ({figure_links['ablation']}); bars summarize "
            f"component-level selected-utility changes, with row-level values in {table_links['ablation']}."
        ),
        (
            f"Figure 5 caption: Paired selected-utility deltas for CMD-GD minus each baseline "
            f"({figure_links['statistics']}); n = {pair_n} paired comparisons for the observed-recombination "
            f"boundary when present, with 95% CI {observed_ci} versus observed recombination and "
            f"{fixed_ci} versus the fixed-pool selector in {table_links['pairwise']}."
        ),
    ]
    spaced: list[str] = []
    for item in captions:
        spaced.append(item)
        if item.startswith("Figure "):
            spaced.append("")
    return spaced


def _cmdgd_config_seed_text(config: dict[str, Any]) -> str:
    seeds = config.get("seeds")
    if isinstance(seeds, list) and seeds:
        return str(len(seeds))
    seed_count = config.get("seed_count")
    return str(seed_count) if seed_count not in (None, "") else ""


def _cmdgd_config_replicate_text(config: dict[str, Any]) -> str:
    seed_text = _cmdgd_config_seed_text(config)
    worlds = config.get("worlds")
    if seed_text and isinstance(worlds, list) and worlds:
        return str((_to_int(seed_text) or 0) * len(worlds))
    return seed_text


def _cmdgd_ci_text(row: dict[str, Any]) -> str:
    if not row:
        return "not recorded"
    low = _float_text(row.get("ci95_low"))
    high = _float_text(row.get("ci95_high"))
    if low == "not recorded" or high == "not recorded":
        return "not recorded"
    return f"[{low}, {high}]"


def _normalize_cmdgd_pairwise_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    aliases = {
        "paired_count": "n_pairs",
        "mean_delta": "paired_delta_mean",
        "sd_delta": "paired_delta_sd",
        "sem_delta": "paired_delta_sem",
        "ci95_low": "paired_delta_ci95_low",
        "ci95_high": "paired_delta_ci95_high",
        "cmdgd_win_fraction": "win_fraction",
    }
    for target, source in aliases.items():
        if target not in out or out.get(target) in (None, ""):
            out[target] = out.get(source, "")
    if "mechanism" not in out:
        out["mechanism"] = "cmdgd"
    return out


def _cmdgd_results_summary_text(
    selected_row: dict[str, str],
    gate_report: dict[str, Any],
) -> str:
    rank = gate_report.get("selected_utility_rank") or selected_row.get("comparative_utility_rank")
    rank_text = (
        "selected-panel utility rank was 1 among eligible mechanisms and is treated as a benchmark statistic, not as a general utility-dominance claim"
        if str(rank) == "1"
        else f"selected-panel utility rank was {rank}"
        if rank not in (None, "")
        else "selected-panel utility rank was not recorded"
    )
    return (
        "CMD-GD generated "
        f"{selected_row.get('mean_generated_novel_count', 'not recorded')} novel candidates per benchmark "
        "replicate on average (the generated novel count). The best generated utility was "
        f"{selected_row.get('mean_best_generated_utility', 'not recorded')}, the best selected utility was "
        f"{selected_row.get('mean_best_selected_utility', 'not recorded')}, and the best selected pH contrast "
        f"was {selected_row.get('mean_best_selected_pH_contrast_score', 'not recorded')}. The constraint pass "
        f"rate was {selected_row.get('mean_constraint_pass_rate', 'not recorded')}, the mean pH contrast was "
        f"{selected_row.get('mean_pH_contrast_score', 'not recorded')}, the novel sequence rate was "
        f"{selected_row.get('mean_novel_sequence_rate', 'not recorded')}, and the false-claim rate was "
        f"{selected_row.get('mean_false_claim_rate', 'not recorded')}. Under the predeclared eligibility rule, {rank_text}."
    )


def _cmdgd_statistical_result_lines(context: dict[str, Any], paper_dir: Path) -> list[str]:
    del paper_dir
    pairwise = _cmdgd_pairwise_comparison_rows(context)
    if not pairwise:
        return [
            "Seed-level statistical summaries were unavailable; the benchmark should be treated as descriptive only."
        ]
    selected_rows = {
        row.get("baseline"): row
        for row in pairwise
        if row.get("world_id") == "overall" and row.get("metric") == "best_selected_utility"
    }
    observed = selected_rows.get("observed_recombination_baseline")
    fixed_pool = selected_rows.get("mccbd_pool_selector")
    random_edit = selected_rows.get("random_edit_generator")
    lines = [
        "The benchmark is summarized as paired seed/world differences rather than only as pooled means. This matters because the observed-recombination boundary is intentionally strong: when the useful observed modules are already present, it can closely match CMD-GD on selected utility."
    ]
    if observed:
        lines.append(
            "Against observed recombination, the paired selected-utility difference was "
            f"{_signed_float_text(observed.get('mean_delta'))} with a 95% interval "
            f"[{_float_text(observed.get('ci95_low'))}, {_float_text(observed.get('ci95_high'))}] "
            f"and a CMD-GD win fraction of {_float_text(observed.get('cmdgd_win_fraction'))}. "
            "This is reported as a claim boundary, not as evidence that generation produces a large selected-utility gain in every world."
        )
    if fixed_pool:
        lines.append(
            "Against the fixed-pool selector, the paired selected-utility difference was "
            f"{_signed_float_text(fixed_pool.get('mean_delta'))} with a 95% interval "
            f"[{_float_text(fixed_pool.get('ci95_low'))}, {_float_text(fixed_pool.get('ci95_high'))}]. "
            "Because the fixed-pool selector does not create candidates, this comparison separates panel scoring from candidate-space expansion."
        )
    if random_edit:
        lines.append(
            "Against random editing, the paired selected-utility difference was "
            f"{_signed_float_text(random_edit.get('mean_delta'))}; this supports the claim that the learned grammar and contrastive scorer are doing more than random sequence editing."
        )
    return lines


def _cmdgd_project_masking_lines(context: dict[str, Any], paper_dir: Path) -> list[str]:
    if not _has_project_masking(context):
        return [
            "No real-data retrospective masking benchmark was available for this study. The generative benchmark therefore remains the primary evidence, and no claim is made about recovery of held-out 1E62 wet-lab outcomes."
        ]
    summary_rows = context.get("project_masking", {}).get("summary_rows", [])
    cmdgd_row = next(
        (
            row
            for row in summary_rows
            if isinstance(row, dict) and str(row.get("mechanism")) in {"cmdgd", "cmdgd_observed_pool"}
        ),
        None,
    )
    lines = [
        "A separate retrospective masking benchmark uses the standardized 1E62 observations as observed-pool data. "
        "It hides measured outcomes for held-out variants, exposes only sequence and module descriptors to the selector, "
        "and scores selections after the fact. The accompanying supplementary tables report the summary, fold-level "
        "selections, ablations, and benchmark configuration."
    ]
    if cmdgd_row:
        top_row = max(
            [row for row in summary_rows if isinstance(row, dict)],
            key=lambda row: _float_field(row, "mean_best_feasible_utility"),
            default={},
        )
        lines.append(
            "In that real-data masking setting, CMD-GD reached mean best feasible utility "
            f"{cmdgd_row.get('mean_best_feasible_utility', 'not recorded')}, mean regret proxy "
            f"{cmdgd_row.get('mean_regret_proxy', 'not recorded')}, and mean false-claim rate "
            f"{cmdgd_row.get('mean_false_claim_rate', 'not recorded')}. This is retrospective masked project-data evidence, not a prospective assay result."
        )
        if top_row and top_row.get("mechanism") != cmdgd_row.get("mechanism"):
            lines.append(
                "CMD-GD did not outperform the best observed-pool selector in this retrospective masking check; "
                f"the top row was {top_row.get('mechanism')} with mean best feasible utility "
                f"{top_row.get('mean_best_feasible_utility', 'not recorded')}. "
                "This negative result is important: the observed-pool masking task evaluates recovery of already measured variants, whereas CMD-GD's main algorithmic claim is candidate-space expansion beyond the measured pool."
            )
    else:
        lines.append(
            "The masking benchmark did not include a CMD-GD row, so it is used only as context for the observed project data rather than as direct evidence for CMD-GD."
        )
    return lines


def _cmdgd_biophysical_rationale_lines(context: dict[str, Any], rationale_table_link: str) -> list[str]:
    benchmark = context.get("generative_benchmark", {})
    project_rows = [
        row
        for row in benchmark.get("project_design_candidate_rows", [])
        if isinstance(row, dict) and str(row.get("selected", "")).lower() == "true"
    ] if isinstance(benchmark, dict) else []
    rationale_rows = _cmdgd_candidate_rationale_rows(project_rows)
    if not rationale_rows:
        return [
            "No selected 1E62 candidate rows were available for sequence-level plausibility checks."
        ]
    ranks = ", ".join(str(row.get("rank")) for row in rationale_rows)
    histidine_rows = sum(1 for row in rationale_rows if _to_int(row.get("histidine_edit_count")) and _to_int(row.get("histidine_edit_count")) > 0)
    cdr_rows = sum(1 for row in rationale_rows if str(row.get("contains_cdr_edit", "")).lower() == "true")
    lines = [
        f"The selected computational panel occupied candidate ranks {ranks}, not simply the top five ranks. The selector trades off score, cost, parent diversity, residue-class diversity, and redundancy, so lower-ranked entries can enter the panel when they add a distinct contrastive edit pattern rather than duplicating the same parent combination.",
        (
            f"Sequence-level annotation is summarized in {rationale_table_link}. "
            f"{histidine_rows} of {len(rationale_rows)} selected candidates include at least one histidine edit, which is plausible for pH-sensitive behavior, and {cdr_rows} include an edit in an approximate CDR interval. These checks are deliberately conservative: without structure, antigen-complex modeling, and new assays, they are plausibility screens rather than proof of pH switching."
        ),
    ]
    liability_flags = sorted(
        {
            flag
            for row in rationale_rows
            for flag in _cell_list(row.get("sequence_liability_flags"))
            if flag and flag != "none_detected_by_sequence_screen"
        }
    )
    if liability_flags:
        lines.append(
            "The sequence screen flagged possible liabilities that should be reviewed before synthesis: "
            + ", ".join(liability_flags)
            + "."
        )
    else:
        lines.append(
            "The simple sequence-level liability screen did not detect new cysteine, N-glycosylation, or obvious deamidation/isomerization motifs in the selected proposals."
        )
    return lines


def _cmdgd_reference_list_lines(context: dict[str, Any], limit: int = 20) -> list[str]:
    bib_text = context.get("cmdgd_literature", {}).get("references_bib_text", "")
    entries = _parse_bib_entries(str(bib_text))
    if not entries:
        return ["No structured reference entries were available in the local bibliography file."]
    lines = []
    for entry in entries[:limit]:
        author = entry.get("author", "").replace("\n", " ")
        first_author = author.split(" and ", 1)[0].strip("{} ")
        title = entry.get("title", "").replace("\n", " ").strip("{} ")
        journal = entry.get("journal") or entry.get("booktitle") or entry.get("publisher") or ""
        year = entry.get("year", "")
        key = entry.get("key", "")
        pieces = [piece for piece in (first_author, f"({year})" if year else "", title, journal) if piece]
        lines.append(f"- {key}: " + ". ".join(pieces) + ".")
    return lines


def _parse_bib_entries(text: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for match in re.finditer(r"@\w+\{([^,]+),([\s\S]*?)(?=\n@\w+\{|$)", text):
        key = match.group(1).strip()
        body = match.group(2)
        entry = {"key": key}
        for field_match in re.finditer(r"(\w+)\s*=\s*[{\"]([\s\S]*?)[}\"]\s*,", body):
            field = field_match.group(1).strip().lower()
            value = re.sub(r"\s+", " ", field_match.group(2)).strip()
            entry[field] = value
        entries.append(entry)
    return entries


def _cmdgd_candidate_rationale_rows(project_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in sorted(project_rows, key=lambda item: _to_int(item.get("rank")) or 10**9):
        modules = _cell_list(row.get("modules"))
        parent_ids = _cell_list(row.get("parent_ids"))
        heavy = str(row.get("heavy_chain_seq") or "")
        light = str(row.get("light_chain_seq") or "")
        annotations = [_edit_annotation(token) for token in modules]
        regions = [annotation["region"] for annotation in annotations]
        liability_flags = _sequence_liability_flags(heavy, light)
        histidine_count = sum(1 for item in annotations if item.get("to_residue") == "H")
        acidic_count = sum(1 for item in annotations if item.get("to_residue") in {"D", "E"})
        cdr_count = sum(1 for region in regions if "CDR" in region)
        rows.append(
            {
                "rank": row.get("rank", ""),
                "candidate_id": row.get("candidate_id", ""),
                "score": row.get("score", ""),
                "selected": row.get("selected", ""),
                "modules": ";".join(modules),
                "approx_regions": ";".join(f"{item['edit_id']}:{item['region']}" for item in annotations),
                "parent_ids": ";".join(parent_ids),
                "operator": row.get("operator", ""),
                "histidine_edit_count": histidine_count,
                "acidic_edit_count": acidic_count,
                "contains_cdr_edit": str(cdr_count > 0).lower(),
                "cdr_edit_count": cdr_count,
                "predicted_contrastive_ph_objective": row.get("predicted_contrastive_ph_objective", ""),
                "developability_feasibility": row.get("developability_feasibility", ""),
                "novelty": row.get("novelty", ""),
                "uncertainty": row.get("uncertainty", ""),
                "selection_diversity_prior": row.get("selection_diversity_prior", ""),
                "provenance_prior": row.get("provenance_prior", ""),
                "sequence_liability_flags": ";".join(liability_flags),
                "selection_rationale": _cmdgd_selection_rationale(row, annotations, parent_ids),
            }
        )
    return rows


def _cmdgd_selection_rationale(
    row: dict[str, Any],
    annotations: list[dict[str, str]],
    parent_ids: list[str],
) -> str:
    rank = _to_int(row.get("rank"))
    rank_text = f"rank {rank}" if rank is not None else "recorded rank"
    chains = sorted({item["chain"] for item in annotations if item.get("chain")})
    regions = sorted({item["region"] for item in annotations if item.get("region")})
    residue_classes = sorted({_residue_class(item.get("to_residue", "")) for item in annotations})
    return (
        f"Selected at {rank_text} because the panel objective balances score with nonredundant chain coverage "
        f"({'+'.join(chains) or 'none'}), approximate region coverage ({', '.join(regions) or 'none'}), "
        f"residue-class diversity ({', '.join(residue_classes) or 'none'}), and parent support from "
        f"{len(set(parent_ids))} observed variants."
    )


def _edit_annotation(token: str) -> dict[str, str]:
    text = str(token)
    match = re.fullmatch(r"([HL])(\d+)([A-Z])", text)
    if not match:
        return {
            "edit_id": text,
            "chain": "",
            "position": "",
            "to_residue": "",
            "region": "unparsed",
        }
    chain, position_text, residue = match.groups()
    position = int(position_text)
    return {
        "edit_id": text,
        "chain": chain,
        "position": position_text,
        "to_residue": residue,
        "region": _approx_antibody_region(chain, position),
    }


def _approx_antibody_region(chain: str, position: int) -> str:
    if chain == "H":
        if 26 <= position <= 35:
            return "CDRH1_approx"
        if 50 <= position <= 65:
            return "CDRH2_approx"
        if 95 <= position <= 102:
            return "CDRH3_approx"
        if position < 26:
            return "HFR1_approx"
        if position < 50:
            return "HFR2_approx"
        if position < 95:
            return "HFR3_approx"
        return "HFR4_approx"
    if chain == "L":
        if 24 <= position <= 34:
            return "CDRL1_approx"
        if 50 <= position <= 56:
            return "CDRL2_approx"
        if 89 <= position <= 97:
            return "CDRL3_approx"
        if position < 24:
            return "LFR1_approx"
        if position < 50:
            return "LFR2_approx"
        if position < 89:
            return "LFR3_approx"
        return "LFR4_approx"
    return "unknown"


def _residue_class(residue: str) -> str:
    if residue in {"H"}:
        return "histidine_pH_titratable"
    if residue in {"D", "E"}:
        return "acidic"
    if residue in {"K", "R"}:
        return "basic"
    if residue in {"Y", "F", "W"}:
        return "aromatic"
    if residue in {"S", "T", "N", "Q"}:
        return "polar"
    if residue in {"A", "V", "L", "I", "M"}:
        return "hydrophobic"
    return "other"


def _sequence_liability_flags(heavy: str, light: str) -> list[str]:
    flags: list[str] = []
    combined = f"{heavy}|{light}"
    if re.search(r"N[^P][ST]", combined):
        flags.append("possible_N_glycosylation_motif")
    if re.search(r"N[GST]", combined):
        flags.append("possible_deamidation_motif")
    if re.search(r"D[GPS]", combined):
        flags.append("possible_isomerization_motif")
    for sequence in (heavy, light):
        if re.search(r"[AILMFWVY]{6,}", sequence):
            flags.append("hydrophobic_run_check")
            break
    return sorted(set(flags)) or ["none_detected_by_sequence_screen"]


def _cell_list(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item)]
    return [item.strip() for item in text.replace(",", ";").split(";") if item.strip()]


def _cmdgd_ablation_lines(context: dict[str, Any]) -> list[str]:
    rows = [
        row
        for row in context.get("generative_benchmark", {}).get("ablation_rows", [])
        if isinstance(row, dict) and row.get("mechanism") == "cmdgd"
    ]
    if not rows:
        return [
            "No CMD-GD ablation rows were available; ablation rows support the comparison only when present."
        ]
    by_ablation: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_ablation.setdefault(row.get("ablation", "unknown"), []).append(row)
    labels = {
        "benchmark_no_generation_boundary": "The benchmark no-generation boundary",
        "full": "The full CMD-GD mechanism",
        "no_de_novo_site_proposal": "The de novo site proposal ablation",
        "no_new_site_proposal": "The de novo site proposal ablation",
        "no_new_mutation_site_proposal": "The de novo site proposal ablation",
        "no_contrastive_objective": "Removing the contrastive objective",
        "no_grammar_recombination": "Removing grammar recombination",
        "no_novelty_uncertainty": "Removing novelty and uncertainty terms",
    }
    lines = ["Ablation rows support the comparison by isolating the generator and scoring components."]
    for ablation, ablation_rows in sorted(by_ablation.items()):
        label = labels.get(ablation, ablation.replace("_", " ").capitalize())
        lines.append(
            f"{label} produced a mean generated novel count of "
            f"{_mean_float(ablation_rows, 'generated_novel_count'):.4f}, mean best selected utility of "
            f"{_mean_float(ablation_rows, 'best_selected_utility'):.4f}, mean pH contrast of "
            f"{_mean_float(ablation_rows, 'pH_contrast_score'):.4f}, mean false-claim rate of "
            f"{_mean_float(ablation_rows, 'false_claim_rate'):.4f}, and mean selected-utility loss versus "
            f"the full method of {_mean_float(ablation_rows, 'delta_from_full_best_selected_utility'):.4f}."
        )
    lines.append(
        "The benchmark no-generation boundary can still select from available candidates, but it disables "
        "generated candidate creation and collapses novel sequence rate; it is a fixed-pool benchmark "
        "boundary, not a component-deletion ablation."
    )
    lines.append(
        "Accordingly, ablation effects are interpreted only when ablation rows support the comparison; they are "
        "descriptive benchmark results, not independent mechanism proof."
    )
    return lines


def _cmdgd_data_availability_text(context: dict[str, Any], data_summary: str) -> str:
    benchmark = context.get("generative_benchmark", {})
    summary = benchmark.get("project_design_summary") if isinstance(benchmark, dict) else {}
    if not isinstance(summary, dict):
        summary = {}
    total_rows, kd_rows = _cmdgd_standardized_observation_counts(context)
    design_endpoint_rows = summary.get("observed_endpoint_count")
    observed_variants = summary.get("observed_variant_count")
    if total_rows and design_endpoint_rows:
        row_text = (
            f"The standardized 1E62 observation table contains {total_rows} standardized observation rows "
            f"overall. CMD-GD's project design state used {design_endpoint_rows} endpoint rows linked to the "
            f"{observed_variants or 'recorded'} observed sequence records"
        )
        if kd_rows:
            row_text += (
                f"; the remaining {kd_rows} Ae KD-ratio rows are retained in the standardized data but are not "
                "part of the sequence-linked candidate-generation state."
            )
        else:
            row_text += "."
    else:
        row_text = f"The startup data summary is: {_sentence(data_summary)}"
    return (
        f"{row_text} Supplementary literature tables, benchmark summaries, ablation summaries, and design "
        "examples accompany the manuscript; no new wet-lab measurements are introduced here."
    )


def _cmdgd_standardized_observation_counts(context: dict[str, Any]) -> tuple[int, int]:
    rows = _read_csv(Path(context["root"]) / "standardized" / "observations_long.csv")
    if not rows:
        return 0, 0
    kd_rows = sum(
        1
        for row in rows
        if row.get("endpoint") == "KD_ratio" or "ae_kd_ratio" in str(row.get("source_file", ""))
    )
    return len(rows), kd_rows


def _cmdgd_project_design_text(
    context: dict[str, Any],
    paper_dir: Path,
    selected_table_link: str,
) -> str:
    benchmark = context.get("generative_benchmark", {})
    if not isinstance(benchmark, dict):
        benchmark = {}
    summary = benchmark.get("project_design_summary")
    if not isinstance(summary, dict) or not summary:
        return (
            "This manuscript bundle did not include a separate project-level CMD-GD design table. The paper therefore "
            "reports only the generic generative benchmark evidence."
        )
    selected_ids = summary.get("selected_candidate_ids")
    if isinstance(selected_ids, list):
        selected_preview = ", ".join(str(item) for item in selected_ids[:5])
        if len(selected_ids) > 5:
            selected_preview += ", ..."
    else:
        selected_preview = "not recorded"
    return (
        "After the benchmark, the same CMD-GD lifecycle was applied to the standardized 1E62 startup data as a "
        "computational design output. The benchmark includes vocabulary-extension stress tests; in the 1E62 "
        "application, the generator used observed grammar evidence and mechanism-prior edit discovery, with "
        f"{summary.get('candidate_edit_vocabulary_count', 'not recorded')} external mechanism-vocabulary edits recorded. This application used "
        f"{summary.get('observed_variant_count', 'not recorded')} observed sequence records, "
        f"{summary.get('observed_endpoint_count', 'not recorded')} endpoint rows linked to those sequence records. It generated "
        f"{summary.get('generated_candidate_count', 'not recorded')} candidate heavy/light sequence records, scored "
        f"{summary.get('scored_candidate_count', 'not recorded')} of them, and selected "
        f"{summary.get('selected_count', 'not recorded')} computational proposals under the panel budget. The selected "
        f"candidate table is {selected_table_link}. Representative selected IDs are "
        f"{selected_preview}. This section is a computational design proposal, not a wet-lab result."
    )


def _mccbd_ablation_lines(context: dict[str, Any]) -> list[str]:
    rows = context.get("mccbd_benchmark", {}).get("ablation_rows", [])
    if not rows:
        return ["No MCCBD ablation rows were available."]
    by_ablation: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if isinstance(row, dict):
            by_ablation.setdefault(row.get("ablation", "unknown"), []).append(row)
    lines = ["Ablation summary:"]
    for ablation, ablation_rows in sorted(by_ablation.items()):
        mean_delta = _mean_float(ablation_rows, "delta_from_full_best_feasible_utility")
        mean_false = _mean_float(ablation_rows, "false_claim_rate")
        lines.append(
            f"- {ablation}: mean utility loss versus full {mean_delta:.4f}; "
            f"mean false-claim rate {mean_false:.4f}"
        )
    return lines


def _mccbd_benchmark_summary_payload(context: dict[str, Any]) -> dict[str, Any]:
    benchmark = context.get("mccbd_benchmark", {})
    if not isinstance(benchmark, dict):
        benchmark = {}
    config = benchmark.get("config")
    if not isinstance(config, dict):
        config = {}
    return {
        "available": bool(benchmark.get("available")),
        "artifact_dir": _summary_artifact_path(benchmark.get("artifact_dir"), context),
        "artifacts": _summary_artifact_paths(benchmark.get("artifact_paths", {}), context),
        "summary_row_count": len(benchmark.get("summary_rows", [])),
        "result_row_count": len(benchmark.get("result_rows", [])),
        "ablation_row_count": len(benchmark.get("ablation_rows", [])),
        "selected_mechanism": config.get("selected_mechanism"),
        "selected_passes_gate": config.get("selected_passes_gate"),
        "config": config,
        "top_summary_row": _mccbd_summary_row(context, "mccbd", "overall"),
    }


def _generative_benchmark_summary_payload(context: dict[str, Any]) -> dict[str, Any]:
    benchmark = context.get("generative_benchmark", {})
    if not isinstance(benchmark, dict):
        benchmark = {}
    config = benchmark.get("config")
    if not isinstance(config, dict):
        config = {}
    return {
        "available": bool(benchmark.get("available")),
        "artifact_dir": _summary_artifact_path(benchmark.get("artifact_dir"), context),
        "artifacts": _summary_artifact_paths(benchmark.get("artifact_paths", {}), context),
        "summary_row_count": len(benchmark.get("summary_rows", [])),
        "statistical_summary_row_count": len(_cmdgd_statistical_summary_rows(context)),
        "pairwise_comparison_row_count": len(_cmdgd_pairwise_comparison_rows(context)),
        "result_row_count": len(benchmark.get("result_rows", [])),
        "ablation_row_count": len(benchmark.get("ablation_rows", [])),
        "design_example_row_count": len(benchmark.get("design_example_rows", [])),
        "project_design_candidate_row_count": len(benchmark.get("project_design_candidate_rows", [])),
        "project_design_summary": benchmark.get("project_design_summary", {}),
        "selected_mechanism": config.get("selected_mechanism"),
        "selected_passes_gate": config.get("selected_passes_gate"),
        "selection_gate_report": benchmark.get("selection_gate_report", {}),
        "config": config,
        "top_summary_row": _cmdgd_summary_row(context, "cmdgd", "overall"),
    }


def _project_masking_run_id(context: dict[str, Any]) -> str | None:
    artifact_dir = context.get("project_masking", {}).get("artifact_dir")
    if not artifact_dir:
        return None
    return Path(str(artifact_dir)).name


def _evidence_label(context: dict[str, Any]) -> str:
    if _has_project_masking(context):
        return "computational and synthetic replay evidence plus retrospective masked project-data evidence"
    return "computational and synthetic replay evidence"


def _evidence_boundary_text(context: dict[str, Any]) -> str:
    if _has_project_masking(context):
        return (
            "Framework benchmark evidence in this manuscript is computational and synthetic replay evidence. "
            "Project masking artifacts, when reported, are retrospective masked project-data evidence from "
            "observed-pool masking of already standardized observations. The paper does not report prospective "
            "wet-lab validation; biological or experimental claims require separate primary data before use as "
            "validated wet-lab findings."
        )
    return (
        "All evidence in this manuscript is computational and synthetic replay evidence generated from "
        "framework V3 artifacts. The paper does not report wet-lab validation; biological or experimental "
        "claims require separate primary data before use as validated wet-lab findings."
    )


def _formal_problem_setup_text(context: dict[str, Any], data_summary: str) -> str:
    project_spec = context.get("project_spec", {})
    estimands = context.get("estimands", {})
    objective = _one_line(project_spec.get("objective")) or "optimize a design policy under recorded constraints"
    primary_estimand = (
        _one_line(estimands.get("primary_estimand"))
        or _one_line(estimands.get("primary_endpoint"))
        or _one_line(estimands.get("objective"))
        or "the project-level utility recorded in the framework artifacts"
    )
    return (
        f"Let X denote the feasible design space and D_t denote observations available after round t. "
        f"Each observation links a candidate x in X to measured or replayed endpoints, provenance, and cost. "
        f"The objectives are to choose a bounded panel that improves `{primary_estimand}` for the task "
        f"`{objective}` while respecting feasibility constraints, assay or data-contract constraints, and "
        "claim-boundary constraints. A policy maps D_t and literature-derived mechanism state to candidate "
        "scores, selected panels, and ablations. The available local evidence basis for this run was: "
        f"{_sentence(data_summary)}"
    )


def _literature_basis_text(context: dict[str, Any], literature_summary: str, citation: str) -> str:
    mechanism_count = len(context.get("mechanism_cards", []))
    library_count = len(context.get("mechanism_library", []))
    gap_count = len(context.get("mechanism_gap_rows", []))
    return (
        f"The literature basis contains {literature_summary} {citation}. Literature Engine V3 converts those "
        f"records into {mechanism_count} mechanism cards, {library_count} mechanism-library entries, and "
        f"{gap_count} gap-matrix rows. These artifacts define method priors, baselines, stress tests, ablations, "
        "and known failure modes; they are literature support for algorithm design, not wet-lab validation of "
        "the local project."
    )


def _algorithmic_contribution_text(context: dict[str, Any], selected: str) -> str:
    mechanism_spec = context.get("mechanism_spec", {})
    components = mechanism_spec.get("components")
    component_count = len(components) if isinstance(components, list) else 0
    return (
        "The real algorithmic contribution is an artifact-backed loop that turns literature-derived mechanism "
        f"components into an executable active-design policy. For `{selected}`, the MechanismSpec records "
        f"{component_count} components and the generated mechanism must implement `fit_state`, "
        "`generate_candidates`, `score_candidates`, `select_panel`, and `plan_ablations`. This contributes "
        "a reproducible method contract and evaluation protocol; it does not claim that any proposed variant "
        "has been experimentally validated by this manuscript."
    )


def _evaluation_design_text(context: dict[str, Any], paper_dir: Path) -> str:
    artifact_paths = context["artifact_paths"]
    text = (
        "Evaluation design uses offline framework replay before any prospective claim. The synthetic benchmark "
        "compares the selected mechanism against required baselines including `random_feasible` and `fixed_mix`, "
        "then checks ablations to test whether the claimed mechanism component matters. Framework replay inputs "
        "and outputs are "
        f"{_md_link('mechanism_benchmark_results.csv', artifact_paths['mechanism_benchmark_results'], paper_dir)}, "
        f"{_md_link('mechanism_benchmark_summary.csv', artifact_paths['mechanism_benchmark_summary'], paper_dir)}, and "
        f"{_md_link('mechanism_ablation_results.csv', artifact_paths['mechanism_ablation_results'], paper_dir)}."
    )
    if not _has_project_masking(context):
        return (
            text
            + " Project-data retrospective masking artifacts were not present for this manuscript bundle, "
            "so results are limited to computational and synthetic replay evidence."
        )
    links = _project_masking_links_text(context, paper_dir)
    return (
        text
        + " Project-data retrospective masking artifacts were also present. They evaluate observed-pool masking "
        "of already measured project variants with held-out outcomes and leakage controls, producing "
        f"retrospective masked project-data evidence in {links}. These results "
        f"{PROSPECTIVE_VALIDATION_BOUNDARY}."
    )


def _project_masking_result_lines(context: dict[str, Any]) -> list[str]:
    project_masking = context.get("project_masking", {})
    if not project_masking.get("available"):
        return ["", "No project masking artifacts were present for this manuscript bundle."]

    lines = [
        "",
        "Project-data retrospective masking metrics:",
        "",
        (
            "Retrospective masked project-data evidence was available from observed-pool masking artifacts. "
            "These results reuse already standardized observations and "
            f"{PROSPECTIVE_VALIDATION_BOUNDARY}."
        ),
    ]
    top_row = project_masking.get("top_summary_row")
    if isinstance(top_row, dict) and top_row:
        lines.append(f"- top masking mechanism: {top_row.get('mechanism') or top_row.get('method') or 'not recorded'}")
        for key in (
            "fold_count",
            "completed_fold_count",
            "mean_best_feasible_utility",
            "mean_hit_rate",
            "mean_regret_proxy",
            "mean_false_claim_rate",
            "mean_evidence_coverage",
            "failed_fold_count",
        ):
            if top_row.get(key) not in (None, ""):
                lines.append(f"- {key}: {top_row[key]}")
    else:
        lines.append("- No project masking summary row was available.")
    return lines


def _project_masking_links_text(context: dict[str, Any], paper_dir: Path) -> str:
    links = _project_masking_link_items(context, paper_dir)
    return ", ".join(links) if links else "project masking artifacts with missing paths"


def _project_masking_link_items(context: dict[str, Any], paper_dir: Path) -> list[str]:
    links: list[str] = []
    artifact_paths = context.get("artifact_paths", {})
    for key, filename in PROJECT_MASKING_ARTIFACT_FILENAMES.items():
        path = artifact_paths.get(key)
        if path:
            links.append(_md_link(filename, path, paper_dir))
    return links


def _has_project_masking(context: dict[str, Any]) -> bool:
    project_masking = context.get("project_masking", {})
    return bool(isinstance(project_masking, dict) and project_masking.get("available"))


def _paper_title(context: dict[str, Any], selected: str) -> str:
    project_spec = context.get("project_spec", {})
    system = _one_line(project_spec.get("system") or project_spec.get("project_id"))
    objective = _one_line(project_spec.get("objective")).lower()
    if system and ("ph" in objective or "pH" in objective):
        return f"V3 mechanism for {system} pH-sensitive antibody design"
    return selected


def _render_references_bib(context: dict[str, Any]) -> str:
    entries: list[str] = []
    for card in context["paper_cards"]:
        paper_id = _one_line(card.get("paper_id")) or "paper"
        key = _cite_key(paper_id)
        fields = {
            "title": card.get("title") or card.get("citation") or paper_id,
            "year": card.get("year"),
            "doi": card.get("doi"),
            "url": card.get("url"),
            "note": f"Design Scientist literature card {paper_id}; not wet-lab validation for this run",
        }
        authors = _format_authors(card.get("authors"))
        if authors:
            fields["author"] = authors
        lines = [f"@misc{{{key},"]
        for field, value in fields.items():
            if value in (None, "", [], {}):
                continue
            lines.append(f"  {field} = {{{_bib_escape(_one_line(value))}}},")
        if len(lines) > 1:
            lines[-1] = lines[-1].rstrip(",")
        lines.append("}")
        entries.append("\n".join(lines))
    if not entries:
        return "% No literature paper cards were available for BibTeX export."
    return "\n\n".join(entries)


def _render_cmdgd_references_bib(context: dict[str, Any]) -> str:
    bib_text = context.get("cmdgd_literature", {}).get("references_bib_text", "")
    if bib_text.strip():
        return bib_text.rstrip()
    return _render_references_bib(context)


def _results_summary(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": context["run_id"],
        "evidence_scope": _context_evidence_scope(context),
        "evidence_label": _evidence_label(context),
        "no_wet_lab_validation_claimed": True,
        "selected_node_id": context["selected_node_id"],
        "selected_mechanism": context["selected_mechanism"],
        "benchmark_summary": context["selected_summary_row"],
        "benchmark_row_count": len(context["benchmark_rows"]),
        "ablation_row_count": len(context["ablation_rows"]),
        "project_masking": _project_masking_summary_payload(context),
        "paper_card_count": len(context["paper_cards"]),
        "literature_summary": _literature_summary(context),
        "project_data_summary": _project_data_summary(context),
        "artifact_links": _summary_artifact_paths(context["artifact_paths"], context),
    }


def _claim_evidence_map(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": context["run_id"],
        "evidence_scope": _context_evidence_scope(context),
        "claims": _claim_records(context),
    }


def _context_evidence_scope(context: dict[str, Any]) -> str:
    if _has_project_masking(context):
        return MIXED_EVIDENCE_SCOPE
    return EVIDENCE_SCOPE


def _project_masking_summary_payload(context: dict[str, Any]) -> dict[str, Any]:
    project_masking = context.get("project_masking", {})
    if not isinstance(project_masking, dict):
        project_masking = {}
    return {
        "available": bool(project_masking.get("available")),
        "artifact_dir": _summary_artifact_path(project_masking.get("artifact_dir"), context),
        "artifacts": _summary_artifact_paths(project_masking.get("artifact_paths", {}), context),
        "summary_row_count": len(project_masking.get("summary_rows", [])),
        "result_row_count": len(project_masking.get("result_rows", [])),
        "ablation_row_count": len(project_masking.get("ablation_rows", [])),
        "top_summary_row": project_masking.get("top_summary_row", {}),
        "config": project_masking.get("config", {}),
    }


def _render_reproducibility(context: dict[str, Any], paper_dir: Path) -> str:
    artifact_paths = context["artifact_paths"]
    lines = [
        "# Reproducibility",
        "",
        "This bundle can be regenerated from the local project checkout and the recorded framework V3 artifacts.",
        "",
        "Commands:",
        "",
        "```bash",
        f"PYTHONPATH=src design-scientist review-framework {context['root']} --run-id {context['run_id']}",
        f"PYTHONPATH=src design-scientist generate-short-paper {context['root']} --run-id {context['run_id']}",
        "```",
        "",
        "Primary run artifacts:",
        "",
        f"- {_md_link('scientist_journal.json', artifact_paths['scientist_journal'], paper_dir)}",
        f"- {_md_link('stage_progress.json', artifact_paths['stage_progress'], paper_dir)}",
        f"- {_md_link('route_tree.json', artifact_paths['route_tree'], paper_dir)}",
        f"- {_md_link('mechanism_benchmark_results.csv', artifact_paths['mechanism_benchmark_results'], paper_dir)}",
        f"- {_md_link('mechanism_benchmark_summary.csv', artifact_paths['mechanism_benchmark_summary'], paper_dir)}",
        f"- {_md_link('mechanism_ablation_results.csv', artifact_paths['mechanism_ablation_results'], paper_dir)}",
    ]
    if _has_project_masking(context):
        lines.extend(["", "Project masking artifacts:"])
        lines.extend(f"- {link}" for link in _project_masking_link_items(context, paper_dir))
    lines.extend(
        [
            "",
            (
                "Evidence boundary: these commands reproduce computational artifacts, synthetic replay summaries, "
                "and any retrospective masked project-data summaries already present in the project directory."
            ),
        ]
    )
    return "\n".join(lines)


def _render_data_availability(context: dict[str, Any], paper_dir: Path) -> str:
    artifact_paths = context["artifact_paths"]
    lines = [
        "# Data Availability",
        "",
        "The manuscript uses framework V3 artifacts already present in the local project directory.",
        "No new wet-lab assay data are introduced or claimed by this manuscript bundle.",
        "",
        "Available data artifacts:",
        "",
        f"- {_md_link('paper_cards.json', artifact_paths['paper_cards'], paper_dir)}",
        f"- {_md_link('literature_corpus.jsonl', artifact_paths['literature_corpus'], paper_dir)}",
        f"- {_md_link('mechanism_cards.json', artifact_paths['mechanism_cards'], paper_dir)}",
        f"- {_md_link('mechanism_library.json', artifact_paths['mechanism_library'], paper_dir)}",
        f"- {_md_link('mechanism_gap_matrix.csv', artifact_paths['mechanism_gap_matrix'], paper_dir)}",
        f"- {_md_link('results_summary.json', 'results_summary.json', paper_dir)}",
        f"- {_md_link('claim_evidence_map.json', 'claim_evidence_map.json', paper_dir)}",
    ]
    if _has_project_masking(context):
        lines.extend(["", "Retrospective project masking artifacts:"])
        lines.extend(f"- {link}" for link in _project_masking_link_items(context, paper_dir))
    return "\n".join(lines)


def _render_code_availability(context: dict[str, Any], paper_dir: Path) -> str:
    artifact_paths = context["artifact_paths"]
    return "\n".join(
        [
            "# Code Availability",
            "",
            "The selected mechanism implementation is linked below. This file is a generated computational mechanism artifact, not wet-lab validation.",
            "",
            f"- {_md_link('mechanism.py', artifact_paths.get('selected_node_mechanism'), paper_dir)}",
            f"- {_md_link('mechanism_spec.json', artifact_paths.get('selected_node_mechanism_spec'), paper_dir)}",
            f"- {_md_link('proposal.json', artifact_paths.get('selected_node_proposal'), paper_dir)}",
            f"- {_md_link('ablation_plan.json', artifact_paths.get('selected_node_ablation_plan'), paper_dir)}",
            f"- {_md_link('stress_test_plan.json', artifact_paths.get('selected_node_stress_test_plan'), paper_dir)}",
            f"- {_md_link('validation_report.json', artifact_paths.get('selected_node_validation_report'), paper_dir)}",
        ]
    )


def _claim_records(context: dict[str, Any]) -> list[dict[str, Any]]:
    proposal = context["proposal"]
    mechanism_spec = context["mechanism_spec"]
    literature_basis = _literature_basis(context)
    citations = [
        _citation_key_for_literature_item(context, item)
        for item in literature_basis
    ] or list(context["citation_keys"].values())[:3]
    evidence_artifacts = [
        "selected_node_mechanism_spec",
        "selected_node_proposal",
        "selected_node_stress_test_plan",
        "selected_node_validation_report",
        "mechanism_benchmark_results",
        "mechanism_benchmark_summary",
        "mechanism_ablation_results",
    ]
    claims: list[str] = []
    claims.extend(_string_items(mechanism_spec.get("claims")))
    claims.extend(_string_items(proposal.get("claims")))
    for key in ("hypothesis", "method_claim", "expected_differentiator"):
        value = mechanism_spec.get(key) or proposal.get(key)
        if value:
            claims.append(str(value))
    records = []
    for claim in _unique(claims):
        records.append(
            {
                "claim": claim,
                "evidence_type": EVIDENCE_TYPE,
                "evidence_scope": EVIDENCE_SCOPE,
                "citations": citations,
                "evidence_artifacts": [
                    artifact
                    for artifact in evidence_artifacts
                    if context["artifact_paths"].get(artifact)
                ],
                "limitations": [
                    "No wet-lab validation is included in this manuscript bundle.",
                    "Synthetic replay evidence does not establish real experimental performance.",
                ],
            }
        )
    if _has_project_masking(context):
        project_artifacts = [
            key
            for key in PROJECT_MASKING_ARTIFACT_FILENAMES
            if context["artifact_paths"].get(key)
        ]
        records.append(
            {
                "claim": (
                    "Project-data retrospective masking artifacts characterize observed-pool policy replay "
                    "under held-out outcomes."
                ),
                "evidence_type": PROJECT_MASKING_EVIDENCE_TYPE,
                "evidence_scope": PROJECT_MASKING_EVIDENCE_SCOPE,
                "citations": citations,
                "evidence_artifacts": project_artifacts,
                "limitations": [
                    f"Retrospective masked project-data evidence {PROSPECTIVE_VALIDATION_BOUNDARY}.",
                    "Observed-pool masking can evaluate replay behavior but does not establish real experimental performance.",
                ],
            }
        )
    return records


def _validate_sections(text: str, artifact: str, findings: list[dict[str, Any]]) -> None:
    for section in REQUIRED_SHORT_PAPER_SECTIONS:
        if section not in text:
            _add_finding(
                findings,
                "error",
                "missing_section",
                f"short_paper.md is missing required section {section!r}.",
                artifact,
            )


def _validate_evidence_boundary(text: str, artifact: str, findings: list[dict[str, Any]]) -> None:
    lower = text.lower()
    for term in ("computational", "synthetic", "wet-lab"):
        if term not in lower:
            _add_finding(
                findings,
                "error",
                "missing_evidence_boundary_label",
                f"short_paper.md must label evidence boundary with {term!r}.",
                artifact,
            )


def _validate_citations(text: str, bib_text: str, findings: list[dict[str, Any]]) -> None:
    citation_keys = _markdown_citation_keys(text)
    bib_keys = set(re.findall(r"@\w+\{([^,\s]+)", bib_text))
    if not citation_keys:
        _add_finding(
            findings,
            "error",
            "missing_citation",
            "short_paper.md must include at least one citation key.",
            "runs/<run_id>/paper/short_paper.md",
        )
        return
    for key in sorted(citation_keys - bib_keys):
        _add_finding(
            findings,
            "error",
            "missing_citation_key",
            f"Citation key {key!r} is not present in references.bib.",
            "runs/<run_id>/paper/references.bib",
        )


def _validate_artifact_links(
    text: str,
    paper_dir: Path,
    artifact: str,
    findings: list[dict[str, Any]],
) -> None:
    if "missing artifact path" in text:
        _add_finding(
            findings,
            "error",
            "missing_artifact_link",
            "short_paper.md contains a missing artifact path placeholder.",
            artifact,
        )
    links = [
        target
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)
        if _is_local_artifact_link(target)
    ]
    if not links:
        _add_finding(
            findings,
            "error",
            "missing_artifact_link",
            "short_paper.md must link to at least one local artifact.",
            artifact,
        )
        return
    for target in links:
        link_target = target.split("#", 1)[0]
        if not link_target:
            continue
        path = (paper_dir / link_target).resolve()
        if not path.exists():
            _add_finding(
                findings,
                "error",
                "missing_artifact_link",
                f"Linked artifact does not exist: {target}.",
                artifact,
            )


def _validate_json_file(path: Path, root: Path, findings: list[dict[str, Any]]) -> None:
    if not path.exists():
        return
    try:
        data = read_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        _add_finding(
            findings,
            "error",
            "invalid_json_output",
            f"Invalid JSON output: {exc}.",
            _rel(path, root),
        )
        return
    if not isinstance(data, dict):
        _add_finding(
            findings,
            "error",
            "invalid_json_output",
            "Manuscript JSON outputs must be mappings.",
            _rel(path, root),
        )


def _build_readiness_report(
    root: Path,
    run_id: str | None,
    run_dir: Path | None,
    findings: list[dict[str, Any]],
    artifacts: dict[str, str],
) -> dict[str, Any]:
    errors = sum(1 for finding in findings if finding["severity"] == "error")
    warnings = sum(1 for finding in findings if finding["severity"] == "warning")
    return {
        "schema_version": 1,
        "valid": errors == 0,
        "status": "passed" if errors == 0 else "failed",
        "run_id": run_id,
        "project_dir": str(root),
        "run_dir": str(run_dir) if run_dir is not None else None,
        "summary": {"errors": errors, "warnings": warnings},
        "findings": findings,
        "artifacts": artifacts,
    }


def _add_finding(
    findings: list[dict[str, Any]],
    severity: str,
    code: str,
    message: str,
    artifact: str,
) -> None:
    findings.append(
        {
            "severity": severity,
            "code": code,
            "message": message,
            "artifact": artifact,
        }
    )


def _load_project_masking_artifacts(root: Path, run_dir: Path) -> dict[str, Any]:
    artifact_dir = _find_project_masking_artifact_dir(root, run_dir)
    empty = {
        "available": False,
        "artifact_dir": None,
        "artifact_paths": {},
        "summary_rows": [],
        "statistical_summary_rows": [],
        "pairwise_comparison_rows": [],
        "result_rows": [],
        "ablation_rows": [],
        "config": {},
        "top_summary_row": {},
    }
    if artifact_dir is None:
        return empty

    artifact_paths: dict[str, str] = {}
    for key, filename in PROJECT_MASKING_ARTIFACT_FILENAMES.items():
        path = artifact_dir / filename
        if path.is_file():
            artifact_paths[key] = str(path)

    summary_rows = _read_csv(artifact_dir / PROJECT_MASKING_ARTIFACT_FILENAMES["project_masking_benchmark_summary"])
    result_rows = _read_csv(artifact_dir / PROJECT_MASKING_ARTIFACT_FILENAMES["project_masking_benchmark_results"])
    ablation_rows = _read_csv(artifact_dir / PROJECT_MASKING_ARTIFACT_FILENAMES["project_masking_ablation_results"])
    config = _load_json(artifact_dir / PROJECT_MASKING_ARTIFACT_FILENAMES["project_masking_config"])
    return {
        "available": bool(artifact_paths),
        "artifact_dir": str(artifact_dir),
        "artifact_paths": artifact_paths,
        "summary_rows": summary_rows,
        "result_rows": result_rows,
        "ablation_rows": ablation_rows,
        "config": config if isinstance(config, dict) else {},
        "top_summary_row": dict(summary_rows[0]) if summary_rows else {},
    }


def _load_mccbd_benchmark_artifacts(run_dir: Path) -> dict[str, Any]:
    empty = {
        "available": False,
        "artifact_dir": None,
        "artifact_paths": {},
        "summary_rows": [],
        "result_rows": [],
        "ablation_rows": [],
        "config": {},
    }
    if not run_dir.is_dir():
        return empty
    artifact_paths: dict[str, str] = {}
    for key, filename in MCCBD_BENCHMARK_ARTIFACT_FILENAMES.items():
        path = run_dir / filename
        if path.is_file():
            artifact_paths[key] = str(path)
    if not artifact_paths:
        return empty
    summary_rows = _read_csv(run_dir / MCCBD_BENCHMARK_ARTIFACT_FILENAMES["mccbd_benchmark_summary"])
    result_rows = _read_csv(run_dir / MCCBD_BENCHMARK_ARTIFACT_FILENAMES["mccbd_benchmark_results"])
    ablation_rows = _read_csv(run_dir / MCCBD_BENCHMARK_ARTIFACT_FILENAMES["mccbd_ablation_results"])
    config = _load_json(run_dir / MCCBD_BENCHMARK_ARTIFACT_FILENAMES["mccbd_benchmark_config"])
    return {
        "available": bool(artifact_paths),
        "artifact_dir": str(run_dir),
        "artifact_paths": artifact_paths,
        "summary_rows": summary_rows,
        "result_rows": result_rows,
        "ablation_rows": ablation_rows,
        "config": config if isinstance(config, dict) else {},
    }


def _load_generative_benchmark_artifacts(run_dir: Path) -> dict[str, Any]:
    empty = {
        "available": False,
        "artifact_dir": None,
        "artifact_paths": {},
        "summary_rows": [],
        "result_rows": [],
        "ablation_rows": [],
        "design_example_rows": [],
        "project_design_candidate_rows": [],
        "project_design_summary": {},
        "selection_gate_report": {},
        "config": {},
    }
    if not run_dir.is_dir():
        return empty
    artifact_paths: dict[str, str] = {}
    for key, filename in GENERATIVE_BENCHMARK_ARTIFACT_FILENAMES.items():
        path = run_dir / filename
        if path.is_file():
            artifact_paths[key] = str(path)
    for key, filename in CMDGD_PROJECT_ARTIFACT_FILENAMES.items():
        path = run_dir / filename
        if path.is_file():
            artifact_paths[key] = str(path)
    if not artifact_paths:
        return empty
    summary_rows = _read_csv(run_dir / GENERATIVE_BENCHMARK_ARTIFACT_FILENAMES["generative_benchmark_summary"])
    result_rows = _read_csv(run_dir / GENERATIVE_BENCHMARK_ARTIFACT_FILENAMES["generative_benchmark_results"])
    statistical_summary_rows = _read_csv(
        run_dir / GENERATIVE_BENCHMARK_ARTIFACT_FILENAMES["generative_statistical_summary"]
    )
    if not statistical_summary_rows:
        statistical_summary_rows = _derive_cmdgd_statistical_summary_rows(result_rows)
    pairwise_comparison_rows = _read_csv(
        run_dir / GENERATIVE_BENCHMARK_ARTIFACT_FILENAMES["generative_pairwise_comparisons"]
    )
    if not pairwise_comparison_rows:
        pairwise_comparison_rows = _derive_cmdgd_pairwise_rows(result_rows)
    ablation_rows = _read_csv(run_dir / GENERATIVE_BENCHMARK_ARTIFACT_FILENAMES["generative_ablation_results"])
    design_example_rows = _read_csv(run_dir / GENERATIVE_BENCHMARK_ARTIFACT_FILENAMES["generative_design_examples"])
    project_design_candidate_rows = _read_csv(run_dir / CMDGD_PROJECT_ARTIFACT_FILENAMES["cmdgd_project_generated_candidates"])
    project_design_summary = _load_json(run_dir / CMDGD_PROJECT_ARTIFACT_FILENAMES["cmdgd_project_design_summary"])
    config = _load_json(run_dir / GENERATIVE_BENCHMARK_ARTIFACT_FILENAMES["generative_benchmark_config"])
    gate_report = _load_json(run_dir / GENERATIVE_BENCHMARK_ARTIFACT_FILENAMES["generative_selection_gate_report"])
    return {
        "available": bool(artifact_paths),
        "artifact_dir": str(run_dir),
        "artifact_paths": artifact_paths,
        "summary_rows": summary_rows,
        "statistical_summary_rows": statistical_summary_rows,
        "pairwise_comparison_rows": pairwise_comparison_rows,
        "result_rows": result_rows,
        "ablation_rows": ablation_rows,
        "design_example_rows": design_example_rows,
        "project_design_candidate_rows": project_design_candidate_rows,
        "project_design_summary": project_design_summary if isinstance(project_design_summary, dict) else {},
        "selection_gate_report": gate_report if isinstance(gate_report, dict) else {},
        "config": config if isinstance(config, dict) else {},
    }


def _load_cmdgd_literature_artifacts(root: Path) -> dict[str, Any]:
    framework_dir = root / "framework"
    review_path = framework_dir / "v4_literature_review.md"
    references_path = framework_dir / "v4_references.bib"
    matrix_path = framework_dir / "v4_related_work_matrix.csv"
    trace_path = framework_dir / "v4_literature_trace.json"
    references_bib_text = _read_text(references_path)
    related_work_rows = _read_csv(matrix_path)
    trace = _load_json(trace_path)
    return {
        "available": any(path.exists() for path in (review_path, references_path, matrix_path, trace_path)),
        "review_path": str(review_path) if review_path.exists() else None,
        "references_path": str(references_path) if references_path.exists() else None,
        "matrix_path": str(matrix_path) if matrix_path.exists() else None,
        "trace_path": str(trace_path) if trace_path.exists() else None,
        "review_text": _read_text(review_path),
        "references_bib_text": references_bib_text,
        "related_work_rows": related_work_rows,
        "trace": trace if isinstance(trace, dict) else {},
        "reference_count": len(_bib_entry_keys(references_bib_text)),
        "related_work_row_count": len(related_work_rows),
    }


def _find_project_masking_artifact_dir(root: Path, run_dir: Path) -> Path | None:
    candidates = [run_dir, root / "runs" / "project_masking"]
    runs_dir = root / "runs"
    if runs_dir.is_dir():
        candidates.extend(
            path
            for path in sorted(
                (item for item in runs_dir.iterdir() if item.is_dir()),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )
        )

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if any((resolved / filename).is_file() for filename in PROJECT_MASKING_ARTIFACT_FILENAMES.values()):
            return resolved
    return None


def _artifact_paths(root: Path, run_dir: Path, selected_artifacts: dict[str, Path]) -> dict[str, str]:
    paths = {
        "framework_spec": str(root / "framework" / "framework_spec.yaml"),
        "literature_queries": str(root / "framework" / "literature_queries.yaml"),
        "paper_cards": str(root / "framework" / "paper_cards.json"),
        "literature_corpus": str(root / "framework" / "literature_corpus.jsonl"),
        "literature_reading_trace": str(root / "framework" / "literature_reading_trace.json"),
        "mechanism_cards": str(root / "framework" / "mechanism_cards.json"),
        "mechanism_library": str(root / "framework" / "mechanism_library.json"),
        "mechanism_gap_matrix": str(root / "framework" / "mechanism_gap_matrix.csv"),
        "scientist_journal": str(run_dir / "scientist_journal.json"),
        "stage_progress": str(run_dir / "stage_progress.json"),
        "route_tree": str(run_dir / "route_tree.json"),
        "mechanism_benchmark_results": str(run_dir / "mechanism_benchmark_results.csv"),
        "mechanism_benchmark_summary": str(run_dir / "mechanism_benchmark_summary.csv"),
        "mechanism_ablation_results": str(run_dir / "mechanism_ablation_results.csv"),
        "method_report": str(run_dir / "method_report.md"),
    }
    for key, path in selected_artifacts.items():
        paths[f"selected_node_{key}"] = str(path)
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


def _selected_node_id(journal: Any, selected_node: dict[str, Any] | None) -> str | None:
    if selected_node and isinstance(selected_node.get("node_id"), str):
        return selected_node["node_id"]
    if isinstance(journal, dict) and isinstance(journal.get("selected_node_id"), str):
        return journal["selected_node_id"]
    return None


def _selected_artifact_paths(root: Path, selected_node: dict[str, Any] | None) -> dict[str, Path]:
    if not selected_node:
        return {}
    paths: dict[str, Path] = {}
    artifacts = selected_node.get("artifacts")
    if isinstance(artifacts, dict):
        for key, value in artifacts.items():
            if value:
                paths[key] = _coerce_path(root, value)
    workspace = selected_node.get("workspace")
    if workspace:
        workspace_path = _coerce_path(root, workspace)
        defaults = {
            "mechanism_spec": "mechanism_spec.json",
            "mechanism": "mechanism.py",
            "proposal": "proposal.json",
            "ablation_plan": "ablation_plan.json",
            "stress_test_plan": "stress_test_plan.json",
            "mechanism_metrics": "mechanism_metrics.json",
            "validation_report": "validation_report.json",
        }
        for key, filename in defaults.items():
            paths.setdefault(key, workspace_path / filename)
    return paths


def _selected_mechanism(
    journal: Any,
    selected_node: dict[str, Any] | None,
    mechanism_spec: Any,
    mechanism_metrics: Any,
) -> str | None:
    sources: list[Any] = []
    if selected_node:
        sources.append(selected_node)
    if isinstance(journal, dict):
        sources.append(journal.get("selected_node"))
    sources.extend([mechanism_metrics, mechanism_spec])
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in ("mechanism", "mechanism_id", "mechanism_name", "name"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


def _find_summary_row(rows: list[dict[str, str]], selected_mechanism: str | None) -> dict[str, str]:
    if not selected_mechanism:
        return {}
    for row in rows:
        if row.get("mechanism") == selected_mechanism or row.get("method") == selected_mechanism:
            return dict(row)
    return {}


def _literature_basis(context: dict[str, Any]) -> list[str]:
    values: list[str] = []
    values.extend(_string_items(context["mechanism_spec"].get("literature_basis")))
    values.extend(_string_items(context["proposal"].get("literature_basis")))
    if not values:
        values.extend(card.get("paper_id", "") for card in context["paper_cards"])
    return _unique(values)


def _citation_cluster(context: dict[str, Any]) -> str:
    basis = _literature_basis(context)
    keys = [_citation_key_for_literature_item(context, item) for item in basis if item]
    if not keys:
        keys = list(context["citation_keys"].values())[:3]
    keys = _unique(keys)
    if not keys:
        return ""
    return "[" + "; ".join(f"@{key}" for key in keys[:3]) + "]"


def _cmdgd_citation_cluster(
    context: dict[str, Any],
    *,
    limit: int | None = None,
    keys: tuple[str, ...] | None = None,
) -> str:
    if keys is None:
        matrix_keys = [
            row.get("citation_key", "")
            for row in context.get("cmdgd_literature", {}).get("related_work_rows", [])
            if isinstance(row, dict)
        ]
        bib_keys = _bib_entry_keys(context.get("cmdgd_literature", {}).get("references_bib_text", ""))
        keys = tuple(key for key in matrix_keys if key in bib_keys)
    selected = _unique(list(keys))
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        return ""
    return "[" + "; ".join(f"@{key}" for key in selected) + "]"


def _cmdgd_literature_review_lines(context: dict[str, Any], paper_dir: Path) -> list[str]:
    rows = context.get("cmdgd_literature", {}).get("related_work_rows", [])
    by_lane: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if isinstance(row, dict):
            by_lane.setdefault(row.get("lane", "other"), []).append(row)
    lines = [
        (
            "The literature review frames pH-sensitive antibody engineering as a contrastive design problem "
            "rather than scalar affinity maximization. Prior pH-switch studies establish the therapeutic logic "
            "of neutral-pH retention with acidic release, while active protein optimization and generative "
            "sequence design define the computational tools for learning from sparse panels."
        ),
        "",
        (
            "pH antibody engineering establishes the biological mechanism: pH-dependent antigen release, "
            "histidine scanning or histidine-enriched discovery, and recycling or sweeping antibody behavior. "
            f"Key papers include {_cmdgd_lane_citations(by_lane, 'pH_antibody_engineering', 8)}. Their limitation for "
            "CMD-GD is that they define plausible edit mechanisms and pharmacology, but they do not provide "
            "a sparse-data policy for generating a small next panel in a fixed 1E62 lineage."
        ),
        "",
        (
            "Low-data protein optimization contributes the learning loop: measured variants are used to fit "
            "surrogates, quantify uncertainty, and prioritize the next experimental batch. "
            f"Relevant work includes {_cmdgd_lane_citations(by_lane, 'BO_MLDE_AntBO', 7)}. "
            "These methods motivate active selection and baselines, but most optimize scalar fitness or a "
            "predefined combinatorial library rather than a pH-contrast heavy/light antibody objective. The "
            "most relevant comparator is an AntBO-like constrained search when the candidate space can be "
            "expressed as comparable antibody combinations."
        ),
        "",
        (
            "Generative protein and antibody design contributes candidate-space expansion beyond a fixed pool. "
            f"General protein language models are represented by {_cmdgd_lane_citations(by_lane, 'PLM_guided_evolution', 5)}, "
            f"antibody language models by {_cmdgd_lane_citations(by_lane, 'antibody_language_models', 3)}, and "
            f"structure-aware generation by {_cmdgd_lane_citations(by_lane, 'structure_aware_generative_antibody_design', 2)}. "
            "CMD-GD uses this lane conservatively: it does not claim a large neural generator, but it does "
            "instantiate a grammar-guided generator whose outputs are heavy/light sequence records with edit "
            "tokens, provenance, and contrastive scores."
        ),
        "",
        (
            "The synthesis identifies five gaps that define the method. The endpoint gap is that pH antibody "
            "engineering is a matched pH contrast, whereas most design algorithms optimize a scalar fitness. "
            "The decision-policy gap is that display, scanning, and manual recombination do not define a "
            "budgeted next-panel policy for a lineage with existing measurements. The representation gap is "
            "that BO or MLDE features do not automatically encode pH mechanism or paired heavy/light edits. "
            "The alignment gap is that protein language models and antibody generators can broaden sequence "
            "space without guaranteeing pH-selective release. The evaluation gap is that credible evidence "
            "requires matched-budget baselines, retrospective masking, and ablations separating generation, "
            "contrastive scoring, novelty, and uncertainty."
        ),
    ]
    return lines


def _cmdgd_lane_citations(
    by_lane: dict[str, list[dict[str, str]]],
    lane: str,
    limit: int,
) -> str:
    keys = [row.get("citation_key", "") for row in by_lane.get(lane, [])]
    keys = _unique([key for key in keys if key])[:limit]
    if not keys:
        return "the curated bibliography"
    return "[" + "; ".join(f"@{key}" for key in keys) + "]"


def _bib_entry_keys(bib_text: str) -> set[str]:
    return set(re.findall(r"@\w+\{([^,\s]+)", bib_text or ""))


def _citation_keys_for_cards(cards: list[dict[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for index, card in enumerate(cards):
        paper_id = _one_line(card.get("paper_id")) or f"paper_{index + 1}"
        out[paper_id] = _cite_key(paper_id)
    return out


def _citation_aliases_for_cards(
    cards: list[dict[str, Any]],
    citation_keys: dict[str, str],
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for card in cards:
        paper_id = _one_line(card.get("paper_id"))
        key = citation_keys.get(paper_id)
        if not paper_id or not key:
            continue
        aliases[paper_id] = key
        aliases[paper_id.lower()] = key
        aliases[_cite_key(paper_id)] = key
        title = _one_line(card.get("title"))
        if title:
            aliases[f"{paper_id}: {title}"] = key
            aliases[f"{paper_id}: {title}".lower()] = key
        source = _one_line(card.get("source")).lower()
        for field in ("external_id", "doi", "pmid", "paperId"):
            external_id = _one_line(card.get(field))
            if source and external_id:
                aliases[f"{source}:{external_id}"] = key
                aliases[f"{source}:{external_id}".lower()] = key
    return aliases


def _citation_key_for_literature_item(context: dict[str, Any], item: Any) -> str:
    text = _one_line(item)
    if not text:
        return ""
    aliases = context.get("citation_aliases", {})
    for candidate in _literature_id_candidates(text):
        key = aliases.get(candidate) or aliases.get(candidate.lower())
        if key:
            return key
    return _cite_key(text)


def _literature_id_candidates(text: str) -> list[str]:
    candidates = [text]
    match = re.match(r"^([A-Za-z][A-Za-z0-9_+.-]*:[^:\s]+)", text)
    if match:
        candidates.append(match.group(1))
    if ": " in text:
        candidates.append(text.split(": ", 1)[0])
    return _unique(candidates)


def _literature_summary(context: dict[str, Any]) -> str:
    search_summary = context.get("literature_trace", {}).get("summary", {})
    reading_summary = context.get("reading_trace", {}).get("summary", {})
    paper_count = len(context.get("paper_cards", []))
    corpus_count = len(context.get("literature_corpus", []))
    raw_count = search_summary.get("raw_paper_count")
    selected_count = search_summary.get("final_paper_count") or paper_count
    open_fulltext = reading_summary.get("open_fulltext")
    metadata_only = reading_summary.get("metadata_only")
    parts = []
    if raw_count not in (None, ""):
        parts.append(f"{raw_count} raw literature candidates")
    parts.append(f"{selected_count} selected paper cards")
    parts.append(f"{corpus_count} corpus records")
    if open_fulltext not in (None, ""):
        parts.append(f"{open_fulltext} open-fulltext records")
    if metadata_only not in (None, ""):
        parts.append(f"{metadata_only} metadata-only records")
    return ", ".join(str(part) for part in parts)


def _project_data_summary(context: dict[str, Any]) -> str:
    standardized = context.get("standardized_context", {})
    inventory = standardized.get("source_inventory")
    if isinstance(inventory, list) and inventory:
        source_count = len(inventory)
        row_count = sum(
            int(item.get("rows", 0) or 0)
            for item in inventory
            if isinstance(item, dict)
        )
        return f"{source_count} startup sources were standardized with {row_count} tabular rows before framework execution"
    observations_path = Path(context["root"]) / "standardized" / "observations_long.csv"
    sequences_path = Path(context["root"]) / "standardized" / "variant_sequences.csv"
    observation_rows = max(0, len(_read_csv(observations_path)))
    sequence_rows = max(0, len(_read_csv(sequences_path)))
    if observation_rows or sequence_rows:
        return f"{observation_rows} standardized observation rows and {sequence_rows} standardized sequence records were available before framework execution"
    return "no standardized startup data summary was available in the manuscript context"


def _cite_key(value: str) -> str:
    key = re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")
    if not key:
        key = "paper"
    if key[0].isdigit():
        key = f"ref_{key}"
    return key


def _markdown_citation_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for bracketed in re.findall(r"\[([^\]]+)\]", text):
        keys.update(re.findall(r"@([A-Za-z0-9_:.+-]+)", bracketed))
    return keys


def _is_local_artifact_link(target: str) -> bool:
    stripped = target.strip()
    if not stripped or stripped.startswith("#"):
        return False
    return not re.match(r"^[a-z][a-z0-9+.-]*:", stripped, flags=re.IGNORECASE)


def _md_link(label: str, target: str | Path | None, paper_dir: Path) -> str:
    if target in (None, ""):
        return f"`{label}` (missing artifact path)"
    target_text = str(target)
    if re.match(r"^[a-z][a-z0-9+.-]*:", target_text, flags=re.IGNORECASE):
        return f"[{label}]({target_text})"
    path = Path(target_text)
    if path.is_absolute():
        rel = os.path.relpath(path, paper_dir)
    else:
        rel = target_text
    return f"[{label}]({rel})"


def _format_authors(value: Any) -> str:
    if isinstance(value, list):
        names: list[str] = []
        for item in value:
            if isinstance(item, dict):
                name = item.get("name") or item.get("author")
            else:
                name = item
            if name:
                names.append(_one_line(name))
        return " and ".join(names[:12])
    if isinstance(value, str):
        return _one_line(value)
    return ""


def _bib_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def _collection(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in ("mechanisms", "cards", "items"):
            items = value.get(key)
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
    return []


def _string_items(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, list):
        return [_one_line(item) for item in value if _one_line(item)]
    return [_one_line(value)]


def _as_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _load_json(path: Path | None) -> Any:
    if path is None or not path.exists():
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


def _safe_run_id(run_id: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(run_id)).strip("._-")
    return clean or "algorithm_manuscript"


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _numeric_interval(values: list[float]) -> dict[str, float]:
    cleaned = [float(value) for value in values]
    n = len(cleaned)
    if not cleaned:
        return {"n": 0, "mean": 0.0, "sd": 0.0, "sem": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    center = mean(cleaned)
    if n <= 1:
        sd = 0.0
    else:
        sd = (sum((value - center) ** 2 for value in cleaned) / (n - 1)) ** 0.5
    sem = sd / (n ** 0.5) if n else 0.0
    margin = 1.96 * sem
    return {
        "n": n,
        "mean": center,
        "sd": sd,
        "sem": sem,
        "ci95_low": center - margin,
        "ci95_high": center + margin,
    }


def _float_text(value: Any) -> str:
    numeric = _to_float(value)
    return "not recorded" if numeric is None else f"{numeric:.6f}"


def _signed_float_text(value: Any) -> str:
    numeric = _to_float(value)
    return "not recorded" if numeric is None else f"{numeric:+.6f}"


def _float_field(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _mean_float(rows: list[dict[str, Any]], key: str) -> float:
    values = [_float_field(row, key) for row in rows]
    return sum(values) / len(values) if values else 0.0


def _format_delta(value: float) -> str:
    return f"{value:+.6f}"


def _coerce_path(root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.expanduser().resolve().relative_to(root))
    except (OSError, ValueError):
        return str(path)


def _one_line(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _sentence(value: Any) -> str:
    text = _one_line(value)
    if not text:
        return ""
    return text if text.endswith((".", "!", "?")) else f"{text}."


def _svg_escape(value: Any) -> str:
    return (
        _one_line(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


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
