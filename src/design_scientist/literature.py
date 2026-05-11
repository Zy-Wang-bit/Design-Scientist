"""Local literature-scaffold generation for method development.

This module intentionally does not perform network search. It creates seed paper
cards and a gap matrix that can be reviewed and replaced by real literature
cards later.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from design_scientist.io import ensure_dir, read_yaml, write_json


METHOD_FAMILIES = [
    {
        "citation": "Seed area: active learning for protein engineering",
        "problem": "Choose informative and high-value variants across repeated wet-lab rounds.",
        "data_regime": "small, noisy, batch-limited biological measurements",
        "design_loop": "design-build-test-learn",
        "algorithm": "surrogate model plus acquisition policy",
        "acquisition_or_policy": "expected improvement, uncertainty sampling, constrained batch selection",
        "constraints": "budget, assay feasibility, sequence feasibility, QC risk",
        "evaluation": "retrospective masking and prospective hit-rate tracking",
        "baselines": ["random feasible", "top observed", "uncertainty sampling"],
        "what_to_reuse": "batch-aware acquisition and retrospective replay",
        "what_not_to_copy": "large-data assumptions and single-objective benchmark focus",
    },
    {
        "citation": "Seed area: Bayesian optimization for biological sequence design",
        "problem": "Optimize sequence variants under sparse observations and uncertainty.",
        "data_regime": "small to medium sequence-function datasets",
        "design_loop": "iterative model update and batch proposal",
        "algorithm": "Bayesian surrogate with constrained acquisition",
        "acquisition_or_policy": "multi-objective utility with feasibility penalties",
        "constraints": "wet-lab budget, developability, measurement noise",
        "evaluation": "regret, best-found utility, hit rate",
        "baselines": ["random", "greedy top-score", "surrogate-only"],
        "what_to_reuse": "uncertainty-aware scoring and constraints",
        "what_not_to_copy": "assuming smooth global design landscapes without matched evidence",
    },
    {
        "citation": "Seed area: adaptive experimental design and optimal design",
        "problem": "Allocate experiments to maximize decision value.",
        "data_regime": "small experimental panels with structured comparisons",
        "design_loop": "state update followed by next-action planning",
        "algorithm": "decision-value acquisition and adaptive allocation",
        "acquisition_or_policy": "state-dependent champion/contrast/control allocation",
        "constraints": "assay capacity, required controls, interpretability needs",
        "evaluation": "decision accuracy and uncertainty reduction",
        "baselines": ["fixed allocation", "pure exploration", "pure exploitation"],
        "what_to_reuse": "adaptive allocation rather than fixed ratios",
        "what_not_to_copy": "abstract optimality criteria that ignore wet-lab feasibility",
    },
    {
        "citation": "Seed area: matched contrasts and causal design in small datasets",
        "problem": "Avoid false module claims from confounded variant combinations.",
        "data_regime": "small structured variant panels",
        "design_loop": "repair missing contrasts and interaction squares",
        "algorithm": "contrast lattice and evidence-tier tracking",
        "acquisition_or_policy": "lattice repair value plus champion-centered contrasts",
        "constraints": "exact-match availability and locked mutation blocks",
        "evaluation": "false-positive advancement rate and matched-edge creation",
        "baselines": ["top ratio", "descriptive ranking", "pure lattice repair"],
        "what_to_reuse": "explicit evidence tiers and matched-edge design",
        "what_not_to_copy": "claiming causal module effects from descriptive comparisons",
    },
    {
        "citation": "Seed area: antibody engineering and pH-dependent binding design",
        "problem": "Improve conditional binding while preserving neutral binding and coverage.",
        "data_regime": "antibody variant panels with pH and antigen/genotype endpoints",
        "design_loop": "module proposal, target-system validation, coverage guardrails",
        "algorithm": "mechanism-aware active design with QC guardrails",
        "acquisition_or_policy": "performance-first score with mechanism and coverage terms",
        "constraints": "neutral retention, acidic release, genotype coverage, expression/QC",
        "evaluation": "best feasible variant, coverage retention, validation contrasts",
        "baselines": ["top pH ratio", "top neutral retention", "human heuristic"],
        "what_to_reuse": "domain constraints and measurement requirements",
        "what_not_to_copy": "treating all antibody systems as directly transferable",
    },
]


def create_seed_paper_cards(project_dir: str | Path) -> Path:
    root = Path(project_dir).expanduser().resolve()
    framework_dir = ensure_dir(root / "framework")
    project = read_yaml(root / "project.yaml") if (root / "project.yaml").exists() else {}
    cards: list[dict[str, Any]] = []
    for idx, family in enumerate(METHOD_FAMILIES, start=1):
        card = dict(family)
        card["paper_id"] = f"seed_method_area_{idx}"
        card["relevance_to_current_project"] = project.get("goal", "general iterative variant design")
        card["review_status"] = "seed_unverified"
        cards.append(card)
    out = framework_dir / "paper_cards.json"
    write_json(out, cards)
    print(f"Wrote seed paper cards: {out}")
    return out


def build_research_gap_matrix(project_dir: str | Path) -> Path:
    root = Path(project_dir).expanduser().resolve()
    framework_dir = ensure_dir(root / "framework")
    out = framework_dir / "research_gap_matrix.csv"
    rows = [
        {
            "method_family": "active_learning_protein_engineering",
            "handles_batches": "yes",
            "handles_constraints": "partial",
            "handles_multi_objective": "partial",
            "works_with_tiny_data": "partial",
            "supports_interpretability": "low",
            "supports_transfer": "partial",
            "likely_baseline": "random feasible; top observed",
            "gap_for_our_task": "Needs explicit evidence tiers and wet-lab decision value.",
        },
        {
            "method_family": "bayesian_optimization_sequence_design",
            "handles_batches": "yes",
            "handles_constraints": "yes",
            "handles_multi_objective": "yes",
            "works_with_tiny_data": "partial",
            "supports_interpretability": "medium",
            "supports_transfer": "partial",
            "likely_baseline": "surrogate-only expected utility",
            "gap_for_our_task": "Must avoid over-trusting smooth surrogate predictions.",
        },
        {
            "method_family": "matched_contrast_lattice",
            "handles_batches": "yes",
            "handles_constraints": "yes",
            "handles_multi_objective": "partial",
            "works_with_tiny_data": "yes",
            "supports_interpretability": "high",
            "supports_transfer": "medium",
            "likely_baseline": "pure lattice repair",
            "gap_for_our_task": "Needs performance-first coupling, not pure interpretability.",
        },
        {
            "method_family": "adaptive_batch_allocation",
            "handles_batches": "yes",
            "handles_constraints": "yes",
            "handles_multi_objective": "yes",
            "works_with_tiny_data": "yes",
            "supports_interpretability": "medium",
            "supports_transfer": "medium",
            "likely_baseline": "fixed allocation policy",
            "gap_for_our_task": "Policy weights must be evaluated and revised over rounds.",
        },
    ]
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote research gap matrix: {out}")
    return out

