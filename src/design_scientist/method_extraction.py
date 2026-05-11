"""Extract method modules from literature paper cards.

The extractor is intentionally deterministic and offline. It consumes the
normalized ``framework/paper_cards.json`` artifact written by the literature
pipeline, groups source papers into reusable method modules, and writes the
framework artifacts needed for downstream method development.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

from design_scientist.io import ensure_dir, read_json, write_json, write_yaml


REQUIRED_POLICY_NAMES = (
    "random_feasible",
    "top_observed",
    "greedy_utility",
    "fixed_mix",
    "pure_uncertainty",
    "pure_lattice_repair",
    "mechanism_aware",
    "codex_generated",
)


METHOD_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "module_id": "active_learning_batch_design",
        "name": "Batch Active Learning for Variant Design",
        "keywords": (
            "active learning",
            "batch active",
            "batch",
            "design-build-test",
            "design build test",
            "protein engineering",
            "sequence design",
            "variant design",
            "surrogate model",
            "surrogate models",
            "acquisition function",
            "acquisition functions",
            "expected improvement",
        ),
        "min_score": 4,
        "problem_setting": (
            "Choose wet-lab variants across repeated design-build-test-learn "
            "rounds while each batch must balance high-value candidates with "
            "information gain."
        ),
        "data_regime": (
            "Small to medium noisy sequence-function measurements with batch "
            "limits, sparse early rounds, and accumulating historical panels."
        ),
        "algorithm_family": "surrogate-assisted active learning / Bayesian optimization",
        "acquisition_policy_mechanism": (
            "Fit or update a surrogate from observed variants, score feasible "
            "candidates by expected utility or improvement plus uncertainty and "
            "coverage terms, then select a constrained batch."
        ),
        "baselines": ("random_feasible", "top_observed", "greedy_utility", "pure_uncertainty"),
        "evaluation_protocol": (
            "Retrospective masking or replay by round, plus prospective best "
            "feasible hit rate, best-observed utility, regret, and calibration checks."
        ),
        "failure_modes": (
            "Surrogate overconfidence in sparse or shifted regions.",
            "Batch diversity terms can displace true high performers.",
            "Noisy measurements can make top-observed labels unstable.",
        ),
        "reusable_ideas": (
            "Round-aware design-state update loop.",
            "Batch acquisition with feasibility filters.",
            "Retrospective policy replay before prospective use.",
        ),
        "gap_for_design_scientist": (
            "Needs explicit evidence tiers, mechanistic contrast checks, and wet-lab "
            "guardrails before source-level active learning is sufficient."
        ),
        "capabilities": {
            "handles_batches": "yes",
            "handles_constraints": "partial",
            "handles_multi_objective": "partial",
            "works_with_tiny_data": "partial",
            "supports_mechanistic_evidence": "low",
            "supports_transfer": "partial",
        },
    },
    {
        "module_id": "bayesian_constrained_optimization",
        "name": "Constrained Bayesian Optimization",
        "keywords": (
            "bayesian optimization",
            "bayesian surrogate",
            "gaussian process",
            "expected improvement",
            "constrained optimization",
            "multi-objective",
            "multi objective",
            "surrogate-only",
            "surrogate only",
            "regret",
        ),
        "min_score": 3,
        "problem_setting": (
            "Optimize sequence or variant utility under sparse observations, "
            "uncertainty, and explicit feasibility constraints."
        ),
        "data_regime": (
            "Small to medium sequence-function datasets where the smoothness and "
            "transfer assumptions of the surrogate must be checked."
        ),
        "algorithm_family": "Bayesian optimization with constrained acquisition",
        "acquisition_policy_mechanism": (
            "Use posterior mean and uncertainty to compute expected improvement "
            "or constrained multi-objective utility, penalizing infeasible or risky "
            "candidates."
        ),
        "baselines": ("random_feasible", "top_observed", "greedy_utility", "pure_uncertainty"),
        "evaluation_protocol": (
            "Measure best-found feasible utility, regret against held-out or later "
            "round observations, hit rate above threshold, and uncertainty calibration."
        ),
        "failure_modes": (
            "Smooth surrogate assumptions can fail on epistatic sequence landscapes.",
            "Uncertainty may be miscalibrated outside observed neighborhoods.",
            "Optimizing scalar utility can hide guardrail failures.",
        ),
        "reusable_ideas": (
            "Uncertainty-aware expected utility.",
            "Explicit feasibility penalties.",
            "Ablations separating model score from acquisition policy.",
        ),
        "gap_for_design_scientist": (
            "Must be coupled to evidence-tier labels and module-level contrasts so "
            "model-derived claims are not treated as direct wet-lab evidence."
        ),
        "capabilities": {
            "handles_batches": "yes",
            "handles_constraints": "yes",
            "handles_multi_objective": "yes",
            "works_with_tiny_data": "partial",
            "supports_mechanistic_evidence": "medium",
            "supports_transfer": "partial",
        },
    },
    {
        "module_id": "adaptive_decision_value_allocation",
        "name": "Adaptive Decision-Value Allocation",
        "keywords": (
            "decision value",
            "decision-value",
            "adaptive allocation",
            "optimal design",
            "state-dependent",
            "state dependent",
            "fixed allocation",
            "control allocation",
            "champion",
            "wet lab design",
            "wet-lab design",
        ),
        "min_score": 3,
        "problem_setting": (
            "Allocate a limited experimental batch to actions that improve the next "
            "scientific or design decision, not only the next model fit."
        ),
        "data_regime": (
            "Small experimental panels where the value of champion expansion, "
            "contrasts, interactions, controls, and repeats changes across rounds."
        ),
        "algorithm_family": "adaptive experimental design / decision-value policy",
        "acquisition_policy_mechanism": (
            "Estimate the marginal decision value of candidate categories and adapt "
            "batch allocation across champion, contrast, lattice repair, exploration, "
            "and control slots."
        ),
        "baselines": ("fixed_mix", "random_feasible", "top_observed", "pure_uncertainty"),
        "evaluation_protocol": (
            "Compare round-level decision accuracy, best feasible variant selected, "
            "uncertainty reduction, and allocation ablations against fixed mixes."
        ),
        "failure_modes": (
            "Decision-value terms can become subjective without a fixed objective.",
            "Adaptive allocation can overreact to noisy early evidence.",
            "Control and repeat slots can be squeezed out by short-term utility.",
        ),
        "reusable_ideas": (
            "State-dependent batch allocation.",
            "Champion-centered decision value.",
            "Explicit fixed-mix and component-removal ablations.",
        ),
        "gap_for_design_scientist": (
            "Needs a stable objective and replay protocol so policy-weight changes "
            "are auditable rather than anecdotal."
        ),
        "capabilities": {
            "handles_batches": "yes",
            "handles_constraints": "yes",
            "handles_multi_objective": "yes",
            "works_with_tiny_data": "yes",
            "supports_mechanistic_evidence": "medium",
            "supports_transfer": "medium",
        },
    },
    {
        "module_id": "matched_contrast_lattice",
        "name": "Matched Contrast Lattice",
        "keywords": (
            "matched contrast",
            "matched contrasts",
            "contrast lattice",
            "lattice repair",
            "confounding",
            "causal",
            "interaction square",
            "interaction squares",
            "evidence tier",
            "evidence-tier",
            "matched-edge",
            "matched edge",
            "descriptive ranking",
        ),
        "min_score": 3,
        "problem_setting": (
            "Avoid false module claims when small variant panels contain confounded "
            "mutation combinations and missing matched comparisons."
        ),
        "data_regime": (
            "Small structured variant panels with repeated backgrounds, module "
            "combinations, missing contrast edges, and limited exact matches."
        ),
        "algorithm_family": "contrast-lattice evidence tracking and repair",
        "acquisition_policy_mechanism": (
            "Identify missing matched edges or interaction squares around champions "
            "and high-value modules, then allocate candidates that repair the lattice "
            "while preserving performance priority."
        ),
        "baselines": ("top_observed", "pure_lattice_repair", "fixed_mix", "random_feasible"),
        "evaluation_protocol": (
            "Track matched-edge creation, false-positive module advancement, "
            "unsupported-claim reduction, and best feasible variant retention."
        ),
        "failure_modes": (
            "Pure lattice repair can spend too much budget on interpretability.",
            "Near-matched contrasts can be mistaken for primary matched evidence.",
            "Sparse lattices may not support interaction claims in one round.",
        ),
        "reusable_ideas": (
            "Evidence tiers separating primary matched, near-matched, descriptive, "
            "and model-derived claims.",
            "Champion-centered contrast completion.",
            "Interaction-square repair as a candidate operator.",
        ),
        "gap_for_design_scientist": (
            "Needs coupling to performance-first acquisition so interpretability work "
            "does not dominate the panel."
        ),
        "capabilities": {
            "handles_batches": "yes",
            "handles_constraints": "yes",
            "handles_multi_objective": "partial",
            "works_with_tiny_data": "yes",
            "supports_mechanistic_evidence": "high",
            "supports_transfer": "medium",
        },
    },
    {
        "module_id": "mechanism_aware_antibody_design",
        "name": "Mechanism-Aware Antibody Design",
        "keywords": (
            "antibody",
            "antibodies",
            "ph-dependent",
            "ph dependent",
            "neutral binding",
            "acidic release",
            "genotype",
            "coverage",
            "developability",
            "antigen",
            "mechanism-aware",
            "mechanism aware",
        ),
        "min_score": 3,
        "problem_setting": (
            "Improve conditional antibody behavior while preserving neutral binding, "
            "target coverage, expression, and other wet-lab feasibility guardrails."
        ),
        "data_regime": (
            "Antibody or variant panels with endpoint-specific measurements such as "
            "neutral binding, acidic release, genotype coverage, and QC outcomes."
        ),
        "algorithm_family": "mechanism-aware active design with guardrail constraints",
        "acquisition_policy_mechanism": (
            "Prioritize expected feasible performance, then use mechanistic contrasts, "
            "coverage checks, and QC guardrails to decide which candidates are safe "
            "and informative enough for the next panel."
        ),
        "baselines": ("top_observed", "greedy_utility", "fixed_mix", "random_feasible"),
        "evaluation_protocol": (
            "Evaluate best feasible variant, primary endpoint improvement, guardrail "
            "retention, source-system transfer, and required validation contrasts."
        ),
        "failure_modes": (
            "Mechanistic rationale may not transfer across antibody backgrounds.",
            "Conditional endpoint gains can hide neutral-binding or coverage losses.",
            "Coverage and QC measurements can be missing for otherwise attractive variants.",
        ),
        "reusable_ideas": (
            "Performance-first scoring with mechanism and coverage terms.",
            "Guardrail measurements as first-class constraints.",
            "Champion, contrast, interaction, and control rationale in recommendations.",
        ),
        "gap_for_design_scientist": (
            "Needs every endpoint claim tied to explicit observation provenance and "
            "stratified measurement type."
        ),
        "capabilities": {
            "handles_batches": "yes",
            "handles_constraints": "yes",
            "handles_multi_objective": "yes",
            "works_with_tiny_data": "yes",
            "supports_mechanistic_evidence": "high",
            "supports_transfer": "medium",
        },
    },
)


def extract_methods(project_dir: str | Path) -> dict[str, Any]:
    """Extract method modules and write framework method artifacts.

    Empty or unusable paper-card inputs produce only
    ``framework/method_extraction_failure.md`` for this run and return a failure
    dictionary. This avoids fabricating method modules when the literature stage
    has not supplied evidence.
    """

    root = Path(project_dir).expanduser().resolve()
    framework_dir = ensure_dir(root / "framework")
    cards_path = framework_dir / "paper_cards.json"

    cards, failure_reason = _load_paper_cards(cards_path)
    if failure_reason:
        return _write_failure(framework_dir, cards_path, failure_reason)

    modules, unassigned = _extract_method_modules(cards)
    artifact_paths = {
        "method_modules": framework_dir / "method_modules.json",
        "literature_map": framework_dir / "literature_map.md",
        "research_gap_matrix": framework_dir / "research_gap_matrix.csv",
        "algorithm_spec": framework_dir / "algorithm_spec.md",
        "method_registry": framework_dir / "method_registry.yaml",
        "method_hypotheses": framework_dir / "method_hypotheses.md",
    }

    method_payload = {
        "version": "0.1",
        "source_path": "framework/paper_cards.json",
        "paper_count": len(cards),
        "source_papers": [_source_paper_summary(card) for card in cards],
        "method_modules": modules,
        "modules": modules,
        "unassigned_papers": unassigned,
    }
    write_json(artifact_paths["method_modules"], method_payload)
    _write_literature_map(artifact_paths["literature_map"], cards, modules, unassigned)
    _write_gap_matrix(artifact_paths["research_gap_matrix"], modules)
    _write_algorithm_spec(artifact_paths["algorithm_spec"], modules)
    registry = _method_registry(modules)
    write_yaml(artifact_paths["method_registry"], registry)
    _write_method_hypotheses(artifact_paths["method_hypotheses"], modules)

    return {
        "status": "ok",
        "paper_count": len(cards),
        "method_module_count": len(modules),
        "method_modules": modules,
        "unassigned_papers": unassigned,
        "registry_policies": [policy["name"] for policy in registry["policies"]],
        "artifacts": {name: str(path) for name, path in artifact_paths.items()},
    }


def _load_paper_cards(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    if not path.exists():
        return [], "missing_paper_cards"
    try:
        raw = read_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return [], f"invalid_paper_cards: {exc}"
    if not isinstance(raw, list):
        return [], "paper_cards_json_must_be_a_list"

    cards = [_normalize_card(card, idx) for idx, card in enumerate(raw, start=1) if isinstance(card, dict)]
    cards = [card for card in cards if _has_card_content(card)]
    if not cards:
        return [], "empty_paper_cards"
    return cards, None


def _write_failure(framework_dir: Path, cards_path: Path, reason: str) -> dict[str, Any]:
    failure_path = framework_dir / "method_extraction_failure.md"
    lines = [
        "# Method Extraction Failure",
        "",
        f"Reason: `{reason}`",
        "",
        f"Expected input: `{cards_path}`",
        "",
        (
            "No method modules, registry, gap matrix, or algorithm specification were "
            "generated for this run because doing so would fabricate methods without "
            "source paper evidence."
        ),
        "",
        "Next step: run `design_scientist.literature_pipeline.run_literature_search` or provide reviewed paper cards, then rerun method extraction.",
        "",
    ]
    failure_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "status": "failure",
        "reason": reason,
        "paper_count": 0,
        "method_module_count": 0,
        "method_modules": [],
        "artifacts": {"failure_report": str(failure_path)},
    }


def _extract_method_modules(cards: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    assignments: dict[str, list[dict[str, Any]]] = {template["module_id"]: [] for template in METHOD_TEMPLATES}
    assigned_papers: set[str] = set()

    for card in cards:
        text = _card_text(card)
        for template in METHOD_TEMPLATES:
            if _keyword_score(text, template["keywords"]) >= int(template["min_score"]):
                assignments[template["module_id"]].append(card)
                assigned_papers.add(card["paper_id"])

    modules = [
        _module_from_template(template, assignments[template["module_id"]])
        for template in METHOD_TEMPLATES
        if assignments[template["module_id"]]
    ]

    if not modules:
        modules = [_unclassified_review_module(cards)]
        assigned_papers.update(card["paper_id"] for card in cards)

    unassigned = [card["paper_id"] for card in cards if card["paper_id"] not in assigned_papers]
    return modules, unassigned


def _module_from_template(template: dict[str, Any], source_cards: list[dict[str, Any]]) -> dict[str, Any]:
    source_ids = [card["paper_id"] for card in source_cards]
    source_baselines = _flatten_list_field(source_cards, "baselines")
    source_reusable = _flatten_list_field(source_cards, "reusable_modules")
    source_reusable.extend(_text_field_values(source_cards, ("what_to_reuse",)))
    source_failures = _flatten_list_field(source_cards, "limitations")
    source_failures.extend(_text_field_values(source_cards, ("what_not_to_copy",)))

    module = {
        "module_id": template["module_id"],
        "name": template["name"],
        "problem_setting": _augment_text(template["problem_setting"], source_cards, ("problem", "problem_setting")),
        "data_regime": _augment_text(template["data_regime"], source_cards, ("data_regime",)),
        "algorithm_family": _augment_text(
            template["algorithm_family"],
            source_cards,
            ("algorithm", "algorithm_family", "method_summary"),
        ),
        "acquisition_policy_mechanism": _augment_text(
            template["acquisition_policy_mechanism"],
            source_cards,
            ("acquisition_or_policy", "acquisition_policy_mechanism"),
        ),
        "baselines": _unique([*template["baselines"], *(_normalize_baseline(item) for item in source_baselines)]),
        "evaluation_protocol": _augment_text(template["evaluation_protocol"], source_cards, ("evaluation",)),
        "failure_modes": _unique([*template["failure_modes"], *source_failures]),
        "reusable_ideas": _unique([*template["reusable_ideas"], *source_reusable]),
        "source_papers": source_ids,
        "source_paper_titles": {card["paper_id"]: card.get("title") or card["paper_id"] for card in source_cards},
        "gap_for_design_scientist": template["gap_for_design_scientist"],
        "capabilities": dict(template["capabilities"]),
        "extraction_status": "literature_backed",
    }
    return module


def _unclassified_review_module(cards: list[dict[str, Any]]) -> dict[str, Any]:
    source_ids = [card["paper_id"] for card in cards]
    baselines = [_normalize_baseline(item) for item in _flatten_list_field(cards, "baselines")]
    reusable = _flatten_list_field(cards, "reusable_modules")
    reusable.extend(_text_field_values(cards, ("what_to_reuse", "method_summary")))
    failures = _flatten_list_field(cards, "limitations")
    failures.extend(_text_field_values(cards, ("what_not_to_copy",)))
    return {
        "module_id": "literature_review_queue",
        "name": "Literature Review Queue",
        "problem_setting": _augment_text(
            "Non-empty paper cards did not match the built-in method families; manual review is required before treating them as executable policies.",
            cards,
            ("problem", "problem_setting"),
        ),
        "data_regime": _augment_text(
            "Unclassified from imported paper cards.",
            cards,
            ("data_regime",),
        ),
        "algorithm_family": _augment_text(
            "unclassified_literature_method",
            cards,
            ("algorithm", "algorithm_family", "method_summary"),
        ),
        "acquisition_policy_mechanism": _augment_text(
            "not_extracted_from_keywords",
            cards,
            ("acquisition_or_policy", "acquisition_policy_mechanism"),
        ),
        "baselines": _unique(item for item in baselines if item),
        "evaluation_protocol": _augment_text(
            "not_extracted_from_keywords",
            cards,
            ("evaluation",),
        ),
        "failure_modes": _unique(
            [
                "Unclassified source cards cannot justify a concrete policy without manual review.",
                *failures,
            ]
        ),
        "reusable_ideas": _unique(reusable or ["Manual review queue for non-empty paper cards."]),
        "source_papers": source_ids,
        "source_paper_titles": {card["paper_id"]: card.get("title") or card["paper_id"] for card in cards},
        "gap_for_design_scientist": "Add reviewed method labels or expand the extractor vocabulary for these papers.",
        "capabilities": {
            "handles_batches": "unknown",
            "handles_constraints": "unknown",
            "handles_multi_objective": "unknown",
            "works_with_tiny_data": "unknown",
            "supports_mechanistic_evidence": "unknown",
            "supports_transfer": "unknown",
        },
        "extraction_status": "needs_manual_review",
    }


def _write_literature_map(
    path: Path,
    cards: list[dict[str, Any]],
    modules: list[dict[str, Any]],
    unassigned: list[str],
) -> None:
    title_by_id = {card["paper_id"]: card.get("title") or card["paper_id"] for card in cards}
    lines = [
        "# Literature Map",
        "",
        f"Input paper cards: {len(cards)}",
        f"Extracted method modules: {len(modules)}",
        "",
        "## Source Papers",
        "",
    ]
    for card in cards:
        source_bits = [str(item) for item in (card.get("source"), card.get("year")) if item not in (None, "")]
        suffix = f" ({', '.join(source_bits)})" if source_bits else ""
        lines.append(f"- `{card['paper_id']}` - {card.get('title') or card['paper_id']}{suffix}")

    lines.extend(["", "## Method Modules", ""])
    for module in modules:
        lines.extend(
            [
                f"### {module['name']}",
                "",
                f"- Module id: `{module['module_id']}`",
                f"- Problem setting: {module['problem_setting']}",
                f"- Data regime: {module['data_regime']}",
                f"- Algorithm family: {module['algorithm_family']}",
                f"- Acquisition/policy mechanism: {module['acquisition_policy_mechanism']}",
                f"- Baselines: {', '.join(module['baselines']) if module['baselines'] else 'none extracted'}",
                f"- Source papers: {', '.join(f'`{paper_id}`' for paper_id in module['source_papers'])}",
                "- Reusable ideas:",
            ]
        )
        lines.extend(f"  - {idea}" for idea in module["reusable_ideas"])
        lines.extend(["- Failure modes:"])
        lines.extend(f"  - {mode}" for mode in module["failure_modes"])
        lines.append("")

    lines.extend(["## Unassigned Papers", ""])
    if unassigned:
        lines.extend(f"- `{paper_id}` - {title_by_id.get(paper_id, paper_id)}" for paper_id in unassigned)
    else:
        lines.append("- None")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_gap_matrix(path: Path, modules: list[dict[str, Any]]) -> None:
    fieldnames = [
        "method_module_id",
        "method_module",
        "source_papers",
        "handles_batches",
        "handles_constraints",
        "handles_multi_objective",
        "works_with_tiny_data",
        "supports_mechanistic_evidence",
        "supports_transfer",
        "baselines_present",
        "evaluation_protocol",
        "failure_modes",
        "reusable_ideas",
        "gap_for_design_scientist",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for module in modules:
            capabilities = module["capabilities"]
            writer.writerow(
                {
                    "method_module_id": module["module_id"],
                    "method_module": module["name"],
                    "source_papers": "; ".join(module["source_papers"]),
                    "handles_batches": capabilities.get("handles_batches", "unknown"),
                    "handles_constraints": capabilities.get("handles_constraints", "unknown"),
                    "handles_multi_objective": capabilities.get("handles_multi_objective", "unknown"),
                    "works_with_tiny_data": capabilities.get("works_with_tiny_data", "unknown"),
                    "supports_mechanistic_evidence": capabilities.get("supports_mechanistic_evidence", "unknown"),
                    "supports_transfer": capabilities.get("supports_transfer", "unknown"),
                    "baselines_present": "; ".join(module["baselines"]),
                    "evaluation_protocol": module["evaluation_protocol"],
                    "failure_modes": "; ".join(module["failure_modes"]),
                    "reusable_ideas": "; ".join(module["reusable_ideas"]),
                    "gap_for_design_scientist": module["gap_for_design_scientist"],
                }
            )


def _write_algorithm_spec(path: Path, modules: list[dict[str, Any]]) -> None:
    module_names = ", ".join(f"`{module['module_id']}`" for module in modules)
    baseline_names = ", ".join(REQUIRED_POLICY_NAMES[:-1])
    lines = [
        "# Algorithm Spec",
        "",
        "Algorithm id: `mechanism_aware_active_design`",
        "",
        "This draft turns the extracted literature modules into a fixed active-design loop. It is a framework specification, not a claim that any source paper already implements the full combined policy.",
        "",
        "## Design Space",
        "",
        "- Candidates are feasible variants, modules, controls, repeats, or contrast-completion designs admitted by the project candidate generator.",
        "- Candidate operators include `add_module`, `remove_module`, `combine_modules`, `complete_missing_edge`, `complete_square`, `repair_neutral_binding`, `validate_genotype_coverage`, and `repeat_or_control`.",
        "- Every candidate carries endpoint requirements, risk flags, provenance references, and a category such as champion, champion contrast, lattice repair, interaction square, control, or repeat.",
        "",
        "## Observation Model",
        "",
        "- Observations are typed assay records with endpoint, measurement type, source table or row provenance, round, variant identity, and evidence tier.",
        "- Evidence tiers remain separate: primary matched, secondary near-matched, descriptive, and model-derived evidence are not silently pooled.",
        "- Surrogate predictions may inform acquisition but remain model-derived until validated by allowed wet-lab observations.",
        "",
        "## Objective",
        "",
        "- Maximize expected best feasible design improvement over the next round while preserving required guardrails.",
        "- Panel score combines expected feasible performance, champion decision value, champion-centered contrast value, interaction or lattice repair value, coverage and QC value, minus cost and risk.",
        "- The default policy must beat or explain failure against these baselines: "
        f"{baseline_names}.",
        "",
        "## Constraints",
        "",
        "- Respect assay budget, required controls, allowed design operators, feasibility filters, endpoint-specific guardrails, and data-contract provenance rules.",
        "- Do not advance unsupported mechanism claims as primary evidence without matched or explicitly labeled near-matched contrasts.",
        "- Maintain measurement-type stratification when endpoints or assay conditions differ.",
        "",
        "## Round Update",
        "",
        "1. Ingest new standardized observations and validation reports.",
        "2. Update evidence cards, design state, champion set, unresolved contrast edges, and unsupported claims.",
        "3. Refit or update utility, uncertainty, and feasibility estimates where data support them.",
        "4. Generate feasible candidates with fixed design operators.",
        "5. Score candidates and panels under registry policies, including baselines and ablations.",
        "6. Select the prospective panel only after validation checks and record policy diagnostics for replay.",
        "",
        "## Policy API",
        "",
        "```python",
        "def select_panel(",
        "    state: dict,",
        "    candidates: list[dict],",
        "    observations: list[dict],",
        "    budget: int,",
        "    policy_config: dict,",
        "    rng_seed: int | None = None,",
        ") -> dict:",
        "    \"\"\"Return panel rows, candidate scores, diagnostics, and baseline comparisons.\"\"\"",
        "```",
        "",
        "Required policy output fields: `panel`, `scores`, `diagnostics`, `baseline_comparison`, `unsupported_claims`, and `artifact_paths`.",
        "",
        "## Literature-Derived Modules",
        "",
        f"Included modules: {module_names or 'none'}.",
        "",
    ]
    for module in modules:
        lines.extend(
            [
                f"### {module['name']}",
                "",
                f"- Source papers: {', '.join(module['source_papers'])}",
                f"- Acquisition/policy mechanism: {module['acquisition_policy_mechanism']}",
                f"- Failure modes to monitor: {'; '.join(module['failure_modes'])}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _method_registry(modules: list[dict[str, Any]]) -> dict[str, Any]:
    module_ids = [module["module_id"] for module in modules]
    active_modules = [
        module["module_id"]
        for module in modules
        if module["module_id"]
        in {
            "active_learning_batch_design",
            "bayesian_constrained_optimization",
            "adaptive_decision_value_allocation",
            "matched_contrast_lattice",
            "mechanism_aware_antibody_design",
        }
    ]
    return {
        "registry_version": "0.1",
        "default_policy": "mechanism_aware",
        "policy_api": {
            "callable": "select_panel",
            "inputs": ["state", "candidates", "observations", "budget", "policy_config", "rng_seed"],
            "outputs": ["panel", "scores", "diagnostics", "baseline_comparison", "unsupported_claims"],
        },
        "policies": [
            {
                "name": "random_feasible",
                "role": "baseline",
                "status": "specified",
                "source_modules": [],
                "selection_rule": "Uniformly sample feasible candidates after mandatory controls.",
                "uses_uncertainty": False,
                "uses_mechanism": False,
                "evaluation": "Sanity baseline for hit rate and best feasible utility.",
            },
            {
                "name": "top_observed",
                "role": "baseline",
                "status": "specified",
                "source_modules": [],
                "selection_rule": "Rank candidates by observed endpoint or nearest champion-derived observed utility.",
                "uses_uncertainty": False,
                "uses_mechanism": False,
                "evaluation": "Exploitation baseline; exposes whether modeling or mechanism terms add value.",
            },
            {
                "name": "greedy_utility",
                "role": "baseline",
                "status": "specified",
                "source_modules": _matching_module_ids(module_ids, "active_learning_batch_design", "bayesian_constrained_optimization"),
                "selection_rule": "Rank by predicted feasible utility with constraints, without explicit exploration or lattice repair terms.",
                "uses_uncertainty": False,
                "uses_mechanism": False,
                "evaluation": "Must be compared to uncertainty-aware and mechanism-aware policies.",
            },
            {
                "name": "fixed_mix",
                "role": "baseline",
                "status": "specified",
                "source_modules": _matching_module_ids(module_ids, "adaptive_decision_value_allocation"),
                "selection_rule": "Allocate fixed fractions to champion, contrast, exploration, lattice repair, and control categories.",
                "uses_uncertainty": "optional",
                "uses_mechanism": "optional",
                "evaluation": "Baseline for adaptive allocation and component ablations.",
            },
            {
                "name": "pure_uncertainty",
                "role": "baseline",
                "status": "specified",
                "source_modules": _matching_module_ids(module_ids, "active_learning_batch_design", "bayesian_constrained_optimization"),
                "selection_rule": "Rank feasible candidates by uncertainty or expected information gain, ignoring predicted performance except guardrails.",
                "uses_uncertainty": True,
                "uses_mechanism": False,
                "evaluation": "Exploration baseline; should not dominate if final design quality matters.",
            },
            {
                "name": "pure_lattice_repair",
                "role": "baseline",
                "status": "specified",
                "source_modules": _matching_module_ids(module_ids, "matched_contrast_lattice"),
                "selection_rule": "Prioritize missing matched edges and interaction squares regardless of predicted performance except hard feasibility filters.",
                "uses_uncertainty": False,
                "uses_mechanism": True,
                "evaluation": "Interpretability baseline; useful for measuring performance-first tradeoffs.",
            },
            {
                "name": "mechanism_aware",
                "role": "default_candidate_policy",
                "status": "draft",
                "source_modules": active_modules,
                "selection_rule": (
                    "Maximize expected feasible performance first, then add decision value, "
                    "champion-centered contrast value, lattice repair, coverage, and QC terms; "
                    "subtract cost and risk."
                ),
                "weights": {
                    "performance": 1.0,
                    "decision_value": 0.7,
                    "contrast_value": 0.45,
                    "lattice_repair": 0.35,
                    "coverage_qc": 0.25,
                    "cost": -0.2,
                    "risk": -0.7,
                },
                "batch_allocation": {
                    "champion_or_high_utility": "40-70%",
                    "champion_contrast": "10-30%",
                    "lattice_repair_or_interaction": "5-25%",
                    "controls_or_repeats": "5-15%",
                },
                "uses_uncertainty": True,
                "uses_mechanism": True,
                "evaluation": "Retrospective replay plus prospective validation report against every required baseline.",
                "ablation_components": [
                    "remove_uncertainty",
                    "remove_decision_value",
                    "remove_contrast_value",
                    "remove_lattice_repair",
                    "remove_coverage_qc",
                ],
            },
            {
                "name": "codex_generated",
                "role": "placeholder",
                "status": "not_implemented",
                "source_modules": [],
                "selection_rule": "Reserved for a future generated policy; must cite source modules and pass benchmark gates before use.",
                "uses_uncertainty": "unknown",
                "uses_mechanism": "unknown",
                "evaluation": "Do not use prospectively until it has paper-card provenance, replay metrics, and validation checks.",
            },
        ],
    }


def _write_method_hypotheses(path: Path, modules: list[dict[str, Any]]) -> None:
    source_modules = ", ".join(f"`{module['module_id']}`" for module in modules)
    lines = [
        "# Method Hypotheses",
        "",
        "## mechanism_aware",
        "",
        (
            "If panel selection prioritizes expected feasible performance while adding "
            "decision value, champion-centered contrasts, lattice repair, coverage, and "
            "QC guardrails, it should outperform simple exploitation and pure repair policies."
        ),
        "",
        "Must beat:",
        "- random_feasible",
        "- top_observed",
        "- greedy_utility",
        "- fixed_mix",
        "- pure_uncertainty",
        "- pure_lattice_repair",
        "",
        f"Literature modules supporting this hypothesis: {source_modules or 'none extracted'}.",
        "",
        "## adaptive_batch_allocation",
        "",
        (
            "If batch fractions adapt to design-state maturity and unresolved evidence gaps, "
            "the policy should beat fixed allocation on best feasible utility and unsupported-claim reduction."
        ),
        "",
        "Must beat:",
        "- fixed_mix",
        "- pure_uncertainty",
        "- pure_lattice_repair",
        "",
        "## matched_contrast_guardrail",
        "",
        (
            "If module claims require exact matched or explicitly labeled near-matched "
            "evidence, false-positive advancement should drop relative to descriptive "
            "module ranking without reducing the best feasible variant selected."
        ),
        "",
        "Must beat:",
        "- top_observed",
        "- greedy_utility",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _normalize_card(card: dict[str, Any], idx: int) -> dict[str, Any]:
    normalized = dict(card)
    title = _clean_str(card.get("title") or card.get("citation") or card.get("paper_id"))
    paper_id = _clean_str(card.get("paper_id"))
    if not paper_id:
        source = _clean_str(card.get("source"))
        source_id = _clean_str(card.get("source_id"))
        doi = _clean_str(card.get("doi"))
        if source and source_id:
            paper_id = f"{source}:{source_id}"
        elif doi:
            paper_id = f"doi:{doi.lower()}"
        elif title:
            paper_id = f"paper:{_slugify(title)}"
        else:
            paper_id = f"paper:{idx}"
    normalized["paper_id"] = paper_id
    if title:
        normalized["title"] = title
    return normalized


def _has_card_content(card: dict[str, Any]) -> bool:
    return bool(_clean_str(card.get("title") or card.get("abstract") or card.get("problem") or card.get("method_summary")))


def _source_paper_summary(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": card["paper_id"],
        "title": card.get("title"),
        "source": card.get("source"),
        "year": card.get("year"),
        "doi": card.get("doi"),
        "url": card.get("url"),
        "review_status": card.get("review_status"),
    }


def _card_text(card: dict[str, Any]) -> str:
    keys = (
        "title",
        "abstract",
        "problem",
        "problem_setting",
        "data_regime",
        "design_loop",
        "algorithm",
        "algorithm_family",
        "method_summary",
        "acquisition_or_policy",
        "acquisition_policy_mechanism",
        "constraints",
        "evaluation",
        "baselines",
        "what_to_reuse",
        "what_not_to_copy",
        "reusable_modules",
        "limitations",
        "tags",
        "citation",
        "relevance_to_current_project",
        "relevance",
    )
    return " ".join(_flatten_value(card.get(key)) for key in keys).lower()


def _keyword_score(text: str, keywords: Iterable[str]) -> int:
    score = 0
    for keyword in keywords:
        normalized = keyword.lower()
        if " " in normalized or "-" in normalized:
            if normalized in text:
                score += 2
            continue
        if re.search(rf"\b{re.escape(normalized)}\b", text):
            score += 1
    return score


def _augment_text(base: str, cards: list[dict[str, Any]], keys: tuple[str, ...], *, max_items: int = 3) -> str:
    values = _text_field_values(cards, keys)
    if not values:
        return base
    cues = "; ".join(values[:max_items])
    return f"{base} Source cues: {cues}."


def _text_field_values(cards: list[dict[str, Any]], keys: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for card in cards:
        for key in keys:
            value = card.get(key)
            if value in (None, "", [], {}):
                continue
            text = _clean_str(_flatten_value(value))
            if text:
                values.append(text)
    return _unique(values)


def _flatten_list_field(cards: list[dict[str, Any]], key: str) -> list[str]:
    values: list[str] = []
    for card in cards:
        raw = card.get(key)
        if isinstance(raw, list):
            values.extend(_clean_str(item) for item in raw)
        elif raw not in (None, "", {}, []):
            values.append(_clean_str(_flatten_value(raw)))
    return _unique(item for item in values if item)


def _flatten_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            parts.append(str(key))
            parts.append(_flatten_value(item))
        return " ".join(parts)
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten_value(item) for item in value)
    return str(value)


def _normalize_baseline(value: Any) -> str:
    text = _clean_str(value)
    if not text:
        return ""
    lowered = text.lower()
    if "random" in lowered:
        return "random_feasible"
    if "top observed" in lowered or "top ratio" in lowered or "observed ratio" in lowered:
        return "top_observed"
    if "top predicted" in lowered or "greedy" in lowered or "surrogate" in lowered or "utility" in lowered:
        return "greedy_utility"
    if "fixed" in lowered or "fixed allocation" in lowered or "fixed mix" in lowered:
        return "fixed_mix"
    if "uncertainty" in lowered or "pure exploration" in lowered:
        return "pure_uncertainty"
    if "lattice" in lowered or "repair" in lowered:
        return "pure_lattice_repair"
    return _slugify(text)


def _matching_module_ids(module_ids: list[str], *allowed: str) -> list[str]:
    allowed_set = set(allowed)
    return [module_id for module_id in module_ids if module_id in allowed_set]


def _unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = _clean_str(value)
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "unnamed"
