---
name: design-scientist
description: "Use when developing or applying a novel computational framework for data-grounded protein, antibody, or wet-lab variant design. This skill covers both framework R&D and project execution: formulating method hypotheses, building active-design algorithms, evaluating them over multi-round historical or prospective data, comparing against baselines, and using the resulting framework to choose new experimental panels. Especially relevant for anti-HBsAg pH-dependent antibody design and similar projects where real experimental data already exist and the goal is to iteratively design better variants."
metadata:
  short-description: Data-grounded active design workflow
---

# Design Scientist

Use this skill to make Codex act as a computational research scientist for real-data variant design. The goal is not merely to operate an existing workflow. The goal is to develop, stress-test, and improve a reusable computational framework that can learn from multiple wet-lab rounds and ultimately produce better variants.

## Core Principle

Separate two modes:

```text
Framework R&D mode: invent and validate the computational method.
Project execution mode: apply the current method to choose the next experimental panel.
```

The framework must be able to run as a persistent state-update loop:

```text
standardized data -> evidence update -> design state update -> candidate generation -> panel selection -> validation -> new data merge
```

Do not restart the scientific strategy from scratch when new data arrive. Update the existing project state, then use the new data to improve both the design state and, when justified, the framework itself.

## Framework R&D Workflow

When the task is to develop the method, not just run it:

Prefer the repository's local CLI entry points for framework R&D. Keep the
scientific logic in the project code and artifacts; do not embed an alternate
workflow inside this skill. A finished local run should normally follow this
shape from the Design-Scientist checkout:

```bash
design-scientist init-framework <project_dir> --domain "<domain>"
design-scientist audit-references <project_dir>
design-scientist run-scientist <project_dir> --offline-fixtures
design-scientist review-framework <project_dir>
```

If working directly from an uninstalled checkout, run with `PYTHONPATH=src` or
install the package editable first. The validation step must pass before
presenting a framework run as complete.

1. State the computational research question and method hypothesis.
2. Run a literature-discovery pass before inventing the method.
3. Extract mechanism cards, mechanism components, stress-test claims, baselines, evaluation protocols, and failure modes from the literature.
4. Define what would count as method progress, not only project success.
5. Formalize the design problem: design space, observations, constraints, objectives, costs, and rounds.
6. Propose one or more algorithmic mechanisms, such as acquisition functions, policy search, contrast-lattice repair, uncertainty models, or transfer across related systems.
7. Implement the method as executable code with fixed artifacts.
8. Compare against strong simple baselines and ablations.
9. Run retrospective masking evaluation when historical data allow it.
10. Identify where the method fails and create the next method variant.
11. Record method claims separately from biological claims.
12. Only then apply the current best method to recommend a wet-lab panel.

## Default Workflow

When the task is to run a project with the current method:

1. Locate or create the project directory.
2. Read `project.yaml`, `data_contract.yaml`, `estimands.yaml`, and current `design_state.json` if present.
3. Use only the data tables allowed by `data_contract.yaml`.
4. If raw data are involved, enter a schema-audit stage first; otherwise do not infer raw CSV schemas ad hoc.
5. Build or update evidence cards before selecting new designs.
6. Generate candidates using fixed design operators.
7. Select a panel with a performance-first mechanism-aware acquisition rule.
8. Compare against simple baselines.
9. Write machine-readable artifacts and a concise decision report.
10. Run validation checks before presenting the result as usable.

## Required Artifacts

When developing the framework, prefer these artifacts:

```text
framework/reference_audit.md
framework/reference_components.json
framework/quest.yaml
framework/research_map.json
framework/findings_memory.jsonl
framework/failure_memory.jsonl
framework/literature_search_plan.json
framework/literature_search_trace.json
framework/citation_graph.json
framework/paper_scores.csv
framework/paper_cards.json
framework/literature_corpus.jsonl
framework/literature_reading_trace.json
framework/mechanism_cards.json
framework/mechanism_library.json
framework/mechanism_gap_matrix.csv
runs/<run_id>/stage_progress.json
runs/<run_id>/route_tree.json
runs/<run_id>/scientist_journal.json
runs/<run_id>/mechanism_benchmark_results.csv
runs/<run_id>/mechanism_benchmark_summary.csv
runs/<run_id>/mechanism_ablation_results.csv
runs/<run_id>/mechanism_nodes/<node_id>/mechanism_spec.json
runs/<run_id>/mechanism_nodes/<node_id>/mechanism.py
runs/<run_id>/mechanism_nodes/<node_id>/proposal.json
runs/<run_id>/mechanism_nodes/<node_id>/ablation_plan.json
runs/<run_id>/mechanism_nodes/<node_id>/stress_test_plan.json
runs/<run_id>/mechanism_nodes/<node_id>/mechanism_metrics.json
runs/<run_id>/mechanism_nodes/<node_id>/validation_report.json
runs/<run_id>/method_report.md
src/
tests/
```

When implementing or updating a biological design project, prefer these artifacts:

```text
project.yaml
data_contract.yaml
estimands.yaml
standardized/
state/design_state.json
state/evidence_cards.json
runs/<run_id>/candidate_pool.csv
runs/<run_id>/panel_recommendation.csv
runs/<run_id>/policy_comparison.csv
runs/<run_id>/decision_report.md
runs/<run_id>/validation_report.json
```

If a project is still being initialized, create the minimal subset needed for the next concrete step.

## Acquisition Logic

Panel selection should prioritize good final designs, while using mechanistic checks only when they improve decision quality.

Use this conceptual scoring shape:

```text
score(panel)
= expected best feasible design improvement
+ champion decision value
+ champion-centered contrast value
+ interaction or lattice repair value
+ coverage / QC guardrail value
- cost
- risk
```

Do not make interpretability the primary goal. Use interpretability to avoid false positives, identify transferable modules, and improve the next round.

Treat acquisition design itself as a research object. Policy weights, batch allocation, uncertainty handling, and contrast value are not fixed truths; they should be explored, ablated, and improved.

## Design Operators

Candidate generation should use explicit operators such as:

```text
add_module(background, module)
remove_module(champion, module)
combine_modules(background, module_a, module_b)
complete_missing_edge(background, module)
complete_square(background, module_a, module_b)
repair_neutral_binding(champion)
validate_genotype_coverage(champion)
repeat_or_control(edge_or_variant)
```

Avoid free-form variant invention unless the user explicitly asks for speculative design.

## Baselines

Every nontrivial panel recommendation should compare against simple alternatives:

```text
random feasible panel
top observed ratio or endpoint screen
top predicted utility
pure uncertainty sampling
pure lattice repair / contrast completion
fixed-mix policy
```

If there is not enough data for a fair offline comparison, state that explicitly and still include a qualitative baseline comparison.

## Validation Rules

Before finalizing, check:

- No synthetic records are used as real evidence.
- Every primary endpoint has provenance to an allowed source or standardized row.
- Measurement types are not mixed without an explicit stratification or warning.
- Descriptive, model-derived, near-matched, and primary matched evidence are labeled separately.
- Recommendations include champion, contrast, interaction, control, and risk rationale where applicable.
- Unsupported conclusions are listed, not hidden.

Use `scripts/validate_project.py` when a project directory already exists.
For framework R&D runs, use `python -m design_scientist.framework_validation
<project_dir> --run-id <run_id>` and treat missing framework spec, literature
queries, paper cards, Literature Engine V3 corpus/reading trace, mechanism
cards/library/gap matrix, Research OS quest/map/memory files, literature
search plan/trace/citation graph/paper scores, mechanism benchmark results,
mechanism ablation results, scientist journal, selected mechanism node
artifacts, or method report as blocking errors. Missing reference audit output
is a warning unless `audit-references` was requested and left partial audit
artifacts. Selected V3 nodes must include `mechanism_spec.json`,
`mechanism.py`, `proposal.json`, `ablation_plan.json`,
`stress_test_plan.json`, `mechanism_metrics.json`, and
`validation_report.json`; architecture clones cannot be selected.

## Anti-HBsAg

For private anti-HBsAg pH-dependent antibody design work, read any
project-local or user-provided context file before making recommendations or
creating artifacts. The public skill package intentionally does not bundle
private experimental paths or project-specific data summaries.

For developing the computational framework itself, read `references/framework_development.md`.

## Output Style

When reporting to the user:

- Lead with what changed or what the current decision is.
- Distinguish data facts, inferred evidence, and proposed next actions.
- Keep claims conservative when evidence is not primary matched.
- Prefer concrete artifact paths over abstract descriptions.
