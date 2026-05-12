# Design Scientist Workflow

Use this reference when implementing or reviewing the reusable design framework. It borrows the useful machinery from AI-Scientist and AI-Scientist-v2, but changes the scientific objective from paper generation to data-grounded design decisions.

## Mapping From AI-Scientist

AI-Scientist v1 uses templates such as `experiment.py`, `plot.py`, `prompt.json`, `seed_ideas.json`, and LaTeX writeups. AI-Scientist-v2 removes most hand-written templates and uses ideation plus agentic tree search, experiment execution, debugging, summaries, writeup, and review.

For Design Scientist, map those components as follows:

```text
AI-Scientist idea generation
-> literature-grounded method ideation and project hypothesis generation

AI-Scientist experiment template
-> project runner with data contracts and required artifacts

AI-Scientist tree-search node
-> one executable policy / analysis / panel-selection implementation

AI-Scientist debug loop
-> artifact validator, provenance guard, leakage guard, and failed-node repair

AI-Scientist baseline and ablation stages
-> baseline panel selectors, policy ablations, retrospective masking evaluation

AI-Scientist paper writeup
-> decision dossier: state update, panel recommendation, risk register

AI-Scientist automated review
-> human-review gate plus deterministic validity checks
```

## Full Workflow

### 0. Project Intake and Contract Lock

Create or update:

```text
project.yaml
data_contract.yaml
estimands.yaml
state/design_state.json
```

The project contract defines the goal, endpoints, allowed data tables, forbidden data sources, design space, cost constraints, and wet-lab feasibility constraints. For anti-HBsAg this includes pH 7.4 retention, pH 6.0 reduction, genotype coverage, expression/QC flags, 1E62 backgrounds, sdAb modules, and allowed standardized tables.

### 1. Data Ingestion and Schema Audit

This replaces AI-Scientist's assumption that a template already knows the dataset.

Tasks:

- inventory raw and standardized files;
- map endpoint semantics;
- preserve source provenance;
- detect duplicated or ambiguous source-cell mappings;
- produce standardized primary, raw-detail, and auxiliary tables;
- forbid downstream raw CSV guessing unless this stage is explicitly reopened.

Output:

```text
standardized/
schema_audit_report.md
validation_summary.json
```

### 1B. Literature Discovery and Method Ideation

This is required when developing a new computational framework. It mirrors AI-Scientist's ideation and literature loop, but the output is not a paper idea. The output is a set of method components and research gaps.

Tasks:

- search active learning, Bayesian optimization, batch design, design-build-test-learn, protein engineering, antibody engineering, and adaptive experimental design literature;
- create paper cards;
- extract reusable algorithmic modules, baselines, and evaluation protocols;
- build a research gap matrix;
- propose method hypotheses that are explicitly grounded in the literature;
- reject method ideas that cannot be benchmarked.

Output:

```text
framework/literature_map.md
framework/paper_cards.json
framework/research_gap_matrix.csv
framework/method_hypotheses.md
```

### 2. Evidence Construction

This stage converts standardized rows into auditable evidence cards.

Evidence types:

- endpoint summaries;
- pH 7.4 vs pH 6.0 paired comparisons;
- exact matched mutation contrasts;
- near-matched exploratory contrasts;
- interaction squares;
- coverage and expression/QC flags;
- model-derived predictions, clearly labeled as such.

Each evidence card records claim, source rows, evidence tier, effect estimate, uncertainty, limitations, and actionability.

### 3. Design State Update

This is the durable project memory. It replaces one-off idea lists.

The state should record:

- current round;
- standardized data version;
- active design goals;
- known champions;
- module status;
- unresolved edges;
- failed or risky designs;
- policy parameters used in prior rounds;
- pending wet-lab or review actions.

### 4. Candidate Generation

Generate candidates with explicit design operators, not free-form invention.

Operators:

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

For anti-HBsAg, this means sdAb data can suggest module logic, while 1E62 candidates should be constrained by target-system backgrounds, coverage, and missing matched contrasts.

### 5. Policy-Node Search

This is the AI-Scientist-v2-style tree-search replacement.

Each node is one executable proposal, such as:

- a literature-grounded method variant;
- a candidate generator variant;
- an acquisition function;
- a batch allocation policy;
- a baseline selector;
- an offline evaluation protocol;
- a visualization/reporting implementation.

The tree manager expands, debugs, and compares nodes, but selection is not based on paper metrics. Selection uses a rubric:

```text
real-data provenance compliance
best feasible design utility
hit-rate or expected improvement
champion decision value
primary matched evidence created
false-positive risk reduction
coverage/QC guardrail value
baseline superiority
unsupported-claim penalty
```

### 6. Baseline and Ablation Stage

Every policy must be compared against simple alternatives:

```text
random feasible panel
top observed ratio or endpoint screen
top predicted utility
pure uncertainty sampling
pure lattice repair
fixed-mix champion/contrast/control policy
```

Ablations should test which score components matter:

- remove champion improvement;
- remove contrast value;
- remove lattice repair value;
- remove coverage/QC guardrail;
- change batch allocation weights.

### 7. Retrospective Evaluation

When enough historical data exist, evaluate policies by masking observations:

```text
start with partial observed data
select a batch
reveal held-out observed data for selected variants
update state
repeat
```

Compare:

- best feasible utility found by round;
- hit rate;
- false-positive advancement rate;
- champion explanation coverage;
- number of useful primary matched edges created;
- genotype or condition coverage retained.

### 8. Panel Selection and Decision Dossier

Panel recommendations should include:

- candidate variant or construct;
- category: champion, champion_contrast, interaction_square, lattice_repair, control, repeat;
- acquisition score and components;
- expected decision value;
- risk flags;
- required measurements;
- reason for inclusion;
- baseline comparison.

The decision dossier replaces the AI-Scientist paper draft. It should include:

```text
state update
top champions
evidence summary
recommended panel
baseline comparison
policy ablation summary
unsupported claims
risk register
wet-lab measurement requirements
```

### 9. Review Gate

Before a recommendation is treated as actionable, run deterministic validation and human review.

Checks:

- no synthetic records used as real evidence;
- only allowlisted standardized tables used outside schema audit;
- endpoint provenance retained;
- measurement types not mixed silently;
- evidence tiers labeled;
- model predictions not reported as wet-lab measurements;
- unsupported claims explicitly listed.

### 10. New Data Merge

When new wet-lab data arrive:

```text
raw intake -> schema audit/update -> evidence rebuild -> design state update -> policy-node search or fixed-policy rerun -> new decision dossier
```

Do not restart from an empty idea prompt. Preserve history and make policy changes explicit.

## Evidence Cards

Evidence cards should be structured enough to audit:

```json
{
  "claim": "module X may improve acidic release",
  "evidence_tier": "primary_matched | secondary_near_matched | descriptive | model_derived",
  "source_rows": [],
  "effect": {},
  "limitations": [],
  "actionability": "advance | validate | monitor | unresolved | reject"
}
```
