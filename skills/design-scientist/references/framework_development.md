# Framework Development Reference

Use this reference when the goal is to create or improve the computational design framework itself, not merely run a predefined workflow.

## Scientific Objective

Develop a computational framework for multi-round, data-grounded variant design in life-science research. The framework should support the common wet-lab loop:

```text
design a batch -> generate data -> learn from data -> design the next batch -> repeat
```

The framework should improve the chance of obtaining the desired variants over multiple rounds. It should also make the learning process auditable enough to avoid chasing false positives, but interpretability is a means to better design, not the final objective.

## What Makes The Agent A Scientist

The agent should not only execute a pipeline. It should formulate and test computational method hypotheses.

A good method hypothesis has this shape:

```text
If we represent the design problem as X,
and select batches using Y,
then across repeated design rounds we should improve Z
relative to baselines B,
because mechanism M better matches the data-generation process.
```

Examples:

- Contrast-lattice repair should reduce false-positive module advancement in small-data antibody design.
- Champion-centered contrasts should improve final hit rate more than generic interpretability experiments.
- Adaptive batch allocation should outperform a fixed 70/20/10 split when data maturity changes across rounds.
- sdAb module-learning plus 1E62 target-system validation may outperform direct 1E62 ratio ranking when 1E62 data are sparse.

## Framework R&D Loop

Use this loop when developing the framework:

```text
problem formalization
-> literature discovery
-> paper cards and method extraction
-> research gap matrix
-> method hypothesis
-> algorithm specification
-> implementation
-> offline/retrospective benchmark
-> baseline and ablation comparison
-> failure analysis
-> method revision
-> prospective panel recommendation
```

This is separate from the biological project loop, although the same data may be used.

## Literature Discovery

Do not invent the framework from intuition alone. Before proposing a new algorithm, run a literature pass modeled after AI-Scientist's ideation process, but aimed at method development rather than paper writing.

### Search Scope

Cover at least these areas when relevant:

- active learning for protein engineering;
- Bayesian optimization for biological sequence design;
- batch Bayesian optimization and constrained acquisition;
- multi-objective optimization with wet-lab constraints;
- adaptive experimental design / optimal design;
- design-build-test-learn cycles;
- directed evolution and machine-learning-guided protein engineering;
- antibody engineering and pH-dependent antibody design;
- causal or matched-contrast analysis for small experimental datasets;
- transfer learning across related protein/antibody systems;
- retrospective benchmark protocols for active design.

### Paper Cards

Create a compact card for each useful paper:

```json
{
  "citation": "",
  "problem": "",
  "data_regime": "",
  "design_loop": "",
  "algorithm": "",
  "acquisition_or_policy": "",
  "constraints": "",
  "evaluation": "",
  "baselines": [],
  "what_to_reuse": "",
  "what_not_to_copy": "",
  "relevance_to_current_project": ""
}
```

### Research Gap Matrix

After paper cards, make a matrix:

```text
method family | handles batches | handles constraints | handles multi-objective | works with tiny data | supports interpretability | supports transfer | likely baseline | gap for our task
```

Use this matrix to justify the new framework. A method hypothesis should say which gap it addresses and which baseline it must beat.

### Method Ideation From Literature

Generate method variants by recombining literature-backed components:

- surrogate model choice;
- uncertainty representation;
- acquisition objective;
- batch allocation rule;
- contrast or lattice repair term;
- transfer-learning prior;
- constraint handling;
- retrospective evaluation protocol.

Reject variants that cannot be evaluated against historical data or at least a clear synthetic/held-out benchmark.

## Algorithmic Components To Explore

### State Representation

Represent the project state as a structured object:

- observed variants and measurements;
- design space;
- known champions;
- candidate modules;
- uncertainty and risk;
- missing contrasts and interaction squares;
- prior rounds and decisions;
- policy parameters and outcomes.

### Evidence Model

Possible evidence models include:

- deterministic paired endpoint summaries;
- matched contrast estimands;
- hierarchical module-effect models;
- Gaussian-process or Bayesian surrogate models;
- ensemble uncertainty over small-data predictors;
- explicit failure-mode classifiers.

The framework may combine interpretable estimands with predictive models. Do not force a black-box-only or interpretability-only approach.

### Candidate Generator

Candidate generation should be operator-based:

- exploit promising champions;
- repair or simplify champions;
- add/remove modules;
- combine modules;
- complete matched edges;
- complete interaction squares;
- test genotype coverage;
- add repeats and controls.

### Acquisition Function

Treat the acquisition function as the central method object:

```text
A(panel | state)
= performance value
+ decision value
+ transfer value
+ contrast/lattice value
+ uncertainty reduction
+ coverage/QC value
- cost
- risk
```

The exact terms and weights are research variables. They should be compared by retrospective evaluation and ablation, not chosen by taste.

### Adaptive Batch Allocation

Do not hard-code a single ratio. Explore policies such as:

- fixed split;
- state-dependent rules;
- contextual bandit over allocation templates;
- Bayesian optimization over policy weights;
- Thompson sampling over candidate classes;
- multi-objective constrained batch optimization.

The policy should shift as the project matures:

- early: more exploration, module learning, and lattice repair;
- middle: more champion expansion and targeted contrasts;
- late: more validation, coverage, developability, and repeats.

## Method Evaluation

A method must be judged as a method, not only by one recommended panel.

Use retrospective masking when possible:

```text
hide part of historical data
let each policy choose a batch
reveal selected outcomes
update state
repeat
```

Compare:

- best feasible variant found by round;
- hit rate;
- false-positive advancement rate;
- regret against best observed feasible variant;
- number of rounds to reach a threshold;
- robustness across random masks;
- quality of decisions under budget constraints;
- whether top candidates remain explainable enough to improve next rounds.

## Baselines

Always include strong simple baselines:

- random feasible panel;
- top ratio;
- top neutral retention;
- top predicted utility;
- Bayesian optimization or surrogate-only selection;
- pure uncertainty sampling;
- pure lattice repair;
- fixed allocation policy;
- human-inspired heuristic if available.

The framework is not credible unless it beats or clarifies when it fails against these baselines.

## Method Report

A method report should separate:

- literature basis;
- research gap;
- biological conclusions;
- computational method claims;
- evaluation protocol;
- baselines;
- ablations;
- failure modes;
- next method revision.

Do not let a successful biological anecdote substitute for method validation.

## Anti-HBsAg As First Benchmark

Use anti-HBsAg as the first benchmark because it has:

- real 1E62 and sdAb data;
- clear pH-dependent objective;
- multi-round design motivation;
- small-data constraints;
- mechanistic module hypotheses;
- target-system transfer questions.

Initial framework question:

```text
Can a performance-first, mechanism-aware active design policy use sdAb module-learning plus 1E62 target-system evidence to choose better anti-HBsAg pH-switch panels than ratio ranking or generic active learning?
```

Initial framework variants to compare:

- ratio ranking;
- surrogate-only predicted utility;
- pure contrast-lattice repair;
- fixed champion/contrast/control split;
- adaptive performance-first mechanism-aware policy;
- champion-centered contrast policy.
