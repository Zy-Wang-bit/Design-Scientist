# Acquisition Reference

The design policy is performance-first and mechanism-aware.

## Conceptual Score

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

## Component Meanings

- `expected best feasible design improvement`: expected improvement in the best usable design after this panel.
- `champion decision value`: value of deciding whether a current top candidate should advance, be modified, or be deprioritized.
- `champion-centered contrast value`: value of minimal matched contrasts explaining why a top candidate works or fails.
- `interaction or lattice repair value`: value of completing missing edges or squares that unlock reusable module knowledge.
- `coverage / QC guardrail value`: value of avoiding pH 7.4 loss, genotype coverage loss, expression failure, assay artifacts, or unsupported conclusions.

## Policy Search

Do not hard-code one batch ratio as the method. Treat allocation weights as policy parameters.

Start with grids or named policies, then consider contextual bandit or Bayesian optimization once offline evaluation exists.

Example policy families:

```text
performance_heavy
balanced_champion_contrast
lattice_repair_heavy
validation_heavy
coverage_guardrail_heavy
```

Choose policies according to project state, not habit.
