# Changelog

## 0.3.0 - 2026-05-26

Formal V3 contract release: OperatorSpec + claim-gated mechanism search.

- Added Literature Engine V3 operator compilation artifacts:
  `operator_specs.json`, `operator_gap_matrix.csv`,
  `operator_evidence_map.json`, and `operator_negative_controls.json`.
- Hardened V3 mechanism-node contracts with lifecycle enforcement,
  operator-to-code trace validation, baseline-wrapper rejection, ghost-operator
  rejection, and fake key-ablation rejection.
- Added benchmark saturation, claim cap, reviewer verdict, unfair-baseline, and
  biology-overclaim validation gates.
- Added `reference_data_sources.json` as the run-level source manifest tying
  literature, operator, selected-node, benchmark, claim, and reviewer artifacts
  to supported claims.
- Added `compile-operators` as a debug/development CLI phase and documented the
  formal release acceptance checklist.
- Kept biological project outputs bounded as computational proposals and
  experimental-design support, not wet-lab validation.
