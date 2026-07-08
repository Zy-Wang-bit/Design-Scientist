# Design Scientist

Local-first framework for data-grounded, multi-round variant design. The project
borrows execution, journal, and tree-search ideas from AI-Scientist-v2, but the
scientific core is a persistent design loop:

```text
standardized data -> evidence update -> design state update -> candidate generation
-> panel selection -> validation -> new data merge
```

The default productization check is now framework-first synthetic replay; real
biological projects should be initialized as new project directories when their
data contracts are ready.

## Install

From this checkout:

```bash
python -m pip install -e ".[dev]"
```

When running without installation, prefix commands with `PYTHONPATH=src`.

## Codex Skill Installation

The Codex skill is packaged in `skills/design-scientist/`. Install it on another
machine after cloning this repository:

```bash
mkdir -p ~/.codex/skills
rm -rf ~/.codex/skills/design-scientist
cp -R skills/design-scientist ~/.codex/skills/design-scientist
```

If the machine does not already have the repository checkout, clone first:

```bash
mkdir -p ~/Research
git clone https://github.com/Zy-Wang-bit/Design-Scientist.git ~/Research/Design-Scientist
cd ~/Research/Design-Scientist
uv sync
mkdir -p ~/.codex/skills
rm -rf ~/.codex/skills/design-scientist
cp -R skills/design-scientist ~/.codex/skills/design-scientist
```

Restart Codex after copying the skill so the new metadata is loaded. The skill
does not contain private anti-HBsAg project data; keep project-specific context
in project directories or local-only skill references.

## Environment Configuration

Live literature search can use these environment variables:

- `S2_API_KEY`: optional Semantic Scholar Graph API key. Without it, live
  Semantic Scholar queries are skipped; offline fixture runs still work.
- `NCBI_EMAIL`: optional contact email for PubMed/NCBI requests.
- `NCBI_API_KEY`: optional NCBI API key for higher-rate PubMed requests.
- `NCBI_TOOL`: optional NCBI tool label; defaults to `design_scientist`.
- `DESIGN_SCIENTIST_ARXIV_USER_AGENT`: optional arXiv User-Agent override.
  arXiv does not use an API key in this framework. The live adapter follows the
  public API rate-limit posture with serialized requests, retry/backoff, and
  traceable skip events after a live arXiv error.
- `DESIGN_SCIENTIST_LITERATURE_FIXTURES`: directory of offline literature
  fixtures used by `--offline-fixtures` tests and smoke runs.

Do not put real key values in README, tests, checked-in configs, or shell
history snippets.

To configure the Semantic Scholar key from a local key file:

```bash
python scripts/configure_s2_api_key.py --input ~/Downloads/S2.txt
```

The script accepts either a raw key or an `S2_API_KEY=...` assignment, writes a
managed marker block, and does not print the key. Add `--smoke` to make one
minimal Semantic Scholar request after writing the shell config:

```bash
python scripts/configure_s2_api_key.py --input ~/Downloads/S2.txt --smoke
```

By default the script updates `~/.zshrc`. On some machines that file may be
root-owned or otherwise not writable by the current user. In that case, using
`~/.zshenv` as the target is acceptable:

```bash
python scripts/configure_s2_api_key.py --input ~/Downloads/S2.txt --zshrc ~/.zshenv
```

## Framework R&D Workflow

Framework R&D is the method-development loop. The current default is V4: a
Research Harness around the MechanismSpec Kernel plus literature mining. The
literature engine builds a
full-text corpus, reading trace, mechanism cards, mechanism library, and gap
matrix, then compiles those cards into AlgorithmOperatorSpec artifacts. The
kernel turns selected operators into executable lifecycle nodes with explicit
components, claims, stress tests, ablations, operator-to-code traces, and
benchmark/claim gates. V4 adds append-only claim, evidence, and mechanism
ledgers plus novelty and verification gates, so weak method claims fail review
instead of being hidden in prose.
`run-scientist` is the V4 full-chain path. Staged commands are
debug/development entry points for inspecting one V3 phase at a time. A normal
successful run also generates and validates a paper bundle; `--skip-paper` is
only for local debugging and cannot produce a review-complete scientist run.

```bash
design-scientist init-framework /tmp/ds_product --domain "protein_variant_design"
design-scientist run-scientist /tmp/ds_product
design-scientist review-framework /tmp/ds_product
```

For local smoke tests or deterministic integration runs, use offline fixtures:

```bash
DESIGN_SCIENTIST_LITERATURE_FIXTURES=tests/fixtures/literature \
  design-scientist run-scientist /tmp/ds_product --offline-fixtures
```

V4 `run-scientist` writes `runs/<run_id>/scientist_journal.json`,
`runs/<run_id>/stage_progress.json`, `runs/<run_id>/route_tree.json`,
`runs/<run_id>/reference_data_sources.json`,
`runs/<run_id>/mechanism_benchmark_results.csv`,
`runs/<run_id>/mechanism_benchmark_summary.csv`,
`runs/<run_id>/mechanism_ablation_results.csv`,
`runs/<run_id>/benchmark_saturation.json`, `runs/<run_id>/claim_cap.json`,
reviewer verdict artifacts `runs/<run_id>/novelty_review.json`,
`runs/<run_id>/baseline_audit.json`, `runs/<run_id>/experiment_review.json`,
`runs/<run_id>/biology_review.json`,
`runs/<run_id>/paper_contribution_review.json`, and
`runs/<run_id>/method_report.md`. It also writes paper artifacts
`runs/<run_id>/paper/short_paper.md`,
`runs/<run_id>/paper/references.bib`,
`runs/<run_id>/paper/results_summary.json`,
`runs/<run_id>/paper/claim_evidence_map.json`, and
`runs/<run_id>/paper/paper_readiness_report.json`. It also writes
`framework/claim_ledger.jsonl`, `framework/evidence_ledger.jsonl`,
`framework/mechanism_ledger.jsonl`,
`framework/research_harness_summary.json`,
`framework/mechanism_cards_v4.json`,
`framework/mechanism_library_v4.json`,
`framework/mechanism_gap_matrix_v4.csv`,
`framework/literature_mine_trace_v4.json`,
`framework/mechanism_graph_v4.json`,
`runs/<run_id>/novelty_audit.json`, and
`runs/<run_id>/verification_ladder.json`. Validate that run with:

```bash
design-scientist review-framework /tmp/ds_product --run-id <run_id>
```

The report is written to `runs/<run_id>/method_report.md`. Validation returns a
non-zero exit code if any critical artifact is missing or malformed.
`init-framework` also creates Research OS control files:
`framework/quest.yaml`, `framework/research_map.json`,
`framework/findings_memory.jsonl`, and `framework/failure_memory.jsonl`.
`audit-references` writes `framework/reference_audit.md` and
`framework/reference_components.json`; missing reference audit artifacts are
reported as provenance warnings unless a partial audit output indicates that the
audit was requested and did not complete cleanly.

`run-scientist` is the recommended full-chain CLI path. It runs
Literature Engine V3 reading, mechanism extraction, MechanismSpec lifecycle
node development, V4 mechanism mining, mechanism replay ranking, research
harness ledgering, novelty/verification gates, report generation, and reviewable
paper-bundle generation. A successful run should
leave:

- Literature trace and ranking artifacts: `framework/literature_search_plan.json`,
  `framework/literature_search_trace.json`, `framework/citation_graph.json`,
  `framework/paper_scores.csv`, `framework/paper_cards.json`, and
  V3 literature artifacts `framework/literature_corpus.jsonl`,
  `framework/literature_reading_trace.json`, `framework/mechanism_cards.json`,
  `framework/mechanism_library.json`, and
  `framework/mechanism_gap_matrix.csv`, plus operator artifacts
  `framework/operator_specs.json`, `framework/operator_gap_matrix.csv`,
  `framework/operator_evidence_map.json`, and
  `framework/operator_negative_controls.json`, plus V4 artifacts
  `framework/mechanism_cards_v4.json`,
  `framework/mechanism_library_v4.json`,
  `framework/mechanism_gap_matrix_v4.csv`,
  `framework/literature_mine_trace_v4.json`, and
  `framework/mechanism_graph_v4.json`.
- Run-level V3 framework artifacts: `runs/<run_id>/scientist_journal.json`,
  `runs/<run_id>/stage_progress.json`, `runs/<run_id>/route_tree.json`,
  `runs/<run_id>/reference_data_sources.json`,
  `runs/<run_id>/mechanism_benchmark_results.csv`,
  `runs/<run_id>/mechanism_benchmark_summary.csv`,
  `runs/<run_id>/mechanism_ablation_results.csv`,
  `runs/<run_id>/benchmark_saturation.json`, `runs/<run_id>/claim_cap.json`,
  reviewer verdict artifacts `novelty_review.json`, `baseline_audit.json`,
  `experiment_review.json`, `biology_review.json`, and
  `paper_contribution_review.json`, and
  `runs/<run_id>/method_report.md`, plus paper artifacts under
  `runs/<run_id>/paper/` and V4 run gates
  `runs/<run_id>/novelty_audit.json` and
  `runs/<run_id>/verification_ladder.json`.
- Selected V3 mechanism-node artifacts: `mechanism_spec.json`,
  `mechanism.py`, `proposal.json`, `ablation_plan.json`,
  `stress_test_plan.json`, `mechanism_metrics.json`,
  `validation_report.json`, and `operator_to_code_trace.json`.

The selected mechanism trace in `scientist_journal.json` links the proposal,
MechanismSpec, benchmark metrics, literature-backed route, stress tests,
ablation plan, operator-to-code trace, benchmark saturation, claim cap,
reviewer verdicts, and validation report used for review.

`reference_data_sources.json` is the run-level source-of-truth manifest for
claims. It ties literature, operator, selected-node, benchmark, claim-gate, and
review artifacts to the claims they support, so reports and paper bundles do not
silently drift away from the evidence chain.

Staged framework commands are debug/development entry points:

```bash
design-scientist literature-search /tmp/ds_product
design-scientist read-literature /tmp/ds_product
design-scientist extract-mechanisms /tmp/ds_product
design-scientist mine-literature /tmp/ds_product
design-scientist compile-operators /tmp/ds_product
design-scientist run-scientist /tmp/ds_product --max-papers 60 --nodes 4 --rounds 3
design-scientist review-framework /tmp/ds_product
```

The operator compilation phase is explicit: `extract-mechanisms` writes
`framework/mechanism_cards.json` and automatically compiles
`framework/operator_specs.json`, `framework/operator_gap_matrix.csv`,
`framework/operator_evidence_map.json`, and
`framework/operator_negative_controls.json`. Use `compile-operators <root>` only
when `framework/mechanism_cards.json` already exists and you want to regenerate
those operator artifacts without rerunning literature reading or mechanism
extraction.

Programmatic entry points:

```python
from design_scientist.framework_validation import review_framework_run
from design_scientist.method_report import write_method_report

write_method_report("/tmp/ds_product", run_id="<run_id>")
report = review_framework_run("/tmp/ds_product", run_id="<run_id>")
```

`review_framework_run` detects V4 harness runs from `scientist_journal.json`
`research_harness_version: "v4"`. It still checks the underlying V3 kernel
artifacts: Literature Engine V3 artifacts,
AlgorithmOperatorSpec artifacts, MechanismSpec node artifacts, required replay
baselines, selected eligibility, architecture-clone blocking, positive key
ablation delta, false-claim-rate guardrails, claim cap, benchmark saturation,
reviewer verdicts, `reference_data_sources.json`, and `method_report.md`. V4
runs additionally require research harness ledgers, V4 mechanism mining outputs,
`novelty_audit.json`, and `verification_ladder.json`.

## Formal Release Checklist

Before tagging a formal framework release, run the acceptance sequence from a
clean checkout:

```bash
python -m pip install -e ".[dev]"
pytest -q

tmp="$(mktemp -d)"
design-scientist init-framework "$tmp/ds_product" --domain protein_variant_design
DESIGN_SCIENTIST_LITERATURE_FIXTURES=tests/fixtures/literature \
  design-scientist run-scientist "$tmp/ds_product" \
  --offline-fixtures --max-papers 20 --nodes 3 --rounds 2
design-scientist review-framework "$tmp/ds_product"
```

Acceptance requires full tests passing, offline fixture full-chain execution,
`review-framework` returning zero errors, `method_report.md` showing operator
artifacts, research harness ledgers, novelty audit, verification ladder, claim
cap, benchmark saturation, and reviewer verdicts, adversarial bad-path tests
passing, and the paper bundle preserving the computational-only evidence
boundary without wet-lab or prospective-validation overclaims.

## Manuscript Bundle

`run-scientist` generates the conservative short-paper bundle by default. Use
this command only to rebuild an existing bundle after editing or inspecting
artifacts:

```bash
design-scientist generate-short-paper /tmp/ds_product --run-id <run_id>
```

The command writes `runs/<run_id>/paper/short_paper.md`,
`references.bib`, `results_summary.json`, `claim_evidence_map.json`,
`reproducibility.md`, `data_availability.md`, `code_availability.md`, and
`paper_readiness_report.json`. The paper layer labels evidence as
computational/synthetic replay evidence and does not claim wet-lab validation.
The readiness report checks required sections, citation keys, and local
artifact links before marking the bundle valid.

For algorithm-method papers, use the MCCBD benchmark path. This path treats
`evidence_calibrated_ucb` as a weighted-score baseline and evaluates
Mechanism-Calibrated Constrained Bayesian Design as the primary algorithm:

```bash
design-scientist run-algorithm-benchmark /tmp/ds_product \
  --run-id mccbd_algorithm --rounds 3 --budget 6
design-scientist generate-algorithm-paper /tmp/ds_product \
  --run-id mccbd_algorithm --algorithm mccbd
```

If standardized project data are available, add a retrospective masking case
study. This is a case study only; it is not prospective wet-lab proof.

```bash
design-scientist run-project-benchmark /tmp/ds_product \
  --run-id mccbd_project_case --budget 2 --folds kfold_5 \
  --mechanisms mccbd evidence_calibrated_ucb random_feasible fixed_mix greedy_observed
design-scientist generate-algorithm-paper /tmp/ds_product \
  --run-id mccbd_algorithm --algorithm mccbd
```

The algorithm-paper bundle writes `manuscript.md`, `references.bib`,
`algorithm_results_summary.json`, `algorithm_claim_evidence_map.json`,
`paper_readiness_report.json`, SVG figures, and CSV tables under
`runs/<run_id>/paper/`.
