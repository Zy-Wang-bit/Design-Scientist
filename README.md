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

Framework R&D is the method-development loop. It should produce literature-backed
method modules, a registry, replay benchmarks, a scientist journal, a method
report, and a validation report before any method is treated as usable.
`run-scientist` is the recommended full-chain CLI path; staged commands are debug/development entry points for inspecting one phase at a time.

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

`run-scientist` writes `runs/<run_id>/scientist_journal.json`,
`runs/<run_id>/benchmark_results.csv`, `runs/<run_id>/ablation_results.csv`, and
`runs/<run_id>/method_report.md`. Validate that run with:

```bash
python -m design_scientist.framework_validation /tmp/ds_product --run-id <run_id>
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

`run-scientist` is the recommended full-chain CLI path. It runs literature
discovery, method extraction, method-node development, synthetic replay ranking,
report generation, and reviewable artifact wiring. A successful run should
leave:

- Literature trace and ranking artifacts: `framework/literature_search_plan.json`,
  `framework/literature_search_trace.json`, `framework/citation_graph.json`,
  `framework/paper_scores.csv`, `framework/paper_cards.json`, and
  `framework/literature_map.md`.
- Run-level framework artifacts: `runs/<run_id>/scientist_journal.json`,
  `runs/<run_id>/stage_progress.json`, `runs/<run_id>/route_tree.json`,
  `runs/<run_id>/benchmark_results.csv`, `runs/<run_id>/benchmark_summary.csv`,
  `runs/<run_id>/ablation_results.csv`, and `runs/<run_id>/method_report.md`.
- Selected method-node artifacts: `proposal.json`, `novelty_report.json`,
  `candidate_policy.json`, `benchmark_metrics.json`, `validation_report.json`,
  `design_space.json`, `candidate_pool.csv`, `candidate_pool.jsonl`, and
  `candidate_pool_diagnostics.json`.

The selected method trace in `scientist_journal.json` links the proposal,
novelty report, benchmark metrics, literature-backed route, and candidate-pool
artifacts used for review. Candidate pools are emitted in both CSV and JSONL so
spreadsheet review and machine replay can use the same contract.

No-baseline invention mode is used for literature-gap method nodes. In that
mode the node should invent from recorded literature gaps rather than declaring
a simple baseline family as its source, and `review-framework` expects
non-clone novelty evidence plus the literature gap/rationale in `proposal.json`.

Staged framework commands are debug/development entry points:

```bash
design-scientist literature-search /tmp/ds_product
design-scientist extract-methods /tmp/ds_product
design-scientist develop-method /tmp/ds_product --run-id debug_method_run
design-scientist benchmark-methods /tmp/ds_product --run-id baseline_synthetic_replay_seed_1729
python -m design_scientist.method_report /tmp/ds_product --run-id debug_method_run
design-scientist review-framework /tmp/ds_product --run-id debug_method_run
```

`benchmark-methods` writes baseline-only replay artifacts. When no `--run-id`
is supplied it uses `baseline_synthetic_replay_seed_1729`; it refuses to write
into an existing scientist run directory.

Programmatic entry points:

```python
from design_scientist.framework_validation import review_framework_run
from design_scientist.method_report import write_method_report

write_method_report("/tmp/ds_product", run_id="<run_id>")
report = review_framework_run("/tmp/ds_product", run_id="<run_id>")
```

`review_framework_run` checks framework spec, literature queries, paper cards,
method modules, algorithm spec, method registry, Research OS files, reference
audit outputs, literature plan/trace/citation graph/paper scores, benchmark
results, optional benchmark summary, ablation results, `scientist_journal.json`,
selected node artifacts including v2 `proposal.json` and `novelty_report.json`
when applicable, and `method_report.md`.
