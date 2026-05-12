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

Framework R&D is the method-development loop. V3 is organized as a
MechanismSpec Kernel plus Literature Engine V3: the literature engine builds a
full-text corpus, reading trace, mechanism cards, mechanism library, and gap
matrix; the kernel turns selected mechanisms into executable lifecycle nodes
with explicit components, claims, stress tests, ablations, and benchmark gates.
`run-scientist` is the V3 full-chain path. Staged commands are
debug/development entry points for inspecting one V3 phase at a time.

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

V3 `run-scientist` writes `runs/<run_id>/scientist_journal.json`,
`runs/<run_id>/stage_progress.json`, `runs/<run_id>/route_tree.json`,
`runs/<run_id>/mechanism_benchmark_results.csv`,
`runs/<run_id>/mechanism_benchmark_summary.csv`,
`runs/<run_id>/mechanism_ablation_results.csv`, and
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

`run-scientist` is the recommended full-chain CLI path. In V3 it runs
Literature Engine V3 reading, mechanism extraction, MechanismSpec lifecycle
node development, mechanism replay ranking, report generation, and reviewable
artifact wiring. A successful run should
leave:

- Literature trace and ranking artifacts: `framework/literature_search_plan.json`,
  `framework/literature_search_trace.json`, `framework/citation_graph.json`,
  `framework/paper_scores.csv`, `framework/paper_cards.json`, and
  V3 literature artifacts `framework/literature_corpus.jsonl`,
  `framework/literature_reading_trace.json`, `framework/mechanism_cards.json`,
  `framework/mechanism_library.json`, and
  `framework/mechanism_gap_matrix.csv`.
- Run-level V3 framework artifacts: `runs/<run_id>/scientist_journal.json`,
  `runs/<run_id>/stage_progress.json`, `runs/<run_id>/route_tree.json`,
  `runs/<run_id>/mechanism_benchmark_results.csv`,
  `runs/<run_id>/mechanism_benchmark_summary.csv`,
  `runs/<run_id>/mechanism_ablation_results.csv`, and
  `runs/<run_id>/method_report.md`.
- Selected V3 mechanism-node artifacts: `mechanism_spec.json`,
  `mechanism.py`, `proposal.json`, `ablation_plan.json`,
  `stress_test_plan.json`, `mechanism_metrics.json`, and
  `validation_report.json`.

The selected mechanism trace in `scientist_journal.json` links the proposal,
MechanismSpec, benchmark metrics, literature-backed route, stress tests,
ablation plan, and validation report used for review.

Staged framework commands are debug/development entry points:

```bash
design-scientist literature-search /tmp/ds_product
design-scientist read-literature /tmp/ds_product
design-scientist extract-mechanisms /tmp/ds_product
design-scientist run-scientist /tmp/ds_product --max-papers 60 --nodes 4 --rounds 3
design-scientist review-framework /tmp/ds_product
```

Programmatic entry points:

```python
from design_scientist.framework_validation import review_framework_run
from design_scientist.method_report import write_method_report

write_method_report("/tmp/ds_product", run_id="<run_id>")
report = review_framework_run("/tmp/ds_product", run_id="<run_id>")
```

`review_framework_run` detects V3 runs from `scientist_journal.json`
`version: "v3"`. For V3 it checks the Literature Engine V3 artifacts,
MechanismSpec node artifacts, required replay baselines, selected eligibility,
architecture-clone blocking, positive key ablation delta, false-claim-rate
guardrails, and `method_report.md`.
