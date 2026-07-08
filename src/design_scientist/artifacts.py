"""Required artifact names and checks."""

from __future__ import annotations

from pathlib import Path


PROJECT_ARTIFACTS = (
    "project.yaml",
    "data_contract.yaml",
    "estimands.yaml",
    "state/design_state.json",
    "state/evidence_cards.json",
)

FRAMEWORK_DIR = "framework"
FRAMEWORK_CACHE_DIR = "framework/cache"
FRAMEWORK_SPEC = "framework/framework_spec.yaml"
LITERATURE_QUERIES = "framework/literature_queries.yaml"
BENCHMARKS_DIR = "benchmarks"
SYNTHETIC_REPLAY_DIR = "benchmarks/synthetic_replay"
RUNS_DIR = "runs"
NODES_DIR = "nodes"

FRAMEWORK_BOOTSTRAP_ARTIFACTS = (
    FRAMEWORK_SPEC,
    LITERATURE_QUERIES,
)

FRAMEWORK_BOOTSTRAP_DIRS = (
    FRAMEWORK_CACHE_DIR,
    SYNTHETIC_REPLAY_DIR,
    RUNS_DIR,
)

LEGACY_FRAMEWORK_ARTIFACTS = (
    *FRAMEWORK_BOOTSTRAP_ARTIFACTS,
    "framework/paper_cards.json",
    "framework/method_modules.json",
    "framework/literature_map.md",
    "framework/research_gap_matrix.csv",
    "framework/method_hypotheses.md",
    "framework/algorithm_spec.md",
    "framework/method_registry.yaml",
)

V3_LITERATURE_CORPUS = "framework/literature_corpus.jsonl"
V3_LITERATURE_READING_TRACE = "framework/literature_reading_trace.json"
V3_MECHANISM_CARDS = "framework/mechanism_cards.json"
V3_MECHANISM_LIBRARY = "framework/mechanism_library.json"
V3_MECHANISM_GAP_MATRIX = "framework/mechanism_gap_matrix.csv"
V3_OPERATOR_SPECS = "framework/operator_specs.json"
V3_OPERATOR_GAP_MATRIX = "framework/operator_gap_matrix.csv"
V3_OPERATOR_EVIDENCE_MAP = "framework/operator_evidence_map.json"
V3_OPERATOR_NEGATIVE_CONTROLS = "framework/operator_negative_controls.json"
V3_REFERENCE_DATA_SOURCES = "reference_data_sources.json"

V3_MECHANISM_SPEC = "mechanism_spec.json"
V3_MECHANISM_IMPLEMENTATION = "mechanism.py"
V3_MECHANISM_PROPOSAL = "proposal.json"
V3_ABLATION_PLAN = "ablation_plan.json"
V3_STRESS_TEST_PLAN = "stress_test_plan.json"
V3_MECHANISM_METRICS = "mechanism_metrics.json"
V3_MECHANISM_VALIDATION_REPORT = "validation_report.json"
V3_OPERATOR_TO_CODE_TRACE = "operator_to_code_trace.json"

V4_CLAIM_LEDGER = "framework/claim_ledger.jsonl"
V4_EVIDENCE_LEDGER = "framework/evidence_ledger.jsonl"
V4_MECHANISM_LEDGER = "framework/mechanism_ledger.jsonl"
V4_RESEARCH_HARNESS_SUMMARY = "framework/research_harness_summary.json"
V4_MECHANISM_CARDS = "framework/mechanism_cards_v4.json"
V4_MECHANISM_LIBRARY = "framework/mechanism_library_v4.json"
V4_MECHANISM_GAP_MATRIX = "framework/mechanism_gap_matrix_v4.csv"
V4_LITERATURE_MINE_TRACE = "framework/literature_mine_trace_v4.json"
V4_MECHANISM_GRAPH = "framework/mechanism_graph_v4.json"
V4_NOVELTY_AUDIT = "novelty_audit.json"
V4_VERIFICATION_LADDER = "verification_ladder.json"

V3_FRAMEWORK_ARTIFACTS = (
    *FRAMEWORK_BOOTSTRAP_ARTIFACTS,
    V3_LITERATURE_CORPUS,
    V3_LITERATURE_READING_TRACE,
    V3_MECHANISM_CARDS,
    V3_MECHANISM_LIBRARY,
    V3_MECHANISM_GAP_MATRIX,
    V3_OPERATOR_SPECS,
    V3_OPERATOR_GAP_MATRIX,
    V3_OPERATOR_EVIDENCE_MAP,
    V3_OPERATOR_NEGATIVE_CONTROLS,
)

FRAMEWORK_ARTIFACTS = V3_FRAMEWORK_ARTIFACTS

V3_MECHANISM_NODE_ARTIFACTS = (
    V3_MECHANISM_SPEC,
    V3_MECHANISM_IMPLEMENTATION,
    V3_MECHANISM_PROPOSAL,
    V3_ABLATION_PLAN,
    V3_STRESS_TEST_PLAN,
    V3_MECHANISM_METRICS,
    V3_MECHANISM_VALIDATION_REPORT,
    V3_OPERATOR_TO_CODE_TRACE,
)

V3_STAGE_SEQUENCE = (
    "literature_retrieval",
    "fulltext_reading",
    "mechanism_extraction",
    "mechanism_ideation",
    "mechanism_implementation",
    "debug_repair",
    "ablation_stress",
    "selection_report",
)

LEGACY_PROJECT_EXECUTION_ARTIFACTS = (
    "framework/legacy_seed_paper_cards.json",
    "framework/policy_registry.yaml",
)

FRAMEWORK_STATE_ARTIFACTS = (
    "state/hypothesis_board.json",
    "state/hypothesis_board.md",
    "state/project_history.json",
)

FRAMEWORK_VALIDATION_ARTIFACTS = FRAMEWORK_ARTIFACTS + FRAMEWORK_STATE_ARTIFACTS

RUN_ARTIFACTS = (
    "candidate_pool.csv",
    "panel_recommendation.csv",
    "policy_comparison.csv",
    "policy_metrics.json",
    "validation_report.json",
    "decision_report.md",
    "human_review_packet.md",
)

NODE_ARTIFACTS = (
    "candidate_pool.csv",
    "panel_recommendation.csv",
    "policy_comparison.csv",
    "policy_metrics.json",
    "validation_report.json",
)


def missing_artifacts(root: str | Path, artifacts: tuple[str, ...]) -> list[str]:
    base = Path(root).expanduser().resolve()
    return [artifact for artifact in artifacts if not (base / artifact).exists()]


def has_complete_project(root: str | Path) -> bool:
    return not missing_artifacts(root, PROJECT_ARTIFACTS)
