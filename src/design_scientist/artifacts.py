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

FRAMEWORK_ARTIFACTS = (
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

V3_MECHANISM_SPEC = "mechanism_spec.json"
V3_MECHANISM_IMPLEMENTATION = "mechanism.py"
V3_MECHANISM_PROPOSAL = "proposal.json"
V3_ABLATION_PLAN = "ablation_plan.json"
V3_STRESS_TEST_PLAN = "stress_test_plan.json"
V3_MECHANISM_METRICS = "mechanism_metrics.json"
V3_MECHANISM_VALIDATION_REPORT = "validation_report.json"

V3_FRAMEWORK_ARTIFACTS = (
    *FRAMEWORK_BOOTSTRAP_ARTIFACTS,
    V3_LITERATURE_CORPUS,
    V3_LITERATURE_READING_TRACE,
    V3_MECHANISM_CARDS,
    V3_MECHANISM_LIBRARY,
    V3_MECHANISM_GAP_MATRIX,
)

V3_MECHANISM_NODE_ARTIFACTS = (
    V3_MECHANISM_SPEC,
    V3_MECHANISM_IMPLEMENTATION,
    V3_MECHANISM_PROPOSAL,
    V3_ABLATION_PLAN,
    V3_STRESS_TEST_PLAN,
    V3_MECHANISM_METRICS,
    V3_MECHANISM_VALIDATION_REPORT,
)

V3_STAGE_SEQUENCE = (
    "read_literature",
    "extract_mechanisms",
    "develop_mechanisms",
    "stress_test_mechanisms",
    "ablate_mechanisms",
    "validate_mechanisms",
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
