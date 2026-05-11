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
    "framework/research_gap_matrix.csv",
    "framework/method_hypotheses.md",
    "framework/policy_registry.yaml",
    "framework/evaluation_protocol.md",
    "framework/benchmark_results.csv",
    "framework/ablation_results.csv",
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
