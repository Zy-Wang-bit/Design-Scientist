"""Core typed artifacts for Design Scientist.

The classes in this module are intentionally plain dataclasses so generated
project artifacts can stay JSON/YAML friendly and easy to inspect.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Literal


EvidenceTier = Literal[
    "primary_matched",
    "secondary_near_matched",
    "descriptive",
    "model_derived",
]

CandidateCategory = Literal[
    "champion",
    "champion_contrast",
    "interaction_square",
    "lattice_repair",
    "control",
    "repeat",
]

FeasibilityStatus = Literal["feasible", "infeasible", "needs_review"]


def to_plain_data(value: Any) -> Any:
    """Convert dataclass/path containers into JSON-safe primitives."""
    if is_dataclass(value):
        return {k: to_plain_data(v) for k, v in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_plain_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_plain_data(v) for v in value]
    return value


@dataclass
class ProjectSpec:
    project_id: str
    name: str
    goal: str
    mode: str = "project_execution"
    target_systems: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class DataTableSpec:
    name: str
    path: str
    role: str
    required: bool = True
    allow_raw_access: bool = False
    measurement_types: list[str] = field(default_factory=list)
    provenance_columns: list[str] = field(default_factory=list)


@dataclass
class DataContract:
    version: str
    project_id: str
    raw_access_policy: str
    allowed_tables: list[DataTableSpec] = field(default_factory=list)
    forbidden_claims: list[str] = field(default_factory=list)
    forbidden_table_patterns: list[str] = field(default_factory=list)


@dataclass
class Estimand:
    estimand_id: str
    name: str
    description: str
    endpoint: str
    evidence_tier: EvidenceTier
    success_direction: str
    thresholds: dict[str, float] = field(default_factory=dict)


@dataclass
class EstimandSet:
    project_id: str
    primary: list[Estimand] = field(default_factory=list)
    secondary: list[Estimand] = field(default_factory=list)
    guardrails: list[Estimand] = field(default_factory=list)


@dataclass
class EvidenceCard:
    evidence_id: str
    claim: str
    evidence_tier: EvidenceTier
    source_tables: list[str] = field(default_factory=list)
    source_rows: list[str] = field(default_factory=list)
    effect: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    actionability: str = "unresolved"
    tags: list[str] = field(default_factory=list)


@dataclass
class DesignState:
    project_id: str
    data_version: str
    current_round: int = 0
    row_counts: dict[str, int] = field(default_factory=dict)
    system_roles: dict[str, str] = field(default_factory=dict)
    known_champions: list[dict[str, Any]] = field(default_factory=list)
    module_status: list[dict[str, Any]] = field(default_factory=list)
    unresolved_edges: list[dict[str, Any]] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    policy_history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DesignOperatorSpec:
    operator_id: str
    category: CandidateCategory
    description: str = ""
    required_endpoints: list[str] = field(default_factory=list)
    default_cost: float = 1.0
    max_generated: int | None = None
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass
class DesignSpace:
    design_space_id: str
    project_id: str
    target_systems: list[str] = field(default_factory=list)
    modules: list[str] = field(default_factory=list)
    backgrounds: list[str] = field(default_factory=list)
    operators: list[DesignOperatorSpec] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    evidence_refs: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateLineage:
    operator: str
    operator_args: dict[str, Any] = field(default_factory=dict)
    parent_ids: list[str] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)


@dataclass
class Candidate:
    candidate_id: str
    operator: str
    category: CandidateCategory
    target_system: str
    background: str | None = None
    target_background: str | None = None
    design_context: dict[str, Any] = field(default_factory=dict)
    modules: list[str] = field(default_factory=list)
    rationale: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    score_components: dict[str, float] = field(default_factory=dict)
    risk_flags: list[str] = field(default_factory=list)
    required_measurements: list[str] = field(default_factory=list)
    variant_id: str | None = None
    design_space_id: str | None = None
    operator_args: dict[str, Any] = field(default_factory=dict)
    lineage: CandidateLineage | None = None
    parent_ids: list[str] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)
    feasibility_status: FeasibilityStatus = "feasible"
    feasibility_reasons: list[str] = field(default_factory=list)
    cost: float = 1.0
    required_endpoints: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.variant_id is None:
            self.variant_id = self.candidate_id
        if self.target_background is None and self.background:
            self.target_background = self.background
        if not self.design_context:
            self.design_context = {
                key: value
                for key, value in {
                    "target_system": self.target_system,
                    "target_background": self.target_background,
                }.items()
                if value
            }
        else:
            self.design_context.setdefault("target_system", self.target_system)
            if self.target_background:
                self.design_context.setdefault("target_background", self.target_background)
        if not self.required_endpoints and self.required_measurements:
            self.required_endpoints = list(self.required_measurements)
        if not self.source_refs and self.evidence_refs:
            self.source_refs = list(self.evidence_refs)
        if not self.parent_ids and self.background:
            self.parent_ids = [self.background]
        if self.lineage is None:
            self.lineage = CandidateLineage(
                operator=self.operator,
                operator_args=dict(self.operator_args),
                parent_ids=list(self.parent_ids),
                source_refs=list(self.source_refs),
            )


@dataclass
class CandidatePool:
    candidate_pool_id: str
    design_space_id: str
    strategy: str
    candidates: list[Candidate] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class PanelRecommendation:
    recommendation_id: str
    design_space_id: str
    strategy: str
    candidate_ids: list[str] = field(default_factory=list)
    rationale: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class AcquisitionPolicy:
    name: str
    description: str
    weights: dict[str, float] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass
class JournalNode:
    node_id: str
    parent_id: str | None
    stage: str
    workspace: str
    status: str = "created"
    score: float = 0.0
    artifacts: dict[str, str] = field(default_factory=dict)
    validation_summary: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


@dataclass
class DesignJournal:
    run_id: str
    project_id: str
    nodes: list[JournalNode] = field(default_factory=list)
    selected_node_id: str | None = None


@dataclass
class FrameworkSpec:
    framework_id: str
    domain: str
    version: str = "0.1"
    objective: str = "Develop and evaluate reusable scientific design methods."
    artifact_root: str = "."
    literature_queries_path: str = "framework/literature_queries.yaml"
    cache_dir: str = "framework/cache"
    benchmark_dir: str = "benchmarks/synthetic_replay"
    runs_dir: str = "runs"
    notes: list[str] = field(default_factory=list)


@dataclass
class LiteratureQuery:
    query_id: str
    domain: str
    query: str
    purpose: str = "method_discovery"
    sources: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)
    status: str = "planned"


@dataclass
class PaperCard:
    paper_id: str
    title: str
    citation: str = ""
    url: str | None = None
    year: int | None = None
    venue: str | None = None
    problem: str = ""
    method_summary: str = ""
    reusable_modules: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    evaluation: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    relevance: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class MethodModule:
    module_id: str
    name: str
    purpose: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)
    source_papers: list[str] = field(default_factory=list)
    implementation_status: str = "proposed"


@dataclass
class AlgorithmSpec:
    algorithm_id: str
    name: str
    objective: str
    modules: list[MethodModule] = field(default_factory=list)
    pseudocode: list[str] = field(default_factory=list)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    baselines: list[str] = field(default_factory=list)
    evaluation_plan: dict[str, Any] = field(default_factory=dict)
    status: str = "draft"


@dataclass
class StrategyNode:
    node_id: str
    parent_id: str | None
    strategy: str
    algorithm_id: str | None = None
    status: str = "created"
    rationale: str = ""
    score: float | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    children: list[str] = field(default_factory=list)


@dataclass
class SyntheticReplayConfig:
    replay_id: str
    domain: str
    dataset_refs: list[str] = field(default_factory=list)
    horizon: int = 1
    batch_size: int = 0
    budget: int | None = None
    metrics: list[str] = field(default_factory=list)
    baselines: list[str] = field(default_factory=list)
    random_seed: int = 0
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    benchmark_id: str
    algorithm_id: str
    config_id: str
    metrics: dict[str, float] = field(default_factory=dict)
    baseline_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    summary: str = ""
    passed: bool | None = None


@dataclass
class ScientistJournal:
    journal_id: str
    framework_id: str
    domain: str
    strategy_nodes: list[StrategyNode] = field(default_factory=list)
    paper_cards: list[PaperCard] = field(default_factory=list)
    algorithms: list[AlgorithmSpec] = field(default_factory=list)
    benchmark_results: list[BenchmarkResult] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    selected_node_id: str | None = None


@dataclass
class ValidationFinding:
    severity: Literal["info", "warning", "error"]
    code: str
    message: str
    artifact: str | None = None


@dataclass
class ValidationReport:
    project_id: str
    run_id: str | None
    valid: bool
    findings: list[ValidationFinding] = field(default_factory=list)


@dataclass
class ModelRequest:
    purpose: str
    messages: list[dict[str, str]]
    response_schema: dict[str, Any] | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    context_refs: list[str] = field(default_factory=list)


@dataclass
class ModelResponse:
    text: str
    structured: dict[str, Any] | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    raw_response: Any | None = None
    transcript_id: str | None = None


@dataclass
class WorkspaceAgentTask:
    objective: str
    workspace: str
    mode: Literal["read_only_plan", "patch", "verify"]
    allowed_paths: list[str] = field(default_factory=list)
    write_policy: str = "read_only"
    commands_allowed: list[str] = field(default_factory=list)
    output_schema: dict[str, Any] | None = None


@dataclass
class WorkspaceAgentResult:
    summary: str
    structured: dict[str, Any] | None = None
    files_touched: list[str] = field(default_factory=list)
    commands_run: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    transcript_id: str | None = None
    returncode: int = 0
