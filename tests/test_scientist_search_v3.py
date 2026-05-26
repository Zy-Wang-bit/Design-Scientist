from __future__ import annotations

import csv
import json
from pathlib import Path
from textwrap import dedent

import pytest

from design_scientist.artifacts import V3_MECHANISM_NODE_ARTIFACTS
from design_scientist.cli import main
from design_scientist.framework import init_framework
from design_scientist.schemas import WorkspaceAgentResult, WorkspaceAgentTask
from design_scientist.scientist_search_v3 import SCIENTIST_V3_STAGES, run_scientist_v3


def test_offline_v3_run_writes_full_chain_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = _init_project(tmp_path, monkeypatch)

    result = run_scientist_v3(
        project,
        max_papers=5,
        nodes=2,
        rounds=1,
        offline_fixtures=True,
    )

    run_dir = project / "runs" / result["run_id"]
    journal_path = run_dir / "scientist_journal.json"
    stage_progress_path = run_dir / "stage_progress.json"
    route_tree_path = run_dir / "route_tree.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    stage_progress = json.loads(stage_progress_path.read_text(encoding="utf-8"))
    route_tree = json.loads(route_tree_path.read_text(encoding="utf-8"))

    assert result["journal_path"] == str(journal_path)
    assert result["selected_node"] is not None
    assert result["selected_node"]["node_id"].startswith("node_")
    assert journal["version"] == "v3"
    assert journal["stage_sequence"] == list(SCIENTIST_V3_STAGES)
    assert stage_progress["stage_sequence"] == list(SCIENTIST_V3_STAGES)
    assert [stage["stage"] for stage in stage_progress["stages"]] == list(SCIENTIST_V3_STAGES)
    assert route_tree["stage_sequence"] == list(SCIENTIST_V3_STAGES)
    assert journal["mechanism_library_path"] == str(project / "framework" / "mechanism_library.json")

    assert (project / "framework" / "paper_cards.json").exists()
    assert (project / "framework" / "literature_corpus.jsonl").exists()
    assert (project / "framework" / "literature_reading_trace.json").exists()
    assert (project / "framework" / "mechanism_cards.json").exists()
    assert (project / "framework" / "mechanism_library.json").exists()
    assert (project / "framework" / "mechanism_gap_matrix.csv").exists()
    assert Path(journal["benchmark"]["mechanism_benchmark_results_path"]).exists()
    assert Path(journal["benchmark"]["mechanism_benchmark_summary_path"]).exists()
    assert Path(journal["benchmark"]["mechanism_ablation_results_path"]).exists()

    with Path(journal["benchmark"]["mechanism_benchmark_summary_path"]).open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        summary_rows = list(csv.DictReader(handle))
    selected_row = next(row for row in summary_rows if row["mechanism"] == result["selected_node"]["mechanism"])
    assert selected_row["selected_eligible"] == "true"
    assert selected_row["mechanism"] not in {"random_feasible", "fixed_mix"}


def test_scientist_journal_records_project_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = _init_project(tmp_path, monkeypatch)
    (project / "state").mkdir(parents=True)
    (project / "state" / "design_state.json").write_text(
        json.dumps(
            {
                "project_id": "project_context_unit",
                "objective": "Choose a validation panel for generic variants.",
                "required_endpoints": ["potency", "specificity"],
                "row_counts": {"standardized_measurements": 11},
                "constraints": {"budget": 8, "required_controls": 2},
                "allowed_sources": ["standardized/measurements.csv"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_scientist_v3(
        project,
        max_papers=5,
        nodes=1,
        rounds=1,
        offline_fixtures=True,
    )

    journal = json.loads(Path(result["journal_path"]).read_text(encoding="utf-8"))
    project_context = journal["project_context"]
    assert project_context["objective"] == "Choose a validation panel for generic variants."
    assert project_context["endpoints"] == ["potency", "specificity"]
    assert project_context["observed_row_counts"] == {"standardized_measurements": 11}
    assert project_context["constraints"] == {"budget": 8, "required_controls": 2}
    assert project_context["allowed_sources"] == ["standardized/measurements.csv"]


def test_saturation_sidecar_writes_next_search_constraints(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = _init_project(tmp_path, monkeypatch)

    def fake_benchmark(
        root: str | Path,
        *,
        run_id: str,
        mechanisms: list[object],
        rounds: int,
    ) -> dict[str, object]:
        del rounds
        run_dir = Path(root) / "runs" / run_id
        generated = next(item["name"] for item in mechanisms if isinstance(item, dict))
        benchmark_path = run_dir / "mechanism_benchmark_results.csv"
        summary_path = run_dir / "mechanism_benchmark_summary.csv"
        ablation_path = run_dir / "mechanism_ablation_results.csv"
        saturation_path = run_dir / "benchmark_saturation.json"
        summary_rows = [
            {
                "mechanism": generated,
                "rank": "1",
                "selected_eligible": "true",
                "mean_best_feasible_utility": "0.50",
                "key_ablation_delta": "0.10",
                "false_claim_rate": "0.00",
            },
            {
                "mechanism": "random_feasible",
                "rank": "2",
                "selected_eligible": "false",
                "mean_best_feasible_utility": "0.40",
                "key_ablation_delta": "0.00",
                "false_claim_rate": "0.00",
            },
            {
                "mechanism": "fixed_mix",
                "rank": "3",
                "selected_eligible": "false",
                "mean_best_feasible_utility": "0.39",
                "key_ablation_delta": "0.00",
                "false_claim_rate": "0.00",
            },
        ]
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)
        benchmark_path.write_text("mechanism,round,best_feasible_utility\n", encoding="utf-8")
        ablation_path.write_text("mechanism,ablation,delta\n", encoding="utf-8")
        saturation_path.write_text(
            json.dumps(
                {
                    "verdict": "reject",
                    "saturated": False,
                    "summary": "benchmark_not_saturated",
                    "blocked_claims": [
                        "Selected mechanism is not clearly above no_generation_boundary boundary.",
                        "Superiority and generative claims are blocked by false or unsupported claims.",
                    ],
                    "boundary_checks": [
                        {
                            "boundary": "no_generation_boundary",
                            "selected_delta": 0.0,
                            "near_boundary": True,
                        },
                        {
                            "boundary": "no_differentiation_boundary",
                            "selected_delta": 0.0,
                            "near_boundary": True,
                        },
                    ],
                    "unsupported_claims": ["unmatched superiority over non-generated boundary"],
                    "worlds": ["epistatic"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "run_id": run_id,
            "mechanism_benchmark_results_path": str(benchmark_path),
            "mechanism_benchmark_summary_path": str(summary_path),
            "mechanism_ablation_results_path": str(ablation_path),
            "benchmark_saturation_path": str(saturation_path),
            "summary": {"mechanism_rankings": summary_rows},
            "config": {"worlds": ["epistatic"]},
        }

    monkeypatch.setattr("design_scientist.scientist_search_v3.run_mechanism_benchmark", fake_benchmark)

    result = run_scientist_v3(
        project,
        max_papers=5,
        nodes=1,
        rounds=1,
        offline_fixtures=True,
    )

    journal = json.loads(Path(result["journal_path"]).read_text(encoding="utf-8"))
    stage_progress = json.loads(Path(result["stage_progress_path"]).read_text(encoding="utf-8"))
    route_tree = json.loads(Path(result["route_tree_path"]).read_text(encoding="utf-8"))

    constraints = journal["next_search_constraints"]
    assert constraints
    constraints_text = json.dumps(constraints, sort_keys=True)
    assert "lower_superiority_or_generative_claim" in constraints_text
    assert "matched_contrast_operator" in constraints_text
    assert "experimental_design_operator" in constraints_text
    assert "leave_family_out" in constraints_text
    assert "causal_matched_stress" in constraints_text
    assert journal["saturation_feedback"]["saturated"] is False
    assert stage_progress["next_search_constraints"] == constraints
    assert route_tree["next_search_constraints"] == constraints
    assert result["next_search_constraints"] == constraints


def test_generated_nodes_contain_v3_mechanism_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = _init_project(tmp_path, monkeypatch)

    result = run_scientist_v3(
        project,
        max_papers=5,
        nodes=1,
        rounds=1,
        offline_fixtures=True,
    )

    journal = json.loads(Path(result["journal_path"]).read_text(encoding="utf-8"))
    node = journal["nodes"][0]
    workspace = Path(node["workspace"])

    assert node["status"] == "completed"
    assert set(V3_MECHANISM_NODE_ARTIFACTS) <= {path.name for path in workspace.iterdir()}
    assert Path(node["artifacts"]["mechanism_spec"]).name == "mechanism_spec.json"
    assert Path(node["artifacts"]["mechanism"]).name == "mechanism.py"
    mechanism_spec = json.loads((workspace / "mechanism_spec.json").read_text(encoding="utf-8"))
    proposal = json.loads((workspace / "proposal.json").read_text(encoding="utf-8"))
    ablation_plan = json.loads((workspace / "ablation_plan.json").read_text(encoding="utf-8"))
    assert mechanism_spec["claims"]
    assert mechanism_spec["literature_basis"]
    assert mechanism_spec["stress_test_requirements"]
    assert mechanism_spec["operator_refs"]
    assert proposal["operator_refs"]
    assert any(
        mechanism_spec["operator_refs"][0] in ablation.get("removed_operator_ids", [])
        for ablation in ablation_plan["ablations"]
        if ablation.get("name") == "key_component_removed"
    )
    operator_trace = json.loads((workspace / "operator_to_code_trace.json").read_text(encoding="utf-8"))
    assert mechanism_spec["operator_refs"][0] in operator_trace["operator_to_code_trace"]


def test_codex_prompt_includes_operator_specs_contract_and_trace_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = _init_project(tmp_path, monkeypatch)
    backend = PromptCaptureV3Backend()

    run_scientist_v3(
        project,
        max_papers=5,
        nodes=1,
        rounds=1,
        use_codex=True,
        backend=backend,
        offline_fixtures=True,
    )

    assert len(backend.tasks) == 1
    objective = backend.tasks[0].objective
    write_policy = backend.tasks[0].write_policy
    assert "Operator specs:" in objective
    assert "operator_active_learning_acquisition" in objective
    assert "operator_refs" in objective
    assert "removed_operator_ids" in objective
    assert "operator_to_code_trace.json" in objective
    assert "operator_to_code_trace.json" in write_policy


def test_codex_node_without_operator_specs_is_not_selected(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = _init_project(tmp_path, monkeypatch)
    backend = MissingOperatorRefsV3Backend()

    result = run_scientist_v3(
        project,
        max_papers=5,
        nodes=1,
        rounds=1,
        use_codex=True,
        backend=backend,
        offline_fixtures=True,
    )

    node = result["nodes"][0]
    assert result["selected_node"] is None
    assert node["status"] == "failed"
    assert not node.get("selected")
    assert any("operator_refs" in reason or "operator_specs" in reason for reason in node["failure_reasons"])


def test_generated_lifecycle_claims_and_stress_worlds_come_from_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = _init_project(tmp_path, monkeypatch)
    backend = ArtifactStressClaimsBackend()

    result = run_scientist_v3(
        project,
        max_papers=5,
        nodes=1,
        rounds=1,
        use_codex=True,
        backend=backend,
        offline_fixtures=True,
    )

    journal = json.loads(Path(result["journal_path"]).read_text(encoding="utf-8"))
    assert journal["benchmark"]["config"]["worlds"] == [
        "confounded_transfer",
        "noisy_endpoint",
    ]


def test_offline_local_fallback_does_not_select_mechanism_aware_wrapper(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = _init_project(tmp_path, monkeypatch)

    result = run_scientist_v3(
        project,
        max_papers=5,
        nodes=3,
        rounds=1,
        offline_fixtures=True,
    )

    assert result["selected_node"] is not None
    selected = result["selected_node"]
    assert "mechanism_aware" not in selected["node_id"]
    assert "random_feasible" not in selected["node_id"]
    assert "fixed_mix" not in selected["node_id"]
    workspace = Path(selected["workspace"])
    proposal = json.loads((workspace / "proposal.json").read_text(encoding="utf-8"))
    assert "mechanism_aware" not in proposal.get("reused_components", [])
    assert "random_feasible" not in proposal.get("reused_components", [])
    assert "fixed_mix" not in proposal.get("reused_components", [])


def test_failing_v3_node_does_not_abort_and_appends_failure_memory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = _init_project(tmp_path, monkeypatch)
    backend = MixedV3Backend()

    result = run_scientist_v3(
        project,
        max_papers=5,
        nodes=2,
        rounds=1,
        use_codex=True,
        backend=backend,
        offline_fixtures=True,
    )

    journal = json.loads(Path(result["journal_path"]).read_text(encoding="utf-8"))
    failed = [node for node in journal["nodes"] if node["status"] == "failed"]
    completed = [node for node in journal["nodes"] if node["status"] == "completed"]
    memory_lines = [
        json.loads(line)
        for line in (project / "framework" / "failure_memory.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(backend.tasks) == 2
    assert failed and failed[0]["node_id"].startswith("node_01")
    assert completed and completed[0]["node_id"].startswith("node_02")
    assert journal["benchmark"]["mechanism_benchmark_results_path"]
    assert Path(journal["benchmark"]["mechanism_benchmark_results_path"]).exists()
    assert any(
        record["record_type"] == "failure"
        and record["component"] == "mechanism_node"
        and record["node_id"] == failed[0]["node_id"]
        for record in memory_lines
    )


def test_duplicate_generated_mechanism_names_fail_before_benchmark(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = _init_project(tmp_path, monkeypatch)
    backend = DuplicateNameV3Backend()

    result = run_scientist_v3(
        project,
        max_papers=5,
        nodes=2,
        rounds=1,
        use_codex=True,
        backend=backend,
        offline_fixtures=True,
    )

    journal = json.loads(Path(result["journal_path"]).read_text(encoding="utf-8"))
    duplicate_nodes = [
        node
        for node in journal["nodes"]
        if any("duplicate generated mechanism name" in reason for reason in node["failure_reasons"])
    ]
    assert len(duplicate_nodes) == 1
    assert duplicate_nodes[0]["status"] == "failed"
    assert "benchmark_metrics" not in duplicate_nodes[0]

    with Path(journal["benchmark"]["mechanism_benchmark_summary_path"]).open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        summary_rows = list(csv.DictReader(handle))
    generated_rows = [row for row in summary_rows if row["mechanism"] == "fake_backend_mechanism"]
    assert len(generated_rows) == 1


def test_all_failed_nodes_do_not_complete_implementation_or_ablation_stages(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = _init_project(tmp_path, monkeypatch)
    backend = AlwaysFailingV3Backend()

    result = run_scientist_v3(
        project,
        max_papers=5,
        nodes=2,
        rounds=1,
        use_codex=True,
        backend=backend,
        offline_fixtures=True,
    )

    stage_progress = json.loads(Path(result["stage_progress_path"]).read_text(encoding="utf-8"))
    stages = {stage["stage"]: stage for stage in stage_progress["stages"]}

    assert result["selected_node"] is None
    assert result["selected_node_id"] is None
    assert stages["mechanism_implementation"]["status"] == "failed"
    assert stages["ablation_stress"]["status"] == "failed"
    assert all(node["status"] == "failed" for node in result["nodes"])
    assert not any(node.get("selected") for node in result["nodes"])


def test_cli_run_scientist_uses_v3_and_rejects_legacy_flag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[str] = []

    def fake_report(root: str | Path, *, run_id: str) -> Path:
        return Path(root) / "runs" / run_id / "method_report.md"

    def fake_v3(root: str | Path, **kwargs) -> dict[str, object]:
        calls.append(f"v3:{Path(root).name}:{kwargs['nodes']}")
        return {
            "run_id": "v3_unit",
            "journal_path": str(Path(root) / "runs" / "v3_unit" / "scientist_journal.json"),
            "selected_node": {"node_id": "node_01_v3"},
        }

    monkeypatch.setattr("design_scientist.method_report.write_method_report", fake_report)
    monkeypatch.setattr("design_scientist.scientist_search_v3.run_scientist_v3", fake_v3)

    assert main(["run-scientist", str(tmp_path / "default"), "--nodes", "4"]) == 0
    with pytest.raises(SystemExit) as exc:
        main(["run-scientist", str(tmp_path / "legacy"), "--legacy-v2", "--nodes", "5"])

    assert exc.value.code == 2
    assert calls == ["v3:default:4"]


def test_cli_returns_failure_when_v3_selects_no_node(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_report(root: str | Path, *, run_id: str) -> Path:
        return Path(root) / "runs" / run_id / "method_report.md"

    def fake_v3(root: str | Path, **kwargs) -> dict[str, object]:
        return {
            "run_id": "v3_none",
            "journal_path": str(Path(root) / "runs" / "v3_none" / "scientist_journal.json"),
            "selected_node": None,
        }

    monkeypatch.setattr("design_scientist.method_report.write_method_report", fake_report)
    monkeypatch.setattr("design_scientist.scientist_search_v3.run_scientist_v3", fake_v3)

    assert main(["run-scientist", str(tmp_path / "project")]) == 1


class MixedV3Backend:
    def __init__(self) -> None:
        self.tasks: list[WorkspaceAgentTask] = []

    def run_task(self, task: WorkspaceAgentTask) -> WorkspaceAgentResult:
        self.tasks.append(task)
        workspace = Path(task.workspace)
        assert task.allowed_paths == [str(workspace.resolve())]
        if workspace.name.startswith("node_01"):
            _write_invalid_mechanism(workspace)
        else:
            _write_valid_mechanism(workspace)
        return WorkspaceAgentResult(
            summary="fake V3 backend completed",
            structured={
                "summary": "fake V3 backend completed",
                "mechanism_name": workspace.name,
                "files_written": ["mechanism.py"],
                "contract_notes": [],
            },
            returncode=0,
        )


class DuplicateNameV3Backend:
    def __init__(self) -> None:
        self.tasks: list[WorkspaceAgentTask] = []

    def run_task(self, task: WorkspaceAgentTask) -> WorkspaceAgentResult:
        self.tasks.append(task)
        _write_valid_mechanism(Path(task.workspace))
        return WorkspaceAgentResult(
            summary="duplicate-name V3 backend completed",
            structured={
                "summary": "duplicate-name V3 backend completed",
                "mechanism_name": "fake_backend_mechanism",
                "files_written": ["mechanism.py"],
                "contract_notes": [],
            },
            returncode=0,
        )


class AlwaysFailingV3Backend:
    def __init__(self) -> None:
        self.tasks: list[WorkspaceAgentTask] = []

    def run_task(self, task: WorkspaceAgentTask) -> WorkspaceAgentResult:
        self.tasks.append(task)
        _write_invalid_mechanism(Path(task.workspace))
        return WorkspaceAgentResult(
            summary="invalid V3 backend completed",
            structured={
                "summary": "invalid V3 backend completed",
                "mechanism_name": Path(task.workspace).name,
                "files_written": ["mechanism.py"],
                "contract_notes": ["intentionally invalid for regression test"],
            },
            returncode=0,
        )


class PromptCaptureV3Backend:
    def __init__(self) -> None:
        self.tasks: list[WorkspaceAgentTask] = []

    def run_task(self, task: WorkspaceAgentTask) -> WorkspaceAgentResult:
        self.tasks.append(task)
        _write_invalid_mechanism(Path(task.workspace))
        return WorkspaceAgentResult(
            summary="captured V3 prompt",
            structured={
                "summary": "captured V3 prompt",
                "mechanism_name": Path(task.workspace).name,
                "files_written": ["mechanism.py"],
                "contract_notes": ["prompt capture only"],
            },
            returncode=0,
        )


class MissingOperatorRefsV3Backend:
    def run_task(self, task: WorkspaceAgentTask) -> WorkspaceAgentResult:
        _write_missing_operator_refs_mechanism(Path(task.workspace))
        return WorkspaceAgentResult(
            summary="missing operator refs V3 backend completed",
            structured={
                "summary": "missing operator refs V3 backend completed",
                "mechanism_name": "missing_operator_refs_mechanism",
                "files_written": ["mechanism.py"],
                "contract_notes": [],
            },
            returncode=0,
        )


class ArtifactStressClaimsBackend:
    def run_task(self, task: WorkspaceAgentTask) -> WorkspaceAgentResult:
        _write_artifact_stress_claim_mechanism(Path(task.workspace))
        return WorkspaceAgentResult(
            summary="artifact stress claims V3 backend completed",
            structured={
                "summary": "artifact stress claims V3 backend completed",
                "mechanism_name": "artifact_stress_claim_mechanism",
                "files_written": ["mechanism.py"],
                "contract_notes": [],
            },
            returncode=0,
        )


def _init_project(tmp_path: Path, monkeypatch) -> Path:
    fixture_dir = Path(__file__).parent / "fixtures" / "literature"
    fulltext_fixture_dir = Path(__file__).parent / "fixtures" / "literature_fulltext"
    monkeypatch.setenv("DESIGN_SCIENTIST_LITERATURE_FIXTURES", str(fixture_dir))
    monkeypatch.setenv("DESIGN_SCIENTIST_FULLTEXT_FIXTURES", str(fulltext_fixture_dir))
    project = tmp_path / "project"
    init_framework(project, domain="protein variant design")
    return project


def _write_invalid_mechanism(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "mechanism.py").write_text(
        "def select_batch(observed, candidates, budget, round_index, rng):\n"
        "    return []\n",
        encoding="utf-8",
    )


def _write_valid_mechanism(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "mechanism.py").write_text(
        dedent(
            """
            from __future__ import annotations

            import json
            from pathlib import Path

            from design_scientist import policies


            MECHANISM_ID = "fake_backend_mechanism"
            OPERATOR_ID = "operator_active_learning_acquisition"
            OPERATOR_SPEC = {
                "operator_id": OPERATOR_ID,
                "mechanism_id": "active_learning_acquisition",
                "objective": {"description": "Expected improvement with uncertainty."},
            }


            def fit_state(context):
                state = dict(context)
                state["operator_weights"] = {OPERATOR_ID: 1.0}
                if state.get("ablation") == "key_component_removed":
                    state["operator_weights"][OPERATOR_ID] = 0.0
                    state["removed_operator_ids"] = [OPERATOR_ID]
                return state


            def generate_candidates(state):
                candidates = []
                for candidate in state["candidate_records"]:
                    item = dict(candidate)
                    item["operator_refs"] = [OPERATOR_ID]
                    candidates.append(item)
                return candidates


            def score_candidates(state, candidates):
                weight = float(state["operator_weights"].get(OPERATOR_ID, 0.0))
                scored = []
                for candidate in candidates:
                    item = dict(candidate)
                    modules = item.get("modules") or ()
                    item["operator_score"] = weight * (len(modules) + float(item.get("predicted_utility", 0.0)))
                    scored.append(item)
                return scored


            def select_panel(state, candidates, budget, rng):
                del state, rng
                ranked = sorted(
                    candidates,
                    key=lambda candidate: (
                        float(candidate.get("operator_score", 0.0)),
                        str(candidate.get("candidate_id") or candidate.get("variant_id") or ""),
                    ),
                    reverse=True,
                )
                return [
                    str(candidate.get("candidate_id") or candidate.get("variant_id"))
                    for candidate in ranked[:budget]
                ]


            def plan_ablations(state):
                del state
                return [
                    {"name": "none"},
                    {
                        "name": "key_component_removed",
                        "removed_operator_ids": [OPERATOR_ID],
                    },
                ]


            def _write_json(path, payload):
                path.write_text(json.dumps(payload, indent=2) + "\\n", encoding="utf-8")


            def run(workspace):
                root = Path(workspace)
                artifacts = {
                    "mechanism_spec.json": {
                        "mechanism_id": MECHANISM_ID,
                        "name": "Fake backend mechanism",
                        "version": "v3",
                        "operator_refs": [OPERATOR_ID],
                        "operator_specs": [OPERATOR_SPEC],
                        "architecture_delta": "Uses lifecycle replay rather than select_batch-only dispatch.",
                    },
                    "proposal.json": {
                        "mechanism_id": MECHANISM_ID,
                        "hypothesis": "Mechanism-aware lifecycle nodes can be benchmarked.",
                        "operator_refs": [OPERATOR_ID],
                    },
                    "ablation_plan.json": {
                        "ablations": [
                            {"name": "none"},
                            {"name": "key_component_removed", "removed_operator_ids": [OPERATOR_ID]},
                        ],
                    },
                    "stress_test_plan.json": {
                        "stress_tests": [{"world_id": "epistatic"}],
                    },
                    "mechanism_metrics.json": {
                        "metrics": {"status": "ready"},
                    },
                    "validation_report.json": {
                        "valid": True,
                        "findings": [],
                    },
                    "operator_to_code_trace.json": {
                        "operator_to_code_trace": {
                            OPERATOR_ID: [
                                "fit_state.operator_weights",
                                "generate_candidates.operator_refs",
                                "score_candidates.operator_score",
                                "select_panel.operator_score_rank",
                                "plan_ablations.removed_operator_ids",
                            ]
                        }
                    },
                }
                for filename, payload in artifacts.items():
                    _write_json(root / filename, payload)
            """
        ).lstrip(),
        encoding="utf-8",
    )


def _write_missing_operator_refs_mechanism(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "mechanism.py").write_text(
        dedent(
            """
            from __future__ import annotations

            import json
            from pathlib import Path


            MECHANISM_ID = "missing_operator_refs_mechanism"


            def fit_state(context):
                state = dict(context)
                state["custom_state"] = {"score_weight": 1.0}
                return state


            def generate_candidates(state):
                return [dict(candidate) for candidate in state["candidate_records"]]


            def score_candidates(state, candidates):
                del state
                scored = []
                for candidate in candidates:
                    item = dict(candidate)
                    item["custom_score"] = len(item.get("modules") or ())
                    scored.append(item)
                return scored


            def select_panel(state, candidates, budget, rng):
                del state, rng
                ranked = sorted(
                    candidates,
                    key=lambda candidate: (
                        float(candidate.get("custom_score", 0.0)),
                        str(candidate.get("candidate_id") or candidate.get("variant_id") or ""),
                    ),
                    reverse=True,
                )
                return [
                    str(candidate.get("candidate_id") or candidate.get("variant_id"))
                    for candidate in ranked[:budget]
                ]


            def plan_ablations(state):
                del state
                return [
                    {"name": "none"},
                    {"name": "key_component_removed", "removed_component": "custom_score"},
                ]


            def _write_json(path, payload):
                path.write_text(json.dumps(payload, indent=2) + "\\n", encoding="utf-8")


            def run(workspace):
                root = Path(workspace)
                artifacts = {
                    "mechanism_spec.json": {
                        "mechanism_id": MECHANISM_ID,
                        "name": "Missing operator refs mechanism",
                        "version": "v3",
                    },
                    "proposal.json": {
                        "mechanism_id": MECHANISM_ID,
                        "hypothesis": "This node omits operator refs and must not be selectable.",
                    },
                    "ablation_plan.json": {
                        "ablations": [
                            {"name": "none"},
                            {"name": "key_component_removed", "removed_component": "custom_score"},
                        ],
                    },
                    "stress_test_plan.json": {
                        "stress_tests": [{"world_id": "epistatic"}],
                    },
                    "mechanism_metrics.json": {
                        "metrics": {"status": "ready"},
                    },
                    "validation_report.json": {
                        "valid": True,
                        "findings": [],
                    },
                }
                for filename, payload in artifacts.items():
                    _write_json(root / filename, payload)
            """
        ).lstrip(),
        encoding="utf-8",
    )


def _write_artifact_stress_claim_mechanism(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "mechanism.py").write_text(
        dedent(
            """
            from __future__ import annotations

            import json
            from pathlib import Path

            from design_scientist import policies


            MECHANISM_ID = "artifact_stress_claim_mechanism"
            OPERATOR_ID = "operator_active_learning_acquisition"
            OPERATOR_SPEC = {
                "operator_id": OPERATOR_ID,
                "mechanism_id": "active_learning_acquisition",
                "objective": {"description": "Expected improvement with uncertainty."},
            }


            def fit_state(context):
                state = dict(context)
                state["operator_weights"] = {OPERATOR_ID: 1.0}
                if state.get("ablation") == "key_component_removed":
                    state["operator_weights"][OPERATOR_ID] = 0.0
                    state["removed_operator_ids"] = [OPERATOR_ID]
                return state


            def generate_candidates(state):
                candidates = []
                for candidate in state["candidate_records"]:
                    item = dict(candidate)
                    item["operator_refs"] = [OPERATOR_ID]
                    candidates.append(item)
                return candidates


            def score_candidates(state, candidates):
                weight = float(state["operator_weights"].get(OPERATOR_ID, 0.0))
                scored = []
                for candidate in candidates:
                    item = dict(candidate)
                    modules = item.get("modules") or ()
                    item["operator_score"] = weight * (len(modules) + float(item.get("predicted_utility", 0.0)))
                    scored.append(item)
                return scored


            def select_panel(state, candidates, budget, rng):
                del state, rng
                ranked = sorted(
                    candidates,
                    key=lambda candidate: (
                        float(candidate.get("operator_score", 0.0)),
                        str(candidate.get("candidate_id") or candidate.get("variant_id") or ""),
                    ),
                    reverse=True,
                )
                return [
                    str(candidate.get("candidate_id") or candidate.get("variant_id"))
                    for candidate in ranked[:budget]
                ]


            def plan_ablations(state):
                del state
                return [
                    {"name": "none"},
                    {
                        "name": "key_component_removed",
                        "removed_operator_ids": [OPERATOR_ID],
                    },
                ]


            def _write_json(path, payload):
                path.write_text(json.dumps(payload, indent=2) + "\\n", encoding="utf-8")


            def run(workspace):
                root = Path(workspace)
                artifacts = {
                    "mechanism_spec.json": {
                        "mechanism_id": MECHANISM_ID,
                        "name": "Artifact stress claim mechanism",
                        "version": "v3",
                        "operator_refs": [OPERATOR_ID],
                        "operator_specs": [OPERATOR_SPEC],
                        "components": [{"component_id": "state_model"}],
                        "claims": ["Noisy assay endpoint handling should be stress tested."],
                        "literature_basis": ["fixture"],
                    },
                    "proposal.json": {
                        "mechanism_id": MECHANISM_ID,
                        "hypothesis": "Artifact claims should drive stress replay selection.",
                        "operator_refs": [OPERATOR_ID],
                    },
                    "ablation_plan.json": {
                        "ablations": [
                            {"name": "none"},
                            {"name": "key_component_removed", "removed_operator_ids": [OPERATOR_ID]},
                        ],
                    },
                    "stress_test_plan.json": {
                        "stress_tests": [{"world_id": "confounded_transfer"}],
                    },
                    "mechanism_metrics.json": {
                        "metrics": {"status": "ready"},
                    },
                    "validation_report.json": {
                        "valid": True,
                        "findings": [],
                    },
                    "operator_to_code_trace.json": {
                        "operator_to_code_trace": {
                            OPERATOR_ID: [
                                "fit_state.operator_weights",
                                "generate_candidates.operator_refs",
                                "score_candidates.operator_score",
                                "select_panel.operator_score_rank",
                                "plan_ablations.removed_operator_ids",
                            ]
                        }
                    },
                }
                for filename, payload in artifacts.items():
                    _write_json(root / filename, payload)
            """
        ).lstrip(),
        encoding="utf-8",
    )
