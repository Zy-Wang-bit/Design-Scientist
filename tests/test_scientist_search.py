from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest

from design_scientist.framework import init_framework
from design_scientist.schemas import WorkspaceAgentResult, WorkspaceAgentTask
from design_scientist.scientist_search import SCIENTIST_STAGES, develop_method, run_scientist_search


class FakeWorkspaceBackend:
    def __init__(self) -> None:
        self.tasks: list[WorkspaceAgentTask] = []

    def run_task(self, task: WorkspaceAgentTask) -> WorkspaceAgentResult:
        self.tasks.append(task)
        workspace = Path(task.workspace)
        if "node_01" in workspace.name:
            method_name = "mechanism_aware"
        else:
            method_name = "random_feasible"
        _write_fake_method_node(workspace, method_name=method_name)
        return WorkspaceAgentResult(
            summary=f"wrote fake node for {method_name}",
            structured={
                "summary": "fake backend completed",
                "policy_name": f"{workspace.name}_invented_policy",
                "policy_entrypoint": "select_batch",
                "files_written": [
                    "proposal.json",
                    "method.py",
                    "manifest.json",
                    "candidate_policy.json",
                    "novelty_report.json",
                    "benchmark_metrics.json",
                    "validation_report.json",
                ],
                "contract_notes": [],
            },
            returncode=0,
        )


AGENT_REQUIRED_FILES = [
    "proposal.json",
    "method.py",
    "manifest.json",
    "candidate_policy.json",
    "novelty_report.json",
    "benchmark_metrics.json",
    "validation_report.json",
]


def _successful_agent_result(workspace: Path, *, policy_name: str | None = None) -> WorkspaceAgentResult:
    return WorkspaceAgentResult(
        summary="fake backend completed",
        structured={
            "summary": "fake backend completed",
            "policy_name": policy_name or f"{workspace.name}_invented_policy",
            "policy_entrypoint": "select_batch",
            "files_written": list(AGENT_REQUIRED_FILES),
            "contract_notes": [],
        },
        returncode=0,
    )


def test_develop_method_writes_local_nodes_and_journal(tmp_path: Path) -> None:
    project = tmp_path / "project"

    result = develop_method(project, nodes=3, use_codex=False, run_id="unit")

    journal_path = project / "runs" / "unit" / "scientist_journal.json"
    stage_path = project / "runs" / "unit" / "stage_progress.json"
    route_tree_path = project / "runs" / "unit" / "route_tree.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    stage_progress = json.loads(stage_path.read_text(encoding="utf-8"))
    route_tree = json.loads(route_tree_path.read_text(encoding="utf-8"))
    selected = journal["selected_node"]
    ranking = journal["benchmark"]["ranking"]

    assert result["journal_path"] == str(journal_path)
    assert len(journal["nodes"]) == 3
    assert all(node["status"] == "completed" for node in journal["nodes"])
    assert journal["failed_nodes"] == []
    assert selected == ranking[0]
    assert selected["benchmark_method"] == "mechanism_aware"
    assert [stage["stage"] for stage in stage_progress["stages"]] == list(SCIENTIST_STAGES)
    assert route_tree["run_id"] == "unit"
    assert len(route_tree["nodes"]) == 3

    selected_record = next(node for node in journal["nodes"] if node["node_id"] == selected["node_id"])
    trace = selected_record["trace"]
    assert trace["proposal"] == selected_record["artifacts"]["proposal"]
    assert trace["benchmark_metrics"] == selected_record["artifacts"]["benchmark_metrics"]
    assert trace["ablation_results"] == journal["benchmark"]["ablation_results_path"]
    assert trace["journal"] == str(journal_path)
    assert selected["trace"] == trace

    proposal_path = Path(selected_record["artifacts"]["proposal"])
    novelty_path = Path(selected_record["artifacts"]["novelty_report"])
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    novelty = json.loads(novelty_path.read_text(encoding="utf-8"))
    assert proposal["proposal_id"] == selected_record["node_id"]
    assert proposal["title"]
    assert proposal["method_claim"]
    assert "method_hypothesis" in proposal
    assert novelty["baseline_clone"] is False
    assert "baseline_overlap" in novelty
    assert "selection_overlap_vs_baselines" in novelty
    metrics_path = Path(
        selected_record["artifacts"]["benchmark_metrics"]
    )
    node_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert node_metrics["synthetic_replay"] == selected["benchmark_metrics"]
    assert node_metrics["ranking_score"] == selected["ranking_score"]


def test_selected_node_metrics_use_multi_world_summary_and_store_rows(tmp_path: Path) -> None:
    project = tmp_path / "project"

    result = develop_method(project, nodes=1, use_codex=False, run_id="multi_world_metrics")
    journal = json.loads(Path(result["journal_path"]).read_text(encoding="utf-8"))
    selected = journal["selected_node"]
    selected_record = journal["nodes"][0]
    node_metrics = json.loads(
        Path(selected_record["artifacts"]["benchmark_metrics"]).read_text(encoding="utf-8")
    )

    assert selected["benchmark_metrics"] == node_metrics["synthetic_replay_summary"]
    assert node_metrics["synthetic_replay"] == node_metrics["synthetic_replay_summary"]
    assert "world_id" not in selected["benchmark_metrics"]
    assert selected["benchmark_metrics"]["world_count"] >= 2
    assert len(node_metrics["synthetic_replay_rows"]) == selected["benchmark_metrics"]["world_count"]
    assert {
        row["world_id"]
        for row in node_metrics["synthetic_replay_rows"]
    } == {
        "additive",
        "epistatic",
        "confounded_transfer",
        "noisy_endpoint",
        "sparse_early_round",
    }


def test_develop_method_uses_fake_workspace_backend_with_node_only_allowed_paths(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    backend = FakeWorkspaceBackend()

    result = develop_method(project, nodes=2, use_codex=True, backend=backend, run_id="codex")

    journal = json.loads(Path(result["journal_path"]).read_text(encoding="utf-8"))
    selected = journal["selected_node"]
    ranking_by_node = {
        row["node_id"]: row
        for row in journal["benchmark"]["ranking"]
    }

    assert len(backend.tasks) == 2
    for task in backend.tasks:
        workspace = Path(task.workspace).resolve()
        assert task.mode == "patch"
        assert task.allowed_paths == [str(workspace)]
        assert task.output_schema is not None
        assert set(task.output_schema["required"]) == {
            "summary",
            "policy_name",
            "policy_entrypoint",
            "files_written",
            "contract_notes",
        }
        assert "synthetic_replay_method" not in task.output_schema["properties"]
        assert "proposal.json" in task.objective
        assert "novelty_report.json" in task.objective
        assert "policy callable" in task.objective
        assert "synthetic_replay_method" not in task.objective

    assert selected["node_id"] in ranking_by_node
    assert selected["benchmark_metrics"] == ranking_by_node[selected["node_id"]]["benchmark_metrics"]
    assert selected["benchmark_method"] == "mechanism_aware"
    selected_record = next(node for node in journal["nodes"] if node["node_id"] == selected["node_id"])
    assert selected_record["agent"]["structured"]["policy_entrypoint"] == "select_batch"
    assert "synthetic_replay_method" not in selected_record["agent"]["structured"]


@pytest.mark.parametrize(
    ("backend_result", "expected_reason"),
    [
        (
            WorkspaceAgentResult(
                summary="codex crashed",
                structured={
                    "summary": "codex crashed",
                    "policy_name": "stale_policy",
                    "policy_entrypoint": "select_batch",
                    "files_written": list(AGENT_REQUIRED_FILES),
                    "contract_notes": [],
                },
                returncode=2,
            ),
            "Codex backend failed with returncode 2",
        ),
        (
            WorkspaceAgentResult(
                summary="bad structured output",
                structured={"summary": "missing required fields"},
                returncode=0,
            ),
            "invalid Codex structured output",
        ),
    ],
)
def test_develop_method_rejects_failed_or_invalid_codex_result_without_stale_execution(
    tmp_path: Path,
    backend_result: WorkspaceAgentResult,
    expected_reason: str,
) -> None:
    project = tmp_path / "project"
    stale_workspace = project / "runs" / "stale_codex" / "nodes" / "node_01_mechanism_aware"
    _write_fake_method_node(stale_workspace, method_name="mechanism_aware")
    (stale_workspace / "stale_marker.txt").write_text("must be removed", encoding="utf-8")

    class BadBackend:
        def run_task(self, task: WorkspaceAgentTask) -> WorkspaceAgentResult:
            assert Path(task.workspace) == stale_workspace.resolve()
            return backend_result

    result = develop_method(
        project,
        nodes=1,
        use_codex=True,
        backend=BadBackend(),
        run_id="stale_codex",
    )

    journal = json.loads(Path(result["journal_path"]).read_text(encoding="utf-8"))
    node = journal["nodes"][0]

    assert node["status"] == "failed"
    assert node["contract"]["executed"] is False
    assert expected_reason in "; ".join(node["failure_reasons"])
    assert journal["selected_node"] is None
    assert not (stale_workspace / "stale_marker.txt").exists()


def test_develop_method_codex_prompt_injects_literature_gap_context(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    framework = project / "framework"
    framework.mkdir(parents=True)
    (framework / "paper_cards.json").write_text(
        json.dumps(
            [
                {
                    "paper_id": "paper_gap_alpha",
                    "title": "Gap-aware protein batch design",
                    "year": 2026,
                    "relevance": "active-learning gap provenance",
                }
            ]
        ),
        encoding="utf-8",
    )
    (framework / "method_modules.json").write_text(
        json.dumps(
            {
                "method_modules": [
                    {
                        "module_id": "adaptive_decision_value_allocation",
                        "name": "Adaptive Decision Value Allocation",
                        "reusable_ideas": ["state-dependent allocation"],
                        "gap_for_design_scientist": "needs explicit gap provenance",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (framework / "research_gap_matrix.csv").write_text(
        "method_module_id,gap_for_design_scientist,reusable_ideas\n"
        "gap_constraints_tiny_data,Needs tiny-data constraints,state-dependent allocation\n",
        encoding="utf-8",
    )
    (framework / "failure_memory.jsonl").write_text(
        json.dumps({"summary": "previous clone failed due to tail-swap overlap"}) + "\n",
        encoding="utf-8",
    )
    backend = FakeWorkspaceBackend()

    develop_method(project, nodes=1, use_codex=True, backend=backend, run_id="context")

    objective = backend.tasks[0].objective
    assert "Top papers" in objective
    assert "paper_gap_alpha" in objective
    assert "Gap-aware protein batch design" in objective
    assert "adaptive_decision_value_allocation" in objective
    assert "gap_constraints_tiny_data" in objective
    assert "previous clone failed due to tail-swap overlap" in objective
    for field in (
        "literature_gap_ids",
        "reused_components",
        "architecture_delta",
        "new_mechanism_claim",
        "planned_ablation",
    ):
        assert field in objective


def test_develop_method_accepts_dot_policy_entrypoint_from_codex_node(
    tmp_path: Path,
) -> None:
    class DotEntrypointBackend:
        def run_task(self, task: WorkspaceAgentTask) -> WorkspaceAgentResult:
            workspace = Path(task.workspace)
            _write_fake_method_node(
                workspace,
                method_name="mechanism_aware",
                policy_entrypoint="method.select_batch",
            )
            return WorkspaceAgentResult(
                summary="wrote dot entrypoint node",
                structured={
                    "summary": "fake backend completed",
                    "policy_name": "dot_entrypoint_policy",
                    "policy_entrypoint": "method.select_batch",
                    "files_written": [
                        "proposal.json",
                        "method.py",
                        "manifest.json",
                        "candidate_policy.json",
                        "novelty_report.json",
                        "benchmark_metrics.json",
                        "validation_report.json",
                    ],
                    "contract_notes": [],
                },
                returncode=0,
            )

    result = develop_method(
        tmp_path / "project",
        nodes=1,
        use_codex=True,
        backend=DotEntrypointBackend(),
        run_id="dot_entrypoint",
    )

    journal = json.loads(Path(result["journal_path"]).read_text(encoding="utf-8"))
    assert journal["selected_node_id"] == "node_01_mechanism_aware"
    assert journal["selected_node"]["benchmark_policy_name"]


def test_develop_method_accepts_codex_no_baseline_callable_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from design_scientist import scientist_search

    project = tmp_path / "project"
    backend = FakeWorkspaceBackend()

    def writer(task: WorkspaceAgentTask) -> WorkspaceAgentResult:
        backend.tasks.append(task)
        workspace = Path(task.workspace)
        if workspace.name.endswith("literature_gap"):
            _write_fake_method_node(workspace, method_name="literature_gap", no_baseline=True)
        else:
            _write_fake_method_node(workspace, method_name="mechanism_aware")
        return WorkspaceAgentResult(
            summary="fake backend completed",
            structured={
                "summary": "fake backend completed",
                "policy_name": f"{workspace.name}_invented_policy",
                "policy_entrypoint": "select_batch",
                "files_written": [
                    "proposal.json",
                    "method.py",
                    "manifest.json",
                    "candidate_policy.json",
                    "novelty_report.json",
                    "benchmark_metrics.json",
                    "validation_report.json",
                ],
                "contract_notes": [],
            },
            returncode=0,
        )

    backend.run_task = writer  # type: ignore[method-assign]
    monkeypatch.setattr(scientist_search, "run_synthetic_benchmark", _fake_benchmark_result)

    result = develop_method(project, nodes=8, use_codex=True, backend=backend, run_id="no_baseline")
    journal = json.loads(Path(result["journal_path"]).read_text(encoding="utf-8"))

    no_baseline_node = next(
        node for node in journal["nodes"] if node["planned_method"] == "literature_gap"
    )
    assert no_baseline_node["status"] == "completed"
    assert no_baseline_node["benchmark_method"] is None
    assert "no supported baseline family declared" not in no_baseline_node["failure_reasons"]
    assert journal["selected_node_id"] == no_baseline_node["node_id"]


def test_develop_method_rejects_post_refresh_overlap_above_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from design_scientist import scientist_search

    project = tmp_path / "project"
    backend = FakeWorkspaceBackend()

    def writer(task: WorkspaceAgentTask) -> WorkspaceAgentResult:
        backend.tasks.append(task)
        _write_fake_method_node(
            Path(task.workspace),
            method_name="mechanism_aware",
            fake_unique_novelty=True,
        )
        return WorkspaceAgentResult(
            summary="fake backend completed",
            structured={
                "summary": "fake backend completed",
                "policy_name": f"{Path(task.workspace).name}_invented_policy",
                "policy_entrypoint": "select_batch",
                "files_written": [
                    "proposal.json",
                    "method.py",
                    "manifest.json",
                    "candidate_policy.json",
                    "novelty_report.json",
                    "benchmark_metrics.json",
                    "validation_report.json",
                ],
                "contract_notes": [],
            },
            returncode=0,
        )

    backend.run_task = writer  # type: ignore[method-assign]

    def high_overlap_benchmark(
        root: Path,
        *,
        run_id: str,
        rounds: int,
        methods: list[object],
    ) -> dict[str, object]:
        return _fake_benchmark_result(
            root,
            run_id=run_id,
            rounds=rounds,
            methods=methods,
            high_overlap_by_method={"node_01_mechanism_aware_invented_policy": 0.86},
        )

    monkeypatch.setattr(scientist_search, "run_synthetic_benchmark", high_overlap_benchmark)

    result = develop_method(project, nodes=1, use_codex=True, backend=backend, run_id="overlap_guard")
    journal = json.loads(Path(result["journal_path"]).read_text(encoding="utf-8"))
    node = journal["nodes"][0]
    novelty = json.loads(Path(node["artifacts"]["novelty_report"]).read_text(encoding="utf-8"))

    assert node["status"] == "failed"
    assert novelty["selection_overlap_vs_baselines"] == 0.86
    assert "post-refresh novelty guard" in "; ".join(node["failure_reasons"])
    assert journal["selected_node"] is None


def test_develop_method_rejects_worst_case_overlap_against_any_default_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from design_scientist import scientist_search

    project = tmp_path / "project"
    backend = FakeWorkspaceBackend()

    def writer(task: WorkspaceAgentTask) -> WorkspaceAgentResult:
        backend.tasks.append(task)
        _write_fake_method_node(
            Path(task.workspace),
            method_name="mechanism_aware",
            fake_unique_novelty=True,
        )
        return _successful_agent_result(Path(task.workspace))

    backend.run_task = writer  # type: ignore[method-assign]

    def worst_case_overlap_benchmark(
        root: Path,
        *,
        run_id: str,
        rounds: int,
        methods: list[object],
    ) -> dict[str, object]:
        del rounds
        method_name = str(getattr(methods[0], "policy_name", None))
        run_dir = root / "runs" / run_id
        rows = [
            {
                "world_id": "world_low_overlap",
                "method": method_name,
                "best_feasible_utility": 1.0,
                "hit_rate": 0.2,
                "evidence_coverage": 0.9,
                "round_efficiency": 0.8,
                "regret_proxy": 0.1,
                "false_claim_rate": 0.0,
                "novelty_score": 0.7,
                "baseline_overlap": 0.10,
                "selection_overlap_random_feasible": 0.10,
                "selection_overlap_fixed_mix": 0.10,
                "selection_overlap_mechanism_aware": 0.10,
                "selection_overlap_top_observed": 0.10,
                "selected_ids_digest": "digest_low",
                "status": "completed",
            },
            {
                "world_id": "world_top_observed_clone",
                "method": method_name,
                "best_feasible_utility": 1.0,
                "hit_rate": 0.2,
                "evidence_coverage": 0.9,
                "round_efficiency": 0.8,
                "regret_proxy": 0.1,
                "false_claim_rate": 0.0,
                "novelty_score": 0.7,
                "baseline_overlap": 0.10,
                "selection_overlap_random_feasible": 0.10,
                "selection_overlap_fixed_mix": 0.10,
                "selection_overlap_mechanism_aware": 0.10,
                "selection_overlap_top_observed": 0.91,
                "selected_ids_digest": "digest_high",
                "status": "completed",
            },
        ]
        return {
            "run_id": run_id,
            "benchmark_results_path": str(run_dir / "benchmark_results.csv"),
            "ablation_results_path": str(run_dir / "ablation_results.csv"),
            "summary_results_path": str(run_dir / "benchmark_summary.csv"),
            "benchmark_results": rows,
            "ablation_results": [],
            "summary": {
                "method_rankings": [
                    {
                        "method": method_name,
                        "world_count": 2,
                        "mean_best_feasible_utility": 1.0,
                        "mean_hit_rate": 0.2,
                        "mean_evidence_coverage": 0.9,
                        "mean_round_efficiency": 0.8,
                        "mean_regret_proxy": 0.1,
                        "mean_false_claim_rate": 0.0,
                        "mean_novelty_score": 0.7,
                        "mean_baseline_overlap": 0.10,
                    }
                ]
            },
        }

    monkeypatch.setattr(scientist_search, "run_synthetic_benchmark", worst_case_overlap_benchmark)

    result = develop_method(project, nodes=1, use_codex=True, backend=backend, run_id="worst_overlap")
    journal = json.loads(Path(result["journal_path"]).read_text(encoding="utf-8"))
    node = journal["nodes"][0]
    novelty = json.loads(Path(node["artifacts"]["novelty_report"]).read_text(encoding="utf-8"))

    assert node["status"] == "failed"
    assert novelty["selection_overlap_vs_baselines"] == 0.91
    assert novelty["nearest_baseline"] == "top_observed"
    assert novelty["nearest_baseline_world_id"] == "world_top_observed_clone"
    assert "post-refresh novelty guard" in "; ".join(node["failure_reasons"])
    assert journal["selected_node"] is None


def test_develop_method_namespaces_duplicate_generated_policy_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from design_scientist import scientist_search

    project = tmp_path / "project"
    backend = FakeWorkspaceBackend()

    def duplicate_name_writer(task: WorkspaceAgentTask) -> WorkspaceAgentResult:
        backend.tasks.append(task)
        workspace = Path(task.workspace)
        method_name = "mechanism_aware" if "node_01" in workspace.name else "fixed_mix"
        _write_fake_method_node(
            workspace,
            method_name=method_name,
            policy_id="shared_generated_policy",
        )
        return _successful_agent_result(workspace, policy_name="shared_generated_policy")

    backend.run_task = duplicate_name_writer  # type: ignore[method-assign]
    monkeypatch.setattr(scientist_search, "run_synthetic_benchmark", _fake_benchmark_result)

    result = develop_method(project, nodes=2, use_codex=True, backend=backend, run_id="duplicate_names")
    journal = json.loads(Path(result["journal_path"]).read_text(encoding="utf-8"))
    policy_names = [
        node["benchmark_policy_name"]
        for node in journal["nodes"]
        if node["benchmark_policy_name"]
    ]

    assert policy_names == [
        "node_01_mechanism_aware_shared_generated_policy",
        "node_02_fixed_mix_shared_generated_policy",
    ]
    assert len(policy_names) == len(set(policy_names))


def test_run_scientist_search_records_literature_snapshot_and_benchmark_trace(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    (project / "project.yaml").parent.mkdir(parents=True)
    (project / "project.yaml").write_text(
        "project_id: unit\nname: Active learning protein engineering\n",
        encoding="utf-8",
    )

    result = run_scientist_search(
        project,
        max_papers=2,
        nodes=2,
        rounds=2,
        use_codex=False,
        offline_fixtures=True,
    )

    journal = json.loads(Path(result["journal_path"]).read_text(encoding="utf-8"))
    selected = journal["selected_node"]

    assert journal["literature_snapshot"]["status"] == "completed"
    assert journal["literature_snapshot"]["paper_count"] >= 1
    assert len(journal["literature_snapshot"]["cards"]) <= 2
    assert journal["benchmark"]["rounds"] == 2
    assert selected["benchmark_metrics"]["rounds"] == 2
    assert journal["selected_node_id"] == selected["node_id"]
    assert (Path(result["run_dir"]) / "stage_progress.json").exists()
    assert (Path(result["run_dir"]) / "route_tree.json").exists()


def test_develop_method_rejects_codex_baseline_clone_from_selection(tmp_path: Path) -> None:
    project = tmp_path / "project"
    backend = FakeWorkspaceBackend()

    def clone_writer(task: WorkspaceAgentTask) -> WorkspaceAgentResult:
        backend.tasks.append(task)
        workspace = Path(task.workspace)
        if "node_01" in workspace.name:
            _write_fake_method_node(workspace, method_name="mechanism_aware", baseline_clone=True)
        else:
            _write_fake_method_node(workspace, method_name="random_feasible")
        return WorkspaceAgentResult(
            summary="fake backend completed",
            structured={
                "summary": "fake backend completed",
                "policy_name": f"{workspace.name}_invented_policy",
                "policy_entrypoint": "select_batch",
                "files_written": [
                    "proposal.json",
                    "method.py",
                    "manifest.json",
                    "candidate_policy.json",
                    "novelty_report.json",
                    "benchmark_metrics.json",
                    "validation_report.json",
                ],
                "contract_notes": [],
            },
            returncode=0,
        )

    backend.run_task = clone_writer  # type: ignore[method-assign]

    result = develop_method(project, nodes=2, use_codex=True, backend=backend, run_id="clone_guard")

    journal = json.loads(Path(result["journal_path"]).read_text(encoding="utf-8"))

    assert journal["nodes"][0]["status"] == "failed"
    assert "baseline clone guard" in "; ".join(journal["nodes"][0]["failure_reasons"])
    assert journal["selected_node_id"] == "node_02_fixed_mix"


def test_develop_method_rejects_behavioral_clone_even_with_fake_novelty_metadata(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    backend = FakeWorkspaceBackend()

    def clone_writer(task: WorkspaceAgentTask) -> WorkspaceAgentResult:
        backend.tasks.append(task)
        workspace = Path(task.workspace)
        if "node_01" in workspace.name:
            _write_fake_method_node(
                workspace,
                method_name="mechanism_aware",
                behavioral_clone=True,
                fake_unique_novelty=True,
            )
        else:
            _write_fake_method_node(workspace, method_name="fixed_mix")
        return WorkspaceAgentResult(
            summary="fake backend completed",
            structured={
                "summary": "fake backend completed",
                "policy_name": f"{workspace.name}_invented_policy",
                "policy_entrypoint": "select_batch",
                "files_written": [
                    "proposal.json",
                    "method.py",
                    "manifest.json",
                    "candidate_policy.json",
                    "novelty_report.json",
                    "benchmark_metrics.json",
                    "validation_report.json",
                ],
                "contract_notes": [],
            },
            returncode=0,
        )

    backend.run_task = clone_writer  # type: ignore[method-assign]

    result = develop_method(project, nodes=2, use_codex=True, backend=backend, run_id="behavior_clone")
    journal = json.loads(Path(result["journal_path"]).read_text(encoding="utf-8"))

    first_node = journal["nodes"][0]
    assert first_node["status"] == "failed"
    assert "behavioral baseline clone" in "; ".join(first_node["failure_reasons"])
    assert journal["selected_node_id"] != first_node["node_id"]


def test_develop_method_isolates_entrypoint_and_replay_failures_with_failure_memory(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    backend = FakeWorkspaceBackend()

    def mixed_failure_writer(task: WorkspaceAgentTask) -> WorkspaceAgentResult:
        backend.tasks.append(task)
        workspace = Path(task.workspace)
        if "node_01" in workspace.name:
            _write_fake_method_node(
                workspace,
                method_name="mechanism_aware",
                policy_entrypoint="external.select_batch",
            )
        elif "node_02" in workspace.name:
            _write_fake_method_node(
                workspace,
                method_name="fixed_mix",
                policy_behavior="raise",
            )
        elif "node_03" in workspace.name:
            _write_fake_method_node(
                workspace,
                method_name="pure_lattice_repair",
                policy_behavior="invalid_selection",
            )
        else:
            _write_fake_method_node(workspace, method_name="greedy_utility")
        return _successful_agent_result(workspace)

    backend.run_task = mixed_failure_writer  # type: ignore[method-assign]

    result = develop_method(project, nodes=4, use_codex=True, backend=backend, run_id="isolated_failures")
    journal_path = Path(result["journal_path"])
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    nodes_by_id = {
        node["node_id"]: node
        for node in journal["nodes"]
    }

    assert journal_path.exists()
    assert nodes_by_id["node_01_mechanism_aware"]["status"] == "failed"
    assert nodes_by_id["node_02_fixed_mix"]["status"] == "failed"
    assert nodes_by_id["node_03_pure_lattice_repair"]["status"] == "failed"
    assert nodes_by_id["node_04_greedy_utility"]["status"] == "completed"
    assert journal["selected_node_id"] == "node_04_greedy_utility"
    assert "Unsupported policy entrypoint module" in "; ".join(
        nodes_by_id["node_01_mechanism_aware"]["failure_reasons"]
    )
    assert "synthetic replay failed" in "; ".join(
        nodes_by_id["node_02_fixed_mix"]["failure_reasons"]
    )
    assert "policy replay exploded" in "; ".join(
        nodes_by_id["node_02_fixed_mix"]["failure_reasons"]
    )
    assert "unavailable candidate" in "; ".join(
        nodes_by_id["node_03_pure_lattice_repair"]["failure_reasons"]
    )

    failure_memory = (project / "framework" / "failure_memory.jsonl").read_text(encoding="utf-8")
    assert "node_01_mechanism_aware" in failure_memory
    assert "node_02_fixed_mix" in failure_memory
    assert "node_03_pure_lattice_repair" in failure_memory


def test_develop_method_rejects_node_writing_to_project_root(tmp_path: Path) -> None:
    project = tmp_path / "project"
    backend = FakeWorkspaceBackend()

    def leaky_writer(task: WorkspaceAgentTask) -> WorkspaceAgentResult:
        backend.tasks.append(task)
        _write_fake_method_node(Path(task.workspace), method_name="mechanism_aware", project_root_write=True)
        return WorkspaceAgentResult(
            summary="fake backend completed",
            structured={
                "summary": "fake backend completed",
                "policy_name": f"{Path(task.workspace).name}_invented_policy",
                "policy_entrypoint": "select_batch",
                "files_written": [
                    "proposal.json",
                    "method.py",
                    "manifest.json",
                    "candidate_policy.json",
                    "novelty_report.json",
                    "benchmark_metrics.json",
                    "validation_report.json",
                ],
                "contract_notes": [],
            },
            returncode=0,
        )

    backend.run_task = leaky_writer  # type: ignore[method-assign]

    result = develop_method(project, nodes=1, use_codex=True, backend=backend, run_id="project_leak")
    journal = json.loads(Path(result["journal_path"]).read_text(encoding="utf-8"))

    node = journal["nodes"][0]
    assert node["status"] == "failed"
    assert "out-of-bounds writes detected" in node["failure_reasons"]
    assert any(path.endswith("project_escape.txt") for path in node["contract"]["out_of_bounds_writes"])
    assert journal["selected_node"] is None


def test_policy_callable_extraction_uses_harness_exported_callables(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    backend = FakeWorkspaceBackend()

    def side_effect_writer(task: WorkspaceAgentTask) -> WorkspaceAgentResult:
        backend.tasks.append(task)
        _write_fake_method_node(Path(task.workspace), method_name="mechanism_aware", import_counter=True)
        return WorkspaceAgentResult(
            summary="fake backend completed",
            structured={
                "summary": "fake backend completed",
                "policy_name": f"{Path(task.workspace).name}_invented_policy",
                "policy_entrypoint": "select_batch",
                "files_written": [
                    "proposal.json",
                    "method.py",
                    "manifest.json",
                    "candidate_policy.json",
                    "novelty_report.json",
                    "benchmark_metrics.json",
                    "validation_report.json",
                ],
                "contract_notes": [],
            },
            returncode=0,
        )

    backend.run_task = side_effect_writer  # type: ignore[method-assign]

    result = develop_method(project, nodes=1, use_codex=True, backend=backend, run_id="import_once")
    journal = json.loads(Path(result["journal_path"]).read_text(encoding="utf-8"))
    workspace = Path(journal["nodes"][0]["workspace"])

    assert (workspace / "import_count.txt").read_text(encoding="utf-8") == "2"


def test_run_scientist_search_strict_mode_fails_when_method_extraction_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from design_scientist import method_extraction

    project = tmp_path / "project"
    init_framework(project, domain="protein variant design")

    def fail_extraction(project_dir: Path) -> dict[str, object]:
        return {"status": "failed", "reason": "adapter unavailable", "method_module_count": 0}

    monkeypatch.setattr(method_extraction, "extract_methods", fail_extraction)

    with pytest.raises(RuntimeError, match="method extraction failed"):
        run_scientist_search(
            project,
            max_papers=2,
            nodes=1,
            rounds=1,
            use_codex=False,
            offline_fixtures=True,
        )

    assert not list((project / "runs").glob("*/scientist_journal.json"))


def _write_fake_method_node(
    workspace: Path,
    *,
    method_name: str,
    no_baseline: bool = False,
    baseline_clone: bool = False,
    behavioral_clone: bool = False,
    fake_unique_novelty: bool = False,
    project_root_write: bool = False,
    import_counter: bool = False,
    policy_entrypoint: str = "select_batch",
    policy_id: str | None = None,
    policy_behavior: str = "normal",
) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    import_counter_source = ""
    if import_counter:
        import_counter_source = (
            "_counter = Path(__file__).with_name('import_count.txt'); "
            "_counter.write_text(str((int(_counter.read_text(encoding='utf-8')) "
            "if _counter.exists() else 0) + 1), encoding='utf-8')"
        )
    project_root_write_source = ""
    if project_root_write:
        project_root_write_source = (
            "(root.parents[3] / 'project_escape.txt').write_text('leak', encoding='utf-8')"
        )
    (workspace / "manifest.json").write_text(
        json.dumps(
            {
                "name": workspace.name,
                "entrypoint": "run",
                "policy_entrypoint": "select_batch",
            }
        ),
        encoding="utf-8",
    )
    (workspace / "method.py").write_text(
        dedent(
            f"""
            from __future__ import annotations

            import json
            from pathlib import Path

            from design_scientist import policies as policy_api

            {import_counter_source}

            METHOD = {method_name!r}
            POLICY_NAME = {(policy_id or workspace.name + "_invented_policy")!r}
            NO_BASELINE = {no_baseline!r}
            BASELINE_CLONE = {baseline_clone!r}
            BEHAVIORAL_CLONE = {behavioral_clone!r}
            FAKE_UNIQUE_NOVELTY = {fake_unique_novelty!r}
            POLICY_BEHAVIOR = {policy_behavior!r}


            def _write(path, data):
                Path(path).write_text(json.dumps(data, indent=2) + "\\n", encoding="utf-8")


            def select_batch(observed, candidates, budget, round_index, rng):
                budget = max(0, budget)
                if budget == 0:
                    return []
                if POLICY_BEHAVIOR == "raise":
                    raise RuntimeError("policy replay exploded")
                if POLICY_BEHAVIOR == "invalid_selection":
                    return ["missing_candidate_id"]
                pool = [
                    str(candidate.get("variant_id") or candidate.get("candidate_id") or candidate.get("id"))
                    for candidate in sorted(
                        candidates,
                        key=lambda record: (
                            record.get("background") != record.get("target_background"),
                            len(record.get("modules") or ()),
                            str(record.get("variant_id") or record.get("candidate_id") or record.get("id")),
                        ),
                    )
                ]
                pool = [candidate_id for candidate_id in pool if candidate_id]
                if NO_BASELINE:
                    return pool[:budget]
                base = getattr(policy_api, METHOD)
                selected = []
                selected_ids = set()
                for candidate_id in pool[: max(1, budget // 2)]:
                    selected.append(candidate_id)
                    selected_ids.add(candidate_id)
                fill = list(base(observed, candidates, budget, round_index, rng))
                if BEHAVIORAL_CLONE:
                    return fill[:budget]
                for candidate_id in fill:
                    if candidate_id not in selected:
                        selected.append(candidate_id)
                        selected_ids.add(candidate_id)
                    if len(selected) >= budget:
                        break
                return selected[:budget]


            select_batch.policy_name = POLICY_NAME


            def run(workspace):
                root = Path(workspace)
                _write(root / "proposal.json", {{
                    "schema_version": 2,
                    "proposal_id": {workspace.name!r},
                    "title": "Fake generated policy for " + METHOD,
                    "summary": "Fake backend node with a deterministic diversity replacement.",
                    "method_claim": "A diversity replacement can reduce baseline overlap.",
                    "method_hypothesis": "A non-enum policy can improve transfer search by perturbing " + METHOD,
                    "literature_basis": ["active learning", "transfer-aware design"],
                    "literature_gap_ids": ["gap_constraints_tiny_data"],
                    "reused_components": [] if NO_BASELINE else [METHOD],
                    "architecture_delta": "Adds a gap-conditioned allocation stage before candidate ranking.",
                    "new_mechanism_claim": "Gap-conditioned allocation changes selection before scoring.",
                    "algorithm_mechanism": "Wrap the baseline signal with a deterministic diversity replacement.",
                    "expected_advantage": "Lower baseline overlap while preserving high-utility first choices.",
                    "failure_modes": ["diversity replacement can discard a good final candidate"],
                    "planned_ablation": "disable_gap_conditioning",
                    "planned_ablations": ["remove_diversity_replacement"],
                }})
                _write(root / "candidate_policy.json", {{
                    "policy_id": POLICY_NAME,
                    "policy_entrypoint": {policy_entrypoint!r},
                    **({{}} if NO_BASELINE else {{"source_baseline_family": METHOD}}),
                }})
                _write(root / "novelty_report.json", {{
                    "schema_version": 2,
                    "baseline_clone": BASELINE_CLONE,
                    "baseline_overlap": 0.10 if FAKE_UNIQUE_NOVELTY else (0.90 if BASELINE_CLONE else 0.50),
                    "selection_overlap_vs_baselines": 0.10 if FAKE_UNIQUE_NOVELTY else (0.90 if BASELINE_CLONE else 0.50),
                    "novelty_score": 0.90 if FAKE_UNIQUE_NOVELTY else (0.10 if BASELINE_CLONE else 0.50),
                }})
                _write(root / "benchmark_metrics.json", {{"method": POLICY_NAME if NO_BASELINE else METHOD, "score": 0.0}})
                _write(root / "validation_report.json", {{"valid": True, "findings": []}})
                {project_root_write_source}
            """
        ).lstrip(),
        encoding="utf-8",
    )


def _fake_benchmark_result(
    root: Path,
    *,
    run_id: str,
    rounds: int,
    methods: list[object],
    high_overlap_by_method: dict[str, float] | None = None,
) -> dict[str, object]:
    high_overlap_by_method = high_overlap_by_method or {}
    rows = []
    summary_rows = []
    for method in methods:
        method_name = str(
            getattr(method, "policy_name", None)
            or getattr(method, "__name__", None)
            or method
        )
        is_literature_gap = "literature_gap" in method_name
        source_overlap = high_overlap_by_method.get(method_name, 0.20)
        row = {
            "world_id": "fake_world",
            "method": method_name,
            "best_feasible_utility": 2.0 if is_literature_gap else 1.0,
            "hit_rate": 1.0 if is_literature_gap else 0.2,
            "evidence_coverage": 0.9,
            "round_efficiency": 0.8,
            "regret_proxy": 0.1,
            "false_claim_rate": 0.0,
            "novelty_score": 0.7,
            "selection_overlap_random_feasible": 0.10,
            "selection_overlap_fixed_mix": 0.20,
            "selection_overlap_mechanism_aware": source_overlap,
            "selected_ids_digest": f"digest_{method_name}",
        }
        rows.append(row)
        summary_rows.append(
            {
                "method": method_name,
                "mean_best_feasible_utility": row["best_feasible_utility"],
                "mean_hit_rate": row["hit_rate"],
                "mean_evidence_coverage": row["evidence_coverage"],
                "mean_round_efficiency": row["round_efficiency"],
                "mean_regret_proxy": row["regret_proxy"],
                "mean_false_claim_rate": row["false_claim_rate"],
                "mean_novelty_score": row["novelty_score"],
            }
        )
    run_dir = root / "runs" / run_id
    return {
        "run_id": run_id,
        "benchmark_results_path": str(run_dir / "benchmark_results.csv"),
        "ablation_results_path": str(run_dir / "ablation_results.csv"),
        "summary_results_path": str(run_dir / "benchmark_summary.csv"),
        "benchmark_results": rows,
        "ablation_results": [],
        "summary": {"method_rankings": summary_rows},
    }
