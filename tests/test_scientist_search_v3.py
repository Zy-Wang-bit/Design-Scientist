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
    assert mechanism_spec["claims"]
    assert mechanism_spec["literature_basis"]
    assert mechanism_spec["stress_test_requirements"]


def test_offline_local_fallback_does_not_select_mechanism_aware_wrapper(
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

    assert result["selected_node"] is not None
    selected = result["selected_node"]
    assert "mechanism_aware" not in selected["node_id"]
    workspace = Path(selected["workspace"])
    proposal = json.loads((workspace / "proposal.json").read_text(encoding="utf-8"))
    assert "mechanism_aware" not in proposal.get("reused_components", [])


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


            def fit_state(context):
                return dict(context)


            def generate_candidates(state):
                return list(state["candidate_records"])


            def score_candidates(state, candidates):
                del state
                return list(candidates)


            def select_panel(state, candidates, budget, rng):
                return policies.mechanism_aware(
                    state["observed_records"],
                    candidates,
                    budget,
                    state["round_index"],
                    rng,
                )


            def plan_ablations(state):
                del state
                return [
                    {"name": "none"},
                    {
                        "name": "key_component_removed",
                        "policy_ablation": "interaction_prior",
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
                        "architecture_delta": "Uses lifecycle replay rather than select_batch-only dispatch.",
                    },
                    "proposal.json": {
                        "mechanism_id": MECHANISM_ID,
                        "hypothesis": "Mechanism-aware lifecycle nodes can be benchmarked.",
                    },
                    "ablation_plan.json": {
                        "ablations": [{"name": "key_component_removed", "policy_ablation": "interaction_prior"}],
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
