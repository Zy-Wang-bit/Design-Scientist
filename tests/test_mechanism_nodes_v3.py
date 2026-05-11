from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

from design_scientist.mechanism_nodes import (
    V3_MECHANISM_NODE_REQUIRED_ARTIFACTS,
    execute_mechanism_node,
    validate_mechanism_node,
)


def test_select_batch_only_node_is_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "nodes" / "select_batch_only"
    _write_mechanism(
        workspace,
        """
        def select_batch(observed, candidates, budget, round_index, rng):
            return list(candidates)[:budget]
        """,
    )

    result = execute_mechanism_node(workspace, guard_roots=[tmp_path])

    assert not result.valid
    assert not result.executed
    assert "contract" in result.failure_kinds
    assert any("select_batch" in error and "lifecycle" in error for error in result.errors)


def test_lifecycle_node_runs_and_writes_all_v3_artifacts(tmp_path: Path) -> None:
    workspace = tmp_path / "nodes" / "valid_lifecycle"
    _write_mechanism(
        workspace,
        """
        LIFECYCLE_IMPORT_MARKER = []

        def fit_state(observed=None, context=None):
            return {"state_id": "state_1", "observed_count": len(observed or [])}

        def generate_candidates(state, design_space=None):
            return [{"candidate_id": "cand_1"}, {"candidate_id": "cand_2"}]

        def score_candidates(state, candidates):
            return {candidate["candidate_id"]: 1.0 for candidate in candidates}

        def select_panel(scored_candidates, budget=1):
            return ["cand_1"][:budget]

        def plan_ablations(state=None):
            return [{"ablation_id": "remove_transfer_model"}]

        def run(workspace):
            import json
            from pathlib import Path

            root = Path(workspace)
            artifacts = {
                "mechanism_spec.json": {
                    "mechanism_id": "mechanism_v3_test",
                    "name": "V3 test mechanism",
                    "version": "v3",
                    "components": [{"component_id": "state", "component_type": "state_model"}],
                },
                "proposal.json": {
                    "mechanism_id": "mechanism_v3_test",
                    "hypothesis": "Lifecycle nodes expose explicit mechanism stages.",
                },
                "ablation_plan.json": {
                    "ablations": [{"ablation_id": "remove_transfer_model"}],
                },
                "stress_test_plan.json": {
                    "worlds": [{"world_id": "sparse_transfer"}],
                    "required_baselines": ["random_feasible"],
                },
                "mechanism_metrics.json": {
                    "metrics": {"utility": 1.0},
                },
                "validation_report.json": {
                    "valid": True,
                    "findings": [],
                },
            }
            for name, payload in artifacts.items():
                (root / name).write_text(json.dumps(payload), encoding="utf-8")
        """,
    )

    result = execute_mechanism_node(workspace, guard_roots=[tmp_path])
    validation = validate_mechanism_node(workspace)

    assert result.valid
    assert result.executed
    assert result.missing_artifacts == []
    assert result.malformed_json == {}
    assert result.out_of_bounds_writes == []
    assert set(result.artifacts) == set(V3_MECHANISM_NODE_REQUIRED_ARTIFACTS) - {"mechanism.py"}
    assert set(result.callables) == {
        "fit_state",
        "generate_candidates",
        "score_candidates",
        "select_panel",
        "plan_ablations",
    }
    assert callable(result.callables["fit_state"])
    assert validation.valid


def test_malformed_json_invalid(tmp_path: Path) -> None:
    workspace = tmp_path / "nodes" / "malformed_json"
    _write_mechanism(
        workspace,
        """
        def fit_state(observed=None, context=None):
            return {}

        def generate_candidates(state, design_space=None):
            return []

        def score_candidates(state, candidates):
            return {}

        def select_panel(scored_candidates, budget=1):
            return []

        def plan_ablations(state=None):
            return []

        def run(workspace):
            import json
            from pathlib import Path

            root = Path(workspace)
            (root / "mechanism_spec.json").write_text("{not-json", encoding="utf-8")
            (root / "proposal.json").write_text(json.dumps({"proposal": "ok"}), encoding="utf-8")
            (root / "ablation_plan.json").write_text(json.dumps({"ablations": [{"ablation_id": "a1"}]}), encoding="utf-8")
            (root / "stress_test_plan.json").write_text(json.dumps({"worlds": [{"world_id": "w1"}]}), encoding="utf-8")
            (root / "mechanism_metrics.json").write_text(json.dumps({"metrics": {"score": 0.0}}), encoding="utf-8")
            (root / "validation_report.json").write_text(json.dumps({"valid": True}), encoding="utf-8")
        """,
    )

    result = execute_mechanism_node(workspace, guard_roots=[tmp_path])

    assert not result.valid
    assert result.executed
    assert "malformed_artifacts" in result.failure_kinds
    assert "mechanism_spec.json" in result.malformed_json


def test_out_of_bounds_write_invalid(tmp_path: Path) -> None:
    workspace = tmp_path / "nodes" / "out_of_bounds_write"
    outside_file = tmp_path / "outside.json"
    _write_mechanism(
        workspace,
        f"""
        def fit_state(observed=None, context=None):
            return {{}}

        def generate_candidates(state, design_space=None):
            return []

        def score_candidates(state, candidates):
            return {{}}

        def select_panel(scored_candidates, budget=1):
            return []

        def plan_ablations(state=None):
            return []

        def run(workspace):
            import json
            from pathlib import Path

            root = Path(workspace)
            artifacts = {{
                "mechanism_spec.json": {{"mechanism_id": "escape", "name": "escape"}},
                "proposal.json": {{"hypothesis": "escape"}},
                "ablation_plan.json": {{"ablations": [{{"ablation_id": "a1"}}]}},
                "stress_test_plan.json": {{"worlds": [{{"world_id": "w1"}}]}},
                "mechanism_metrics.json": {{"metrics": {{"score": 1.0}}}},
                "validation_report.json": {{"valid": True}},
            }}
            for name, payload in artifacts.items():
                (root / name).write_text(json.dumps(payload), encoding="utf-8")
            Path({str(outside_file)!r}).write_text(json.dumps({{"escaped": True}}), encoding="utf-8")
        """,
    )

    result = execute_mechanism_node(workspace, guard_roots=[tmp_path])

    assert not result.valid
    assert result.executed
    assert "path_guard" in result.failure_kinds
    assert str(outside_file) in result.out_of_bounds_writes


def test_missing_lifecycle_callable_invalid(tmp_path: Path) -> None:
    workspace = tmp_path / "nodes" / "missing_lifecycle"
    _write_mechanism(
        workspace,
        """
        def fit_state(observed=None, context=None):
            return {}

        def generate_candidates(state, design_space=None):
            return []

        def score_candidates(state, candidates):
            return {}

        def select_panel(scored_candidates, budget=1):
            return []
        """,
    )

    result = execute_mechanism_node(workspace, guard_roots=[tmp_path])

    assert not result.valid
    assert not result.executed
    assert "contract" in result.failure_kinds
    assert any("plan_ablations" in error for error in result.errors)


def _write_mechanism(workspace: Path, body: str) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "mechanism.py").write_text(dedent(body).strip() + "\n", encoding="utf-8")
