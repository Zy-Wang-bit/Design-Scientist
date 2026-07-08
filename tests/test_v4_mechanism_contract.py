from pathlib import Path

from design_scientist.mechanism_nodes import execute_mechanism_node
from design_scientist.schemas import MechanismComponent, MechanismSpec


def test_v4_rejects_select_batch_only_node(tmp_path: Path):
    node = tmp_path / "node"
    node.mkdir()
    (node / "mechanism.py").write_text(
        "def select_batch(observed, candidates, budget):\n"
        "    return []\n",
        encoding="utf-8",
    )

    result = execute_mechanism_node(
        node,
        guard_roots=[tmp_path],
        require_generated_artifacts=False,
    )

    assert result.valid is False
    assert any("select_batch" in error for error in result.errors)


def test_v4_accepts_full_lifecycle_node_without_generated_artifacts(tmp_path: Path):
    node = tmp_path / "node"
    node.mkdir()
    (node / "mechanism.py").write_text(
        "def fit_state(observations=None, literature=None, prior_state=None):\n"
        "    return {'n': len(observations or [])}\n"
        "def generate_candidates(state, design_space=None):\n"
        "    return [{'candidate_id': 'c1'}]\n"
        "def score_candidates(state, candidates):\n"
        "    return [{'candidate_id': 'c1', 'score': 1.0}]\n"
        "def select_panel(scored_candidates, budget):\n"
        "    return ['c1']\n"
        "def plan_ablations(mechanism_spec=None):\n"
        "    return [{'name': 'remove_generator'}]\n",
        encoding="utf-8",
    )

    result = execute_mechanism_node(
        node,
        guard_roots=[tmp_path],
        require_generated_artifacts=False,
    )

    assert result.valid is True
    assert result.missing_artifacts == []


def test_mechanism_spec_records_components():
    spec = MechanismSpec(
        mechanism_id="mechanism_001",
        name="stateful generator",
        hypothesis="stateful generation improves sparse data design",
        components=[
            MechanismComponent(
                component_id="state_model",
                component_type="state",
                description="tracks evidence across rounds",
                literature_refs=["paper_001"],
            )
        ],
    )

    assert spec.components[0].component_id == "state_model"
