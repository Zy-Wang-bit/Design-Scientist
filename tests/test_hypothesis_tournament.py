from __future__ import annotations

from design_scientist.hypothesis_tournament import (
    FakeHypothesisAgent,
    run_hypothesis_tournament,
)


def test_tournament_rejects_mechanism_without_gap() -> None:
    agents = [
        FakeHypothesisAgent(
            name="generator",
            output={
                "mechanism_id": "m1",
                "hypothesis": "reuse fixed_mix with a new name",
                "literature_gap": "",
                "components": ["fixed_mix"],
            },
        )
    ]

    result = run_hypothesis_tournament(
        mechanism_library=[],
        gap_matrix=[],
        failure_memory=[],
        agents=agents,
    )

    assert result["selected_for_implementation"] == []
    assert result["candidates"][0]["status"] == "rejected"
    assert "missing_literature_gap" in result["candidates"][0]["reasons"]


def test_tournament_selects_accepted_candidates_in_agent_order() -> None:
    agents = [
        FakeHypothesisAgent(
            name="generator_a",
            output={
                "mechanism_id": "m1",
                "hypothesis": "condition acquisition on sparse transfer gaps",
                "literature_gap": "tiny wet-lab campaigns lack transfer-aware acquisition",
                "components": ["gap_conditioned_scoring"],
                "failure_signature": "sig-new-a",
            },
        ),
        FakeHypothesisAgent(
            name="generator_b",
            output={
                "mechanism_id": "m2",
                "hypothesis": "repair missing contrast edges before selection",
                "literature_gap": "contrast completion is rarely tied to pH-switch objectives",
                "components": [{"name": "contrast_repair"}],
                "failure_signature": "sig-new-b",
            },
        ),
    ]

    result = run_hypothesis_tournament(
        mechanism_library=[],
        gap_matrix=[],
        failure_memory=[],
        agents=agents,
    )

    assert result["selected_for_implementation"] == ["m1", "m2"]
    assert [candidate["status"] for candidate in result["candidates"]] == [
        "accepted",
        "accepted",
    ]
    assert result["candidates"][0]["proposal"]["hypothesis"].startswith(
        "condition acquisition"
    )
    assert result["candidates"][0]["skeptic_notes"] == []
    assert result["candidates"][0]["novelty_notes"] == []
    assert result["candidates"][0]["engineering_notes"] == []


def test_tournament_rejects_missing_components_baseline_name_and_known_failure() -> None:
    agents = [
        FakeHypothesisAgent(
            name="missing_components",
            output={
                "mechanism_id": "m1",
                "hypothesis": "state a gap but no mechanism parts",
                "literature_gap": "gap exists",
                "components": [],
            },
        ),
        FakeHypothesisAgent(
            name="baseline_copy",
            output={
                "mechanism_id": "fixed_mix",
                "hypothesis": "rename the baseline",
                "literature_gap": "gap exists",
                "components": ["quota"],
            },
        ),
        FakeHypothesisAgent(
            name="known_failure",
            output={
                "mechanism_id": "m3",
                "hypothesis": "repeat a failed mechanism",
                "literature_gap": "gap exists",
                "components": ["repair"],
                "failure_signature": "sig-repeat",
            },
        ),
    ]

    result = run_hypothesis_tournament(
        mechanism_library=[
            {"mechanism_id": "fixed_mix", "kind": "baseline"},
            {"mechanism_id": "novel_graph", "kind": "candidate"},
        ],
        gap_matrix=[],
        failure_memory=[{"failure_signature": "sig-repeat"}],
        agents=agents,
    )

    assert result["selected_for_implementation"] == []
    assert "missing_components" in result["candidates"][0]["reasons"]
    assert "copies_baseline_name" in result["candidates"][1]["reasons"]
    assert "known_failure_signature" in result["candidates"][2]["reasons"]
