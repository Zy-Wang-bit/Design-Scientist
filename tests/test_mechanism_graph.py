from __future__ import annotations

from design_scientist.mechanism_graph import build_mechanism_graph


def test_build_mechanism_graph_returns_components_gaps_and_edges() -> None:
    cards = [
        {
            "mechanism_id": "transfer_prior_design",
            "source_paper_ids": ["paper_b"],
            "state_model": "Target-task state with source-task priors.",
            "candidate_generation": "Generate target candidates.",
            "acquisition_objective": "Balance target utility and transfer risk.",
            "uncertainty_model": "",
            "transfer_model": "Source-to-target prior.",
            "stress_tests": ["source_ablation"],
            "reusable_components": ["transfer prior", "source-target validation"],
        },
        {
            "mechanism_id": "active_learning_acquisition",
            "source_paper_ids": ["paper_a"],
            "state_model": "Surrogate state.",
            "candidate_generation": "Generate feasible candidates.",
            "acquisition_objective": "Expected improvement.",
            "uncertainty_model": "Posterior variance.",
            "transfer_model": "",
            "stress_tests": [],
            "reusable_components": ["surrogate model", "acquisition policy"],
        },
    ]

    graph = build_mechanism_graph(cards)

    assert set(graph) == {"components", "gaps", "edges"}
    assert [component["component_id"] for component in graph["components"]] == [
        "acquisition_policy",
        "source_target_validation",
        "surrogate_model",
        "transfer_prior",
    ]
    assert graph["edges"][0] == {
        "source": "active_learning_acquisition",
        "target": "acquisition_policy",
        "edge_type": "uses_component",
    }
    assert {
        "mechanism_id": "active_learning_acquisition",
        "missing_fields": ["transfer_model", "stress_tests"],
    } in graph["gaps"]
    assert {
        "mechanism_id": "transfer_prior_design",
        "missing_fields": ["uncertainty_model"],
    } in graph["gaps"]
