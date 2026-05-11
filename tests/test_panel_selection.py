from __future__ import annotations

from design_scientist.acquisition import default_policy
from design_scientist.candidates import generate_candidates
from design_scientist.panel import (
    compare_baselines,
    fixed_mix,
    pure_lattice_repair,
    random_feasible,
    select_panel,
    top_predicted_utility,
)
from design_scientist.schemas import Candidate


def test_panel_selection_respects_budget_and_baselines() -> None:
    state = {
        "module_status": [
            {"module_id": "HD110H"},
            {"module_id": "HG56H"},
            {"module_id": "HN54H"},
        ],
        "unresolved_edges": [
            {"system": "1E62", "module_id": "HA23K", "base_variant": "com1", "reason": "repair edge"}
        ],
    }
    candidates = generate_candidates(state, [], strategy="mechanism_aware")
    panel, score = select_panel(candidates, default_policy("mechanism_aware"), budget=4)
    baselines = compare_baselines(candidates, budget=4)

    assert sum(candidate.cost for candidate in panel) <= 4
    assert all("candidate" not in str(candidate.background) for candidate in panel)
    assert score > 0
    assert any(candidate.category == "lattice_repair" for candidate in candidates)
    assert any(item.startswith("fixed_mix:") for item in baselines)


def test_panel_selection_uses_cost_budget_not_row_count() -> None:
    candidates = [
        Candidate(
            candidate_id="expensive_best",
            operator="complete_square",
            category="interaction_square",
            target_system="1E62",
            background="real_bg",
            modules=["HD110H", "HG56H"],
            score_components={"performance": 5.0},
            cost=2.0,
        ),
        Candidate(
            candidate_id="expensive_second",
            operator="complete_missing_edge",
            category="lattice_repair",
            target_system="1E62",
            background="real_bg",
            modules=["HN54H"],
            score_components={"performance": 4.0},
            cost=2.0,
        ),
        Candidate(
            candidate_id="cheap_fill",
            operator="add_module",
            category="champion",
            target_system="1E62",
            background="real_bg",
            modules=["HV105H"],
            score_components={"performance": 1.0},
            cost=1.0,
        ),
    ]

    panel, _score, diagnostics = select_panel(
        candidates,
        default_policy("mechanism_aware"),
        budget=3,
        return_diagnostics=True,
    )

    assert [candidate.candidate_id for candidate in panel] == ["expensive_best", "cheap_fill"]
    assert sum(candidate.cost for candidate in panel) == 3.0
    assert diagnostics["total_cost"] == 3.0
    assert diagnostics["cost_budget"] == 3.0


def test_panel_selectors_exclude_infeasible_and_forbidden_candidates() -> None:
    candidates = [
        Candidate(
            candidate_id="valid_singleton",
            operator="add_module",
            category="champion",
            target_system="1E62",
            background="real_bg",
            modules=["HD110H"],
            score_components={"performance": 0.1},
        ),
        Candidate(
            candidate_id="infeasible_high_score",
            operator="add_module",
            category="champion",
            target_system="1E62",
            background="real_bg",
            modules=["HG56H"],
            score_components={"performance": 10.0},
            feasibility_status="infeasible",
        ),
        Candidate(
            candidate_id="forbidden_pair",
            operator="complete_square",
            category="interaction_square",
            target_system="1E62",
            background="real_bg",
            modules=["KD31N", "SY92F"],
            score_components={"performance": 9.0, "lattice_repair": 9.0},
        ),
    ]

    panel, _score = select_panel(candidates, default_policy("mechanism_aware"), budget=3)
    selector_outputs = [
        panel,
        random_feasible(candidates, budget=3),
        top_predicted_utility(candidates, budget=3),
        pure_lattice_repair(candidates, budget=3),
        fixed_mix(candidates, budget=3),
    ]

    for selected in selector_outputs:
        assert [candidate.candidate_id for candidate in selected] == ["valid_singleton"]
