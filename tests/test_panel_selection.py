from __future__ import annotations

from design_scientist.acquisition import default_policy
from design_scientist.candidates import generate_candidates
from design_scientist.panel import compare_baselines, select_panel


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

    assert len(panel) == 4
    assert score > 0
    assert any(candidate.category == "lattice_repair" for candidate in candidates)
    assert any(item.startswith("fixed_mix:") for item in baselines)

