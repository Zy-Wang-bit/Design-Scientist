from __future__ import annotations

from design_scientist.novelty_audit import audit_mechanism_novelty


def test_audit_rejects_component_level_baseline_clone() -> None:
    mechanism = {
        "name": "topk wrapper",
        "components": ["rank_by_score", "top_k_select", "random_tie_break"],
        "architecture_delta": "Wraps the same top-k selector.",
        "literature_refs": ["Smith 2024"],
    }
    baselines = [
        {
            "name": "top_observed",
            "components": ["rank_by_score", "top_k_select", "random_tie_break"],
        }
    ]

    audit = audit_mechanism_novelty(mechanism, baselines)

    assert audit["verdict"] == "reject"
    assert audit["baseline_clone"] is True
    assert audit["nearest_baseline"] == "top_observed"
    assert audit["component_overlap"] == 1.0
    assert any("component clone" in finding for finding in audit["findings"])


def test_audit_rejects_weak_delta_against_component_baseline() -> None:
    mechanism = {
        "name": "one-term tweak",
        "components": [
            "rank_by_score",
            "top_k_select",
            "random_tie_break",
            "novelty_penalty",
        ],
        "architecture_delta": "Adds one scoring term to the baseline.",
        "literature_refs": ["Lee 2025"],
    }
    baselines = [
        {
            "name": "top_observed",
            "components": ["rank_by_score", "top_k_select", "random_tie_break"],
        }
    ]

    audit = audit_mechanism_novelty(mechanism, baselines)

    assert audit["verdict"] == "reject"
    assert audit["baseline_clone"] is False
    assert audit["weak_delta"] is True
    assert audit["new_components"] == ["novelty_penalty"]
    assert any("weak delta" in finding for finding in audit["findings"])


def test_audit_allows_helper_reuse_with_architecture_delta_and_refs() -> None:
    mechanism = {
        "name": "contrast lattice repair",
        "components": [
            {"component_id": "shared_normalize", "component_type": "helper"},
            "contrast_state_model",
            "lattice_gap_detector",
            "repair_panel_selector",
        ],
        "architecture_delta": (
            "Separates state fitting, lattice gap detection, and repair panel "
            "selection instead of ranking candidates directly."
        ),
        "literature_refs": ["Nguyen 2024", "Patel 2025"],
    }
    baselines = [
        {
            "name": "uncertainty_sampling",
            "components": [
                {"component_id": "shared_normalize", "component_type": "helper"},
                "uncertainty_score",
                "top_k_select",
            ],
        }
    ]

    audit = audit_mechanism_novelty(mechanism, baselines)

    assert audit["verdict"] == "pass"
    assert audit["baseline_clone"] is False
    assert audit["weak_delta"] is False
    assert audit["missing_required_fields"] == []


def test_audit_requires_architecture_delta_and_literature_refs() -> None:
    mechanism = {
        "name": "under-specified mechanism",
        "components": ["contrast_state_model", "repair_panel_selector"],
    }

    audit = audit_mechanism_novelty(mechanism, baselines=[])

    assert audit["verdict"] == "reject"
    assert audit["missing_required_fields"] == ["architecture_delta", "literature_refs"]
