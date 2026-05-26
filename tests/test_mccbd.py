from __future__ import annotations

import random

import pytest

from design_scientist.algorithms import evidence_calibrated_ucb, mccbd


def test_fit_state_learns_favorable_module_and_ranks_candidates_above_neutral() -> None:
    observed = [
        {"variant_id": "base_a", "modules": ["base"], "observed_utility": 0.10, "true_feasible": True},
        {"variant_id": "fav_a", "modules": ["base", "fav"], "observed_utility": 1.00, "true_feasible": True},
        {"variant_id": "fav_b", "modules": ["fav"], "observed_utility": 0.90, "true_feasible": True},
        {"variant_id": "neutral_a", "modules": ["neutral"], "observed_utility": 0.20, "true_feasible": True},
    ]
    candidates = [
        {"candidate_id": "candidate_fav", "modules": ["fav"], "cost": 1.0},
        {"candidate_id": "candidate_neutral", "modules": ["neutral"], "cost": 1.0},
    ]

    state = mccbd.fit_state(observed, candidates)
    scored = mccbd.score_candidates(observed, candidates, fitted_state=state)
    by_id = {row["candidate_id"]: row for row in scored}

    assert by_id["candidate_fav"]["posterior_mean"] > by_id["candidate_neutral"]["posterior_mean"]
    assert by_id["candidate_fav"]["score"] > by_id["candidate_neutral"]["score"]
    assert any(name == "module:fav" for name in state.feature_names)


def test_constrained_ei_avoids_infeasible_high_mean_risky_candidate() -> None:
    observed = [
        {"id": "good_feasible", "module_id": "good", "observed_utility": 0.70, "true_feasible": True},
        {"id": "risky_high", "module_id": "risky", "observed_utility": 1.30, "true_feasible": False},
        {"id": "risky_repeat", "module_id": "risky", "observed_utility": 1.10, "true_feasible": False},
        {"id": "baseline", "module_id": "base", "observed_utility": 0.20, "true_feasible": True},
    ]
    candidates = [
        {"candidate_id": "risky_candidate", "module_id": "risky", "cost": 1.0},
        {"candidate_id": "good_candidate", "module_id": "good", "cost": 1.0},
    ]

    scored = mccbd.score_candidates(observed, candidates)
    by_id = {row["candidate_id"]: row for row in scored}
    selected = mccbd.select_batch(
        observed,
        candidates,
        budget=1,
        round_index=0,
        rng=random.Random(9),
    )

    assert by_id["risky_candidate"]["posterior_mean"] > by_id["good_candidate"]["posterior_mean"]
    assert by_id["risky_candidate"]["feasibility_probability"] < by_id["good_candidate"]["feasibility_probability"]
    assert by_id["risky_candidate"]["constrained_expected_improvement"] < by_id["good_candidate"]["constrained_expected_improvement"]
    assert selected == ["good_candidate"]


def test_batch_selection_respects_cost_budget_and_avoids_redundant_second_choice() -> None:
    observed = [
        {"variant_id": "anchor_a", "modules": ["anchor", "helper"], "observed_utility": 1.00, "true_feasible": True},
        {"variant_id": "anchor_b", "modules": ["anchor"], "observed_utility": 0.92, "true_feasible": True},
        {"variant_id": "distinct_a", "modules": ["distinct", "support"], "observed_utility": 0.88, "true_feasible": True},
        {"variant_id": "baseline", "modules": ["baseline"], "observed_utility": 0.20, "true_feasible": True},
    ]
    candidates = [
        {"candidate_id": "anchor_panel_a", "modules": ["anchor", "helper", "novel_a"], "cost": 1.0},
        {"candidate_id": "anchor_panel_b", "modules": ["anchor", "helper", "novel_b"], "cost": 1.0},
        {"candidate_id": "distinct_panel", "modules": ["distinct", "support", "novel_c"], "cost": 1.0},
        {"candidate_id": "too_expensive", "modules": ["anchor", "helper", "novel_d"], "cost": 3.0},
    ]

    selected = mccbd.select_batch(
        observed,
        candidates,
        budget=2,
        round_index=0,
        rng=random.Random(5),
    )

    assert "too_expensive" not in selected
    assert len(selected) == 2
    assert "distinct_panel" in selected
    assert not {"anchor_panel_a", "anchor_panel_b"} <= set(selected)


def test_weighted_ucb_is_distinct_baseline_and_mccbd_exposes_posterior_ei_components() -> None:
    observed = [
        {"variant_id": "base", "modules": [], "observed_utility": 0.10, "true_feasible": True},
        {"variant_id": "m1", "modules": ["M1"], "observed_utility": 0.80, "true_feasible": True},
    ]
    candidates = [{"candidate_id": "candidate", "modules": ["M1", "M2"], "cost": 1.0}]

    mccbd_row = mccbd.score_candidates(observed, candidates)[0]
    ucb_row = evidence_calibrated_ucb.score_candidates(observed, candidates)[0]

    assert mccbd.mechanism_lifecycle().name == "mccbd"
    assert evidence_calibrated_ucb.mechanism_lifecycle().name == "evidence_calibrated_ucb"
    assert "posterior_mean" in mccbd_row
    assert "posterior_std" in mccbd_row
    assert "constrained_expected_improvement" in mccbd_row
    assert "feasibility_probability" in mccbd_row
    assert "information_gain_proxy" in mccbd_row
    assert "constrained_expected_improvement" in mccbd_row["score_components"]
    assert "predicted_utility" in ucb_row["score_components"]
    assert "constrained_expected_improvement" not in ucb_row["score_components"]


def test_select_batch_returns_candidate_ids_and_never_observed_ids() -> None:
    observed = [
        {"variant_id": "already_tested", "modules": ["winner"], "observed_utility": 1.0, "true_feasible": True},
        {"variant_id": "control", "modules": ["control"], "observed_utility": 0.1, "true_feasible": True},
    ]
    candidates = [
        {"candidate_id": "already_tested", "modules": ["winner"], "cost": 1.0},
        {"id": "fresh_id_field", "modules": ["winner", "new"], "cost": 1.0},
        {"candidate_id": "fresh_candidate_field", "modules": ["control", "new"], "cost": 1.0},
    ]

    selected = mccbd.select_batch(
        observed,
        candidates,
        budget=2,
        round_index=0,
        rng=random.Random(2),
    )

    assert selected
    assert all(isinstance(candidate_id, str) for candidate_id in selected)
    assert "already_tested" not in selected
    assert set(selected) <= {"fresh_id_field", "fresh_candidate_field"}


def test_candidate_feasibility_status_does_not_override_observed_constraint_posterior() -> None:
    observed = [
        {"variant_id": "safe_a", "modules": ["safe"], "observed_utility": 0.70, "true_feasible": True},
        {"variant_id": "risky_a", "modules": ["risky"], "observed_utility": 1.20, "true_feasible": False},
        {"variant_id": "risky_b", "modules": ["risky"], "observed_utility": 1.10, "true_feasible": False},
    ]
    candidates = [
        {"candidate_id": "risky_candidate", "modules": ["risky"], "feasibility_status": "feasible"},
        {"candidate_id": "safe_candidate", "modules": ["safe"], "feasibility_status": "feasible"},
    ]

    by_id = {row["candidate_id"]: row for row in mccbd.score_candidates(observed, candidates)}

    assert by_id["risky_candidate"]["feasibility_probability"] < 0.5
    assert by_id["safe_candidate"]["feasibility_probability"] > by_id["risky_candidate"]["feasibility_probability"]


def test_feature_scale_calibration_prevents_long_unseen_architecture_from_winning_by_variance_only() -> None:
    observed = [
        {"variant_id": "compact_good", "modules": ["good"], "observed_utility": 0.80, "true_feasible": True},
        {"variant_id": "compact_good_b", "modules": ["good", "helper"], "observed_utility": 0.78, "true_feasible": True},
        {"variant_id": "base", "modules": ["base"], "observed_utility": 0.10, "true_feasible": True},
    ]
    candidates = [
        {"candidate_id": "compact_candidate", "modules": ["good", "helper"], "cost": 1.0},
        {
            "candidate_id": "long_unseen_candidate",
            "modules": ["u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8"],
            "cost": 1.0,
        },
    ]

    selected = mccbd.select_batch(
        observed,
        candidates,
        budget=1,
        round_index=0,
        rng=random.Random(11),
    )

    assert selected == ["compact_candidate"]
