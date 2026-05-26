from __future__ import annotations

import random
from pathlib import Path

import pytest

from design_scientist.algorithms import evidence_calibrated_ucb


def _observed_records() -> list[dict[str, object]]:
    return [
        {
            "variant_id": "bgA__WT",
            "background": "bgA",
            "modules": [],
            "observed_utility": 0.20,
            "endpoint_values": {"primary": 0.20},
        },
        {
            "variant_id": "bgA__M1",
            "background": "bgA",
            "modules": ["M1"],
            "observed_utility": 0.78,
            "endpoint_values": {"primary": 0.78},
        },
        {
            "variant_id": "bgA__M2",
            "background": "bgA",
            "modules": ["M2"],
            "observed_utility": 0.70,
            "endpoint_values": {"primary": 0.70},
        },
        {
            "variant_id": "bgA__M1_M2",
            "background": "bgA",
            "modules": ["M1", "M2"],
            "observed_utility": 0.92,
            "endpoint_values": {"primary": 0.92},
        },
        {
            "variant_id": "bgA__M3",
            "background": "bgA",
            "modules": ["M3"],
            "observed_utility": 0.35,
            "endpoint_values": {"primary": 0.35},
        },
    ]


def _candidate_records() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": "bgB__M1_M2_M3",
            "background": "bgB",
            "modules": ["M1", "M2", "M3"],
            "uncertainty": 0.65,
            "cost": 1.0,
            "target_background": "bgB",
            "required_measurements": ["primary"],
            "missing_endpoints": [],
            "source_refs": [{"kind": "screen", "id": "plate-1"}],
        },
        {
            "candidate_id": "bgB__M1",
            "background": "bgB",
            "modules": ["M1"],
            "uncertainty": 0.10,
            "cost": 1.0,
            "target_background": "bgB",
            "required_measurements": ["primary"],
            "missing_endpoints": [],
        },
        {
            "candidate_id": "bgB__M3",
            "background": "bgB",
            "modules": ["M3"],
            "uncertainty": 0.20,
            "cost": 1.0,
            "target_background": "bgB",
            "required_measurements": ["primary"],
            "missing_endpoints": [],
        },
    ]


def _neighborhood_observed_records() -> list[dict[str, object]]:
    return [
        {
            "variant_id": "top_arch_a",
            "modules": ["core_a", "core_b", "bridge"],
            "observed_utility": 0.95,
            "endpoint_values": {"primary": 0.95},
            "true_feasible": True,
        },
        {
            "variant_id": "top_arch_b",
            "modules": ["core_a", "core_c", "bridge"],
            "observed_utility": 0.82,
            "endpoint_values": {"primary": 0.82},
            "true_feasible": True,
        },
        {
            "variant_id": "bad_arch_a",
            "modules": ["core_a", "toxic_a", "toxic_b"],
            "observed_utility": -0.70,
            "endpoint_values": {"primary": -0.70},
            "true_feasible": False,
        },
        {
            "variant_id": "background",
            "modules": ["background"],
            "observed_utility": 0.05,
            "endpoint_values": {"primary": 0.05},
            "true_feasible": True,
        },
    ]


def test_select_batch_prefers_high_utility_uncertain_candidate() -> None:
    selected = evidence_calibrated_ucb.select_batch(
        _observed_records(),
        _candidate_records(),
        budget=1,
        round_index=1,
        rng=random.Random(17),
    )

    assert selected == ["bgB__M1_M2_M3"]


def test_select_batch_respects_infeasible_status_and_cost_budget() -> None:
    observed = _observed_records()
    candidates = [
        {
            "candidate_id": "blocked_high_score",
            "background": "bgB",
            "modules": ["M1", "M2"],
            "uncertainty": 0.99,
            "cost": 1.0,
            "feasibility_status": "infeasible",
        },
        {
            "candidate_id": "too_expensive",
            "background": "bgB",
            "modules": ["M1", "M2", "M3"],
            "uncertainty": 0.99,
            "cost": 3.0,
        },
        {
            "candidate_id": "affordable_a",
            "background": "bgB",
            "modules": ["M1"],
            "uncertainty": 0.20,
            "cost": 1.0,
        },
        {
            "candidate_id": "affordable_b",
            "background": "bgB",
            "modules": ["M2"],
            "uncertainty": 0.20,
            "cost": 1.0,
        },
    ]

    selected = evidence_calibrated_ucb.select_batch(
        observed,
        candidates,
        budget=2,
        round_index=0,
        rng=random.Random(11),
    )

    assert set(selected) == {"affordable_a", "affordable_b"}


def test_score_candidates_exposes_interpretable_formula_components() -> None:
    scored = evidence_calibrated_ucb.score_candidates(
        _observed_records(),
        [
            {
                "candidate_id": "bgB__M1_M4",
                "background": "bgB",
                "modules": ["M1", "M4"],
                "uncertainty": 0.30,
                "risk_flags": ["retention_shift", "qc_guardrail"],
                "cost": 1.5,
                "required_measurements": ["primary", "secondary"],
                "missing_endpoints": ["secondary"],
                "source_refs": [{"kind": "future_holdout", "id": "reserved"}],
            }
        ],
        round_index=2,
        beta=0.40,
    )

    assert len(scored) == 1
    result = scored[0]
    components = result["score_components"]
    expected_keys = {
        "predicted_utility",
        "uncertainty",
        "uncertainty_bonus",
        "top_neighbor_bonus",
        "motif_enrichment_bonus",
        "contrast_bonus",
        "coverage_bonus",
        "bad_neighbor_penalty",
        "retention_risk_penalty",
        "guardrail_risk_penalty",
        "cost_penalty",
        "leakage_penalty",
        "missing_evidence_penalty",
        "total_score",
    }
    assert expected_keys <= set(components)

    formula_score = (
        components["predicted_utility"]
        + components["uncertainty_bonus"]
        + components["top_neighbor_bonus"]
        + components["motif_enrichment_bonus"]
        + components["contrast_bonus"]
        + components["coverage_bonus"]
        - components["bad_neighbor_penalty"]
        - components["retention_risk_penalty"]
        - components["guardrail_risk_penalty"]
        - components["cost_penalty"]
        - components["leakage_penalty"]
        - components["missing_evidence_penalty"]
    )
    assert result["candidate_id"] == "bgB__M1_M4"
    assert result["score"] == pytest.approx(formula_score)
    assert components["total_score"] == pytest.approx(result["score"])
    assert components["leakage_penalty"] > 0
    assert components["missing_evidence_penalty"] > 0
    assert components["retention_risk_penalty"] > 0


def test_score_candidates_uses_top_neighbor_enrichment_and_bad_neighbor_penalty() -> None:
    scored = evidence_calibrated_ucb.score_candidates(
        _neighborhood_observed_records(),
        [
            {
                "candidate_id": "top_neighbor_architecture",
                "modules": ["core_a", "core_b", "core_c", "bridge"],
                "uncertainty": 0.0,
                "cost": 1.0,
            },
            {
                "candidate_id": "bad_neighbor_architecture",
                "modules": ["core_a", "toxic_a", "toxic_b", "bridge"],
                "uncertainty": 0.0,
                "cost": 1.0,
            },
        ],
        round_index=0,
        beta=0.0,
    )

    by_id = {row["candidate_id"]: row for row in scored}
    top_components = by_id["top_neighbor_architecture"]["score_components"]
    bad_components = by_id["bad_neighbor_architecture"]["score_components"]

    assert top_components["top_neighbor_bonus"] > 0
    assert top_components["motif_enrichment_bonus"] > 0
    assert bad_components["bad_neighbor_penalty"] > top_components["bad_neighbor_penalty"]
    assert by_id["top_neighbor_architecture"]["score"] > by_id["bad_neighbor_architecture"]["score"]


def test_select_batch_uses_batch_architecture_delta_after_first_pick() -> None:
    observed = [
        {
            "variant_id": "high_anchor",
            "modules": ["anchor", "helper"],
            "observed_utility": 1.0,
            "endpoint_values": {"primary": 1.0},
            "true_feasible": True,
        },
        {
            "variant_id": "mid_alt",
            "modules": ["alt", "support"],
            "observed_utility": 0.55,
            "endpoint_values": {"primary": 0.55},
            "true_feasible": True,
        },
        {
            "variant_id": "low_risk",
            "modules": ["risk"],
            "observed_utility": -0.40,
            "endpoint_values": {"primary": -0.40},
            "true_feasible": False,
        },
    ]
    candidates = [
        {
            "candidate_id": "near_duplicate_a",
            "modules": ["anchor", "helper", "x1"],
            "uncertainty": 0.10,
            "cost": 1.0,
        },
        {
            "candidate_id": "near_duplicate_b",
            "modules": ["anchor", "helper", "x2"],
            "uncertainty": 0.10,
            "cost": 1.0,
        },
        {
            "candidate_id": "distinct_architecture",
            "modules": ["alt", "support", "x3"],
            "uncertainty": 0.10,
            "cost": 1.0,
        },
    ]

    selected = evidence_calibrated_ucb.select_batch(
        observed,
        candidates,
        budget=2,
        round_index=1,
        rng=random.Random(5),
        beta=0.10,
    )

    assert "distinct_architecture" in selected
    assert not {"near_duplicate_a", "near_duplicate_b"} <= set(selected)


def test_algorithm_uses_generic_m1_m2_m3_modules() -> None:
    scored = evidence_calibrated_ucb.score_candidates(
        _observed_records(),
        _candidate_records(),
        round_index=0,
        beta=0.35,
    )

    by_id = {row["candidate_id"]: row for row in scored}
    assert by_id["bgB__M1_M2_M3"]["score"] > by_id["bgB__M3"]["score"]
    assert by_id["bgB__M1_M2_M3"]["score_components"]["coverage_bonus"] > 0
    assert by_id["bgB__M1_M2_M3"]["score_components"]["contrast_bonus"] > 0


def test_mechanism_lifecycle_matches_v3_shape_without_project_imports() -> None:
    lifecycle = evidence_calibrated_ucb.mechanism_lifecycle()
    for name in (
        "fit_state",
        "generate_candidates",
        "score_candidates",
        "select_panel",
        "plan_ablations",
    ):
        assert callable(getattr(lifecycle, name))

    context = {
        "observed_records": _observed_records(),
        "candidate_records": _candidate_records(),
        "round_index": 1,
    }
    state = lifecycle.fit_state(context)
    candidates = lifecycle.generate_candidates(state)
    scored = lifecycle.score_candidates(state, candidates)
    selected = lifecycle.select_panel(state, candidates, budget=1, rng=random.Random(7))
    ablations = lifecycle.plan_ablations(state)

    assert scored[0]["score_components"]["total_score"] == pytest.approx(scored[0]["score"])
    assert selected == ["bgB__M1_M2_M3"]
    assert {"name": "none"} in ablations


def test_algorithm_source_has_no_project_specific_constants_or_fixtures() -> None:
    source = Path(evidence_calibrated_ucb.__file__).read_text()

    assert "1E62" not in source
    assert "HD110H" not in source
    assert "HG56H" not in source
    assert "anti-HBsAg" not in source
    assert "synthetic" not in source.lower()
