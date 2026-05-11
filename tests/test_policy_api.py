from __future__ import annotations

import inspect
import random

from design_scientist import policies
from design_scientist.acquisition import default_policy
from design_scientist.panel import select_panel
from design_scientist.schemas import Candidate
from design_scientist.synthetic_replay import DEFAULT_METHODS
from design_scientist.synthetic_replay import SyntheticObservation, SyntheticVariant


def _toy_observed() -> list[dict[str, object]]:
    return [
        {
            "variant_id": "screening_bg__WT",
            "background": "screening_bg",
            "modules": (),
            "observed_utility": 0.52,
            "target_background": "transfer_bg",
        },
        {
            "variant_id": "transfer_bg__WT",
            "background": "transfer_bg",
            "modules": (),
            "observed_utility": 0.49,
            "target_background": "transfer_bg",
        },
        {
            "variant_id": "screening_bg__HD110H",
            "background": "screening_bg",
            "modules": ("HD110H",),
            "observed_utility": 0.69,
            "target_background": "transfer_bg",
        },
        {
            "variant_id": "transfer_bg__HG56H",
            "background": "transfer_bg",
            "modules": ("HG56H",),
            "observed_utility": 0.63,
            "target_background": "transfer_bg",
        },
    ]


def _toy_candidates() -> list[dict[str, object]]:
    return [
        {
            "variant_id": "transfer_bg__HD110H",
            "background": "transfer_bg",
            "modules": ("HD110H",),
            "target_background": "transfer_bg",
        },
        {
            "variant_id": "transfer_bg__HD110H_HG56H",
            "background": "transfer_bg",
            "modules": ("HD110H", "HG56H"),
            "target_background": "transfer_bg",
        },
        {
            "variant_id": "screening_bg__HN54H",
            "background": "screening_bg",
            "modules": ("HN54H",),
            "target_background": "transfer_bg",
        },
        {
            "variant_id": "transfer_bg__LT47Y",
            "background": "transfer_bg",
            "modules": ("LT47Y",),
            "target_background": "transfer_bg",
        },
    ]


def test_default_methods_are_policy_api_callables() -> None:
    assert tuple(policies.DEFAULT_POLICY_NAMES) == DEFAULT_METHODS

    expected_parameters = ["observed", "candidates", "budget", "round_index", "rng"]
    candidate_ids = {str(candidate["variant_id"]) for candidate in _toy_candidates()}
    for method in DEFAULT_METHODS:
        policy = policies.get_policy(method)
        assert list(inspect.signature(policy).parameters) == expected_parameters

        selected = policy(
            _toy_observed(),
            _toy_candidates(),
            budget=2,
            round_index=1,
            rng=random.Random(17),
        )

        assert isinstance(selected, list)
        assert len(selected) <= 2
        assert len(selected) == len(set(selected))
        assert set(selected) <= candidate_ids


def test_random_policy_is_fixed_seed_deterministic() -> None:
    first = policies.random_feasible(
        _toy_observed(),
        _toy_candidates(),
        budget=3,
        round_index=1,
        rng=random.Random(123),
    )
    second = policies.random_feasible(
        _toy_observed(),
        _toy_candidates(),
        budget=3,
        round_index=1,
        rng=random.Random(123),
    )

    assert first == second


def test_select_batch_uses_mechanism_aware_default_policy() -> None:
    selected = policies.select_batch(
        _toy_observed(),
        _toy_candidates(),
        budget=2,
        round_index=1,
        rng=random.Random(5),
    )

    assert selected == policies.mechanism_aware(
        _toy_observed(),
        _toy_candidates(),
        budget=2,
        round_index=1,
        rng=random.Random(5),
    )


def test_policy_api_accepts_real_like_candidate_records() -> None:
    observed = [
        {
            "candidate_id": "screening_bg__WT",
            "background": "screening_bg",
            "modules": [],
            "utility": 0.51,
            "target_background": "transfer_bg",
        },
        {
            "candidate_id": "transfer_bg__WT",
            "background": "transfer_bg",
            "modules": [],
            "utility": 0.48,
            "target_background": "transfer_bg",
        },
        {
            "candidate_id": "screening_bg__HD110H",
            "background": "screening_bg",
            "modules": ["HD110H"],
            "utility": 0.68,
            "target_background": "transfer_bg",
        },
    ]
    candidates = [
        {
            "candidate_id": "transfer_bg__HD110H",
            "background": "transfer_bg",
            "modules": ["HD110H"],
            "target_background": "transfer_bg",
            "score_components": {"performance": 0.7},
            "risk_flags": [],
        },
        {
            "candidate_id": "transfer_bg__HG56H",
            "background": "transfer_bg",
            "modules": ["HG56H"],
            "target_background": "transfer_bg",
            "score_components": {"performance": 0.6},
            "risk_flags": ["transfer_unvalidated"],
        },
        {
            "candidate_id": "transfer_bg__HD110H_HG56H",
            "background": "transfer_bg",
            "modules": ["HD110H", "HG56H"],
            "target_background": "transfer_bg",
            "score_components": {"performance": 0.8},
            "risk_flags": [],
        },
    ]

    normalized = policies.to_policy_record(candidates[0])
    selected = policies.select_batch(
        observed,
        candidates,
        budget=2,
        round_index=1,
        rng=random.Random(19),
    )

    assert normalized["variant_id"] == "transfer_bg__HD110H"
    assert normalized["modules"] == ("HD110H",)
    assert len(selected) == 2
    assert set(selected) <= {str(candidate["candidate_id"]) for candidate in candidates}


def test_policy_api_accepts_synthetic_variant_records_directly() -> None:
    observed = [
        SyntheticObservation(
            variant=SyntheticVariant("screening_bg__WT", "screening_bg", ()),
            true_endpoints={},
            observed_endpoints={},
            true_utility=0.50,
            observed_utility=0.50,
            true_feasible=False,
        ),
        SyntheticObservation(
            variant=SyntheticVariant("transfer_bg__WT", "transfer_bg", ()),
            true_endpoints={},
            observed_endpoints={},
            true_utility=0.47,
            observed_utility=0.47,
            true_feasible=False,
        ),
        SyntheticObservation(
            variant=SyntheticVariant("screening_bg__HD110H", "screening_bg", ("HD110H",)),
            true_endpoints={},
            observed_endpoints={},
            true_utility=0.69,
            observed_utility=0.69,
            true_feasible=False,
        ),
    ]
    candidates = [
        SyntheticVariant("transfer_bg__HD110H", "transfer_bg", ("HD110H",)),
        SyntheticVariant("transfer_bg__HG56H", "transfer_bg", ("HG56H",)),
        SyntheticVariant("transfer_bg__HD110H_HG56H", "transfer_bg", ("HD110H", "HG56H")),
    ]

    selected = policies.select_batch(
        observed,
        candidates,
        budget=2,
        round_index=1,
        rng=random.Random(23),
    )

    assert len(selected) == 2
    assert set(selected) <= {candidate.variant_id for candidate in candidates}


def test_policy_selectors_exclude_infeasible_and_forbidden_candidate_records() -> None:
    observed = [
        {
            "variant_id": "real_bg__WT",
            "background": "real_bg",
            "modules": (),
            "observed_utility": 0.50,
            "target_background": "real_bg",
        }
    ]
    candidates = [
        {
            "candidate_id": "infeasible_high_score",
            "background": "real_bg",
            "modules": ["HD110H"],
            "score": 99.0,
            "target_background": "real_bg",
            "feasibility_status": "infeasible",
            "cost": 1.0,
        },
        {
            "candidate_id": "forbidden_pair",
            "background": "real_bg",
            "modules": ["KD31N", "SY92F"],
            "score": 98.0,
            "target_background": "real_bg",
            "feasibility_status": "feasible",
            "cost": 1.0,
        },
        {
            "candidate_id": "valid_singleton",
            "background": "real_bg",
            "modules": ["HG56H"],
            "score": 0.1,
            "target_background": "real_bg",
            "feasibility_status": "feasible",
            "cost": 1.0,
        },
    ]

    for method in policies.DEFAULT_POLICY_NAMES:
        selected = policies.get_policy(method)(
            observed,
            candidates,
            budget=3,
            round_index=1,
            rng=random.Random(31),
        )

        assert selected == ["valid_singleton"]


def test_policy_records_from_real_candidates_preserve_target_context() -> None:
    candidate = Candidate(
        candidate_id="real_bg__HD110H",
        operator="add_module",
        category="champion",
        target_system="1E62",
        background="real_bg",
        modules=["HD110H"],
        score_components={"performance": 0.5},
    )

    normalized = policies.to_policy_record(candidate)
    selected = policies.mechanism_aware(
        observed=[],
        candidates=[candidate],
        budget=1,
        round_index=1,
        rng=random.Random(37),
    )

    assert normalized["target_background"] == "real_bg"
    assert normalized["design_context"] == {
        "target_system": "1E62",
        "target_background": "real_bg",
    }
    assert selected == ["real_bg__HD110H"]


def test_select_panel_is_budgeted_subset_deterministic_and_diagnostic() -> None:
    candidates = [
        Candidate(
            candidate_id="candidate_c",
            operator="add_module",
            category="champion",
            target_system="1E62",
            background="bg",
            modules=["HD110H"],
            score_components={"performance": 0.5},
            risk_flags=["transfer_unvalidated"],
            required_measurements=["pH7.4 binding"],
        ),
        Candidate(
            candidate_id="candidate_a",
            operator="complete_square",
            category="interaction_square",
            target_system="1E62",
            background="bg",
            modules=["HD110H", "HG56H"],
            score_components={"performance": 0.5},
            required_measurements=["pH6.0 binding"],
        ),
        Candidate(
            candidate_id="candidate_b",
            operator="complete_missing_edge",
            category="lattice_repair",
            target_system="1E62",
            background="bg",
            modules=["HG56H"],
            score_components={"performance": 0.5},
            required_measurements=["pH7.4 binding"],
        ),
    ]

    panel, score, diagnostics = select_panel(
        candidates,
        default_policy("mechanism_aware"),
        budget=2,
        return_diagnostics=True,
    )
    reversed_panel, reversed_score, reversed_diagnostics = select_panel(
        list(reversed(candidates)),
        default_policy("mechanism_aware"),
        budget=2,
        return_diagnostics=True,
    )
    candidate_ids = {candidate.candidate_id for candidate in candidates}
    selected_ids = [candidate.candidate_id for candidate in panel]

    assert selected_ids == [candidate.candidate_id for candidate in reversed_panel]
    assert score == reversed_score
    assert len(panel) == 2
    assert set(selected_ids) <= candidate_ids
    assert diagnostics["selected_count"] == 2
    assert diagnostics["budget"] == 2
    assert diagnostics["subset_of_pool"] is True
    assert diagnostics["coverage"]["category_counts"]
    assert diagnostics["risk"]["risky_candidate_count"] >= 0
    assert "top_predicted_utility" in diagnostics["overlap"]
    assert diagnostics == reversed_diagnostics
