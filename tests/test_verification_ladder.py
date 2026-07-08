from __future__ import annotations

from design_scientist.verification_ladder import run_verification_ladder


def test_verification_ladder_passes_ordered_stages() -> None:
    checks = {
        "contract": True,
        "toy_invariant": True,
        "synthetic_replay": {"passed": True, "ablation": True},
        "retrospective_masking": True,
        "structure_proxy": True,
        "human_review": True,
    }

    result = run_verification_ladder("contrast_lattice", checks)

    assert result["passed"] is True
    assert [stage["name"] for stage in result["stages"]] == [
        "contract",
        "toy_invariant",
        "synthetic_replay",
        "retrospective_masking",
        "structure_proxy",
        "human_review",
    ]
    assert {stage["status"] for stage in result["stages"]} == {"passed"}


def test_verification_ladder_skips_later_stages_after_first_failure() -> None:
    checks = {
        "contract": True,
        "toy_invariant": {"passed": False, "reason": "mass balance invariant failed"},
        "synthetic_replay": True,
        "retrospective_masking": True,
        "structure_proxy": True,
        "human_review": True,
    }

    result = run_verification_ladder("broken_mechanism", checks)

    assert result["passed"] is False
    statuses = {stage["name"]: stage["status"] for stage in result["stages"]}
    assert statuses["contract"] == "passed"
    assert statuses["toy_invariant"] == "failed"
    assert statuses["synthetic_replay"] == "skipped"
    assert statuses["human_review"] == "skipped"
    assert result["blocking_stage"] == "toy_invariant"


def test_strong_algorithm_claim_requires_replay_and_ablation_support() -> None:
    checks = {
        "contract": True,
        "toy_invariant": True,
        "synthetic_replay": {"passed": True},
        "retrospective_masking": True,
        "structure_proxy": True,
        "human_review": True,
    }
    strong_claims = [
        {"claim_id": "algorithmic_novelty", "claim_type": "algorithm", "strength": "strong"}
    ]

    result = run_verification_ladder("no_ablation", checks, strong_claims)

    assert result["passed"] is False
    assert any(
        finding["claim_id"] == "algorithmic_novelty" and "ablation" in finding["reason"]
        for finding in result["claim_findings"]
    )


def test_biological_claim_cannot_be_supported_only_by_structure_proxy() -> None:
    checks = {
        "contract": True,
        "toy_invariant": True,
        "synthetic_replay": {"passed": True, "ablation": True},
        "retrospective_masking": True,
        "structure_proxy": True,
        "human_review": True,
    }
    strong_claims = [
        {
            "claim_id": "binding_mode",
            "claim_type": "biology",
            "strength": "strong",
            "supported_by": ["structure_proxy"],
        }
    ]

    result = run_verification_ladder("structure_only", checks, strong_claims)

    assert result["passed"] is False
    assert any(
        finding["claim_id"] == "binding_mode" and "structure_proxy" in finding["reason"]
        for finding in result["claim_findings"]
    )
