from __future__ import annotations

from design_scientist.candidates import (
    build_design_space,
    candidate_pool_diagnostics,
    generate_candidate_pool,
    generate_candidates,
)
from design_scientist.schemas import CandidatePool, DesignSpace, to_plain_data


def test_build_design_space_is_stable_and_canonicalizes_target_systems() -> None:
    first = build_design_space(_state(), _evidence())
    second = build_design_space(_state(), list(reversed(_evidence())))

    assert isinstance(first, DesignSpace)
    assert first == second
    assert first.design_space_id == "anti_hbsag:1E62:HD110H-HG56H-HN54H-HV105H"
    assert first.target_systems == ["1E62"]
    assert [operator.operator_id for operator in first.operators] == [
        "add_module",
        "remove_module",
        "complete_missing_edge",
        "complete_square",
        "validate_genotype_coverage",
        "repeat_or_control",
    ]


def test_candidate_pool_records_operator_lineage_and_candidate_metadata() -> None:
    design_space = build_design_space(_state(), _evidence())
    pool = generate_candidate_pool(design_space, _state(), _evidence())
    candidate = next(item for item in pool.candidates if item.operator == "add_module")

    assert isinstance(pool, CandidatePool)
    assert candidate.variant_id == candidate.candidate_id
    assert candidate.design_space_id == design_space.design_space_id
    assert candidate.operator_args == {
        "background": "1E62_background_candidate",
        "module": "HD110H",
    }
    assert candidate.lineage.operator == "add_module"
    assert candidate.lineage.operator_args == candidate.operator_args
    assert candidate.parent_ids == ["1E62_background_candidate"]
    assert candidate.source_refs == ["role_split_sdab_1E62"]
    assert candidate.feasibility_status == "feasible"
    assert candidate.feasibility_reasons == []
    assert candidate.cost == 1.0
    assert candidate.required_endpoints == ["pH7.4 binding", "pH6.0 binding", "Ae/B/D1 genotype coverage"]


def test_generate_candidate_pool_is_deterministic() -> None:
    design_space = build_design_space(_state(), _evidence())

    first = generate_candidate_pool(design_space, _state(), _evidence(), strategy="mechanism_aware")
    second = generate_candidate_pool(design_space, _state(), _evidence(), strategy="mechanism_aware")

    assert to_plain_data(first) == to_plain_data(second)
    assert [candidate.variant_id for candidate in first.candidates] == [
        candidate.variant_id for candidate in second.candidates
    ]


def test_candidate_pool_diagnostics_marks_generator_limited_run_for_small_pools() -> None:
    tiny_state = {
        "project_id": "anti_hbsag",
        "system_roles": {"1e+62": "target_system_design_genotype_coverage"},
        "module_status": [{"module_id": "HD110H"}],
        "unresolved_edges": [],
    }
    design_space = build_design_space(tiny_state, [])
    pool = generate_candidate_pool(design_space, tiny_state, [])
    diagnostics = candidate_pool_diagnostics(pool)

    assert diagnostics["candidate_count"] == len(pool.candidates)
    assert diagnostics["generator_limited_run"] is True
    assert "small_candidate_pool" in diagnostics["reasons"]


def test_generate_candidates_compatibility_returns_enriched_candidates_without_stale_onee62() -> None:
    candidates = generate_candidates(_state(), _evidence(), strategy="mechanism_aware")
    serialized = repr(to_plain_data(candidates))

    assert candidates
    assert all(candidate.variant_id == candidate.candidate_id for candidate in candidates)
    assert all(candidate.design_space_id for candidate in candidates)
    assert "1e+62" not in serialized


def _state() -> dict[str, object]:
    return {
        "project_id": "anti_hbsag",
        "system_roles": {
            "sdAb": "module_learning",
            "1e+62": "target_system_design_genotype_coverage",
        },
        "module_status": [
            {"module_id": "HV105H"},
            {"module_id": "HD110H"},
            {"module_id": "HG56H"},
            {"module_id": "HN54H"},
        ],
        "unresolved_edges": [
            {"system": "1e+62", "module_id": "HA23K", "base_variant": "com1", "reason": "repair edge"},
            {"system": "1E62", "module_id": "HG57R", "base_variant": "com2", "reason": "repair second edge"},
        ],
    }


def _evidence() -> list[dict[str, object]]:
    return [
        {"evidence_id": "other_evidence", "source_tables": ["module_summary"]},
        {"evidence_id": "role_split_sdab_1E62", "source_tables": ["module_summary"]},
    ]
