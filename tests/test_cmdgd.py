from __future__ import annotations

import random
from typing import Any

from design_scientist.algorithms import cmdgd


BASE_HEAVY = "A" * 70
BASE_LIGHT = "C" * 65
CONTEXT_HEAVY = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
CONTEXT_LIGHT = "YWVTSRQPNMLKIHGFEDCAYWVTSRQPNMLKIHGFEDCA"


def _mutate(sequence: str, position: int, residue: str) -> str:
    index = position - 1
    return f"{sequence[:index]}{residue}{sequence[index + 1:]}"


def _observed_variants() -> list[dict[str, Any]]:
    return [
        {
            "variant_id": "base",
            "heavy_chain_seq": BASE_HEAVY,
            "light_chain_seq": BASE_LIGHT,
            "source_refs": [{"kind": "round", "id": "r0"}],
        },
        {
            "variant_id": "h30y",
            "heavy_chain_seq": _mutate(BASE_HEAVY, 30, "Y"),
            "light_chain_seq": BASE_LIGHT,
            "source_refs": [{"kind": "round", "id": "r1"}],
        },
        {
            "variant_id": "l50h",
            "heavy_chain_seq": BASE_HEAVY,
            "light_chain_seq": _mutate(BASE_LIGHT, 50, "H"),
            "source_refs": [{"kind": "round", "id": "r1"}],
        },
        {
            "variant_id": "h35w_bad",
            "heavy_chain_seq": _mutate(BASE_HEAVY, 35, "W"),
            "light_chain_seq": BASE_LIGHT,
            "source_refs": [{"kind": "round", "id": "r1"}],
        },
    ]


def _observed_endpoints() -> list[dict[str, Any]]:
    return [
        {
            "variant_id": "base",
            "ph6_binding": 0.20,
            "ph7_binding": 0.20,
            "expression": 0.90,
        },
        {
            "variant_id": "h30y",
            "ph6_binding": 0.92,
            "ph7_binding": 0.24,
            "expression": 0.88,
        },
        {
            "variant_id": "l50h",
            "ph6_binding": 0.70,
            "ph7_binding": 0.31,
            "expression": 0.82,
        },
        {
            "variant_id": "h35w_bad",
            "ph6_binding": 0.34,
            "ph7_binding": 0.76,
            "expression": 0.42,
        },
    ]


def _flat_observed_endpoints() -> list[dict[str, Any]]:
    return [
        {
            "variant_id": record["variant_id"],
            "ph6_binding": 0.60,
            "ph7_binding": 0.40,
            "expression": 0.80,
        }
        for record in _observed_variants()
    ]


def _context_observed_variants() -> list[dict[str, Any]]:
    return [
        {
            "variant_id": "base",
            "heavy_chain_seq": CONTEXT_HEAVY,
            "light_chain_seq": CONTEXT_LIGHT,
            "source_refs": [{"kind": "round", "id": "ctx-r0"}],
        },
        {
            "variant_id": "h8y",
            "heavy_chain_seq": _mutate(CONTEXT_HEAVY, 8, "Y"),
            "light_chain_seq": CONTEXT_LIGHT,
            "source_refs": [{"kind": "round", "id": "ctx-r1"}],
        },
        {
            "variant_id": "h16f",
            "heavy_chain_seq": _mutate(CONTEXT_HEAVY, 16, "F"),
            "light_chain_seq": CONTEXT_LIGHT,
            "source_refs": [{"kind": "round", "id": "ctx-r1"}],
        },
        {
            "variant_id": "l11h",
            "heavy_chain_seq": CONTEXT_HEAVY,
            "light_chain_seq": _mutate(CONTEXT_LIGHT, 11, "H"),
            "source_refs": [{"kind": "round", "id": "ctx-r1"}],
        },
        {
            "variant_id": "h21p_bad",
            "heavy_chain_seq": _mutate(CONTEXT_HEAVY, 21, "P"),
            "light_chain_seq": CONTEXT_LIGHT,
            "source_refs": [{"kind": "round", "id": "ctx-r1"}],
        },
    ]


def _context_observed_endpoints() -> list[dict[str, Any]]:
    return [
        {
            "variant_id": "base",
            "ph6_binding": 0.20,
            "ph7_binding": 0.20,
            "expression": 0.92,
        },
        {
            "variant_id": "h8y",
            "ph6_binding": 0.90,
            "ph7_binding": 0.22,
            "expression": 0.88,
        },
        {
            "variant_id": "h16f",
            "ph6_binding": 0.82,
            "ph7_binding": 0.26,
            "expression": 0.86,
        },
        {
            "variant_id": "l11h",
            "ph6_binding": 0.76,
            "ph7_binding": 0.28,
            "expression": 0.84,
        },
        {
            "variant_id": "h21p_bad",
            "ph6_binding": 0.28,
            "ph7_binding": 0.64,
            "expression": 0.35,
        },
    ]


def _context_vocabulary() -> list[dict[str, Any]]:
    return [
        {
            "token": "H25W",
            "chain": "H",
            "position": 25,
            "residue": "W",
            "annotation": "contrast",
        }
    ]


def _observed_edit_positions(state: cmdgd.CMDGDState) -> set[tuple[str, int]]:
    return {
        (module.chain, module.position)
        for module in state.edit_modules
        if "design_vocabulary" not in module.parent_ids
    }


def test_fit_state_extracts_readable_heavy_and_light_edit_ids() -> None:
    state = cmdgd.fit_state(
        BASE_HEAVY,
        BASE_LIGHT,
        _observed_variants(),
        _observed_endpoints(),
    )

    edit_ids = {module.edit_id for module in state.edit_modules}

    assert "H30Y" in edit_ids
    assert "L50H" in edit_ids
    assert all(isinstance(module.edit_id, str) for module in state.edit_modules)


def test_generate_candidates_creates_unobserved_sequences_with_provenance() -> None:
    state = cmdgd.fit_state(
        BASE_HEAVY,
        BASE_LIGHT,
        _observed_variants(),
        _observed_endpoints(),
    )
    candidates = cmdgd.generate_candidates(state, max_candidates=12)
    observed_sequences = {
        (record["heavy_chain_seq"], record["light_chain_seq"])
        for record in _observed_variants()
    }

    assert any(
        (candidate["heavy_chain_seq"], candidate["light_chain_seq"])
        not in observed_sequences
        for candidate in candidates
    )
    for candidate in candidates:
        assert isinstance(candidate["candidate_id"], str)
        assert isinstance(candidate["variant_id"], str)
        assert candidate["parent_ids"]
        assert all(isinstance(parent_id, str) for parent_id in candidate["parent_ids"])
        assert candidate["source_refs"]
        assert candidate["operator"]
        assert isinstance(candidate["operator"], str)
        assert isinstance(candidate["design_context"], dict)
        assert isinstance(candidate["cost"], float)
        assert candidate["feasibility_status"] in {"feasible", "review", "infeasible"}
        assert all(isinstance(edit["id"], str) for edit in candidate["edits"])


def test_generate_candidates_includes_higher_order_recombinations() -> None:
    state = cmdgd.fit_state(
        BASE_HEAVY,
        BASE_LIGHT,
        _observed_variants(),
        _observed_endpoints(),
    )

    candidates = cmdgd.generate_candidates(state, max_candidates=32)

    assert any(
        candidate["operator"] == "beam_recombine_modules"
        and len(candidate["edits"]) >= 3
        for candidate in candidates
    )


def test_candidate_edit_vocabulary_expands_design_space_beyond_observed_edits() -> None:
    state = cmdgd.fit_state(
        BASE_HEAVY,
        BASE_LIGHT,
        _observed_variants(),
        _observed_endpoints(),
        candidate_edit_vocabulary=[
            {"token": "H40F", "chain": "H", "position": 40, "residue": "F", "annotation": "contrast"}
        ],
    )

    candidates = cmdgd.generate_candidates(state, max_candidates=32)

    assert "H40F" in {module.edit_id for module in state.edit_modules}
    assert any(
        candidate["operator"] == "beam_recombine_modules"
        and "H40F" in {edit["id"] for edit in candidate["edits"]}
        for candidate in candidates
    )


def test_generate_candidates_proposes_de_novo_sites_outside_observed_and_vocabulary_positions() -> None:
    state = cmdgd.fit_state(
        CONTEXT_HEAVY,
        CONTEXT_LIGHT,
        _context_observed_variants(),
        _context_observed_endpoints(),
        candidate_edit_vocabulary=_context_vocabulary(),
        config=cmdgd.CMDGDConfig(max_de_novo_site_proposals=8),
    )

    candidates = cmdgd.generate_candidates(state, max_candidates=80)
    observed_positions = _observed_edit_positions(state)
    vocabulary_positions = {("H", 25)}
    de_novo_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("uses_de_novo_site_proposal") is True
    ]

    assert de_novo_candidates
    for candidate in de_novo_candidates:
        de_novo_edit_ids = set(candidate["de_novo_site_edit_ids"])
        assert de_novo_edit_ids
        assert candidate["operator"] in {
            "de_novo_site_proposal",
            "de_novo_site_guided_recombine",
        }
        assert any(
            ref.get("kind") == "cmdgd_de_novo_site_proposal"
            for ref in candidate["source_refs"]
            if isinstance(ref, dict)
        )
        new_positions = {
            (edit["chain"], edit["position"])
            for edit in candidate["edits"]
            if edit["id"] in de_novo_edit_ids
        }
        assert new_positions
        assert new_positions.isdisjoint(observed_positions)
        assert new_positions.isdisjoint(vocabulary_positions)
        assert candidate["cmdgd_trace"]["de_novo_site_proposal"][
            "uses_de_novo_site_proposal"
        ] is True


def test_disabling_de_novo_site_proposal_removes_new_site_generation() -> None:
    full_state = cmdgd.fit_state(
        CONTEXT_HEAVY,
        CONTEXT_LIGHT,
        _context_observed_variants(),
        _context_observed_endpoints(),
        candidate_edit_vocabulary=_context_vocabulary(),
    )
    disabled_state = cmdgd.fit_state(
        CONTEXT_HEAVY,
        CONTEXT_LIGHT,
        _context_observed_variants(),
        _context_observed_endpoints(),
        candidate_edit_vocabulary=_context_vocabulary(),
        config=cmdgd.CMDGDConfig(enable_de_novo_site_proposal=False),
    )

    full_candidates = cmdgd.generate_candidates(full_state, max_candidates=80)
    disabled_candidates = cmdgd.generate_candidates(disabled_state, max_candidates=80)

    assert any(candidate.get("uses_de_novo_site_proposal") for candidate in full_candidates)
    assert not any(
        candidate.get("uses_de_novo_site_proposal") for candidate in disabled_candidates
    )
    assert full_state.component_trace["de_novo_site_proposal"]["enabled"] is True
    assert disabled_state.component_trace["de_novo_site_proposal"]["enabled"] is False


def test_scoring_and_trace_expose_de_novo_site_component() -> None:
    state = cmdgd.fit_state(
        CONTEXT_HEAVY,
        CONTEXT_LIGHT,
        _context_observed_variants(),
        _context_observed_endpoints(),
        candidate_edit_vocabulary=_context_vocabulary(),
        config=cmdgd.CMDGDConfig(max_de_novo_site_proposals=8),
    )
    candidate = next(
        candidate
        for candidate in cmdgd.generate_candidates(state, max_candidates=80)
        if candidate.get("uses_de_novo_site_proposal") is True
    )

    scored = cmdgd.score_candidates(state, [candidate])

    assert len(scored) == 1
    score_components = scored[0]["score_components"]
    trace = scored[0]["cmdgd_trace"]["de_novo_site_proposal"]
    assert score_components["de_novo_site_proposal_prior"] > 0.0
    assert trace["enabled"] is True
    assert trace["uses_de_novo_site_proposal"] is True
    assert trace["candidate_de_novo_site_edit_ids"] == candidate["de_novo_site_edit_ids"]
    assert trace["active_de_novo_site_proposal_prior"] == score_components[
        "de_novo_site_proposal_prior"
    ]


def test_score_and_select_panel_use_generated_candidates_not_observed_ids() -> None:
    state = cmdgd.fit_state(
        BASE_HEAVY,
        BASE_LIGHT,
        _observed_variants(),
        _observed_endpoints(),
    )
    candidates = cmdgd.generate_candidates(state, max_candidates=12)
    leaked_observed_id = {
        **candidates[0],
        "candidate_id": "h30y",
        "variant_id": "h30y",
        "source_refs": [{"kind": "leak_check", "id": "observed-id"}],
    }

    scored = cmdgd.score_candidates(state, [*candidates, leaked_observed_id])
    selected = cmdgd.select_panel(state, scored, budget=2, rng=random.Random(7))

    assert scored
    assert "h30y" not in {row["candidate_id"] for row in scored}
    assert selected
    assert all(isinstance(candidate_id, str) for candidate_id in selected)
    assert "h30y" not in selected
    assert set(selected).isdisjoint({"base", "l50h", "h35w_bad"})


def test_ablation_modes_change_generation_or_scoring_and_lifecycle_exposes_v3_api() -> None:
    full_state = cmdgd.fit_state(
        BASE_HEAVY,
        BASE_LIGHT,
        _observed_variants(),
        _observed_endpoints(),
    )
    no_recombination_state = cmdgd.fit_state(
        BASE_HEAVY,
        BASE_LIGHT,
        _observed_variants(),
        _observed_endpoints(),
        config=cmdgd.CMDGDConfig(enable_grammar_recombination=False),
    )
    full_candidates = cmdgd.generate_candidates(full_state, max_candidates=12)
    ablated_candidates = cmdgd.generate_candidates(no_recombination_state, max_candidates=12)

    assert {candidate["operator"] for candidate in full_candidates} != {
        candidate["operator"] for candidate in ablated_candidates
    }

    no_contrast_state = cmdgd.fit_state(
        BASE_HEAVY,
        BASE_LIGHT,
        _observed_variants(),
        _observed_endpoints(),
        config=cmdgd.CMDGDConfig(enable_contrastive_objective=False),
    )
    full_scores = cmdgd.score_candidates(full_state, full_candidates)
    no_contrast_scores = cmdgd.score_candidates(no_contrast_state, full_candidates)
    assert [
        row["score_components"]["contrastive_ph_objective"]
        for row in full_scores
    ] != [
        row["score_components"]["contrastive_ph_objective"]
        for row in no_contrast_scores
    ]

    lifecycle = cmdgd.mechanism_lifecycle()
    context = lifecycle.fit_state(
        {
            "base_heavy_chain_seq": BASE_HEAVY,
            "base_light_chain_seq": BASE_LIGHT,
            "observed_records": _observed_variants(),
            "observed_endpoints": _observed_endpoints(),
        }
    )
    generated = lifecycle.generate_candidates(context)
    scored = lifecycle.score_candidates(context, generated)
    selected = lifecycle.select_panel(context, scored, budget=2, rng=random.Random(11))
    ablations = lifecycle.plan_ablations(context)

    assert lifecycle.name == "cmd_gd"
    assert generated
    assert scored
    assert selected
    assert {row["name"] for row in ablations} >= {
        "full",
        "no_contrastive_objective",
        "no_grammar_recombination",
        "no_de_novo_site_proposal",
        "no_novelty_uncertainty",
    }


def test_scoring_differentiates_flat_endpoint_candidates_by_residue_and_provenance() -> None:
    state = cmdgd.fit_state(
        BASE_HEAVY,
        BASE_LIGHT,
        _observed_variants(),
        _flat_observed_endpoints(),
    )
    candidates = [
        {
            "candidate_id": "local_h30f_round",
            "heavy_chain_seq": _mutate(BASE_HEAVY, 30, "F"),
            "light_chain_seq": BASE_LIGHT,
            "edits": [
                {
                    "id": "H30F",
                    "chain": "H",
                    "position": 30,
                    "from": "A",
                    "to": "F",
                    "source_edit_id": "H30Y",
                }
            ],
            "parent_ids": ["h30y"],
            "operator": "local_grammar_substitution",
            "source_refs": [{"kind": "round", "id": "r1"}],
            "cost": 1.0,
            "feasibility_status": "feasible",
        },
        {
            "candidate_id": "local_l50y_curated",
            "heavy_chain_seq": BASE_HEAVY,
            "light_chain_seq": _mutate(BASE_LIGHT, 50, "Y"),
            "edits": [
                {
                    "id": "L50Y",
                    "chain": "L",
                    "position": 50,
                    "from": "C",
                    "to": "Y",
                    "source_edit_id": "L50H",
                }
            ],
            "parent_ids": ["l50h", "design_vocabulary"],
            "operator": "local_grammar_substitution",
            "source_refs": [{"kind": "curated_vocabulary", "id": "cdr-prior"}],
            "cost": 1.0,
            "feasibility_status": "feasible",
        },
        {
            "candidate_id": "local_h35f_liability_parent",
            "heavy_chain_seq": _mutate(BASE_HEAVY, 35, "F"),
            "light_chain_seq": BASE_LIGHT,
            "edits": [
                {
                    "id": "H35F",
                    "chain": "H",
                    "position": 35,
                    "from": "A",
                    "to": "F",
                    "source_edit_id": "H35W",
                }
            ],
            "parent_ids": ["h35w_bad"],
            "operator": "local_grammar_substitution",
            "source_refs": [{"kind": "round", "id": "r1"}],
            "cost": 1.0,
            "feasibility_status": "feasible",
        },
    ]

    scored = cmdgd.score_candidates(state, candidates)

    assert len(scored) == len(candidates)
    assert len({row["score"] for row in scored}) == len(candidates)
    assert len(
        {
            (
                row["score_components"]["residue_combination_prior"],
                row["score_components"]["provenance_prior"],
            )
            for row in scored
        }
    ) == len(candidates)


def test_cmdgd_exposes_component_level_trace_fields() -> None:
    state = cmdgd.fit_state(
        BASE_HEAVY,
        BASE_LIGHT,
        _observed_variants(),
        _observed_endpoints(),
        candidate_edit_vocabulary=[
            {"token": "H40F", "chain": "H", "position": 40, "residue": "F", "annotation": "contrast"}
        ],
    )

    candidates = cmdgd.generate_candidates(state, max_candidates=8)
    scored = cmdgd.score_candidates(state, candidates)

    assert state.component_trace["grammar_learning"]["observed_variant_count"] == 4
    assert state.component_trace["vocabulary_expansion"]["accepted_edit_count"] == 1
    assert candidates
    assert scored
    candidate_trace = candidates[0]["cmdgd_trace"]
    score_trace = scored[0]["cmdgd_trace"]
    assert set(candidate_trace) >= {
        "grammar_learning",
        "vocabulary_expansion",
        "generation",
        "feasibility_guardrails",
    }
    assert set(score_trace) >= {
        "contrastive_scoring",
        "feasibility_guardrails",
        "selection_diversity",
    }
    assert isinstance(score_trace["selection_diversity"]["diversity_key"], str)


def test_lifecycle_ablation_knobs_disable_components_without_disabling_scoring() -> None:
    vocabulary = [
        {"token": "H40F", "chain": "H", "position": 40, "residue": "F", "annotation": "contrast"}
    ]
    full_state = cmdgd.fit_state(
        BASE_HEAVY,
        BASE_LIGHT,
        _observed_variants(),
        _observed_endpoints(),
        candidate_edit_vocabulary=vocabulary,
    )
    full_candidates = cmdgd.generate_candidates(full_state, max_candidates=16)
    full_scores = cmdgd.score_candidates(full_state, full_candidates)

    no_ph_state = cmdgd.fit_state(
        BASE_HEAVY,
        BASE_LIGHT,
        _observed_variants(),
        _observed_endpoints(),
        candidate_edit_vocabulary=vocabulary,
        config=cmdgd.CMDGDConfig(enable_ph_contrast=False),
    )
    no_ph_scores = cmdgd.score_candidates(no_ph_state, full_candidates)
    assert full_scores and no_ph_scores
    assert [row["score"] for row in full_scores] != [row["score"] for row in no_ph_scores]
    assert all(
        row["cmdgd_trace"]["contrastive_scoring"]["enabled"] is False
        for row in no_ph_scores
    )

    no_guardrail_state = cmdgd.fit_state(
        BASE_HEAVY,
        BASE_LIGHT,
        _observed_variants(),
        _observed_endpoints(),
        candidate_edit_vocabulary=vocabulary,
        config=cmdgd.CMDGDConfig(enable_feasibility_guardrails=False),
    )
    no_guardrail_scores = cmdgd.score_candidates(no_guardrail_state, full_candidates)
    assert [row["score"] for row in full_scores] != [
        row["score"] for row in no_guardrail_scores
    ]
    assert all(
        row["cmdgd_trace"]["feasibility_guardrails"]["enabled"] is False
        for row in no_guardrail_scores
    )

    no_vocabulary_state = cmdgd.fit_state(
        BASE_HEAVY,
        BASE_LIGHT,
        _observed_variants(),
        _observed_endpoints(),
        candidate_edit_vocabulary=vocabulary,
        config=cmdgd.CMDGDConfig(enable_vocabulary_expansion=False),
    )
    assert "H40F" in {module.edit_id for module in full_state.edit_modules}
    assert "H40F" not in {module.edit_id for module in no_vocabulary_state.edit_modules}
    assert full_state.component_trace["vocabulary_expansion"]["enabled"] is True
    assert no_vocabulary_state.component_trace["vocabulary_expansion"]["enabled"] is False

    no_generation_state = cmdgd.fit_state(
        BASE_HEAVY,
        BASE_LIGHT,
        _observed_variants(),
        _observed_endpoints(),
        candidate_edit_vocabulary=vocabulary,
        config=cmdgd.CMDGDConfig(enable_generation=False),
    )
    assert cmdgd.generate_candidates(no_generation_state, max_candidates=16) == []
    no_generation_scores = cmdgd.score_candidates(no_generation_state, full_candidates)
    assert no_generation_scores

    ablations = cmdgd.mechanism_lifecycle().plan_ablations({"algorithm_state": full_state})
    assert {row["name"] for row in ablations} >= {
        "no_ph_contrast",
        "no_feasibility_guardrails",
        "no_vocabulary_expansion",
        "no_de_novo_site_proposal",
        "no_generation",
    }
