from __future__ import annotations

import inspect
import random
from typing import Any

from design_scientist.algorithms import pcig


BASE_HEAVY = "EVQLVESGGGLVQPGGSL"
BASE_LIGHT = "DIQMTQSPSSLSASVGDR"


def _mutate(sequence: str, position: int, residue: str) -> str:
    index = position - 1
    return f"{sequence[:index]}{residue}{sequence[index + 1:]}"


def _observed_variants() -> list[dict[str, Any]]:
    return [
        {
            "variant_id": "base",
            "heavy_chain_seq": BASE_HEAVY,
            "light_chain_seq": BASE_LIGHT,
            "observed_pH_contrast": 0.10,
            "observed_utility": 0.45,
            "observed_feasible": True,
        },
        {
            "variant_id": "h4f",
            "heavy_chain_seq": _mutate(BASE_HEAVY, 4, "F"),
            "light_chain_seq": BASE_LIGHT,
            "observed_pH_contrast": 0.22,
            "observed_utility": 0.58,
            "observed_feasible": True,
        },
        {
            "variant_id": "h13d",
            "heavy_chain_seq": _mutate(BASE_HEAVY, 13, "D"),
            "light_chain_seq": BASE_LIGHT,
            "observed_pH_contrast": 0.25,
            "observed_utility": 0.56,
            "observed_feasible": True,
        },
        {
            "variant_id": "l4e",
            "heavy_chain_seq": BASE_HEAVY,
            "light_chain_seq": _mutate(BASE_LIGHT, 4, "E"),
            "observed_pH_contrast": 0.21,
            "observed_utility": 0.54,
            "observed_feasible": True,
        },
    ]


def _visible_sites() -> list[dict[str, int | str]]:
    return [
        *({"chain": "H", "position": position} for position in (4, 7, 8, 13, 15)),
        *({"chain": "L", "position": position} for position in (4, 6, 9, 13, 16, 18)),
    ]


def _base_only_variants(heavy: str, light: str) -> list[dict[str, Any]]:
    return [
        {
            "variant_id": "base",
            "heavy_chain_seq": heavy,
            "light_chain_seq": light,
            "observed_pH_contrast": 0.10,
            "observed_utility": 0.45,
            "observed_feasible": True,
        }
    ]


def _legacy_fallback_sites(heavy_len: int, light_len: int) -> list[dict[str, int | str]]:
    return [
        *(
            {"chain": "H", "position": position}
            for position in (10, 14, 18, 22, 26, 30, 34, 38)
            if position <= heavy_len
        ),
        *(
            {"chain": "L", "position": position}
            for position in (8, 12, 16, 20, 24, 28, 32)
            if position <= light_len
        ),
    ]


def test_pcig_does_not_import_or_wrap_cmdgd() -> None:
    source = inspect.getsource(pcig)

    assert "algorithms.cmdgd" not in source
    assert "import cmdgd" not in source
    assert pcig.mechanism_lifecycle().name == "ph_switch_graph"


def test_pcig_builds_site_graph_and_generates_new_positions() -> None:
    state = pcig.fit_state(
        BASE_HEAVY,
        BASE_LIGHT,
        _observed_variants(),
        [],
        observed_mutation_sites=[
            {"chain": "H", "position": 4},
            {"chain": "H", "position": 13},
            {"chain": "L", "position": 4},
        ],
        visible_mutation_sites=_visible_sites(),
    )

    assert state.component_trace["state_model"] == "protonation_coupled_site_graph"
    assert state.component_trace["legacy_rejected_method_reused"] is False
    assert state.counterfactual_site_edits

    candidates = pcig.generate_candidates(state, max_candidates=48)
    generated_sites = {
        (edit["chain"], edit["position"])
        for candidate in candidates
        for edit in candidate["edits"]
        if edit["source"] == "counterfactual_site"
    }

    assert generated_sites
    assert generated_sites.isdisjoint(state.observed_sites | state.vocabulary_sites)
    assert any(candidate["uses_de_novo_site_proposal"] for candidate in candidates)
    assert all(
        candidate["design_context"]["legacy_rejected_method_reused"] is False
        for candidate in candidates
    )


def test_pcig_counterfactual_search_does_not_depend_on_fixed_fallback_positions() -> None:
    heavy = "EVQLVHSGGGLVQPGGSLYTSA"
    light = "DIQMTQSPSSLSASVGDRYTS"
    legacy_sites = _legacy_fallback_sites(len(heavy), len(light))
    legacy_site_keys = {
        (str(site["chain"]), int(site["position"])) for site in legacy_sites
    }

    state = pcig.fit_state(
        heavy,
        light,
        _base_only_variants(heavy, light),
        [],
        observed_mutation_sites=[],
        visible_mutation_sites=legacy_sites,
    )

    generated_sites = {
        (edit.chain, edit.position) for edit in state.counterfactual_site_edits
    }

    assert generated_sites
    assert generated_sites.isdisjoint(legacy_site_keys)
    assert generated_sites.isdisjoint(state.observed_sites | state.vocabulary_sites)


def test_pcig_counterfactual_search_finds_non_hardcoded_sites_at_different_lengths() -> None:
    sequence_pairs = [
        ("EVQLVESGGGLVQP", "DIQMTQSPSSLS"),
        ("EVQLVHSGGGLVQPGGSLYTSADKTT", "DIQMTQSPSSLSASVGDRYTSDN"),
    ]
    generated_by_length: list[set[tuple[str, int]]] = []

    for heavy, light in sequence_pairs:
        legacy_sites = _legacy_fallback_sites(len(heavy), len(light))
        legacy_site_keys = {
            (str(site["chain"]), int(site["position"])) for site in legacy_sites
        }
        state = pcig.fit_state(
            heavy,
            light,
            _base_only_variants(heavy, light),
            [],
            observed_mutation_sites=[],
            visible_mutation_sites=legacy_sites,
        )
        candidates = pcig.generate_candidates(state, max_candidates=16)
        generated_sites = {
            (edit["chain"], edit["position"])
            for candidate in candidates
            for edit in candidate["edits"]
            if edit["source"] == "counterfactual_site"
        }

        assert generated_sites
        assert generated_sites.isdisjoint(legacy_site_keys)
        assert all(
            position <= (len(heavy) if chain == "H" else len(light))
            for chain, position in generated_sites
        )
        generated_by_length.append(generated_sites)

    assert generated_by_length[0] != generated_by_length[1]


def test_pcig_scores_and_selects_without_cmdgd_trace() -> None:
    state = pcig.fit_state(
        BASE_HEAVY,
        BASE_LIGHT,
        _observed_variants(),
        [],
        observed_mutation_sites=[
            {"chain": "H", "position": 4},
            {"chain": "H", "position": 13},
            {"chain": "L", "position": 4},
        ],
        visible_mutation_sites=_visible_sites(),
    )
    candidates = pcig.generate_candidates(state, max_candidates=48)
    scored = pcig.score_candidates(state, candidates)
    selected = pcig.select_panel(state, scored, 5, random.Random(0))

    assert selected
    assert all("score_components" in candidate for candidate in scored)
    assert all(
        candidate["score_components"]["legacy_rejected_method_reused"] is False
        for candidate in scored
    )
    assert all(
        "transition_context_prior" in candidate["score_components"]
        for candidate in scored
    )
    assert all(
        "literature_transition_prior" in candidate["score_components"]
        for candidate in scored
    )
    assert state.component_trace["literature_transition_calibration"]["enabled"] is True
    assert any(scored_candidate["candidate_id"] in selected for scored_candidate in scored)


def test_pcig_literature_transition_prior_is_componentized() -> None:
    assert pcig._literature_calibrated_transition_prior("Y", "H") > 0.8
    assert pcig._literature_calibrated_transition_prior("R", "H") < 0.2

    enabled = pcig.fit_state(
        BASE_HEAVY,
        BASE_LIGHT,
        _observed_variants(),
        [],
        observed_mutation_sites=[
            {"chain": "H", "position": 4},
            {"chain": "H", "position": 13},
            {"chain": "L", "position": 4},
        ],
        visible_mutation_sites=_visible_sites(),
    )
    disabled = pcig.fit_state(
        {
            "base_heavy_chain_seq": BASE_HEAVY,
            "base_light_chain_seq": BASE_LIGHT,
            "observed_variants": _observed_variants(),
            "observed_mutation_sites": [
                {"chain": "H", "position": 4},
                {"chain": "H", "position": 13},
                {"chain": "L", "position": 4},
            ],
            "visible_mutation_sites": _visible_sites(),
            "ablation": "no_literature_transition_calibration",
        }
    )

    assert enabled.component_trace["literature_transition_calibration"]["enabled"] is True
    assert disabled.component_trace["literature_transition_calibration"]["enabled"] is False


def test_pcig_panel_is_utility_first_but_keeps_counterfactual_probe_when_affordable() -> None:
    state = pcig.fit_state(
        BASE_HEAVY,
        BASE_LIGHT,
        _observed_variants(),
        [],
        observed_mutation_sites=[
            {"chain": "H", "position": 4},
            {"chain": "H", "position": 13},
            {"chain": "L", "position": 4},
        ],
        visible_mutation_sites=_visible_sites(),
    )
    candidates = pcig.generate_candidates(state, max_candidates=64)
    scored = pcig.score_candidates(state, candidates)
    selected = pcig.select_panel(state, scored, 5, random.Random(0))
    by_id = {str(candidate["candidate_id"]): candidate for candidate in scored}

    assert len(selected) >= 3
    assert sum(float(by_id[candidate_id]["cost"]) for candidate_id in selected) <= 5.0
    assert (
        sum(
            1
            for candidate_id in selected
            if by_id[candidate_id].get("uses_de_novo_site_proposal") is True
        )
        >= 1
    )
    assert any(
        by_id[candidate_id].get("uses_de_novo_site_proposal") is True
        for candidate_id in selected
    )
    assert all("score_components" in by_id[candidate_id] for candidate_id in selected)


def test_pcig_selector_boundary_modes_are_explicit() -> None:
    state = pcig.fit_state(
        BASE_HEAVY,
        BASE_LIGHT,
        _observed_variants(),
        [],
        observed_mutation_sites=[
            {"chain": "H", "position": 4},
            {"chain": "H", "position": 13},
            {"chain": "L", "position": 4},
        ],
        visible_mutation_sites=_visible_sites(),
    )
    candidates = pcig.generate_candidates(state, max_candidates=64)
    scored = pcig.score_candidates(state, candidates)

    anchor_state = pcig.fit_state(
        {
            "base_heavy_chain_seq": BASE_HEAVY,
            "base_light_chain_seq": BASE_LIGHT,
            "observed_variants": _observed_variants(),
            "observed_mutation_sites": [
                {"chain": "H", "position": 4},
                {"chain": "H", "position": 13},
                {"chain": "L", "position": 4},
            ],
            "visible_mutation_sites": _visible_sites(),
            "ablation": "anchor_only_panel",
        }
    )
    new_site_state = pcig.fit_state(
        {
            "base_heavy_chain_seq": BASE_HEAVY,
            "base_light_chain_seq": BASE_LIGHT,
            "observed_variants": _observed_variants(),
            "observed_mutation_sites": [
                {"chain": "H", "position": 4},
                {"chain": "H", "position": 13},
                {"chain": "L", "position": 4},
            ],
            "visible_mutation_sites": _visible_sites(),
            "ablation": "new_site_only_panel",
        }
    )
    no_reserve_state = pcig.fit_state(
        {
            "base_heavy_chain_seq": BASE_HEAVY,
            "base_light_chain_seq": BASE_LIGHT,
            "observed_variants": _observed_variants(),
            "observed_mutation_sites": [
                {"chain": "H", "position": 4},
                {"chain": "H", "position": 13},
                {"chain": "L", "position": 4},
            ],
            "visible_mutation_sites": _visible_sites(),
            "ablation": "no_new_site_reserve",
        }
    )

    by_id = {str(candidate["candidate_id"]): candidate for candidate in scored}
    anchor_selected = pcig.select_panel(anchor_state, scored, 5, random.Random(0))
    new_site_selected = pcig.select_panel(new_site_state, scored, 5, random.Random(0))
    no_reserve_selected = pcig.select_panel(no_reserve_state, scored, 5, random.Random(0))

    assert anchor_selected
    assert all(not by_id[candidate_id].get("uses_de_novo_site_proposal") for candidate_id in anchor_selected)
    assert new_site_selected
    assert all(by_id[candidate_id].get("uses_de_novo_site_proposal") for candidate_id in new_site_selected)
    assert no_reserve_selected
    assert no_reserve_state.config.selection_mode == "no_new_site_reserve"


def test_pcig_probe_reserve_is_mechanism_level_and_survives_string_flags() -> None:
    state = pcig.fit_state(
        {
            "base_heavy_chain_seq": BASE_HEAVY,
            "base_light_chain_seq": BASE_LIGHT,
            "observed_variants": _observed_variants(),
            "observed_mutation_sites": [
                {"chain": "H", "position": 4},
                {"chain": "H", "position": 13},
                {"chain": "L", "position": 4},
            ],
            "visible_mutation_sites": _visible_sites(),
            "pcig_config": {
                "min_counterfactual_probe_fraction": 0.50,
                "max_counterfactual_probe_reserve": 2,
            },
        }
    )
    candidates = pcig.score_candidates(state, pcig.generate_candidates(state, max_candidates=64))
    round_tripped = []
    for candidate in candidates:
        row = dict(candidate)
        if row.get("uses_de_novo_site_proposal") is True:
            row["uses_de_novo_site_proposal"] = "true"
        round_tripped.append(row)

    selected = pcig.select_panel(state, round_tripped, 5, random.Random(0))
    by_id = {str(candidate["candidate_id"]): candidate for candidate in round_tripped}

    assert (
        sum(1 for candidate_id in selected if pcig._uses_de_novo_site(by_id[candidate_id]))
        >= 2
    )
