from __future__ import annotations

import pytest

from design_scientist.sequence_annotation import annotate_candidate

HEAVY_1E62 = "EMQLVESGGGLVQPGGSLRLSCAASGFTFSDYWMNWVRQAPGQGYTWHHHVKLKSNNYATHYAPSVAGRFTISRDDSKNSVYLQMNSLKTEDTAVYYCASGFDYWGQGTLVTVSS"
LIGHT_1E62 = "DIVMSQSPDSLAVSLGERATITCKSSQSLLYSSDQNNYLAWSQQKPGQPPKLLIYWASIRDSGVPDRFSGSGSGTDFTLTISSLQAEDVAVYYCHQYYTYPFTFGQGTKLEIK"


def _sequence(length: int, residues: dict[int, str]) -> str:
    chars = ["A"] * length
    for one_based_position, residue in residues.items():
        chars[one_based_position - 1] = residue
    return "".join(chars)


def test_parse_vh_vl_tokens_and_marks_regions_as_sequence_heuristic() -> None:
    heavy = _sequence(120, {30: "S"})
    light = _sequence(110, {24: "K", 44: "Q"})

    annotation = annotate_candidate(
        heavy_sequence=heavy,
        light_sequence=light,
        edit_notation="VH:S30H;VL:K24H;VL:Q44H",
    )

    assert [(item.chain, item.one_based_position, item.wildtype, item.mutant) for item in annotation.mutations] == [
        ("VH", 30, "S", "H"),
        ("VL", 24, "K", "H"),
        ("VL", 44, "Q", "H"),
    ]
    assert annotation.mutations[0].region == "cdr_like"
    assert annotation.mutations[2].region == "framework"
    assert all(item.region_is_heuristic for item in annotation.mutations)
    assert all("sequence-index heuristic" in item.region_note for item in annotation.mutations)
    assert all("not IMGT/Kabat" in item.region_note for item in annotation.mutations)


def test_invalid_tokens_are_reported_or_rejected() -> None:
    heavy = _sequence(60, {30: "S"})
    light = _sequence(60, {24: "K"})

    annotation = annotate_candidate(
        heavy_sequence=heavy,
        light_sequence=light,
        edit_notation="VH:S30H;bad-token;VL:K24H",
    )

    assert [(item.chain, item.one_based_position) for item in annotation.mutations] == [("VH", 30), ("VL", 24)]
    assert any("bad-token" in warning for warning in annotation.warnings)

    with pytest.raises(ValueError, match="bad-token"):
        annotate_candidate(
            heavy_sequence=heavy,
            light_sequence=light,
            edit_notation="VH:S30H;bad-token",
            strict=True,
        )


def test_glycosylation_nxs_t_motif_created_is_flagged() -> None:
    heavy = _sequence(80, {10: "S", 11: "A", 12: "T"})
    light = _sequence(60, {})

    annotation = annotate_candidate(
        heavy_sequence=heavy,
        light_sequence=light,
        edit_notation="VH:S10N",
    )

    assert annotation.mutations[0].flags["glycosylation_motif_created"] is True
    assert annotation.new_site_count == 0
    assert annotation.risk_count == 1


def test_selected_candidate_multi_mutation_summary_counts_developability_risk() -> None:
    heavy = _sequence(120, {30: "S"})
    light = _sequence(110, {24: "K", 44: "Q"})

    annotation = annotate_candidate(
        heavy_sequence=heavy,
        light_sequence=light,
        edit_notation="VH:S30H;VL:K24H;VL:Q44H",
        candidate_id="selected-001",
    )

    assert annotation.candidate_id == "selected-001"
    assert annotation.mutation_count == 3
    assert annotation.flag_counts["histidine_switch"] == 3
    assert annotation.flag_counts["charge_change"] == 2
    assert annotation.risk_count == 2
    assert annotation.new_site_count == 0


def test_new_site_count_comes_from_declared_new_site_notation() -> None:
    heavy = _sequence(120, {30: "S"})
    light = _sequence(110, {24: "K", 44: "Q"})

    annotation = annotate_candidate(
        heavy_sequence=heavy,
        light_sequence=light,
        edit_notation="VH:S30H;VL:K24H;VL:Q44H",
        new_site_notation=["VH:S30H"],
    )

    assert annotation.new_site_count == 1
    assert [item.is_new_site for item in annotation.mutations] == [True, False, False]


def test_real_antibody_sequences_get_imgt_numbering_when_anarci_is_available() -> None:
    annotation = annotate_candidate(
        heavy_sequence=HEAVY_1E62,
        light_sequence=LIGHT_1E62,
        edit_notation="VH:S30H;VL:K24H;VL:Q44H",
        new_site_notation=["VH:S30H"],
    )

    assert len(annotation.mutations) == 3
    if annotation.mutations[0].numbering_source != "anarci_abnumber":
        pytest.skip("ANARCI/abnumber numbering backend is not available in this pytest runtime")
    assert annotation.mutations[0].numbering_source == "anarci_abnumber"
    assert annotation.mutations[0].numbering_scheme == "imgt"
    assert annotation.mutations[0].numbering_position == "H35"
    assert annotation.mutations[0].numbering_region == "CDR1"
    assert annotation.mutations[0].region == "CDR1"
    assert annotation.mutations[0].region_is_heuristic is False
    assert "IMGT numbering" in annotation.mutations[0].region_note
    assert annotation.mutations[1].numbering_position == "L24"
    assert annotation.mutations[2].numbering_position == "L44"
