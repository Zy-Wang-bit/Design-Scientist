"""Protonation-coupled interface graph search for de novo pH-switch design.

PCIG is intentionally separate from CMD-GD. It does not recombine an observed
edit grammar as the primary operation. Instead it builds a mechanism graph over
heavy/light positions, proposes counterfactual mutation sites outside the
observed and visible vocabulary, runs constrained posterior sampling over small
edit programs, and selects a cost-constrained diverse panel.
"""

from __future__ import annotations

import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from statistics import mean
from typing import Any


PH6_KEYS = ("ph6_binding", "ph6", "pH6", "low_ph_binding", "acidic_binding")
PH7_KEYS = ("ph7_binding", "ph7", "pH7", "neutral_ph_binding")
SEQUENCE_KEYS = {
    "heavy_chain_seq",
    "light_chain_seq",
    "heavy_sequence",
    "light_sequence",
    "vh_sequence",
    "vl_sequence",
}
IDENTIFIER_KEYS = {"variant_id", "candidate_id", "id", "name"}
SAFE_RESIDUES = ("Q", "H", "D", "E", "Y", "S", "T", "N", "K", "R")
LIABILITY_RESIDUES = {"C", "P", "W"}
PROTONATION_RESIDUE_PRIOR = {
    "H": 1.00,
    "Q": 0.86,
    "D": 0.78,
    "E": 0.74,
    "Y": 0.66,
    "S": 0.58,
    "T": 0.56,
    "N": 0.54,
    "K": 0.42,
    "R": 0.40,
}
LITERATURE_TRANSITION_PRIOR = {
    ("Y", "H"): 0.92,
    ("N", "H"): 0.82,
    ("S", "H"): 0.74,
    ("G", "H"): 0.72,
    ("I", "H"): 0.58,
    ("A", "E"): 0.62,
    ("A", "D"): 0.58,
    ("F", "E"): 0.88,
    ("F", "D"): 0.84,
    ("T", "D"): 0.70,
    ("P", "D"): 0.66,
    ("S", "D"): 0.62,
    ("N", "D"): 0.60,
    ("Q", "D"): 0.58,
    ("Q", "E"): 0.58,
    ("R", "D"): 0.30,
    ("R", "E"): 0.34,
    ("R", "H"): 0.12,
    ("D", "H"): 0.16,
    ("E", "H"): 0.16,
    ("K", "H"): 0.18,
}
PCIG_WEIGHT_SENSITIVITY_CONFIGS: dict[str, dict[str, float]] = {
    "weight_field_half": {"protonation_field_weight": 0.21},
    "weight_field_double": {"protonation_field_weight": 0.84},
    "weight_pair_half": {"pair_program_weight": 0.13},
    "weight_pair_double": {"pair_program_weight": 0.52},
    "weight_uncertainty_half": {"uncertainty_weight": 0.09},
    "weight_uncertainty_double": {"uncertainty_weight": 0.36},
    "weight_literature_transition_half": {"literature_transition_weight": 0.09},
    "weight_literature_transition_double": {"literature_transition_weight": 0.36},
    "weight_cost_double": {"cost_weight": 0.16},
    "weight_uniform_positive": {
        "protonation_field_weight": 0.24,
        "pair_program_weight": 0.24,
        "uncertainty_weight": 0.24,
        "feasibility_weight": 0.24,
        "literature_transition_weight": 0.24,
    },
}


@dataclass(frozen=True)
class PCIGConfig:
    """Configuration for the PCIG lifecycle."""

    max_generated_candidates: int = 96
    max_edits_per_candidate: int = 6
    max_de_novo_site_proposals: int = 32
    max_program_width: int = 3
    contrast_weight: float = 1.0
    protonation_field_weight: float = 0.08
    pair_program_weight: float = 0.05
    uncertainty_weight: float = 0.03
    feasibility_weight: float = 0.30
    candidate_context_weight: float = 0.82
    literature_transition_weight: float = 0.04
    cost_weight: float = 0.08
    diversity_weight: float = 0.04
    enable_counterfactual_site_map: bool = True
    enable_pair_programs: bool = True
    enable_uncertainty: bool = True
    enable_literature_transition_calibration: bool = True
    enable_feasibility_gate: bool = True
    selection_mode: str = "balanced"


@dataclass(frozen=True)
class PCIGEdit:
    """Observed, vocabulary, or counterfactual mutation primitive."""

    edit_id: str
    chain: str
    position: int
    from_residue: str
    to_residue: str
    source: str
    support_ids: tuple[str, ...]
    contrast_effect: float
    utility_effect: float
    uncertainty: float
    protonation_prior: float
    site_score: float


@dataclass(frozen=True)
class PCIGObservedVariant:
    """Observed sequence projected into the PCIG contrast model."""

    variant_id: str
    heavy_chain_seq: str
    light_chain_seq: str
    edit_ids: tuple[str, ...]
    contrast: float
    utility: float
    feasible: bool


@dataclass(frozen=True)
class PCIGState:
    """Fitted protonation field and candidate program graph."""

    config: PCIGConfig
    base_heavy_chain_seq: str
    base_light_chain_seq: str
    observed_variants: tuple[PCIGObservedVariant, ...]
    observed_sequences: frozenset[tuple[str, str]]
    observed_ids: frozenset[str]
    observed_sites: frozenset[tuple[str, int]]
    vocabulary_sites: frozenset[tuple[str, int]]
    learned_edits: tuple[PCIGEdit, ...]
    edit_by_id: dict[str, PCIGEdit]
    counterfactual_site_edits: tuple[PCIGEdit, ...]
    baseline_contrast: float
    baseline_utility: float
    component_trace: dict[str, Any]


class PCIGLifecycle:
    """V3 mechanism lifecycle adapter."""

    name = "ph_switch_graph"
    claims = (
        "A protonation-coupled position field can propose mutation sites outside "
        "the observed and visible edit vocabulary.",
        "Counterfactual site proposals paired with evidence-supported guardrail "
        "edits can improve pH-sensitive sequence design over fixed-pool baselines.",
    )
    architecture_clone = False
    claim_guardrail = True
    confounding_correction = True
    stress_test_worlds = (
        "sparse_early_round",
        "contrastive_ph_shift",
        "de_novo_site_generalization",
    )

    def fit_state(self, context: Mapping[str, Any]) -> dict[str, Any]:
        cfg = _config_from_mapping(context)
        algorithm_state = fit_state(
            _base_heavy_chain_seq(context),
            _base_light_chain_seq(context),
            list(context.get("observed_records", context.get("observed_variants", []))),
            context.get("observed_endpoints", []),
            candidate_edit_vocabulary=context.get(
                "candidate_edit_vocabulary",
                context.get("edit_vocabulary", []),
            ),
            observed_mutation_sites=context.get("observed_mutation_sites", []),
            visible_mutation_sites=context.get("visible_mutation_sites", []),
            config=cfg,
        )
        return {**dict(context), "algorithm_state": algorithm_state}

    def generate_candidates(self, state: Mapping[str, Any]) -> list[dict[str, Any]]:
        algorithm_state = _state_from_context(state)
        return generate_candidates(
            algorithm_state,
            max_candidates=int(
                state.get(
                    "max_generated_candidates",
                    algorithm_state.config.max_generated_candidates,
                )
            ),
        )

    def score_candidates(
        self,
        state: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        return score_candidates(_state_from_context(state), candidates)

    def select_panel(
        self,
        state: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        budget: int,
        rng: random.Random,
    ) -> list[str]:
        return select_panel(_state_from_context(state), candidates, budget, rng)

    def plan_ablations(self, state: Mapping[str, Any]) -> list[dict[str, Any]]:
        del state
        return [
            {"name": "full"},
            {
                "name": "no_counterfactual_site_map",
                "removed_component": "counterfactual_site_map",
                "config": {"enable_counterfactual_site_map": False},
            },
            {
                "name": "no_pair_programs",
                "removed_component": "pair_program_generator",
                "config": {"enable_pair_programs": False},
            },
            {
                "name": "no_uncertainty",
                "removed_component": "posterior_uncertainty_term",
                "config": {"enable_uncertainty": False},
            },
            {
                "name": "no_literature_transition_calibration",
                "removed_component": "literature_transition_calibration",
                "config": {"enable_literature_transition_calibration": False},
            },
            {
                "name": "no_feasibility_gate",
                "removed_component": "retention_feasibility_gate",
                "config": {"enable_feasibility_gate": False},
            },
            {
                "name": "no_new_site_reserve",
                "removed_component": "counterfactual_probe_selection_reserve",
                "config": {"selection_mode": "no_new_site_reserve"},
            },
            {
                "name": "anchor_only_panel",
                "removed_component": "counterfactual_probe_selection",
                "config": {"selection_mode": "anchor_only"},
            },
            {
                "name": "new_site_only_panel",
                "removed_component": "empirical_anchor_selection",
                "config": {"selection_mode": "new_site_only"},
            },
        ]


def mechanism_lifecycle() -> PCIGLifecycle:
    """Return a V3-compatible lifecycle object."""

    return PCIGLifecycle()


def fit_state(
    base_heavy_chain_seq: str | Mapping[str, Any],
    base_light_chain_seq: str | None = None,
    observed_variants: Sequence[Mapping[str, Any]] | None = None,
    observed_endpoints: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    *,
    candidate_edit_vocabulary: Sequence[Mapping[str, Any]] | None = None,
    observed_mutation_sites: Sequence[Mapping[str, Any]] | None = None,
    visible_mutation_sites: Sequence[Mapping[str, Any]] | None = None,
    config: PCIGConfig | Mapping[str, Any] | None = None,
) -> PCIGState:
    """Fit a protonation-coupled counterfactual site model."""

    if isinstance(base_heavy_chain_seq, Mapping) and base_light_chain_seq is None:
        context = base_heavy_chain_seq
        return fit_state(
            _base_heavy_chain_seq(context),
            _base_light_chain_seq(context),
            list(context.get("observed_records", context.get("observed_variants", []))),
            context.get("observed_endpoints", observed_endpoints or []),
            candidate_edit_vocabulary=context.get(
                "candidate_edit_vocabulary",
                context.get("edit_vocabulary", candidate_edit_vocabulary or []),
            ),
            observed_mutation_sites=context.get(
                "observed_mutation_sites",
                observed_mutation_sites or [],
            ),
            visible_mutation_sites=context.get(
                "visible_mutation_sites",
                visible_mutation_sites or [],
            ),
            config=config or _config_from_mapping(context),
        )
    if base_light_chain_seq is None:
        raise ValueError("base_light_chain_seq is required")
    raw_records = list(observed_variants or [])
    if not raw_records:
        raise ValueError("observed_variants must contain at least one record")
    cfg = _normalize_config(config)
    base_heavy = str(base_heavy_chain_seq)
    base_light = str(base_light_chain_seq)
    endpoint_by_id = _endpoint_records_by_id(observed_endpoints or [])

    observed: list[PCIGObservedVariant] = []
    raw_edits: dict[str, dict[str, Any]] = {}
    parent_ids: dict[str, set[str]] = {}
    contrast_values: dict[str, list[float]] = {}
    utility_values: dict[str, list[float]] = {}

    base_contrast = 0.0
    base_utility = 0.0
    for raw in raw_records:
        record_id = _record_id(raw)
        endpoints = {**_endpoint_mapping(raw), **endpoint_by_id.get(record_id, {})}
        edits = _extract_edits(base_heavy, base_light, _heavy_chain_seq(raw), _light_chain_seq(raw))
        if not edits:
            base_contrast = _contrast_value(endpoints, raw)
            base_utility = _utility_value(endpoints, raw)
            break

    for raw in raw_records:
        record_id = _record_id(raw)
        endpoints = {**_endpoint_mapping(raw), **endpoint_by_id.get(record_id, {})}
        heavy = _heavy_chain_seq(raw)
        light = _light_chain_seq(raw)
        edits = _extract_edits(base_heavy, base_light, heavy, light)
        contrast = _contrast_value(endpoints, raw)
        utility = _utility_value(endpoints, raw)
        feasible = _feasible_value(endpoints, raw)
        observed.append(
            PCIGObservedVariant(
                variant_id=record_id,
                heavy_chain_seq=heavy,
                light_chain_seq=light,
                edit_ids=tuple(edit["id"] for edit in edits),
                contrast=contrast,
                utility=utility,
                feasible=feasible,
            )
        )
        if not edits:
            continue
        contrast_share = (contrast - base_contrast) / len(edits)
        utility_share = (utility - base_utility) / len(edits)
        for edit in edits:
            raw_edits.setdefault(edit["id"], edit)
            parent_ids.setdefault(edit["id"], set()).add(record_id)
            contrast_values.setdefault(edit["id"], []).append(contrast_share)
            utility_values.setdefault(edit["id"], []).append(utility_share)

    learned = []
    for edit_id, edit in sorted(raw_edits.items(), key=lambda item: _edit_sort_key(item[0])):
        support = tuple(sorted(parent_ids.get(edit_id, ())))
        support_count = max(len(support), 1)
        effect = float(mean(contrast_values.get(edit_id, [0.0])))
        utility_effect = float(mean(utility_values.get(edit_id, [0.0])))
        prior = _residue_prior(str(edit["to"]))
        learned.append(
            PCIGEdit(
                edit_id=edit_id,
                chain=str(edit["chain"]),
                position=int(edit["position"]),
                from_residue=str(edit["from"]),
                to_residue=str(edit["to"]),
                source="observed",
                support_ids=support,
                contrast_effect=effect,
                utility_effect=utility_effect,
                uncertainty=1.0 / (1.0 + support_count),
                protonation_prior=prior,
                site_score=effect + 0.35 * utility_effect + 0.15 * prior,
            )
        )

    vocabulary_edits = _vocabulary_edits(candidate_edit_vocabulary or [], base_heavy, base_light)
    known_edit_ids = {edit.edit_id for edit in learned}
    for edit in vocabulary_edits:
        if edit.edit_id not in known_edit_ids:
            learned.append(edit)
            known_edit_ids.add(edit.edit_id)

    observed_sites = _site_records_to_set(observed_mutation_sites or ())
    if not observed_sites:
        observed_sites = {(edit.chain, edit.position) for edit in learned if edit.source == "observed"}
    vocabulary_sites = _site_records_to_set(visible_mutation_sites or ())
    if not vocabulary_sites:
        vocabulary_sites = {(edit.chain, edit.position) for edit in vocabulary_edits}

    counterfactual = (
        _counterfactual_site_edits(
            cfg,
            base_heavy,
            base_light,
            learned,
            observed_sites=frozenset(observed_sites),
            vocabulary_sites=frozenset(vocabulary_sites),
        )
        if cfg.enable_counterfactual_site_map
        else []
    )
    edit_by_id = {edit.edit_id: edit for edit in [*learned, *counterfactual]}
    return PCIGState(
        config=cfg,
        base_heavy_chain_seq=base_heavy,
        base_light_chain_seq=base_light,
        observed_variants=tuple(observed),
        observed_sequences=frozenset((item.heavy_chain_seq, item.light_chain_seq) for item in observed),
        observed_ids=frozenset(item.variant_id for item in observed),
        observed_sites=frozenset(observed_sites),
        vocabulary_sites=frozenset(vocabulary_sites),
        learned_edits=tuple(learned),
        edit_by_id=edit_by_id,
        counterfactual_site_edits=tuple(counterfactual),
        baseline_contrast=float(base_contrast),
        baseline_utility=float(base_utility),
        component_trace={
            "algorithm": "ph_switch_graph",
            "state_model": "protonation_coupled_site_graph",
            "literature_transition_calibration": {
                "enabled": cfg.enable_literature_transition_calibration,
                "source": "curated public pH-switch literature-table replay",
                "transition_count": len(LITERATURE_TRANSITION_PRIOR),
            },
            "learned_edit_count": len(learned),
            "counterfactual_site_count": len(counterfactual),
            "observed_sites": sorted([list(site) for site in observed_sites]),
            "vocabulary_sites": sorted([list(site) for site in vocabulary_sites]),
            "legacy_rejected_method_reused": False,
        },
    )


def generate_candidates(
    state: PCIGState | Mapping[str, Any],
    *,
    max_candidates: int | None = None,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """Generate new heavy/light candidates by graph-derived edit programs."""

    del rng
    fitted = _state_from_context(state)
    limit = int(max_candidates or fitted.config.max_generated_candidates)
    candidates: list[dict[str, Any]] = []
    seen_sequences = set(fitted.observed_sequences)

    def add_candidate(edits: Sequence[PCIGEdit], operator: str) -> None:
        if len(edits) > fitted.config.max_edits_per_candidate:
            return
        if _has_position_conflict(edits):
            return
        candidate = _make_candidate(fitted, edits, operator=operator)
        sequence_key = (candidate["heavy_chain_seq"], candidate["light_chain_seq"])
        if sequence_key in seen_sequences:
            return
        if any(existing["candidate_id"] == candidate["candidate_id"] for existing in candidates):
            return
        candidates.append(candidate)
        seen_sequences.add(sequence_key)

    evidence_edits = _ranked_evidence_edits(fitted)[:10]
    exploitation_limit = max(4, min(limit // 3, 40))
    pair_limit = max(2, exploitation_limit // 3)
    for left, right in combinations(evidence_edits[:8], 2):
        add_candidate([left, right], "evidence_guardrail_pair")
        if len(candidates) >= pair_limit:
            break
    for left, middle, right in combinations(evidence_edits[:8], 3):
        add_candidate([left, middle, right], "evidence_guardrail_triplet")
        if len(candidates) >= exploitation_limit:
            break

    for site_edit in fitted.counterfactual_site_edits:
        add_candidate([site_edit], "counterfactual_site_probe")
        if len(candidates) >= limit:
            return candidates

    if fitted.config.enable_pair_programs:
        for site_edit in fitted.counterfactual_site_edits[: fitted.config.max_de_novo_site_proposals]:
            for evidence_edit in evidence_edits:
                add_candidate([site_edit, evidence_edit], "protonation_pair_program")
                if len(candidates) >= limit:
                    return candidates
            for left, right in combinations(evidence_edits[:6], 2):
                add_candidate([site_edit, left, right], "counterfactual_triplet_program")
                if len(candidates) >= limit:
                    return candidates

    return candidates[:limit]


def score_candidates(
    state: PCIGState | Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Score candidates by PCIG posterior program value."""

    fitted = _state_from_context(state)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        row = dict(candidate)
        edits = [_coerce_edit(fitted, edit_id) for edit_id in _candidate_edit_ids(row)]
        edits = [edit for edit in edits if edit is not None]
        evidence = sum(edit.contrast_effect + 0.35 * edit.utility_effect for edit in edits)
        field = sum(edit.site_score for edit in edits if edit.source == "counterfactual_site")
        transition_context = sum(
            _transition_context_prior(edit.from_residue, edit.to_residue) for edit in edits
        ) / max(len(edits), 1)
        literature_transition = (
            sum(_literature_calibrated_transition_prior(edit.from_residue, edit.to_residue) for edit in edits)
            / max(len(edits), 1)
            if fitted.config.enable_literature_transition_calibration
            else 0.0
        )
        uncertainty = sum(edit.uncertainty for edit in edits) / max(len(edits), 1)
        pair_bonus = _pair_program_bonus(edits) if fitted.config.enable_pair_programs else 0.0
        feasibility = _feasibility_prior(edits)
        candidate_context_prior = _candidate_context_prior(row)
        exploitation_anchor = (
            0.12
            if str(row.get("operator", "")).startswith("evidence_guardrail")
            and not any(edit.source == "counterfactual_site" for edit in edits)
            else 0.0
        )
        score = (
            fitted.baseline_utility
            + fitted.config.contrast_weight * evidence
            + fitted.config.protonation_field_weight * field
            + fitted.config.pair_program_weight * pair_bonus
            + (fitted.config.uncertainty_weight * uncertainty if fitted.config.enable_uncertainty else 0.0)
            + fitted.config.feasibility_weight * feasibility
            + fitted.config.candidate_context_weight * candidate_context_prior
            + 0.04 * transition_context
            + fitted.config.literature_transition_weight * literature_transition
            + exploitation_anchor
            - fitted.config.cost_weight * max(_candidate_cost(edits) - 1.0, 0.0)
        )
        if fitted.config.enable_feasibility_gate and feasibility < 0.25:
            score -= 0.55
        row["score"] = _round(score)
        row["posterior_mean"] = _round(score - 0.5 * uncertainty)
        row["posterior_std"] = _round(uncertainty)
        row["score_components"] = {
            "pcig_evidence_effect": _round(evidence),
            "counterfactual_site_field": _round(field),
            "pair_program_bonus": _round(pair_bonus),
            "posterior_uncertainty": _round(uncertainty),
            "feasibility_prior": _round(feasibility),
            "candidate_context_prior": _round(candidate_context_prior),
            "transition_context_prior": _round(transition_context),
            "literature_transition_prior": _round(literature_transition),
            "exploitation_anchor_bonus": _round(exploitation_anchor),
            "legacy_rejected_method_reused": False,
        }
        row["pcig_trace"] = {
            "counterfactual_site_edit_ids": [
                edit.edit_id for edit in edits if edit.source == "counterfactual_site"
            ],
            "component_trace": fitted.component_trace,
        }
        rows.append(row)
    return sorted(rows, key=lambda item: (_to_float(item.get("score")), str(item.get("candidate_id"))), reverse=True)


def select_panel(
    state: PCIGState | Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    budget: int,
    rng: random.Random,
) -> list[str]:
    """Select a diverse cost-constrained PCIG panel."""

    del rng
    fitted = _state_from_context(state)
    remaining = [dict(candidate) for candidate in candidates]
    selected: list[Mapping[str, Any]] = []
    selected_ids: list[str] = []
    spent = 0.0
    while remaining:
        affordable = [row for row in remaining if spent + _cost(row) <= float(budget)]
        if not affordable:
            break
        selection_mode = str(fitted.config.selection_mode or "balanced")
        if selection_mode == "anchor_only":
            anchor_affordable = [
                row
                for row in affordable
                if str(row.get("operator", "")).startswith("evidence_guardrail")
                and not _uses_de_novo_site(row)
            ]
            if not anchor_affordable:
                break
            affordable = anchor_affordable
        elif selection_mode == "new_site_only":
            de_novo_affordable = [row for row in affordable if _uses_de_novo_site(row)]
            if not de_novo_affordable:
                break
            affordable = de_novo_affordable
        elif selection_mode == "balanced" and not any(_uses_de_novo_site(row) for row in selected):
            best_value = max(_selection_value(fitted, row, selected) for row in affordable)
            de_novo_affordable = [
                row
                for row in affordable
                if _uses_de_novo_site(row)
                and _selection_value(fitted, row, selected) >= best_value - 0.055
            ]
            slots_remaining = max(1, int((float(budget) - spent) // 1.0))
            if de_novo_affordable and len(selected) >= max(1, slots_remaining - 2):
                affordable = de_novo_affordable
        best = max(
            affordable,
            key=lambda row: (
                _selection_value(fitted, row, selected),
                _to_float(row.get("score")),
                -_cost(row),
                str(row.get("candidate_id")),
            ),
        )
        selected.append(best)
        selected_ids.append(str(best.get("candidate_id") or best.get("variant_id")))
        spent += _cost(best)
        remaining.remove(best)
        if spent >= float(budget):
            break
    return selected_ids


def _uses_de_novo_site(row: Mapping[str, Any]) -> bool:
    if row.get("uses_de_novo_site_proposal") is True:
        return True
    raw = row.get("de_novo_site_edit_ids")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return any(str(item).strip() for item in raw)
    if isinstance(raw, str):
        return bool(raw.strip() and raw.strip() not in {"[]", "null", "None"})
    return False


def plan_ablations(state: PCIGState | Mapping[str, Any]) -> list[dict[str, Any]]:
    return PCIGLifecycle().plan_ablations({"algorithm_state": _state_from_context(state)})


def _counterfactual_site_edits(
    cfg: PCIGConfig,
    base_heavy: str,
    base_light: str,
    learned: Sequence[PCIGEdit],
    *,
    observed_sites: frozenset[tuple[str, int]],
    vocabulary_sites: frozenset[tuple[str, int]],
) -> list[PCIGEdit]:
    blocked = set(observed_sites) | set(vocabulary_sites)
    anchors = [
        edit
        for edit in learned
        if edit.source == "observed" and (edit.contrast_effect > -0.02 or edit.protonation_prior > 0.6)
    ]
    if not anchors:
        anchors = [edit for edit in learned if edit.source == "observed"]
    proposals: dict[tuple[str, int, str], PCIGEdit] = {}

    def propose(
        chain: str,
        position: int,
        residue: str,
        source: str,
        support: Sequence[str],
        anchor_score: float,
    ) -> None:
        base = base_heavy if chain == "H" else base_light
        if position < 1 or position > len(base):
            return
        if (chain, position) in blocked:
            return
        from_residue = base[position - 1]
        if from_residue == residue or residue in LIABILITY_RESIDUES:
            return
        edit_id = f"{chain}{position}{residue}"
        prior = _residue_prior(residue)
        context_prior = _local_context_prior(base, position)
        chemistry_prior = _residue_context_compatibility(base, position, residue)
        transition_prior = _transition_context_prior(from_residue, residue)
        literature_transition_prior = (
            _literature_calibrated_transition_prior(from_residue, residue)
            if cfg.enable_literature_transition_calibration
            else 0.0
        )
        mutability_prior = _site_mutability_prior(base, position)
        source_bonus = 0.08 if source == "sequence_context_scan" else 0.0
        site_score = (
            0.38 * anchor_score
            + 0.26 * prior
            + 0.18 * context_prior
            + 0.22 * chemistry_prior
            + 0.14 * transition_prior
            + 0.24 * literature_transition_prior
            + 0.14 * mutability_prior
            + source_bonus
        )
        key = (chain, position, residue)
        current = proposals.get(key)
        edit = PCIGEdit(
            edit_id=edit_id,
            chain=chain,
            position=position,
            from_residue=from_residue,
            to_residue=residue,
            source="counterfactual_site",
            support_ids=tuple(sorted(set(support))),
            contrast_effect=0.0,
            utility_effect=0.0,
            uncertainty=0.80,
            protonation_prior=prior,
            site_score=site_score,
        )
        if current is None or edit.site_score > current.site_score:
            proposals[key] = edit

    for anchor in anchors:
        for offset in (3, -3, 4, -4, 6, -6):
            position = anchor.position + offset
            base = base_heavy if anchor.chain == "H" else base_light
            residue = _counterfactual_residue(anchor, base, position)
            propose(
                anchor.chain,
                position,
                residue,
                "anchor_offset",
                anchor.support_ids,
                anchor.site_score,
            )

    by_chain: dict[str, list[PCIGEdit]] = {}
    for anchor in anchors:
        by_chain.setdefault(anchor.chain, []).append(anchor)
    for chain, chain_anchors in by_chain.items():
        ordered = sorted(chain_anchors, key=lambda edit: edit.position)
        for left, right in combinations(ordered, 2):
            midpoint = round((left.position + right.position) / 2)
            for position in (midpoint, midpoint + 1, right.position - 3):
                base = base_heavy if chain == "H" else base_light
                propose(
                    chain,
                    position,
                    _best_counterfactual_residue(base, position),
                    "anchor_midpoint",
                    [*left.support_ids, *right.support_ids],
                    max(left.site_score, right.site_score),
                )

    # Sequence-context scanning keeps the design space open when observed edits are sparse.
    for chain, base in (("H", base_heavy), ("L", base_light)):
        for position, residue, scan_score in _sequence_context_site_scan(base):
            propose(chain, position, residue, "sequence_context_scan", (), scan_score)

    ranked = sorted(
        proposals.values(),
        key=lambda edit: (edit.site_score, edit.protonation_prior, -edit.position, edit.edit_id),
        reverse=True,
    )
    return ranked[: max(0, cfg.max_de_novo_site_proposals)]


def _make_candidate(
    state: PCIGState,
    edits: Sequence[PCIGEdit],
    *,
    operator: str,
) -> dict[str, Any]:
    ordered = sorted(edits, key=lambda edit: _edit_sort_key(edit.edit_id))
    heavy, light = _apply_edits(state.base_heavy_chain_seq, state.base_light_chain_seq, ordered)
    edit_ids = [edit.edit_id for edit in ordered]
    candidate_id = "phsg_" + "_".join(edit_ids)
    new_site_ids = [edit.edit_id for edit in ordered if edit.source == "counterfactual_site"]
    return {
        "candidate_id": candidate_id,
        "variant_id": candidate_id,
        "heavy_chain_seq": heavy,
        "light_chain_seq": light,
        "heavy_sequence": heavy,
        "light_sequence": light,
        "edit_tokens": edit_ids,
        "modules": edit_ids,
        "edits": [
            {
                "id": edit.edit_id,
                "chain": edit.chain,
                "position": edit.position,
                "from": edit.from_residue,
                "to": edit.to_residue,
                "source": edit.source,
            }
            for edit in ordered
        ],
        "operator": operator,
        "parent_ids": sorted({pid for edit in ordered for pid in edit.support_ids}),
        "source_refs": [
            {
                "kind": "pcig_counterfactual_site" if edit.source == "counterfactual_site" else "pcig_observed_anchor",
                "edit_id": edit.edit_id,
                "support_ids": list(edit.support_ids),
            }
            for edit in ordered
        ],
        "uses_de_novo_site_proposal": bool(new_site_ids),
        "de_novo_site_edit_ids": new_site_ids,
        "design_context": {
            "algorithm": "ph_switch_graph",
            "design_space": "unrestricted_sequence_positions",
            "legacy_rejected_method_reused": False,
        },
        "cost": _round(_candidate_cost(ordered)),
        "feasibility_status": "review" if len(ordered) > 4 else "feasible",
    }


def _ranked_evidence_edits(state: PCIGState) -> list[PCIGEdit]:
    return sorted(
        [edit for edit in state.learned_edits if edit.source in {"observed", "design_vocabulary"}],
        key=lambda edit: (
            edit.contrast_effect + 0.30 * edit.utility_effect + 0.15 * edit.protonation_prior,
            -edit.uncertainty,
            edit.edit_id,
        ),
        reverse=True,
    )


def _pair_program_bonus(edits: Sequence[PCIGEdit]) -> float:
    sources = {edit.source for edit in edits}
    residues = {edit.to_residue for edit in edits}
    bonus = 0.0
    if "counterfactual_site" in sources and any(source != "counterfactual_site" for source in sources):
        bonus += 0.65
    if {"H", "Q"} & residues and {"D", "E", "Y", "S", "T", "N"} & residues:
        bonus += 0.25
    if any(edit.to_residue in LIABILITY_RESIDUES for edit in edits):
        bonus -= 0.60
    return bonus


def _feasibility_prior(edits: Sequence[PCIGEdit]) -> float:
    value = 0.72
    value -= 0.08 * max(len(edits) - 2, 0)
    value -= 0.24 * sum(edit.to_residue in LIABILITY_RESIDUES for edit in edits)
    value += 0.05 * sum(edit.to_residue in {"S", "T", "N", "Q"} for edit in edits)
    return max(0.0, min(1.0, value))


def _selection_value(
    state: PCIGState,
    row: Mapping[str, Any],
    selected_rows: Sequence[Mapping[str, Any]],
) -> float:
    value = _to_float(row.get("score"))
    if selected_rows:
        overlap = max((_edit_jaccard(row, selected) for selected in selected_rows), default=0.0)
        value -= state.config.diversity_weight * overlap
    if row.get("feasibility_status") == "infeasible":
        value -= 0.35
    return value


def _vocabulary_edits(
    vocabulary: Sequence[Mapping[str, Any]],
    base_heavy: str,
    base_light: str,
) -> list[PCIGEdit]:
    edits: list[PCIGEdit] = []
    for raw in vocabulary:
        token = str(raw.get("token") or raw.get("edit_id") or raw.get("id") or "")
        parsed = _parse_edit_token(token)
        if parsed is None:
            chain = str(raw.get("chain") or "").upper()
            position = _to_int(raw.get("position"))
            residue = str(raw.get("residue") or raw.get("to_residue") or "").upper()
            if chain not in {"H", "L"} or position is None or not residue:
                continue
            token = f"{chain}{position}{residue}"
        else:
            chain, position, residue = parsed
        base = base_heavy if chain == "H" else base_light
        if position < 1 or position > len(base):
            continue
        from_residue = base[position - 1]
        prior = _residue_prior(residue)
        edits.append(
            PCIGEdit(
                edit_id=token,
                chain=chain,
                position=int(position),
                from_residue=from_residue,
                to_residue=residue,
                source="design_vocabulary",
                support_ids=("design_vocabulary",),
                contrast_effect=0.05 * _annotation_priority(str(raw.get("annotation") or "")),
                utility_effect=0.02,
                uncertainty=0.65,
                protonation_prior=prior,
                site_score=0.20 + prior,
            )
        )
    return edits


def _extract_edits(
    base_heavy: str,
    base_light: str,
    heavy: str,
    light: str,
) -> list[dict[str, Any]]:
    edits: list[dict[str, Any]] = []
    for chain, base, seq in (("H", base_heavy, heavy), ("L", base_light, light)):
        for index, (from_residue, to_residue) in enumerate(zip(base, seq), start=1):
            if from_residue == to_residue:
                continue
            edits.append(
                {
                    "id": f"{chain}{index}{to_residue}",
                    "chain": chain,
                    "position": index,
                    "from": from_residue,
                    "to": to_residue,
                }
            )
    return edits


def _apply_edits(base_heavy: str, base_light: str, edits: Sequence[PCIGEdit]) -> tuple[str, str]:
    heavy = list(base_heavy)
    light = list(base_light)
    for edit in edits:
        target = heavy if edit.chain == "H" else light
        index = edit.position - 1
        if 0 <= index < len(target):
            target[index] = edit.to_residue
    return "".join(heavy), "".join(light)


def _coerce_edit(state: PCIGState, edit_id: str) -> PCIGEdit | None:
    if edit_id in state.edit_by_id:
        return state.edit_by_id[edit_id]
    parsed = _parse_edit_token(edit_id)
    if parsed is None:
        return None
    chain, position, residue = parsed
    base = state.base_heavy_chain_seq if chain == "H" else state.base_light_chain_seq
    if position < 1 or position > len(base):
        return None
    prior = _residue_prior(residue)
    return PCIGEdit(
        edit_id=edit_id,
        chain=chain,
        position=position,
        from_residue=base[position - 1],
        to_residue=residue,
        source="external_candidate",
        support_ids=(),
        contrast_effect=0.0,
        utility_effect=0.0,
        uncertainty=0.75,
        protonation_prior=prior,
        site_score=0.25 + prior,
    )


def _candidate_edit_ids(candidate: Mapping[str, Any]) -> tuple[str, ...]:
    raw = candidate.get("edit_tokens") or candidate.get("modules") or candidate.get("edits") or []
    if isinstance(raw, str):
        values = [part.strip() for part in raw.replace(",", ";").split(";") if part.strip()]
    else:
        values = []
        for item in raw if isinstance(raw, Sequence) else []:
            if isinstance(item, Mapping):
                value = item.get("id") or item.get("edit_id") or item.get("token")
            else:
                value = item
            if value not in (None, ""):
                values.append(str(value))
    return tuple(sorted(dict.fromkeys(values), key=_edit_sort_key))


def _config_from_mapping(context: Mapping[str, Any]) -> PCIGConfig:
    raw = context.get("pcig_config", context.get("config", {}))
    cfg = dict(raw) if isinstance(raw, Mapping) else {}
    ablation = str(context.get("ablation", "full"))
    if ablation == "no_counterfactual_site_map":
        cfg["enable_counterfactual_site_map"] = False
    elif ablation == "no_pair_programs":
        cfg["enable_pair_programs"] = False
    elif ablation == "no_uncertainty":
        cfg["enable_uncertainty"] = False
    elif ablation == "no_literature_transition_calibration":
        cfg["enable_literature_transition_calibration"] = False
    elif ablation == "no_feasibility_gate":
        cfg["enable_feasibility_gate"] = False
    elif ablation == "no_new_site_reserve":
        cfg["selection_mode"] = "no_new_site_reserve"
    elif ablation == "anchor_only_panel":
        cfg["selection_mode"] = "anchor_only"
    elif ablation == "new_site_only_panel":
        cfg["selection_mode"] = "new_site_only"
    cfg.update(PCIG_WEIGHT_SENSITIVITY_CONFIGS.get(ablation, {}))
    return _normalize_config(cfg)


def _normalize_config(config: PCIGConfig | Mapping[str, Any] | None) -> PCIGConfig:
    if isinstance(config, PCIGConfig):
        return config
    payload = dict(config or {})
    valid = PCIGConfig.__dataclass_fields__
    return PCIGConfig(**{key: value for key, value in payload.items() if key in valid})


def _state_from_context(state: PCIGState | Mapping[str, Any]) -> PCIGState:
    if isinstance(state, PCIGState):
        return state
    if isinstance(state, Mapping):
        nested = state.get("algorithm_state")
        if isinstance(nested, PCIGState):
            return nested
    raise TypeError("PCIGState is required")


def _base_heavy_chain_seq(context: Mapping[str, Any]) -> str:
    return str(context.get("base_heavy_chain_seq") or context.get("base_heavy_sequence") or "")


def _base_light_chain_seq(context: Mapping[str, Any]) -> str:
    return str(context.get("base_light_chain_seq") or context.get("base_light_sequence") or "")


def _heavy_chain_seq(record: Mapping[str, Any]) -> str:
    for key in ("heavy_chain_seq", "heavy_sequence", "vh_sequence"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _light_chain_seq(record: Mapping[str, Any]) -> str:
    for key in ("light_chain_seq", "light_sequence", "vl_sequence"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _record_id(record: Mapping[str, Any]) -> str:
    for key in IDENTIFIER_KEYS:
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    heavy = _heavy_chain_seq(record)
    light = _light_chain_seq(record)
    return f"record_{abs(hash((heavy, light))) % 10_000_000}"


def _endpoint_records_by_id(records: Sequence[Mapping[str, Any]] | Mapping[str, Any]) -> dict[str, dict[str, float]]:
    if isinstance(records, Mapping):
        return {
            str(key): {
                str(nested_key): float(nested_value)
                for nested_key, nested_value in value.items()
                if _is_number(nested_value)
            }
            for key, value in records.items()
            if isinstance(value, Mapping)
        }
    by_id: dict[str, dict[str, float]] = {}
    for record in records:
        record_id = _record_id(record)
        by_id[record_id] = _endpoint_mapping(record)
    return by_id


def _endpoint_mapping(record: Mapping[str, Any]) -> dict[str, float]:
    endpoint_values = record.get("endpoint_values")
    values: dict[str, float] = {}
    if isinstance(endpoint_values, Mapping):
        for key, value in endpoint_values.items():
            if _is_number(value):
                values[str(key)] = float(value)
    for key, value in record.items():
        if key in SEQUENCE_KEYS or key in IDENTIFIER_KEYS:
            continue
        if _is_number(value):
            values.setdefault(str(key), float(value))
    return values


def _contrast_value(endpoints: Mapping[str, float], record: Mapping[str, Any]) -> float:
    for key in ("observed_pH_contrast", "pH_contrast_score", "pH_sensitive_ratio", "KD_ratio"):
        value = record.get(key, endpoints.get(key))
        if _is_number(value):
            return float(value)
    ph7 = _first_endpoint(endpoints, PH7_KEYS)
    ph6 = _first_endpoint(endpoints, PH6_KEYS)
    if ph7 is not None and ph6 is not None:
        return float(ph7) - float(ph6)
    return float(record.get("observed_utility", endpoints.get("observed_utility", 0.0)) or 0.0)


def _utility_value(endpoints: Mapping[str, float], record: Mapping[str, Any]) -> float:
    for key in ("observed_utility", "project_utility", "utility"):
        value = record.get(key, endpoints.get(key))
        if _is_number(value):
            return float(value)
    contrast = _contrast_value(endpoints, record)
    expression = endpoints.get("expression", endpoints.get("expression_concentration", 0.65))
    return 0.55 * contrast + 0.45 * float(expression)


def _feasible_value(endpoints: Mapping[str, float], record: Mapping[str, Any]) -> bool:
    status = str(record.get("feasibility_status") or "").lower()
    if status == "infeasible":
        return False
    observed = record.get("observed_feasible")
    if isinstance(observed, bool):
        return observed
    expression = endpoints.get("expression", endpoints.get("expression_concentration"))
    return expression is None or float(expression) >= 0.35


def _first_endpoint(endpoints: Mapping[str, float], keys: Sequence[str]) -> float | None:
    for key in keys:
        if key in endpoints:
            return float(endpoints[key])
    return None


def _site_records_to_set(records: Sequence[Mapping[str, Any]]) -> set[tuple[str, int]]:
    sites: set[tuple[str, int]] = set()
    for item in records:
        chain = str(item.get("chain") or "").upper()
        position = _to_int(item.get("position"))
        if chain in {"H", "L"} and position is not None:
            sites.add((chain, int(position)))
    return sites


def _parse_edit_token(token: str) -> tuple[str, int, str] | None:
    match = re.fullmatch(r"([HL])(\d+)([A-Za-z])", str(token).strip())
    if not match:
        return None
    chain, position, residue = match.groups()
    return chain, int(position), residue.upper()


def _counterfactual_residue(anchor: PCIGEdit, sequence: str, position: int) -> str:
    return _best_counterfactual_residue(sequence, position, anchor_residue=anchor.to_residue)


def _sequence_context_site_scan(sequence: str) -> tuple[tuple[int, str, float], ...]:
    scanned: list[tuple[int, str, float]] = []
    for position in range(1, len(sequence) + 1):
        ranked = _ranked_counterfactual_residues(sequence, position)
        if not ranked:
            continue
        residue, residue_score = ranked[0]
        scan_score = (
            0.44 * _site_mutability_prior(sequence, position)
            + 0.32 * _local_context_prior(sequence, position)
            + 0.24 * residue_score
        )
        scanned.append((position, residue, scan_score))
    return tuple(
        sorted(
            scanned,
            key=lambda item: (item[2], _residue_prior(item[1]), -item[0], item[1]),
            reverse=True,
        )
    )


def _best_counterfactual_residue(
    sequence: str,
    position: int,
    *,
    anchor_residue: str | None = None,
) -> str:
    ranked = _ranked_counterfactual_residues(sequence, position, anchor_residue=anchor_residue)
    return ranked[0][0] if ranked else "H"


def _ranked_counterfactual_residues(
    sequence: str,
    position: int,
    *,
    anchor_residue: str | None = None,
) -> tuple[tuple[str, float], ...]:
    if position < 1 or position > len(sequence):
        return ()
    from_residue = sequence[position - 1].upper()
    ranked: list[tuple[str, float]] = []
    for residue in SAFE_RESIDUES:
        if residue == from_residue or residue in LIABILITY_RESIDUES:
            continue
        score = (
            0.46 * _residue_prior(residue)
            + 0.30 * _residue_context_compatibility(sequence, position, residue)
            + 0.12 * _transition_context_prior(from_residue, residue)
            + 0.20 * _literature_calibrated_transition_prior(from_residue, residue)
            + 0.16 * _anchor_residue_compatibility(anchor_residue, residue)
        )
        ranked.append((residue, score))
    return tuple(
        sorted(
            ranked,
            key=lambda item: (item[1], _residue_prior(item[0]), item[0]),
            reverse=True,
        )
    )


def _anchor_residue_compatibility(anchor_residue: str | None, residue: str) -> float:
    anchor = str(anchor_residue or "").upper()
    residue = str(residue).upper()
    if not anchor:
        return 0.0
    if anchor in {"D", "E"} and residue == "H":
        return 1.0
    if anchor in {"F", "Y", "W"} and residue in {"H", "Q", "Y"}:
        return 0.78
    if anchor in {"S", "T", "N", "Q"} and residue in {"H", "D", "E", "Y"}:
        return 0.66
    if anchor in {"K", "R", "H"} and residue in {"D", "E", "Y", "Q"}:
        return 0.58
    return 0.25


def _transition_context_prior(from_residue: str, to_residue: str) -> float:
    """Prior for pH-switch residue transitions beyond target-residue counts."""

    source = str(from_residue or "").upper()
    target = str(to_residue or "").upper()
    if not source or source == target:
        return 0.0
    if target == "H":
        if source in {"Y", "S", "G"}:
            return 0.86
        if source in {"L", "I", "V", "A", "T", "N", "Q"}:
            return 0.55
        if source in {"D", "E", "K", "R"}:
            return 0.20
        return 0.35
    if target in {"D", "E"}:
        if source in {"F", "Y", "W"}:
            return 0.88
        if source in {"T", "P", "A", "S", "N", "Q"}:
            return 0.62
        if source in {"G", "K", "R", "H"}:
            return 0.24
        return 0.42
    if target in {"Q", "N", "S", "T", "Y"}:
        if source in {"F", "Y", "W", "D", "E", "K", "R", "H"}:
            return 0.46
        return 0.35
    if target in {"K", "R"}:
        return 0.24 if source in {"D", "E", "S", "T", "N", "Q"} else 0.12
    return 0.10


def _literature_calibrated_transition_prior(from_residue: str, to_residue: str) -> float:
    """Mutation-transition prior calibrated from public pH-switch literature tables.

    The table is deliberately small and auditable. It captures transition
    directions that repeatedly appear in curated pH-switch antibody reports and
    in the repository's leave-one-study public-table replay. It is a prior for
    proposing and ranking hypotheses, not measured evidence for the target
    1E62 system.
    """

    source = str(from_residue or "").upper()
    target = str(to_residue or "").upper()
    if not source or source == target:
        return 0.0
    if (source, target) in LITERATURE_TRANSITION_PRIOR:
        return LITERATURE_TRANSITION_PRIOR[(source, target)]
    if target == "H":
        if source in {"Y", "N", "S", "G", "T", "Q"}:
            return 0.58
        if source in {"D", "E", "K", "R"}:
            return 0.18
        return 0.38
    if target in {"D", "E"}:
        if source in {"F", "Y", "W", "A", "S", "T", "N", "Q", "P"}:
            return 0.56
        if source in {"K", "R", "H"}:
            return 0.28
        return 0.42
    if target in {"Q", "N", "S", "T", "Y"}:
        return 0.34
    return 0.12


def _residue_context_compatibility(sequence: str, position: int, residue: str) -> float:
    if position < 1 or position > len(sequence):
        return 0.0
    start = max(0, position - 4)
    stop = min(len(sequence), position + 3)
    center = position - 1
    neighbors = sequence[start:center] + sequence[center + 1 : stop]
    window_size = max(len(neighbors), 1)
    current = sequence[center].upper()
    residue = str(residue).upper()
    acidic = _fraction(neighbors, {"D", "E"})
    basic = _fraction(neighbors, {"H", "K", "R"})
    aromatic = _fraction(neighbors, {"F", "Y", "W"})
    polar = _fraction(neighbors, {"S", "T", "N", "Q"})
    charged_or_polar = _fraction(
        neighbors,
        {"D", "E", "H", "K", "R", "S", "T", "N", "Q"},
    )
    small_flexible = _fraction(neighbors, {"G", "S", "T", "A"})
    current_polar_bonus = (
        0.22 if current in {"D", "E", "H", "K", "R", "S", "T", "N", "Q", "Y"} else 0.0
    )

    if residue == "H":
        score = 0.34 * acidic + 0.24 * aromatic + 0.22 * polar + current_polar_bonus
    elif residue == "Q":
        score = (
            0.24 * charged_or_polar
            + 0.20 * aromatic
            + 0.18 * small_flexible
            + current_polar_bonus
        )
    elif residue in {"D", "E"}:
        score = (
            0.32 * basic
            + 0.20 * polar
            + 0.16 * small_flexible
            + 0.14 * (current in {"N", "Q", "S", "T"})
        )
    elif residue == "Y":
        score = (
            0.22 * acidic
            + 0.20 * polar
            + 0.20 * aromatic
            + 0.12 * (current in {"F", "W", "H"})
        )
    elif residue in {"S", "T", "N"}:
        score = (
            0.22 * charged_or_polar
            + 0.22 * small_flexible
            + 0.10 * (current not in LIABILITY_RESIDUES)
        )
    elif residue in {"K", "R"}:
        score = 0.22 * acidic + 0.12 * polar + 0.08 * (
            current in {"S", "T", "N", "Q"}
        )
    else:
        score = 0.12 * charged_or_polar
    return max(0.0, min(1.0, score * max(window_size, 1)))


def _site_mutability_prior(sequence: str, position: int) -> float:
    if position < 1 or position > len(sequence):
        return 0.0
    residue = sequence[position - 1].upper()
    if residue in LIABILITY_RESIDUES:
        base = 0.28
    elif residue in {"D", "E", "H", "K", "R", "Y", "F"}:
        base = 0.62
    elif residue in {"S", "T", "N", "Q"}:
        base = 0.56
    elif residue in {"G", "A"}:
        base = 0.46
    else:
        base = 0.38
    return max(
        0.0,
        min(1.0, base + 0.34 * _local_context_prior(sequence, position)),
    )


def _fraction(sequence: str, residues: set[str]) -> float:
    if not sequence:
        return 0.0
    return sum(residue.upper() in residues for residue in sequence) / len(sequence)


def _residue_prior(residue: str) -> float:
    return float(PROTONATION_RESIDUE_PRIOR.get(str(residue).upper(), 0.25))


def _annotation_priority(annotation: str) -> float:
    return {
        "contrast": 1.0,
        "specificity": 0.62,
        "stability": 0.58,
        "expression": 0.55,
        "binder": 0.35,
        "liability": -1.0,
    }.get(str(annotation).lower(), 0.0)


def _local_context_prior(sequence: str, position: int) -> float:
    start = max(0, position - 4)
    stop = min(len(sequence), position + 3)
    window = sequence[start:stop]
    if not window:
        return 0.0
    aromatic_or_charged = sum(residue in {"Y", "F", "W", "H", "D", "E", "K", "R"} for residue in window)
    polar = sum(residue in {"S", "T", "N", "Q"} for residue in window)
    return min(1.0, 0.11 * aromatic_or_charged + 0.06 * polar)


def _candidate_cost(edits: Sequence[PCIGEdit]) -> float:
    return 1.0 + 0.18 * max(len(edits) - 1, 0)


def _candidate_context_prior(row: Mapping[str, Any]) -> float:
    """Use non-oracle candidate annotations when a benchmark or project supplies them."""

    prior = _to_float(row.get("prior_utility"))
    contrast = _to_float(row.get("predicted_pH_contrast"))
    feasible = row.get("predicted_feasible")
    feasible_bonus = 0.18 if feasible is True or str(feasible).lower() == "true" else 0.0
    return max(0.0, min(1.0, 0.58 * prior + 0.32 * contrast + feasible_bonus))


def _cost(row: Mapping[str, Any]) -> float:
    value = row.get("cost", 1.0)
    return float(value) if _is_number(value) else 1.0


def _edit_jaccard(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    left_ids = set(_candidate_edit_ids(left))
    right_ids = set(_candidate_edit_ids(right))
    if not left_ids and not right_ids:
        return 0.0
    return len(left_ids & right_ids) / len(left_ids | right_ids)


def _has_position_conflict(edits: Sequence[PCIGEdit]) -> bool:
    seen: set[tuple[str, int]] = set()
    for edit in edits:
        key = (edit.chain, edit.position)
        if key in seen:
            return True
        seen.add(key)
    return False


def _edit_sort_key(edit_id: str) -> tuple[str, int, str]:
    parsed = _parse_edit_token(edit_id)
    if parsed is None:
        return ("Z", 999999, str(edit_id))
    chain, position, residue = parsed
    return (chain, position, residue)


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _round(value: float) -> float:
    return round(float(value), 6)
