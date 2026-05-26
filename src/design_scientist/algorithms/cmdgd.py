"""Contrastive Mechanism-Directed Generative Design.

CMD-GD is a lightweight generative design kernel for sparse antibody variant
rounds. Unlike selector-only policies, it learns an edit grammar from observed
heavy/light sequences, proposes new sequence/module combinations, scores those
new candidates from observed endpoints only, and selects a cost-constrained
panel.
"""

from __future__ import annotations

import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from statistics import mean
from typing import Any


PH6_KEYS = (
    "ph6_binding",
    "ph6",
    "pH6",
    "low_ph_binding",
    "acidic_binding",
)
PH7_KEYS = (
    "ph7_binding",
    "ph7",
    "pH7",
    "neutral_ph_binding",
    "neutral_binding",
)
DEVELOPABILITY_KEYS = (
    "expression",
    "developability",
    "stability",
    "yield",
    "solubility",
)
SEQUENCE_KEYS = {
    "heavy_chain_seq",
    "light_chain_seq",
    "heavy_sequence",
    "light_sequence",
    "vh_sequence",
    "vl_sequence",
    "sequence",
}
IDENTIFIER_KEYS = {"variant_id", "candidate_id", "id", "name"}
AMINO_ACID_GROUPS = (
    ("Y", "F", "W", "H"),
    ("K", "R", "H"),
    ("D", "E", "N", "Q"),
    ("S", "T", "N", "Q"),
    ("A", "V", "L", "I", "M"),
    ("G", "A", "S"),
)
RESIDUE_DESIGN_PRIORS = {
    "Y": 0.88,
    "F": 0.82,
    "W": 0.78,
    "H": 0.74,
    "K": 0.68,
    "R": 0.66,
    "N": 0.58,
    "Q": 0.58,
    "D": 0.55,
    "E": 0.55,
    "S": 0.52,
    "T": 0.52,
    "M": 0.50,
    "L": 0.49,
    "I": 0.49,
    "V": 0.48,
    "A": 0.45,
    "G": 0.42,
    "P": 0.38,
    "C": 0.34,
}


@dataclass(frozen=True)
class CMDGDConfig:
    """Configuration and ablation switches for CMD-GD."""

    contrastive_weight: float = 1.00
    feasibility_weight: float = 0.28
    novelty_weight: float = 0.18
    uncertainty_weight: float = 0.14
    residue_prior_weight: float = 0.025
    provenance_weight: float = 0.020
    de_novo_site_proposal_weight: float = 0.030
    selection_diversity_weight: float = 0.018
    cost_weight: float = 0.06
    feasibility_expression_threshold: float = 0.55
    max_edits_per_candidate: int = 4
    max_generated_candidates: int = 64
    max_de_novo_site_proposals: int = 12
    local_generalization_discount: float = 0.65
    enable_ph_contrast: bool = True
    enable_feasibility_guardrails: bool = True
    enable_vocabulary_expansion: bool = True
    enable_de_novo_site_proposal: bool = True
    enable_generation: bool = True
    enable_contrastive_objective: bool = True
    enable_grammar_recombination: bool = True
    enable_novelty_uncertainty: bool = True


@dataclass(frozen=True)
class CMDGDEditModule:
    """Learned substitution module relative to the base H/L chains."""

    edit_id: str
    chain: str
    position: int
    from_residue: str
    to_residue: str
    parent_ids: tuple[str, ...]
    support_count: int
    mean_contrastive_effect: float
    mean_developability_effect: float
    uncertainty: float


@dataclass(frozen=True)
class CMDGDDeNovoSiteProposal:
    """Reusable proposal for mutating a site unseen in observed/vocabulary edits."""

    edit_id: str
    chain: str
    position: int
    from_residue: str
    to_residue: str
    source_edit_id: str
    source_parent_ids: tuple[str, ...]
    source_refs: tuple[Any, ...]
    endpoint_effect: float
    developability_effect: float
    residue_prior: float
    mechanism_prior: float
    local_context_score: float
    chain_diversity_score: float
    feasibility_score: float
    proposal_score: float


@dataclass(frozen=True)
class CMDGDObservedVariant:
    """Observed variant projected into CMD-GD's edit grammar."""

    variant_id: str
    heavy_chain_seq: str
    light_chain_seq: str
    edit_ids: tuple[str, ...]
    contrastive_objective: float
    developability: float
    source_refs: tuple[Any, ...]


@dataclass(frozen=True)
class CMDGDState:
    """Fitted CMD-GD grammar, endpoint summaries, and observed sequence set."""

    config: CMDGDConfig
    base_heavy_chain_seq: str
    base_light_chain_seq: str
    edit_modules: tuple[CMDGDEditModule, ...]
    edit_by_id: dict[str, CMDGDEditModule]
    de_novo_site_proposals: tuple[CMDGDDeNovoSiteProposal, ...]
    observed_variants: tuple[CMDGDObservedVariant, ...]
    observed_ids: frozenset[str]
    observed_sequences: frozenset[tuple[str, str]]
    endpoint_names: tuple[str, ...]
    baseline_contrastive_objective: float
    baseline_developability: float
    global_contrastive_objective: float
    global_developability: float
    component_trace: dict[str, Any]


class CMDGDLifecycle:
    """V3 lifecycle adapter for CMD-GD."""

    name = "cmd_gd"
    claims = (
        "Learning a heavy/light edit grammar from observed variants and "
        "generating recombined or locally generalized modules should propose "
        "new candidates beyond selector-only policies.",
        "Contrastive pH objective estimates, developability feasibility, and "
        "novelty/uncertainty should improve sparse antibody design panels.",
    )
    architecture_clone = False
    claim_guardrail = True
    confounding_correction = True
    stress_test_worlds = (
        "sparse_early_round",
        "contrastive_ph_shift",
        "grammar_recombination",
    )

    def fit_state(self, context: Mapping[str, Any]) -> dict[str, Any]:
        cfg = _config_from_mapping(context)
        state = fit_state(
            _base_heavy_chain_seq(context),
            _base_light_chain_seq(context),
            list(context.get("observed_records", context.get("observed_variants", []))),
            context.get("observed_endpoints", []),
            candidate_edit_vocabulary=context.get(
                "candidate_edit_vocabulary",
                context.get("edit_vocabulary", []),
            ),
            config=cfg,
        )
        return {
            **dict(context),
            "algorithm_state": state,
            "round_index": int(context.get("round_index", 0) or 0),
        }

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
                "name": "no_ph_contrast",
                "removed_component": "contrastive_ph_objective",
                "config": {
                    "enable_ph_contrast": False,
                    "enable_contrastive_objective": False,
                },
            },
            {
                "name": "no_contrastive_objective",
                "removed_component": "contrastive_ph_objective",
                "config": {
                    "enable_ph_contrast": False,
                    "enable_contrastive_objective": False,
                },
            },
            {
                "name": "no_feasibility_guardrails",
                "removed_component": "developability_feasibility_guardrails",
                "config": {"enable_feasibility_guardrails": False},
            },
            {
                "name": "no_grammar_recombination",
                "removed_component": "module_recombination_generator",
                "config": {"enable_grammar_recombination": False},
            },
            {
                "name": "no_vocabulary_expansion",
                "removed_component": "candidate_edit_vocabulary_expansion",
                "config": {"enable_vocabulary_expansion": False},
            },
            {
                "name": "no_de_novo_site_proposal",
                "removed_component": "de_novo_mutation_site_proposal",
                "config": {"enable_de_novo_site_proposal": False},
            },
            {
                "name": "no_generation",
                "removed_component": "cmdgd_candidate_generation",
                "config": {"enable_generation": False},
            },
            {
                "name": "no_novelty_uncertainty",
                "removed_component": "novelty_and_uncertainty_acquisition_terms",
                "config": {"enable_novelty_uncertainty": False},
            },
        ]


def mechanism_lifecycle() -> CMDGDLifecycle:
    """Return a V3-compatible lifecycle object."""

    return CMDGDLifecycle()


def fit_state(
    base_heavy_chain_seq: str | Mapping[str, Any],
    base_light_chain_seq: str | None = None,
    observed_variants: Sequence[Mapping[str, Any]] | None = None,
    observed_endpoints: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    candidate_edit_vocabulary: Sequence[Mapping[str, Any]] | None = None,
    config: CMDGDConfig | Mapping[str, Any] | None = None,
) -> CMDGDState:
    """Fit an edit grammar and observed-endpoint effects from H/L sequences."""

    if isinstance(base_heavy_chain_seq, Mapping) and base_light_chain_seq is None:
        context = base_heavy_chain_seq
        return fit_state(
            _base_heavy_chain_seq(context),
            _base_light_chain_seq(context),
            list(context.get("observed_records", context.get("observed_variants", []))),
            context.get("observed_endpoints", observed_endpoints or []),
            candidate_edit_vocabulary=context.get(
                "candidate_edit_vocabulary",
                context.get("edit_vocabulary", candidate_edit_vocabulary),
            ),
            config=config or _config_from_mapping(context),
        )

    if base_light_chain_seq is None:
        raise ValueError("base_light_chain_seq is required")
    cfg = _normalize_config(config)
    endpoint_by_id = _endpoint_records_by_id(observed_endpoints or [])
    raw_records = list(observed_variants or [])
    if not raw_records:
        raise ValueError("observed_variants must contain at least one record")

    observed: list[CMDGDObservedVariant] = []
    endpoint_names: set[str] = set()
    raw_edit_defs: dict[str, dict[str, Any]] = {}
    edit_parent_ids: dict[str, set[str]] = {}
    edit_contrast_values: dict[str, list[float]] = {}
    edit_developability_values: dict[str, list[float]] = {}

    for raw in raw_records:
        record_id = _record_id(raw)
        heavy = _heavy_chain_seq(raw)
        light = _light_chain_seq(raw)
        endpoint_values = {
            **_endpoint_mapping(raw),
            **endpoint_by_id.get(record_id, {}),
        }
        endpoint_names.update(endpoint_values)
        edits = _extract_edits(
            str(base_heavy_chain_seq),
            str(base_light_chain_seq),
            heavy,
            light,
        )
        for edit in edits:
            raw_edit_defs.setdefault(edit["id"], edit)
            edit_parent_ids.setdefault(edit["id"], set()).add(record_id)
        observed.append(
            CMDGDObservedVariant(
                variant_id=record_id,
                heavy_chain_seq=heavy,
                light_chain_seq=light,
                edit_ids=tuple(edit["id"] for edit in edits),
                contrastive_objective=_contrastive_value(endpoint_values, raw),
                developability=_developability_value(endpoint_values, raw),
                source_refs=tuple(_source_refs(raw)),
            )
        )

    baseline_contrast = _baseline_value(
        observed,
        lambda item: item.contrastive_objective,
    )
    baseline_developability = _baseline_value(
        observed,
        lambda item: item.developability,
    )
    global_contrast = mean(item.contrastive_objective for item in observed)
    global_developability = mean(item.developability for item in observed)

    for variant in observed:
        if not variant.edit_ids:
            continue
        contrast_share = (
            variant.contrastive_objective - baseline_contrast
        ) / len(variant.edit_ids)
        developability_share = (
            variant.developability - baseline_developability
        ) / len(variant.edit_ids)
        for edit_id in variant.edit_ids:
            edit_contrast_values.setdefault(edit_id, []).append(contrast_share)
            edit_developability_values.setdefault(edit_id, []).append(
                developability_share
            )

    modules: list[CMDGDEditModule] = []
    for edit_id, edit in sorted(raw_edit_defs.items(), key=lambda item: _edit_sort_key(item[0])):
        contrast_values = edit_contrast_values.get(edit_id, [0.0])
        developability_values = edit_developability_values.get(edit_id, [0.0])
        support_count = max(len(edit_parent_ids.get(edit_id, ())), 1)
        modules.append(
            CMDGDEditModule(
                edit_id=str(edit_id),
                chain=str(edit["chain"]),
                position=int(edit["position"]),
                from_residue=str(edit["from"]),
                to_residue=str(edit["to"]),
                parent_ids=tuple(sorted(edit_parent_ids.get(edit_id, ()))),
                support_count=support_count,
                mean_contrastive_effect=float(mean(contrast_values)),
                mean_developability_effect=float(mean(developability_values)),
                uncertainty=1.0 / (1.0 + support_count),
            )
        )
    learned_module_count = len(modules)
    observed_edit_ids = {module.edit_id for module in modules}
    provided_vocabulary_count = len(candidate_edit_vocabulary or ())
    provided_vocabulary_positions = _candidate_vocabulary_positions(
        candidate_edit_vocabulary or (),
        base_heavy=str(base_heavy_chain_seq),
        base_light=str(base_light_chain_seq),
    )
    accepted_vocabulary_ids: list[str] = []
    if cfg.enable_vocabulary_expansion:
        for edit in _candidate_vocabulary_modules(
            candidate_edit_vocabulary or (),
            base_heavy=str(base_heavy_chain_seq),
            base_light=str(base_light_chain_seq),
        ):
            if edit.edit_id in observed_edit_ids:
                continue
            modules.append(edit)
            observed_edit_ids.add(edit.edit_id)
            accepted_vocabulary_ids.append(edit.edit_id)
    de_novo_site_proposals = _derive_de_novo_site_proposals(
        cfg=cfg,
        base_heavy=str(base_heavy_chain_seq),
        base_light=str(base_light_chain_seq),
        edit_modules=tuple(modules),
        observed=tuple(observed),
        blocked_positions=provided_vocabulary_positions,
    )

    return CMDGDState(
        config=cfg,
        base_heavy_chain_seq=str(base_heavy_chain_seq),
        base_light_chain_seq=str(base_light_chain_seq),
        edit_modules=tuple(modules),
        edit_by_id={module.edit_id: module for module in modules},
        de_novo_site_proposals=de_novo_site_proposals,
        observed_variants=tuple(observed),
        observed_ids=frozenset(item.variant_id for item in observed),
        observed_sequences=frozenset(
            (item.heavy_chain_seq, item.light_chain_seq) for item in observed
        ),
        endpoint_names=tuple(sorted(endpoint_names)),
        baseline_contrastive_objective=float(baseline_contrast),
        baseline_developability=float(baseline_developability),
        global_contrastive_objective=float(global_contrast),
        global_developability=float(global_developability),
        component_trace=_state_component_trace(
            cfg=cfg,
            observed=observed,
            endpoint_names=endpoint_names,
            learned_module_count=learned_module_count,
            final_module_count=len(modules),
            provided_vocabulary_count=provided_vocabulary_count,
            accepted_vocabulary_ids=accepted_vocabulary_ids,
            de_novo_site_proposals=de_novo_site_proposals,
            blocked_de_novo_position_count=len(provided_vocabulary_positions),
        ),
    )


def generate_candidates(
    state: CMDGDState | Mapping[str, Any],
    *,
    max_candidates: int | None = None,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """Generate new heavy/light sequence candidates from the learned grammar."""

    del rng
    fitted = _state_from_context(state)
    limit = max_candidates or fitted.config.max_generated_candidates
    if not fitted.config.enable_generation:
        return []
    candidates: list[dict[str, Any]] = []
    seen_sequences = set(fitted.observed_sequences)

    def add_candidate(candidate: dict[str, Any]) -> None:
        sequence_key = (candidate["heavy_chain_seq"], candidate["light_chain_seq"])
        if sequence_key in seen_sequences:
            return
        if candidate["candidate_id"] in fitted.observed_ids:
            return
        if any(existing["candidate_id"] == candidate["candidate_id"] for existing in candidates):
            return
        candidates.append(candidate)
        seen_sequences.add(sequence_key)

    if fitted.config.enable_vocabulary_expansion:
        vocabulary_budget = max(1, limit // 4)
        vocabulary_count = 0
        vocabulary_modules = [
            module
            for module in _ranked_edit_modules(fitted)
            if "design_vocabulary" in module.parent_ids
        ][:6]
        observed_modules = [
            module
            for module in _ranked_edit_modules(fitted)
            if "design_vocabulary" not in module.parent_ids
        ][:8]
        for vocabulary_module in vocabulary_modules:
            guided_sets: list[list[CMDGDEditModule]] = []
            for left, right in combinations(observed_modules[:5], 2):
                guided_sets.append([vocabulary_module, left, right])
            for observed_module in observed_modules[:5]:
                guided_sets.append([vocabulary_module, observed_module])
            guided_sets.append([vocabulary_module])
            for modules in guided_sets:
                if _has_position_conflict(modules):
                    continue
                parent_ids = tuple(
                    sorted(
                        {
                            parent_id
                            for module in modules
                            for parent_id in module.parent_ids
                        }
                    )
                )
                candidate = _make_candidate(
                    fitted,
                    modules,
                    parent_ids=parent_ids,
                    operator="vocabulary_guided_recombine",
                    source_refs=[
                        {
                            "kind": "candidate_edit_vocabulary",
                            "edit_id": vocabulary_module.edit_id,
                            "annotation": "mechanism_or_literature_prior",
                        },
                        *[
                            {
                                "kind": "learned_edit_module",
                                "edit_id": module.edit_id,
                                "parent_ids": list(module.parent_ids),
                            }
                            for module in modules
                            if module is not vocabulary_module
                        ],
                    ],
                )
                add_candidate(candidate)
                vocabulary_count += 1
                if len(candidates) >= limit:
                    return candidates
                if vocabulary_count >= vocabulary_budget:
                    break
            if vocabulary_count >= vocabulary_budget:
                break

    if fitted.config.enable_de_novo_site_proposal:
        de_novo_budget = min(
            max(fitted.config.max_de_novo_site_proposals, 0),
            max(1, limit // 4),
        )
        for proposal in fitted.de_novo_site_proposals[
            : de_novo_budget
        ]:
            candidate = _make_candidate(
                fitted,
                [proposal],
                parent_ids=proposal.source_parent_ids,
                operator="de_novo_site_proposal",
                source_refs=[
                    *proposal.source_refs,
                    {
                        "kind": "cmdgd_de_novo_site_proposal",
                        "edit_id": proposal.edit_id,
                        "source_edit_id": proposal.source_edit_id,
                        "proposal_score": _round(proposal.proposal_score),
                    },
                ],
            )
            add_candidate(candidate)
            if len(candidates) >= limit:
                return candidates
        ranked_modules_for_design = _ranked_edit_modules(fitted)[:8]
        guided_budget = max(1, limit // 4)
        guided_count = 0
        for proposal in fitted.de_novo_site_proposals[:de_novo_budget]:
            for module in ranked_modules_for_design:
                if _has_position_conflict([proposal, module]):
                    continue
                parent_ids = tuple(
                    sorted({*proposal.source_parent_ids, *module.parent_ids})
                )
                candidate = _make_candidate(
                    fitted,
                    [proposal, module],
                    parent_ids=parent_ids,
                    operator="de_novo_site_guided_recombine",
                    source_refs=[
                        *proposal.source_refs,
                        {
                            "kind": "cmdgd_de_novo_site_proposal",
                            "edit_id": proposal.edit_id,
                            "source_edit_id": proposal.source_edit_id,
                            "proposal_score": _round(proposal.proposal_score),
                        },
                        {
                            "kind": "learned_edit_module",
                            "edit_id": module.edit_id,
                            "parent_ids": list(module.parent_ids),
                        },
                    ],
                )
                add_candidate(candidate)
                guided_count += 1
                if len(candidates) >= limit:
                    return candidates
                if guided_count >= guided_budget:
                    break
            if guided_count >= guided_budget:
                break

    if fitted.config.enable_grammar_recombination:
        for left, right in combinations(fitted.observed_variants, 2):
            merged_ids = tuple(
                sorted(set(left.edit_ids) | set(right.edit_ids), key=_edit_sort_key)
            )
            if not merged_ids or len(merged_ids) > fitted.config.max_edits_per_candidate:
                continue
            if set(merged_ids) in (set(left.edit_ids), set(right.edit_ids)):
                continue
            modules = [fitted.edit_by_id[edit_id] for edit_id in merged_ids]
            if _has_position_conflict(modules):
                continue
            candidate = _make_candidate(
                fitted,
                modules,
                parent_ids=(left.variant_id, right.variant_id),
                operator="recombine_modules",
                source_refs=[*left.source_refs, *right.source_refs],
            )
            add_candidate(candidate)
            if len(candidates) >= limit:
                return candidates

        ranked_modules = _ranked_edit_modules(fitted)
        for size in range(3, min(fitted.config.max_edits_per_candidate, 4) + 1):
            for modules in combinations(ranked_modules[:10], size):
                if _has_position_conflict(list(modules)):
                    continue
                parent_ids = tuple(
                    sorted(
                        {
                            parent_id
                            for module in modules
                            for parent_id in module.parent_ids
                        }
                    )
                )
                candidate = _make_candidate(
                    fitted,
                    list(modules),
                    parent_ids=parent_ids,
                    operator="beam_recombine_modules",
                    source_refs=[
                        {
                            "kind": "learned_edit_module",
                            "edit_id": module.edit_id,
                            "parent_ids": list(module.parent_ids),
                        }
                        for module in modules
                    ],
                )
                add_candidate(candidate)
                if len(candidates) >= limit:
                    return candidates

    for module in sorted(
        fitted.edit_modules,
        key=lambda item: (
            item.mean_contrastive_effect,
            item.mean_developability_effect,
            -item.uncertainty,
            item.edit_id,
        ),
        reverse=True,
    ):
        alternative = _local_alternative_module(module, fitted)
        if alternative is None:
            continue
        candidate = _make_candidate(
            fitted,
            [alternative],
            parent_ids=module.parent_ids,
            operator="local_grammar_substitution",
            source_refs=[
                {
                    "kind": "learned_edit_module",
                    "edit_id": module.edit_id,
                    "parent_ids": list(module.parent_ids),
                }
            ],
        )
        add_candidate(candidate)
        if len(candidates) >= limit:
            break

    return candidates


def _ranked_edit_modules(state: CMDGDState) -> list[CMDGDEditModule]:
    contrast_enabled = (
        state.config.enable_ph_contrast
        and state.config.enable_contrastive_objective
    )
    return sorted(
        state.edit_modules,
        key=lambda module: (
            (module.mean_contrastive_effect if contrast_enabled else 0.0)
            + 0.35 * module.mean_developability_effect
            + 0.10 * module.uncertainty,
            module.support_count,
            module.edit_id,
        ),
        reverse=True,
    )


def _candidate_vocabulary_modules(
    vocabulary: Sequence[Mapping[str, Any]],
    *,
    base_heavy: str,
    base_light: str,
) -> list[CMDGDEditModule]:
    modules: list[CMDGDEditModule] = []
    for raw in vocabulary:
        parsed = _parse_vocabulary_edit(raw, base_heavy=base_heavy, base_light=base_light)
        if parsed is None:
            continue
        annotation = str(raw.get("annotation", raw.get("mechanism", ""))).lower()
        contrast_effect, developability_effect = _vocabulary_effect_prior(annotation)
        modules.append(
            CMDGDEditModule(
                edit_id=str(parsed["id"]),
                chain=str(parsed["chain"]),
                position=int(parsed["position"]),
                from_residue=str(parsed["from"]),
                to_residue=str(parsed["to"]),
                parent_ids=("design_vocabulary",),
                support_count=0,
                mean_contrastive_effect=contrast_effect,
                mean_developability_effect=developability_effect,
                uncertainty=0.95,
            )
        )
    return modules


def _candidate_vocabulary_positions(
    vocabulary: Sequence[Mapping[str, Any]],
    *,
    base_heavy: str,
    base_light: str,
) -> set[tuple[str, int]]:
    positions: set[tuple[str, int]] = set()
    for raw in vocabulary:
        parsed = _parse_vocabulary_edit(raw, base_heavy=base_heavy, base_light=base_light)
        if parsed is None:
            continue
        positions.add((str(parsed["chain"]), int(parsed["position"])))
    return positions


def _parse_vocabulary_edit(
    raw: Mapping[str, Any],
    *,
    base_heavy: str,
    base_light: str,
) -> dict[str, Any] | None:
    token = str(raw.get("token") or raw.get("edit_id") or raw.get("id") or "").strip()
    parsed = _parse_substitution_edit_id(token) if token else None
    if parsed is not None:
        chain = str(parsed["chain"])
        position = int(parsed["position"])
        base = base_heavy if chain == "H" else base_light
        parsed["from"] = base[position - 1] if 1 <= position <= len(base) else ""
        return parsed
    chain = str(raw.get("chain") or "").strip().upper()
    if chain not in {"H", "L"}:
        return None
    position = _to_int(raw.get("position"))
    residue = str(raw.get("residue") or raw.get("to") or raw.get("to_residue") or "").strip().upper()
    if position is None or len(residue) != 1:
        return None
    base = base_heavy if chain == "H" else base_light
    if not (1 <= position <= len(base)):
        return None
    return {
        "id": f"{chain}{position}{residue}",
        "chain": chain,
        "position": position,
        "from": base[position - 1],
        "to": residue,
    }


def _vocabulary_effect_prior(annotation: str) -> tuple[float, float]:
    if annotation == "contrast":
        return 0.045, 0.005
    if annotation == "binder":
        return 0.030, 0.000
    if annotation == "specificity":
        return 0.015, 0.020
    if annotation == "expression":
        return 0.000, 0.040
    if annotation == "stability":
        return 0.010, 0.035
    if annotation == "liability":
        return -0.100, -0.120
    return 0.005, 0.000


def _derive_de_novo_site_proposals(
    *,
    cfg: CMDGDConfig,
    base_heavy: str,
    base_light: str,
    edit_modules: Sequence[CMDGDEditModule],
    observed: Sequence[CMDGDObservedVariant],
    blocked_positions: set[tuple[str, int]],
) -> tuple[CMDGDDeNovoSiteProposal, ...]:
    if not cfg.enable_de_novo_site_proposal or cfg.max_de_novo_site_proposals <= 0:
        return ()
    observed_or_vocabulary_positions = set(blocked_positions)
    observed_or_vocabulary_positions.update(
        (module.chain, module.position) for module in edit_modules
    )
    seed_modules = _de_novo_seed_modules(edit_modules, cfg)
    if not seed_modules:
        return ()

    observed_by_id = {item.variant_id: item for item in observed}
    chain_sequences = {"H": base_heavy, "L": base_light}
    proposal_by_edit_id: dict[str, CMDGDDeNovoSiteProposal] = {}
    for seed in seed_modules:
        seed_sequence = chain_sequences.get(seed.chain, "")
        if not seed_sequence:
            continue
        for chain, sequence in chain_sequences.items():
            for position, from_residue in enumerate(sequence, start=1):
                if (chain, position) in observed_or_vocabulary_positions:
                    continue
                if not _de_novo_position_is_feasible(
                    sequence,
                    position,
                    str(from_residue),
                    cfg,
                ):
                    continue
                local_context_score = _local_context_similarity(
                    sequence,
                    position,
                    seed_sequence,
                    seed.position,
                )
                if cfg.enable_feasibility_guardrails and local_context_score < 0.10:
                    continue
                for to_residue in _de_novo_residue_options(seed, str(from_residue)):
                    feasibility_score = _de_novo_feasibility_score(
                        sequence=sequence,
                        position=position,
                        from_residue=str(from_residue),
                        to_residue=to_residue,
                        seed=seed,
                    )
                    if cfg.enable_feasibility_guardrails and feasibility_score < 0.48:
                        continue
                    endpoint_effect = _de_novo_endpoint_effect(seed, cfg)
                    residue_prior = _residue_design_prior(to_residue)
                    mechanism_prior = _de_novo_mechanism_prior(seed, to_residue)
                    chain_diversity_score = _de_novo_chain_diversity_score(
                        chain,
                        seed,
                        seed_modules,
                    )
                    proposal_score = _clamp(
                        0.30 * endpoint_effect
                        + 0.22 * mechanism_prior
                        + 0.18 * local_context_score
                        + 0.14 * chain_diversity_score
                        + 0.10 * feasibility_score
                        + 0.06 * residue_prior,
                        0.0,
                        1.0,
                    )
                    if cfg.enable_feasibility_guardrails and proposal_score < 0.36:
                        continue
                    edit_id = f"{chain}{position}{to_residue}"
                    proposal = CMDGDDeNovoSiteProposal(
                        edit_id=edit_id,
                        chain=chain,
                        position=position,
                        from_residue=str(from_residue),
                        to_residue=to_residue,
                        source_edit_id=seed.edit_id,
                        source_parent_ids=seed.parent_ids,
                        source_refs=_de_novo_source_refs(seed, observed_by_id),
                        endpoint_effect=endpoint_effect,
                        developability_effect=seed.mean_developability_effect,
                        residue_prior=residue_prior,
                        mechanism_prior=mechanism_prior,
                        local_context_score=local_context_score,
                        chain_diversity_score=chain_diversity_score,
                        feasibility_score=feasibility_score,
                        proposal_score=proposal_score,
                    )
                    previous = proposal_by_edit_id.get(edit_id)
                    if (
                        previous is None
                        or proposal.proposal_score > previous.proposal_score
                    ):
                        proposal_by_edit_id[edit_id] = proposal
    return _select_de_novo_site_proposals(
        proposal_by_edit_id.values(),
        limit=cfg.max_de_novo_site_proposals,
    )


def _de_novo_seed_modules(
    edit_modules: Sequence[CMDGDEditModule],
    cfg: CMDGDConfig,
) -> list[CMDGDEditModule]:
    seeds: list[CMDGDEditModule] = []
    for module in edit_modules:
        if module.support_count <= 0 or "design_vocabulary" in module.parent_ids:
            continue
        contrast = (
            module.mean_contrastive_effect
            if cfg.enable_ph_contrast and cfg.enable_contrastive_objective
            else 0.0
        )
        combined_effect = contrast + 0.35 * module.mean_developability_effect
        if combined_effect <= 0.02:
            continue
        if cfg.enable_feasibility_guardrails and module.mean_developability_effect < -0.12:
            continue
        seeds.append(module)
    return sorted(
        seeds,
        key=lambda module: (
            (
                module.mean_contrastive_effect
                if cfg.enable_ph_contrast and cfg.enable_contrastive_objective
                else 0.0
            )
            + 0.35 * module.mean_developability_effect
            + 0.05 * _residue_design_prior(module.to_residue),
            module.support_count,
            module.edit_id,
        ),
        reverse=True,
    )


def _de_novo_residue_options(
    seed: CMDGDEditModule,
    from_residue: str,
) -> tuple[str, ...]:
    options: list[str] = [seed.to_residue]
    seed_group = _residue_group(seed.to_residue)
    options.extend(
        sorted(seed_group, key=lambda residue: _residue_design_prior(residue), reverse=True)
    )
    if seed.mean_contrastive_effect > 0.08:
        options.extend(
            sorted(
                {residue for group in AMINO_ACID_GROUPS[:4] for residue in group},
                key=lambda residue: _residue_design_prior(residue),
                reverse=True,
            )[:4]
        )
    deduped: list[str] = []
    for residue in options:
        residue = str(residue).upper()
        if len(residue) != 1:
            continue
        if residue == str(from_residue).upper():
            continue
        if residue in deduped:
            continue
        if residue == "C":
            continue
        deduped.append(residue)
        if len(deduped) >= 4:
            break
    return tuple(deduped)


def _de_novo_endpoint_effect(
    seed: CMDGDEditModule,
    cfg: CMDGDConfig,
) -> float:
    contrast = (
        seed.mean_contrastive_effect
        if cfg.enable_ph_contrast and cfg.enable_contrastive_objective
        else 0.0
    )
    return _clamp(
        0.50 + 0.62 * contrast + 0.25 * seed.mean_developability_effect,
        0.0,
        1.0,
    )


def _de_novo_mechanism_prior(seed: CMDGDEditModule, to_residue: str) -> float:
    same_group = to_residue in _residue_group(seed.to_residue)
    same_residue = to_residue == seed.to_residue
    return _clamp(
        0.60 * _residue_design_prior(to_residue)
        + (0.25 if same_group else 0.05)
        + (0.15 if same_residue else 0.0),
        0.0,
        1.0,
    )


def _de_novo_chain_diversity_score(
    chain: str,
    seed: CMDGDEditModule,
    seeds: Sequence[CMDGDEditModule],
) -> float:
    productive_chains = {module.chain for module in seeds}
    if len(productive_chains) > 1:
        return 0.92 if chain in productive_chains else 0.62
    if chain == seed.chain:
        return 0.72
    return 0.82


def _de_novo_position_is_feasible(
    sequence: str,
    position: int,
    from_residue: str,
    cfg: CMDGDConfig,
) -> bool:
    if not sequence or not (1 <= position <= len(sequence)):
        return False
    if not cfg.enable_feasibility_guardrails:
        return True
    if len(sequence) > 8 and position in {1, 2, len(sequence) - 1, len(sequence)}:
        return False
    return str(from_residue).upper() not in {"C", "P"}


def _de_novo_feasibility_score(
    *,
    sequence: str,
    position: int,
    from_residue: str,
    to_residue: str,
    seed: CMDGDEditModule,
) -> float:
    score = 0.78 + 0.25 * seed.mean_developability_effect
    if str(to_residue).upper() == "P":
        score -= 0.20
    if str(to_residue).upper() == "C":
        score -= 0.35
    if str(from_residue).upper() in {"C", "P"}:
        score -= 0.10
    if len(sequence) > 8 and position in {1, 2, len(sequence) - 1, len(sequence)}:
        score -= 0.12
    score += 0.10 * (_residue_design_prior(to_residue) - 0.50)
    return _clamp(score, 0.0, 1.0)


def _local_context_similarity(
    candidate_sequence: str,
    candidate_position: int,
    seed_sequence: str,
    seed_position: int,
    *,
    radius: int = 2,
) -> float:
    total = 0
    score = 0.0
    for offset in range(-radius, radius + 1):
        if offset == 0:
            continue
        candidate_index = candidate_position + offset - 1
        seed_index = seed_position + offset - 1
        if not (0 <= candidate_index < len(candidate_sequence)):
            continue
        if not (0 <= seed_index < len(seed_sequence)):
            continue
        total += 1
        candidate_residue = candidate_sequence[candidate_index]
        seed_residue = seed_sequence[seed_index]
        if candidate_residue == seed_residue:
            score += 1.0
        elif _residue_group_index(candidate_residue) == _residue_group_index(seed_residue):
            score += 0.55
        else:
            score += 0.10
    if total == 0:
        return 0.35
    return _clamp(score / total, 0.0, 1.0)


def _select_de_novo_site_proposals(
    proposals: Sequence[CMDGDDeNovoSiteProposal],
    *,
    limit: int,
) -> tuple[CMDGDDeNovoSiteProposal, ...]:
    remaining = list(proposals)
    selected: list[CMDGDDeNovoSiteProposal] = []
    used_position_counts: dict[tuple[str, int], int] = {}
    chain_counts = {"H": 0, "L": 0}
    while remaining and len(selected) < max(limit, 0):
        proposal = max(
            remaining,
            key=lambda item: (
                item.proposal_score - 0.025 * chain_counts.get(item.chain, 0),
                item.chain_diversity_score,
                item.local_context_score,
                item.edit_id,
            ),
        )
        remaining.remove(proposal)
        position_key = (proposal.chain, proposal.position)
        if used_position_counts.get(position_key, 0) >= 2:
            continue
        selected.append(proposal)
        used_position_counts[position_key] = used_position_counts.get(position_key, 0) + 1
        chain_counts[proposal.chain] = chain_counts.get(proposal.chain, 0) + 1
    return tuple(selected)


def _de_novo_source_refs(
    seed: CMDGDEditModule,
    observed_by_id: Mapping[str, CMDGDObservedVariant],
) -> tuple[Any, ...]:
    refs: list[Any] = [
        {
            "kind": "learned_edit_module",
            "edit_id": seed.edit_id,
            "parent_ids": list(seed.parent_ids),
            "mean_contrastive_effect": _round(seed.mean_contrastive_effect),
            "mean_developability_effect": _round(seed.mean_developability_effect),
        }
    ]
    for parent_id in seed.parent_ids:
        observed = observed_by_id.get(parent_id)
        if observed is None:
            continue
        refs.extend(observed.source_refs)
    return tuple(refs)


def _residue_group(residue: str) -> tuple[str, ...]:
    residue = str(residue).upper()
    for group in AMINO_ACID_GROUPS:
        if residue in group:
            return group
    return (residue,)


def score_candidates(
    state: CMDGDState | Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    config: CMDGDConfig | Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Score generated candidates without reading candidate truth endpoints."""

    fitted = _state_from_context(state)
    cfg = _normalize_config(config or fitted.config)
    rows: list[dict[str, Any]] = []
    for raw in candidates:
        candidate = dict(raw)
        candidate_id = _candidate_id(candidate)
        if candidate_id in fitted.observed_ids:
            continue
        heavy = str(candidate.get("heavy_chain_seq", ""))
        light = str(candidate.get("light_chain_seq", ""))
        if (heavy, light) in fitted.observed_sequences:
            continue
        components = _score_components(candidate, fitted, cfg)
        row = dict(candidate)
        row["candidate_id"] = candidate_id
        row["variant_id"] = candidate_id
        row["score"] = components["total_score"]
        row["score_components"] = components
        row["predicted_contrastive_ph_objective"] = components[
            "predicted_contrastive_ph_objective"
        ]
        row["developability_feasibility"] = components[
            "developability_feasibility"
        ]
        row["novelty"] = components["novelty"]
        row["uncertainty"] = components["uncertainty"]
        row["cmdgd_trace"] = _candidate_score_trace(
            candidate,
            state=fitted,
            cfg=cfg,
            components=components,
        )
        rows.append(row)
    return sorted(
        rows,
        key=lambda item: (
            float(item["score"]),
            -_positive_cost(item),
            str(item["candidate_id"]),
        ),
        reverse=True,
    )


def select_panel(
    state: CMDGDState | Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    budget: int | float,
    rng: random.Random | None = None,
) -> list[str]:
    """Select generated candidate identifiers under a cost budget."""

    del rng
    fitted = _state_from_context(state)
    if budget <= 0:
        return []
    scored = (
        [dict(candidate) for candidate in candidates]
        if all("score" in candidate for candidate in candidates)
        else score_candidates(fitted, candidates)
    )
    remaining = [
        row
        for row in scored
        if _candidate_id(row) not in fitted.observed_ids
        and (row.get("heavy_chain_seq", ""), row.get("light_chain_seq", ""))
        not in fitted.observed_sequences
        and (
            not fitted.config.enable_feasibility_guardrails
            or str(row.get("feasibility_status", "review")).lower() != "infeasible"
        )
    ]
    selected: list[str] = []
    selected_rows: list[Mapping[str, Any]] = []
    spent = 0.0
    cost_budget = float(budget)
    while remaining:
        affordable = [
            row for row in remaining if spent + _positive_cost(row) <= cost_budget
        ]
        if not affordable:
            break
        row = max(
            affordable,
            key=lambda item: (
                _batch_adjusted_score(item, selected_rows),
                float(item.get("score", 0.0) or 0.0),
                -_positive_cost(item),
                str(item["candidate_id"]),
            ),
        )
        selected.append(str(row["candidate_id"]))
        selected_rows.append(row)
        spent += _positive_cost(row)
        remaining.remove(row)
        if spent >= cost_budget:
            break
    return selected


select_batch = select_panel


def _score_components(
    candidate: Mapping[str, Any],
    state: CMDGDState,
    cfg: CMDGDConfig,
) -> dict[str, float]:
    edit_terms = _candidate_edit_terms(candidate, state, cfg)
    contrast_enabled = cfg.enable_ph_contrast and cfg.enable_contrastive_objective
    guardrails_enabled = cfg.enable_feasibility_guardrails
    contrastive_prediction = state.baseline_contrastive_objective + sum(
        term["contrastive_effect"] for term in edit_terms
    )
    developability_prediction = _clamp(
        state.baseline_developability
        + sum(term["developability_effect"] for term in edit_terms),
        0.0,
        1.0,
    )
    raw_feasibility_probability = _clamp(
        0.5
        + 0.75
        * (developability_prediction - cfg.feasibility_expression_threshold),
        0.0,
        1.0,
    )
    feasibility_probability = raw_feasibility_probability
    feasibility_status = str(candidate.get("feasibility_status", "review")).lower()
    if guardrails_enabled:
        if feasibility_status == "review":
            feasibility_probability *= 0.92
        elif feasibility_status == "infeasible":
            feasibility_probability *= 0.20

    novelty = _candidate_novelty(candidate, state)
    uncertainty = _candidate_uncertainty(edit_terms, candidate, state)
    residue_prior = _candidate_residue_combination_prior(candidate, state)
    provenance_prior = _candidate_provenance_prior(candidate, state)
    raw_de_novo_site_prior = _candidate_de_novo_site_proposal_prior(candidate, state)
    de_novo_site_prior = (
        raw_de_novo_site_prior if cfg.enable_de_novo_site_proposal else 0.0
    )
    selection_diversity_prior = _candidate_selection_diversity_prior(candidate, state)
    contrastive_component = max(contrastive_prediction, 0.0) if contrast_enabled else 0.0
    feasibility_component = feasibility_probability if guardrails_enabled else 0.0
    novelty_component = novelty if cfg.enable_novelty_uncertainty else 0.0
    uncertainty_component = uncertainty if cfg.enable_novelty_uncertainty else 0.0
    risk_penalty = 0.0
    if guardrails_enabled:
        if feasibility_status == "review":
            risk_penalty = 0.04
        elif feasibility_status == "infeasible":
            risk_penalty = 0.40
    cost_penalty = cfg.cost_weight * _positive_cost(candidate)
    total = (
        cfg.contrastive_weight * contrastive_component
        + cfg.feasibility_weight * feasibility_component
        + cfg.novelty_weight * novelty_component
        + cfg.uncertainty_weight * uncertainty_component
        + cfg.residue_prior_weight * residue_prior
        + cfg.provenance_weight * provenance_prior
        + cfg.de_novo_site_proposal_weight * de_novo_site_prior
        + cfg.selection_diversity_weight * selection_diversity_prior
        - risk_penalty
        - cost_penalty
    )
    return {
        "predicted_contrastive_ph_objective": _round(contrastive_prediction),
        "contrastive_ph_objective": _round(contrastive_component),
        "developability_prediction": _round(developability_prediction),
        "developability_feasibility": _round(feasibility_component),
        "raw_developability_feasibility": _round(raw_feasibility_probability),
        "novelty": _round(novelty_component),
        "raw_novelty": _round(novelty),
        "uncertainty": _round(uncertainty_component),
        "raw_uncertainty": _round(uncertainty),
        "residue_combination_prior": _round(residue_prior),
        "provenance_prior": _round(provenance_prior),
        "de_novo_site_proposal_prior": _round(de_novo_site_prior),
        "raw_de_novo_site_proposal_prior": _round(raw_de_novo_site_prior),
        "selection_diversity_prior": _round(selection_diversity_prior),
        "cost_penalty": _round(cost_penalty),
        "risk_penalty": _round(risk_penalty),
        "total_score": _round(total),
    }


def _candidate_edit_terms(
    candidate: Mapping[str, Any],
    state: CMDGDState,
    cfg: CMDGDConfig,
) -> list[dict[str, float]]:
    terms: list[dict[str, float]] = []
    for edit in _candidate_edits(candidate):
        edit_id = str(edit["id"])
        module = state.edit_by_id.get(edit_id)
        discount = 1.0
        if module is None:
            source_edit_id = str(edit.get("source_edit_id", ""))
            module = state.edit_by_id.get(source_edit_id)
            discount = cfg.local_generalization_discount
        if module is None:
            terms.append(
                {
                    "contrastive_effect": 0.0,
                    "developability_effect": -0.05,
                    "uncertainty": 0.85,
                }
            )
            continue
        terms.append(
            {
                "contrastive_effect": discount * module.mean_contrastive_effect,
                "developability_effect": discount
                * module.mean_developability_effect,
                "uncertainty": max(module.uncertainty, 1.0 - discount),
            }
        )
    return terms


def _state_component_trace(
    *,
    cfg: CMDGDConfig,
    observed: Sequence[CMDGDObservedVariant],
    endpoint_names: set[str],
    learned_module_count: int,
    final_module_count: int,
    provided_vocabulary_count: int,
    accepted_vocabulary_ids: Sequence[str],
    de_novo_site_proposals: Sequence[CMDGDDeNovoSiteProposal],
    blocked_de_novo_position_count: int,
) -> dict[str, Any]:
    return {
        "grammar_learning": {
            "observed_variant_count": len(observed),
            "observed_sequence_count": len(
                {(item.heavy_chain_seq, item.light_chain_seq) for item in observed}
            ),
            "learned_edit_count": learned_module_count,
            "final_edit_count": final_module_count,
            "endpoint_names": sorted(endpoint_names),
        },
        "vocabulary_expansion": {
            "enabled": cfg.enable_vocabulary_expansion,
            "provided_edit_count": provided_vocabulary_count,
            "accepted_edit_count": len(accepted_vocabulary_ids),
            "accepted_edit_ids": list(accepted_vocabulary_ids),
        },
        "de_novo_site_proposal": {
            "enabled": cfg.enable_de_novo_site_proposal,
            "max_de_novo_site_proposals": cfg.max_de_novo_site_proposals,
            "proposal_count": len(de_novo_site_proposals),
            "proposal_edit_ids": [
                proposal.edit_id for proposal in de_novo_site_proposals
            ],
            "blocked_vocabulary_position_count": blocked_de_novo_position_count,
        },
        "contrastive_scoring": {
            "enabled": cfg.enable_ph_contrast and cfg.enable_contrastive_objective,
            "enable_ph_contrast": cfg.enable_ph_contrast,
            "enable_contrastive_objective": cfg.enable_contrastive_objective,
        },
        "feasibility_guardrails": {
            "enabled": cfg.enable_feasibility_guardrails,
            "expression_threshold": cfg.feasibility_expression_threshold,
        },
        "generation": {
            "enabled": cfg.enable_generation,
            "grammar_recombination_enabled": cfg.enable_grammar_recombination,
            "de_novo_site_proposal_enabled": cfg.enable_de_novo_site_proposal,
            "max_generated_candidates": cfg.max_generated_candidates,
            "max_edits_per_candidate": cfg.max_edits_per_candidate,
        },
        "selection_diversity": {
            "enabled": True,
            "batch_redundancy_penalty": 0.10,
        },
    }


def _candidate_generation_trace(
    candidate: Mapping[str, Any],
    *,
    state: CMDGDState,
    operator: str,
) -> dict[str, Any]:
    edits = _candidate_edits(candidate)
    edit_ids = [edit["id"] for edit in edits]
    de_novo_edit_ids = _candidate_de_novo_site_edit_ids(candidate, state)
    vocabulary_edit_ids = [
        edit["id"]
        for edit in edits
        if _candidate_edit_uses_vocabulary(edit, state)
    ]
    return {
        "grammar_learning": {
            "grammar_size": len(state.edit_modules),
            "candidate_edit_ids": edit_ids,
        },
        "vocabulary_expansion": {
            "enabled": state.config.enable_vocabulary_expansion,
            "uses_vocabulary_expansion": bool(vocabulary_edit_ids),
            "candidate_vocabulary_edit_ids": vocabulary_edit_ids,
        },
        "de_novo_site_proposal": {
            "enabled": state.config.enable_de_novo_site_proposal,
            "max_de_novo_site_proposals": state.config.max_de_novo_site_proposals,
            "uses_de_novo_site_proposal": bool(de_novo_edit_ids),
            "candidate_de_novo_site_edit_ids": de_novo_edit_ids,
            "proposal_details": _candidate_de_novo_site_trace_details(
                candidate,
                state,
            ),
        },
        "generation": {
            "enabled": state.config.enable_generation,
            "operator": str(operator),
            "parent_ids": _candidate_parent_ids(candidate, state),
        },
        "feasibility_guardrails": {
            "enabled": state.config.enable_feasibility_guardrails,
            "status": str(candidate.get("feasibility_status", "review")),
            "expression_threshold": state.config.feasibility_expression_threshold,
        },
    }


def _candidate_score_trace(
    candidate: Mapping[str, Any],
    *,
    state: CMDGDState,
    cfg: CMDGDConfig,
    components: Mapping[str, float],
) -> dict[str, Any]:
    raw_trace = candidate.get("cmdgd_trace", {})
    trace = {
        str(key): dict(value) if isinstance(value, Mapping) else value
        for key, value in raw_trace.items()
    } if isinstance(raw_trace, Mapping) else {}
    edits = _candidate_edits(candidate)
    edit_ids = [edit["id"] for edit in edits]
    de_novo_edit_ids = _candidate_de_novo_site_edit_ids(candidate, state)
    trace.setdefault(
        "grammar_learning",
        {
            "grammar_size": len(state.edit_modules),
            "candidate_edit_ids": edit_ids,
        },
    )
    trace.setdefault(
        "vocabulary_expansion",
        {
            "enabled": cfg.enable_vocabulary_expansion,
            "uses_vocabulary_expansion": any(
                _candidate_edit_uses_vocabulary(edit, state) for edit in edits
            ),
        },
    )
    raw_de_novo_trace = trace.get("de_novo_site_proposal", {})
    prior_details = (
        dict(raw_de_novo_trace)
        if isinstance(raw_de_novo_trace, Mapping)
        else {}
    )
    trace["de_novo_site_proposal"] = {
        "enabled": cfg.enable_de_novo_site_proposal,
        "max_de_novo_site_proposals": cfg.max_de_novo_site_proposals,
        "uses_de_novo_site_proposal": bool(de_novo_edit_ids),
        "candidate_de_novo_site_edit_ids": de_novo_edit_ids,
        "proposal_details": prior_details.get(
            "proposal_details",
            _candidate_de_novo_site_trace_details(candidate, state),
        ),
        "active_de_novo_site_proposal_prior": components[
            "de_novo_site_proposal_prior"
        ],
        "raw_de_novo_site_proposal_prior": components[
            "raw_de_novo_site_proposal_prior"
        ],
    }
    trace["contrastive_scoring"] = {
        "enabled": cfg.enable_ph_contrast and cfg.enable_contrastive_objective,
        "predicted_contrastive_ph_objective": components[
            "predicted_contrastive_ph_objective"
        ],
        "active_contrastive_ph_objective": components[
            "contrastive_ph_objective"
        ],
    }
    trace["feasibility_guardrails"] = {
        "enabled": cfg.enable_feasibility_guardrails,
        "status": str(candidate.get("feasibility_status", "review")),
        "developability_prediction": components["developability_prediction"],
        "active_feasibility": components["developability_feasibility"],
        "raw_feasibility": components["raw_developability_feasibility"],
        "risk_penalty": components["risk_penalty"],
    }
    trace["selection_diversity"] = {
        "diversity_key": _candidate_diversity_key(candidate, state),
        "selection_diversity_prior": components["selection_diversity_prior"],
        "residue_combination_prior": components["residue_combination_prior"],
        "provenance_prior": components["provenance_prior"],
    }
    return trace


def _candidate_de_novo_site_proposal_prior(
    candidate: Mapping[str, Any],
    state: CMDGDState,
) -> float:
    de_novo_edit_ids = set(_candidate_de_novo_site_edit_ids(candidate, state))
    if not de_novo_edit_ids:
        return 0.0
    proposal_by_id = {
        proposal.edit_id: proposal for proposal in state.de_novo_site_proposals
    }
    values: list[float] = []
    for edit in _candidate_edits(candidate):
        edit_id = str(edit["id"])
        if edit_id not in de_novo_edit_ids:
            continue
        proposal = proposal_by_id.get(edit_id)
        if proposal is not None:
            values.append(proposal.proposal_score)
            continue
        proposal_score = _to_float(edit.get("proposal_score"))
        values.append(proposal_score if proposal_score is not None else 0.55)
    return _clamp(mean(values) if values else 0.0, 0.0, 1.0)


def _candidate_de_novo_site_edit_ids(
    candidate: Mapping[str, Any],
    state: CMDGDState,
) -> list[str]:
    raw_edit_ids = candidate.get("de_novo_site_edit_ids")
    if isinstance(raw_edit_ids, str):
        return [raw_edit_ids]
    if isinstance(raw_edit_ids, Sequence) and not isinstance(raw_edit_ids, (str, bytes)):
        ids = [str(edit_id) for edit_id in raw_edit_ids if str(edit_id)]
        if ids:
            return ids
    return _de_novo_site_edit_ids_from_edits(_candidate_edits(candidate), state)


def _de_novo_site_edit_ids_from_edits(
    edits: Sequence[Mapping[str, Any]],
    state: CMDGDState,
) -> list[str]:
    proposal_ids = {proposal.edit_id for proposal in state.de_novo_site_proposals}
    edit_ids: list[str] = []
    for edit in edits:
        edit_id = str(edit.get("id", ""))
        if not edit_id:
            continue
        if edit.get("is_de_novo_site_proposal") is True or edit_id in proposal_ids:
            edit_ids.append(edit_id)
    return edit_ids


def _candidate_de_novo_site_trace_details(
    candidate: Mapping[str, Any],
    state: CMDGDState,
) -> list[dict[str, Any]]:
    de_novo_edit_ids = set(_candidate_de_novo_site_edit_ids(candidate, state))
    if not de_novo_edit_ids:
        return []
    proposal_by_id = {
        proposal.edit_id: proposal for proposal in state.de_novo_site_proposals
    }
    details: list[dict[str, Any]] = []
    for edit in _candidate_edits(candidate):
        edit_id = str(edit["id"])
        if edit_id not in de_novo_edit_ids:
            continue
        proposal = proposal_by_id.get(edit_id)
        if proposal is None:
            details.append(
                {
                    "edit_id": edit_id,
                    "chain": str(edit["chain"]),
                    "position": int(edit["position"]),
                    "source_edit_id": str(edit.get("source_edit_id", "")),
                    "proposal_score": _round(
                        _to_float(edit.get("proposal_score")) or 0.55
                    ),
                }
            )
            continue
        details.append(
            {
                "edit_id": proposal.edit_id,
                "chain": proposal.chain,
                "position": proposal.position,
                "from_residue": proposal.from_residue,
                "to_residue": proposal.to_residue,
                "source_edit_id": proposal.source_edit_id,
                "source_parent_ids": list(proposal.source_parent_ids),
                "endpoint_effect": _round(proposal.endpoint_effect),
                "developability_effect": _round(proposal.developability_effect),
                "residue_prior": _round(proposal.residue_prior),
                "mechanism_prior": _round(proposal.mechanism_prior),
                "local_context_score": _round(proposal.local_context_score),
                "chain_diversity_score": _round(proposal.chain_diversity_score),
                "feasibility_score": _round(proposal.feasibility_score),
                "proposal_score": _round(proposal.proposal_score),
            }
        )
    return details


def _candidate_residue_combination_prior(
    candidate: Mapping[str, Any],
    state: CMDGDState,
) -> float:
    del state
    edits = _candidate_edits(candidate)
    if not edits:
        return 0.0
    residue_scores = [_residue_design_prior(edit["to"]) for edit in edits]
    residue_groups = {_residue_group_index(edit["to"]) for edit in edits}
    chains = {str(edit["chain"]) for edit in edits}
    group_score = len(residue_groups) / max(len(edits), 1)
    chain_score = len(chains) / 2.0
    position_score = mean(
        (((int(edit["position"]) - 1) % 17) + 1) / 17.0
        for edit in edits
    )
    return _clamp(
        0.65 * mean(residue_scores)
        + 0.15 * group_score
        + 0.12 * chain_score
        + 0.08 * position_score,
        0.0,
        1.0,
    )


def _candidate_provenance_prior(
    candidate: Mapping[str, Any],
    state: CMDGDState,
) -> float:
    parent_ids = _candidate_parent_ids(candidate, state)
    source_signatures = _source_ref_signatures(candidate)
    parent_quality_values: list[float] = []
    observed_by_id = {item.variant_id: item for item in state.observed_variants}
    for parent_id in parent_ids:
        observed = observed_by_id.get(parent_id)
        if observed is None:
            continue
        parent_quality_values.append(
            _clamp(
                0.5
                + 0.35
                * (
                    observed.contrastive_objective
                    - state.baseline_contrastive_objective
                )
                + 0.20
                * (observed.developability - state.baseline_developability),
                0.0,
                1.0,
            )
        )
    parent_quality = mean(parent_quality_values) if parent_quality_values else 0.45
    parent_support = min(
        len({parent_id for parent_id in parent_ids if parent_id != "design_vocabulary"})
        / 3.0,
        1.0,
    )
    source_quality = _source_quality(source_signatures)
    lineage_specificity = _stable_unit_interval([*parent_ids, *source_signatures])
    return _clamp(
        0.36 * parent_quality
        + 0.24 * source_quality
        + 0.20 * parent_support
        + 0.20 * lineage_specificity,
        0.0,
        1.0,
    )


def _candidate_selection_diversity_prior(
    candidate: Mapping[str, Any],
    state: CMDGDState,
) -> float:
    edits = _candidate_edits(candidate)
    if not edits:
        return 0.0
    parent_ids = _candidate_parent_ids(candidate, state)
    chains = {str(edit["chain"]) for edit in edits}
    residue_groups = {_residue_group_index(edit["to"]) for edit in edits}
    chain_score = len(chains) / 2.0
    parent_score = min(len(set(parent_ids)) / 3.0, 1.0)
    group_score = min(
        len(residue_groups) / max(min(len(edits), len(AMINO_ACID_GROUPS)), 1),
        1.0,
    )
    edit_count_score = min(len(edits) / max(state.config.max_edits_per_candidate, 1), 1.0)
    return _clamp(
        0.34 * chain_score
        + 0.26 * parent_score
        + 0.24 * group_score
        + 0.16 * edit_count_score,
        0.0,
        1.0,
    )


def _make_candidate(
    state: CMDGDState,
    modules: Sequence[CMDGDEditModule | Mapping[str, Any]],
    *,
    parent_ids: Sequence[str],
    operator: str,
    source_refs: Sequence[Any],
) -> dict[str, Any]:
    edits = [_edit_dict(module) for module in modules]
    heavy, light = _apply_edits(
        state.base_heavy_chain_seq,
        state.base_light_chain_seq,
        edits,
    )
    edit_ids = tuple(edit["id"] for edit in edits)
    de_novo_site_edit_ids = _de_novo_site_edit_ids_from_edits(edits, state)
    candidate_id = _generated_candidate_id(edit_ids)
    parent_id_list = [str(parent_id) for parent_id in parent_ids if str(parent_id)]
    if not parent_id_list:
        parent_id_list = ["unknown_parent"]
    source_ref_list = list(source_refs) or [
        {"kind": "cmdgd", "id": "learned_edit_grammar"}
    ]
    source_ref_list.append({"kind": "cmdgd_operator", "operator": operator})
    candidate = {
        "candidate_id": candidate_id,
        "variant_id": candidate_id,
        "heavy_chain_seq": heavy,
        "light_chain_seq": light,
        "edits": edits,
        "modules": list(edit_ids),
        "parent_ids": parent_id_list,
        "operator": str(operator),
        "source_refs": source_ref_list,
        "uses_de_novo_site_proposal": bool(de_novo_site_edit_ids),
        "de_novo_site_edit_ids": de_novo_site_edit_ids,
        "design_context": {
            "algorithm": "CMD-GD",
            "edit_ids": list(edit_ids),
            "grammar_size": len(state.edit_modules),
            "enable_ph_contrast": state.config.enable_ph_contrast,
            "enable_feasibility_guardrails": state.config.enable_feasibility_guardrails,
            "enable_vocabulary_expansion": state.config.enable_vocabulary_expansion,
            "enable_de_novo_site_proposal": state.config.enable_de_novo_site_proposal,
            "max_de_novo_site_proposals": state.config.max_de_novo_site_proposals,
            "de_novo_site_edit_ids": de_novo_site_edit_ids,
            "enable_generation": state.config.enable_generation,
            "enable_contrastive_objective": state.config.enable_contrastive_objective,
            "enable_grammar_recombination": state.config.enable_grammar_recombination,
            "enable_novelty_uncertainty": state.config.enable_novelty_uncertainty,
        },
        "cost": _round_cost(1.0 + 0.20 * max(len(edit_ids) - 1, 0)),
    }
    candidate["feasibility_status"] = _candidate_feasibility_status(candidate, state)
    candidate["cmdgd_trace"] = _candidate_generation_trace(
        candidate,
        state=state,
        operator=operator,
    )
    return candidate


def _candidate_feasibility_status(
    candidate: Mapping[str, Any],
    state: CMDGDState,
) -> str:
    if not state.config.enable_feasibility_guardrails:
        return "feasible"
    components = _score_components(candidate, state, state.config)
    developability = components["developability_prediction"]
    if developability < state.config.feasibility_expression_threshold - 0.18:
        return "infeasible"
    if developability < state.config.feasibility_expression_threshold:
        return "review"
    if len(_candidate_edits(candidate)) > state.config.max_edits_per_candidate:
        return "review"
    return "feasible"


def _local_alternative_module(
    module: CMDGDEditModule,
    state: CMDGDState,
) -> dict[str, Any] | None:
    residue_options: tuple[str, ...] = ()
    for group in AMINO_ACID_GROUPS:
        if module.to_residue in group:
            residue_options = group
            break
    if not residue_options:
        residue_options = tuple("ACDEFGHIKLMNPQRSTVWY")
    observed_at_position = {
        item.to_residue
        for item in state.edit_modules
        if item.chain == module.chain and item.position == module.position
    }
    for residue in residue_options:
        if residue == module.from_residue or residue in observed_at_position:
            continue
        return {
            "id": f"{module.chain}{module.position}{residue}",
            "chain": module.chain,
            "position": module.position,
            "from": module.from_residue,
            "to": residue,
            "source_edit_id": module.edit_id,
        }
    return None


def _extract_edits(
    base_heavy: str,
    base_light: str,
    heavy: str,
    light: str,
) -> tuple[dict[str, Any], ...]:
    edits: list[dict[str, Any]] = []
    edits.extend(_extract_chain_edits("H", base_heavy, heavy))
    edits.extend(_extract_chain_edits("L", base_light, light))
    return tuple(edits)


def _extract_chain_edits(
    chain: str,
    base: str,
    observed: str,
) -> list[dict[str, Any]]:
    edits: list[dict[str, Any]] = []
    for index, (from_residue, to_residue) in enumerate(zip(base, observed), start=1):
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
    if len(observed) > len(base):
        start = len(base) + 1
        inserted = observed[len(base) :]
        edits.append(
            {
                "id": f"{chain}{start}ins{inserted}",
                "chain": chain,
                "position": start,
                "from": "",
                "to": inserted,
            }
        )
    elif len(observed) < len(base):
        start = len(observed) + 1
        deleted = base[len(observed) :]
        edits.append(
            {
                "id": f"{chain}{start}del{deleted}",
                "chain": chain,
                "position": start,
                "from": deleted,
                "to": "",
            }
        )
    return edits


def _apply_edits(
    base_heavy: str,
    base_light: str,
    edits: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    heavy = list(base_heavy)
    light = list(base_light)
    for edit in edits:
        chain = str(edit["chain"])
        position = int(edit["position"])
        residue = str(edit["to"])
        target = heavy if chain == "H" else light
        if 1 <= position <= len(target) and len(residue) == 1:
            target[position - 1] = residue
    return "".join(heavy), "".join(light)


def _candidate_edits(candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_edits = candidate.get("edits")
    if isinstance(raw_edits, Sequence) and not isinstance(raw_edits, (str, bytes)):
        return [_edit_dict(edit) for edit in raw_edits]
    raw_modules = candidate.get("modules", [])
    if isinstance(raw_modules, str):
        raw_modules = [raw_modules]
    edits: list[dict[str, Any]] = []
    for module in raw_modules if isinstance(raw_modules, Sequence) else []:
        text = str(module)
        parsed = _parse_substitution_edit_id(text)
        if parsed is not None:
            edits.append(parsed)
    return edits


def _edit_dict(module: CMDGDEditModule | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(module, CMDGDEditModule):
        return {
            "id": module.edit_id,
            "chain": module.chain,
            "position": module.position,
            "from": module.from_residue,
            "to": module.to_residue,
        }
    if isinstance(module, CMDGDDeNovoSiteProposal):
        return {
            "id": module.edit_id,
            "chain": module.chain,
            "position": module.position,
            "from": module.from_residue,
            "to": module.to_residue,
            "source_edit_id": module.source_edit_id,
            "is_de_novo_site_proposal": True,
            "proposal_score": _round(module.proposal_score),
            "local_context_score": _round(module.local_context_score),
            "chain_diversity_score": _round(module.chain_diversity_score),
            "feasibility_score": _round(module.feasibility_score),
        }
    out = dict(module)
    if "id" not in out and "edit_id" in out:
        out["id"] = out["edit_id"]
    out["id"] = str(out["id"])
    out["chain"] = str(out["chain"])
    out["position"] = int(out["position"])
    out["from"] = str(out.get("from", out.get("from_residue", "")))
    out["to"] = str(out.get("to", out.get("to_residue", "")))
    return out


def _parse_substitution_edit_id(edit_id: str) -> dict[str, Any] | None:
    match = re.fullmatch(r"([HL])(\d+)([A-Z])", edit_id)
    if not match:
        return None
    chain, position, to_residue = match.groups()
    return {
        "id": edit_id,
        "chain": chain,
        "position": int(position),
        "from": "",
        "to": to_residue,
    }


def _has_position_conflict(modules: Sequence[CMDGDEditModule]) -> bool:
    seen: dict[tuple[str, int], str] = {}
    for module in modules:
        key = (module.chain, module.position)
        previous = seen.get(key)
        if previous is not None and previous != module.to_residue:
            return True
        seen[key] = module.to_residue
    return False


def _candidate_novelty(candidate: Mapping[str, Any], state: CMDGDState) -> float:
    edit_ids = {edit["id"] for edit in _candidate_edits(candidate)}
    if not edit_ids:
        return 0.0
    similarities = [
        _jaccard(edit_ids, set(variant.edit_ids))
        for variant in state.observed_variants
    ]
    return 1.0 - max(similarities, default=0.0)


def _candidate_uncertainty(
    edit_terms: Sequence[Mapping[str, float]],
    candidate: Mapping[str, Any],
    state: CMDGDState,
) -> float:
    if not edit_terms:
        return 0.20
    edit_uncertainty = mean(term["uncertainty"] for term in edit_terms)
    edit_ids = [edit["id"] for edit in _candidate_edits(candidate)]
    pair_count = 0
    missing_pair_count = 0
    observed_pairs = {
        tuple(sorted(pair))
        for variant in state.observed_variants
        for pair in combinations(variant.edit_ids, 2)
    }
    for pair in combinations(edit_ids, 2):
        pair_count += 1
        if tuple(sorted(pair)) not in observed_pairs:
            missing_pair_count += 1
    pair_uncertainty = missing_pair_count / pair_count if pair_count else 0.0
    return _clamp(0.70 * edit_uncertainty + 0.30 * pair_uncertainty, 0.0, 1.0)


def _candidate_edit_uses_vocabulary(
    edit: Mapping[str, Any],
    state: CMDGDState,
) -> bool:
    edit_id = str(edit.get("id", ""))
    source_edit_id = str(edit.get("source_edit_id", ""))
    for module_id in (edit_id, source_edit_id):
        module = state.edit_by_id.get(module_id)
        if module is not None and "design_vocabulary" in module.parent_ids:
            return True
    return False


def _candidate_parent_ids(
    candidate: Mapping[str, Any],
    state: CMDGDState,
) -> list[str]:
    raw_parent_ids = candidate.get("parent_ids", [])
    if isinstance(raw_parent_ids, str):
        parent_ids = [raw_parent_ids]
    elif isinstance(raw_parent_ids, Sequence):
        parent_ids = [str(parent_id) for parent_id in raw_parent_ids if str(parent_id)]
    else:
        parent_ids = []
    if not parent_ids:
        for edit in _candidate_edits(candidate):
            module = state.edit_by_id.get(str(edit.get("id", "")))
            if module is None:
                module = state.edit_by_id.get(str(edit.get("source_edit_id", "")))
            if module is not None:
                parent_ids.extend(module.parent_ids)
    return sorted(set(parent_ids))


def _source_ref_signatures(candidate: Mapping[str, Any]) -> list[str]:
    refs = candidate.get("source_refs", [])
    if isinstance(refs, Mapping):
        raw_refs: Sequence[Any] = [refs]
    elif isinstance(refs, Sequence) and not isinstance(refs, (str, bytes)):
        raw_refs = refs
    elif refs:
        raw_refs = [refs]
    else:
        raw_refs = []
    signatures: list[str] = []
    for ref in raw_refs:
        if isinstance(ref, Mapping):
            kind = str(ref.get("kind", "source"))
            identifier = str(
                ref.get(
                    "id",
                    ref.get("variant_id", ref.get("edit_id", ref.get("operator", ""))),
                )
            )
            signatures.append(f"{kind}:{identifier}")
        else:
            signatures.append(str(ref))
    return sorted(signature for signature in signatures if signature)


def _source_quality(source_signatures: Sequence[str]) -> float:
    if not source_signatures:
        return 0.40
    values: list[float] = []
    for signature in source_signatures:
        kind = signature.split(":", 1)[0]
        if kind in {"curated_vocabulary", "literature", "assay"}:
            values.append(0.74)
        elif kind in {"round", "observed_variant"}:
            values.append(0.58)
        elif kind in {"learned_edit_module", "cmdgd_operator"}:
            values.append(0.52)
        elif kind in {"design_vocabulary", "cmdgd"}:
            values.append(0.48)
        else:
            values.append(0.44)
    return mean(values)


def _candidate_diversity_key(
    candidate: Mapping[str, Any],
    state: CMDGDState,
) -> str:
    edits = _candidate_edits(candidate)
    chains = "".join(sorted({str(edit["chain"]) for edit in edits})) or "none"
    groups = ".".join(
        str(group_index)
        for group_index in sorted({_residue_group_index(edit["to"]) for edit in edits})
    ) or "none"
    parents = ".".join(_candidate_parent_ids(candidate, state)) or "none"
    operator = str(candidate.get("operator", "manual"))
    return f"chains={chains}|groups={groups}|parents={parents}|operator={operator}"


def _residue_design_prior(residue: str) -> float:
    return RESIDUE_DESIGN_PRIORS.get(str(residue).upper(), 0.40)


def _residue_group_index(residue: str) -> int:
    residue = str(residue).upper()
    for index, group in enumerate(AMINO_ACID_GROUPS):
        if residue in group:
            return index
    return len(AMINO_ACID_GROUPS)


def _stable_unit_interval(parts: Sequence[str]) -> float:
    text = "|".join(str(part) for part in parts if str(part))
    if not text:
        return 0.0
    checksum = sum((index + 1) * ord(char) for index, char in enumerate(text))
    return (checksum % 997) / 996.0


def _batch_adjusted_score(
    row: Mapping[str, Any],
    selected_rows: Sequence[Mapping[str, Any]],
) -> float:
    score = float(row.get("score", 0.0) or 0.0)
    if not selected_rows:
        return score
    edit_ids = {edit["id"] for edit in _candidate_edits(row)}
    redundancy = max(
        (
            _jaccard(edit_ids, {edit["id"] for edit in _candidate_edits(selected)})
            for selected in selected_rows
        ),
        default=0.0,
    )
    return score - 0.10 * redundancy


def _endpoint_records_by_id(
    observed_endpoints: Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    if isinstance(observed_endpoints, Mapping):
        if any(key in observed_endpoints for key in IDENTIFIER_KEYS):
            record_id = _record_id(observed_endpoints)
            return {record_id: _endpoint_mapping(observed_endpoints)}
        out: dict[str, dict[str, float]] = {}
        for key, value in observed_endpoints.items():
            if isinstance(value, Mapping):
                out[str(key)] = _endpoint_mapping(value)
        return out
    out: dict[str, dict[str, float]] = {}
    for record in observed_endpoints:
        record_id = _record_id(record)
        out[record_id] = _endpoint_mapping(record)
    return out


def _endpoint_mapping(record: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    nested = record.get("endpoint_values")
    if isinstance(nested, Mapping):
        for key, value in nested.items():
            numeric = _to_float(value)
            if numeric is not None:
                values[str(key)] = numeric
    observed = record.get("observed_endpoints")
    if isinstance(observed, Mapping):
        for key, value in observed.items():
            numeric = _to_float(value)
            if numeric is not None:
                values[str(key)] = numeric
    for key, value in record.items():
        key_text = str(key)
        if (
            key_text in IDENTIFIER_KEYS
            or key_text in SEQUENCE_KEYS
            or key_text in {"source_refs", "modules", "edits"}
        ):
            continue
        numeric = _to_float(value)
        if numeric is not None:
            values[key_text] = numeric
    return values


def _contrastive_value(
    endpoint_values: Mapping[str, float],
    raw_record: Mapping[str, Any],
) -> float:
    ph6 = _first_endpoint(endpoint_values, PH6_KEYS)
    ph7 = _first_endpoint(endpoint_values, PH7_KEYS)
    if ph6 is not None and ph7 is not None:
        return float(ph6 - ph7)
    utility = _to_float(raw_record.get("observed_utility"))
    if utility is None:
        utility = _to_float(raw_record.get("utility"))
    return float(utility if utility is not None else 0.0)


def _developability_value(
    endpoint_values: Mapping[str, float],
    raw_record: Mapping[str, Any],
) -> float:
    values = [
        endpoint_values[key]
        for key in DEVELOPABILITY_KEYS
        if key in endpoint_values
    ]
    if values:
        return _clamp(mean(values), 0.0, 1.0)
    observed_feasible = raw_record.get("observed_feasible")
    if isinstance(observed_feasible, bool):
        return 1.0 if observed_feasible else 0.0
    status = str(raw_record.get("feasibility_status", "")).lower()
    if status == "feasible":
        return 0.80
    if status == "infeasible":
        return 0.20
    return 0.60


def _first_endpoint(
    endpoint_values: Mapping[str, float],
    keys: Sequence[str],
) -> float | None:
    lower_lookup = {key.lower(): value for key, value in endpoint_values.items()}
    for key in keys:
        if key in endpoint_values:
            return endpoint_values[key]
        if key.lower() in lower_lookup:
            return lower_lookup[key.lower()]
    return None


def _baseline_value(
    observed: Sequence[CMDGDObservedVariant],
    getter: Any,
) -> float:
    base_values = [getter(item) for item in observed if not item.edit_ids]
    if base_values:
        return float(mean(base_values))
    return float(mean(getter(item) for item in observed))


def _config_from_mapping(context: Mapping[str, Any]) -> CMDGDConfig:
    raw = context.get("cmdgd_config", context.get("config"))
    return _normalize_config(raw)


def _normalize_config(
    config: CMDGDConfig | Mapping[str, Any] | None,
) -> CMDGDConfig:
    if isinstance(config, CMDGDConfig):
        return config
    if isinstance(config, Mapping):
        allowed = CMDGDConfig.__dataclass_fields__
        return CMDGDConfig(
            **{key: value for key, value in config.items() if key in allowed}
        )
    return CMDGDConfig()


def _state_from_context(state: CMDGDState | Mapping[str, Any]) -> CMDGDState:
    if isinstance(state, CMDGDState):
        return state
    algorithm_state = state.get("algorithm_state") if isinstance(state, Mapping) else None
    if isinstance(algorithm_state, CMDGDState):
        return algorithm_state
    raise ValueError("CMD-GD state is missing algorithm_state")


def _base_heavy_chain_seq(context: Mapping[str, Any]) -> str:
    for key in ("base_heavy_chain_seq", "base_heavy_seq", "base_vh_sequence"):
        value = context.get(key)
        if value not in (None, ""):
            return str(value)
    observed = list(context.get("observed_records", context.get("observed_variants", [])))
    if observed:
        return _heavy_chain_seq(observed[0])
    raise ValueError("context is missing base_heavy_chain_seq")


def _base_light_chain_seq(context: Mapping[str, Any]) -> str:
    for key in ("base_light_chain_seq", "base_light_seq", "base_vl_sequence"):
        value = context.get(key)
        if value not in (None, ""):
            return str(value)
    observed = list(context.get("observed_records", context.get("observed_variants", [])))
    if observed:
        return _light_chain_seq(observed[0])
    raise ValueError("context is missing base_light_chain_seq")


def _record_id(record: Mapping[str, Any]) -> str:
    for key in ("variant_id", "candidate_id", "id", "name"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    raise ValueError("record is missing variant_id/candidate_id/id/name")


def _candidate_id(record: Mapping[str, Any]) -> str:
    for key in ("candidate_id", "variant_id", "id", "name"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    edits = _candidate_edits(record)
    if edits:
        return _generated_candidate_id(tuple(edit["id"] for edit in edits))
    raise ValueError("candidate is missing candidate_id/variant_id/id/name")


def _heavy_chain_seq(record: Mapping[str, Any]) -> str:
    for key in ("heavy_chain_seq", "heavy_sequence", "vh_sequence", "heavy"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    raise ValueError(f"{_record_id(record)} is missing heavy chain sequence")


def _light_chain_seq(record: Mapping[str, Any]) -> str:
    for key in ("light_chain_seq", "light_sequence", "vl_sequence", "light"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    raise ValueError(f"{_record_id(record)} is missing light chain sequence")


def _source_refs(record: Mapping[str, Any]) -> list[Any]:
    refs = record.get("source_refs", [])
    if isinstance(refs, list):
        return refs
    if isinstance(refs, tuple):
        return list(refs)
    if refs:
        return [refs]
    return [{"kind": "observed_variant", "variant_id": _record_id(record)}]


def _positive_cost(candidate: Mapping[str, Any]) -> float:
    value = _to_float(candidate.get("cost"))
    if value is None or value <= 0.0:
        return 1.0
    return float(value)


def _edit_sort_key(edit_id: str) -> tuple[int, int, str]:
    match = re.match(r"([HL])(\d+)(.*)", str(edit_id))
    if not match:
        return (2, 10**9, str(edit_id))
    chain, position, suffix = match.groups()
    return (0 if chain == "H" else 1, int(position), suffix)


def _generated_candidate_id(edit_ids: Sequence[str]) -> str:
    body = "_".join(str(edit_id) for edit_id in sorted(edit_ids, key=_edit_sort_key))
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", body).strip("_")
    return f"cmdgd_{safe or 'base'}"


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _round(value: float) -> float:
    return round(float(value), 6)


def _round_cost(value: float) -> float:
    return round(float(value), 3)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        if value is None or value == "":
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
