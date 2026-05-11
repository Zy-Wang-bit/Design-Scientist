"""Candidate generation with explicit design operators."""

from __future__ import annotations

from collections import Counter
from typing import Any

from design_scientist.schemas import (
    Candidate,
    CandidateLineage,
    CandidatePool,
    DesignOperatorSpec,
    DesignSpace,
)


CANONICAL_TARGET_SYSTEM = "1E62"
STALE_TARGET_SYSTEM = "1e+62"
PREFERRED_MODULE_ORDER = ["HD110H", "HG56H", "HN54H", "HV105H"]


def _canonicalize_identifier(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace(STALE_TARGET_SYSTEM, CANONICAL_TARGET_SYSTEM)
    return value


def _canonicalize_data(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(_canonicalize_identifier(key)): _canonicalize_data(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_canonicalize_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_canonicalize_data(item) for item in value)
    return _canonicalize_identifier(value)


def _module_sort_key(module_id: str) -> tuple[int, str]:
    if module_id in PREFERRED_MODULE_ORDER:
        return (PREFERRED_MODULE_ORDER.index(module_id), module_id)
    return (len(PREFERRED_MODULE_ORDER), module_id)


def _module_ids(state: dict[str, Any]) -> list[str]:
    modules = {
        str(module.get("module_id"))
        for module in state.get("module_status", [])
        if isinstance(module, dict) and module.get("module_id")
    }
    return sorted(modules, key=_module_sort_key)


def _target_systems(state: dict[str, Any], config: dict[str, Any]) -> list[str]:
    configured = config.get("target_systems")
    if isinstance(configured, list) and configured:
        systems = [str(_canonicalize_identifier(system)) for system in configured]
        return sorted(set(systems))

    systems: list[str] = []
    system_roles = state.get("system_roles", {})
    if isinstance(system_roles, dict):
        for system, role in system_roles.items():
            role_text = str(role)
            if "target" in role_text or "design" in role_text:
                systems.append(str(_canonicalize_identifier(system)))

    for edge in state.get("unresolved_edges", []):
        if isinstance(edge, dict) and edge.get("system"):
            systems.append(str(_canonicalize_identifier(edge["system"])))

    unique = sorted(set(systems))
    return unique or [CANONICAL_TARGET_SYSTEM]


def _evidence_refs(evidence: list[dict[str, Any]] | list[Any]) -> list[str]:
    refs: set[str] = set()
    for item in evidence:
        evidence_id = None
        if isinstance(item, dict):
            evidence_id = item.get("evidence_id")
        else:
            evidence_id = getattr(item, "evidence_id", None)
        if evidence_id:
            refs.add(str(_canonicalize_identifier(evidence_id)))
    return sorted(refs)


def _default_operator_specs() -> list[DesignOperatorSpec]:
    return [
        DesignOperatorSpec(
            operator_id="add_module",
            category="champion",
            description="Add a reusable module to a target-system background.",
            required_endpoints=["pH7.4 binding", "pH6.0 binding", "Ae/B/D1 genotype coverage"],
            default_cost=1.0,
        ),
        DesignOperatorSpec(
            operator_id="remove_module",
            category="champion_contrast",
            description="Remove a module from a champion to test dependence.",
            required_endpoints=["pH7.4 binding", "pH6.0 binding"],
            default_cost=1.0,
        ),
        DesignOperatorSpec(
            operator_id="complete_missing_edge",
            category="lattice_repair",
            description="Create a primary matched edge missing from the contrast lattice.",
            required_endpoints=["matched pH7.4 endpoint", "matched pH6.0 endpoint"],
            default_cost=1.0,
            max_generated=24,
        ),
        DesignOperatorSpec(
            operator_id="complete_square",
            category="interaction_square",
            description="Complete a two-module square to test interaction rather than additivity.",
            required_endpoints=["all square corners at pH7.4 and pH6.0"],
            default_cost=2.0,
            max_generated=1,
        ),
        DesignOperatorSpec(
            operator_id="validate_genotype_coverage",
            category="control",
            description="Validate neutral-pH genotype coverage for an advancing champion.",
            required_endpoints=["Ae pH7.4", "B pH7.4", "D1 pH7.4", "pH6.0 paired checks"],
            default_cost=1.0,
        ),
        DesignOperatorSpec(
            operator_id="repeat_or_control",
            category="repeat",
            description="Add a repeat or control measurement for QC and risk reduction.",
            required_endpoints=["repeat pH7.4", "repeat pH6.0", "QC flag"],
            default_cost=1.0,
        ),
    ]


def build_design_space(
    state: dict[str, Any],
    evidence: list[dict[str, Any]] | list[Any],
    config: dict[str, Any] | None = None,
) -> DesignSpace:
    canonical_state = _canonicalize_data(state)
    canonical_evidence = _canonicalize_data(evidence)
    canonical_config = _canonicalize_data(config or {})

    modules = _module_ids(canonical_state)
    target_systems = _target_systems(canonical_state, canonical_config)
    project_id = str(canonical_state.get("project_id") or canonical_config.get("project_id") or "unknown_project")
    backgrounds = canonical_config.get("backgrounds")
    if not isinstance(backgrounds, list) or not backgrounds:
        target = target_systems[0]
        backgrounds = [f"{target}_background_candidate", f"{target}_champion_candidate"]
    backgrounds = [str(_canonicalize_identifier(background)) for background in backgrounds]
    design_space_id = str(
        canonical_config.get("design_space_id")
        or f"{project_id}:{'-'.join(target_systems)}:{'-'.join(modules) if modules else 'no_modules'}"
    )

    return DesignSpace(
        design_space_id=design_space_id,
        project_id=project_id,
        target_systems=target_systems,
        modules=modules,
        backgrounds=backgrounds,
        operators=_default_operator_specs(),
        constraints=dict(canonical_config.get("constraints", {})),
        evidence_refs=_evidence_refs(canonical_evidence),
        config=canonical_config,
    )


def _lineage(
    operator: str,
    operator_args: dict[str, Any],
    parent_ids: list[str],
    source_refs: list[str],
) -> CandidateLineage:
    return CandidateLineage(
        operator=operator,
        operator_args=dict(operator_args),
        parent_ids=list(parent_ids),
        source_refs=list(source_refs),
    )


def add_module(
    background: str,
    module: str,
    evidence_id: str | None = None,
    *,
    design_space_id: str | None = None,
    target_system: str = CANONICAL_TARGET_SYSTEM,
) -> Candidate:
    background = str(_canonicalize_identifier(background))
    module = str(_canonicalize_identifier(module))
    target_system = str(_canonicalize_identifier(target_system))
    source_refs = [str(_canonicalize_identifier(evidence_id))] if evidence_id else []
    operator_args = {"background": background, "module": module}
    required_endpoints = ["pH7.4 binding", "pH6.0 binding", "Ae/B/D1 genotype coverage"]
    candidate_id = f"add_{module}_to_{background}"
    return Candidate(
        candidate_id=candidate_id,
        variant_id=candidate_id,
        design_space_id=design_space_id,
        operator="add_module",
        category="champion",
        target_system=target_system,
        background=background,
        modules=[module],
        rationale=f"Test whether module {module} improves pH-switch behavior on {background}.",
        evidence_refs=source_refs,
        score_components={"performance": 0.7, "decision_value": 0.45, "contrast_value": 0.1},
        risk_flags=["sdAb_to_1E62_transfer_unvalidated"],
        required_measurements=list(required_endpoints),
        required_endpoints=required_endpoints,
        operator_args=operator_args,
        lineage=_lineage("add_module", operator_args, [background], source_refs),
        parent_ids=[background],
        source_refs=source_refs,
        cost=1.0,
    )


def remove_module(
    champion: str,
    module: str,
    *,
    design_space_id: str | None = None,
    target_system: str = CANONICAL_TARGET_SYSTEM,
) -> Candidate:
    champion = str(_canonicalize_identifier(champion))
    module = str(_canonicalize_identifier(module))
    target_system = str(_canonicalize_identifier(target_system))
    operator_args = {"champion": champion, "module": module}
    required_endpoints = ["pH7.4 binding", "pH6.0 binding"]
    candidate_id = f"remove_{module}_from_{champion}"
    return Candidate(
        candidate_id=candidate_id,
        variant_id=candidate_id,
        design_space_id=design_space_id,
        operator="remove_module",
        category="champion_contrast",
        target_system=target_system,
        background=champion,
        modules=[module],
        rationale=f"Create champion-centered contrast to test dependence on {module}.",
        score_components={"decision_value": 0.65, "contrast_value": 0.7, "performance": 0.25},
        required_measurements=list(required_endpoints),
        required_endpoints=required_endpoints,
        operator_args=operator_args,
        lineage=_lineage("remove_module", operator_args, [champion], []),
        parent_ids=[champion],
        cost=1.0,
    )


def complete_missing_edge(edge: dict[str, Any], *, design_space_id: str | None = None) -> Candidate:
    edge = _canonicalize_data(edge)
    module = str(edge.get("module_id", "unknown_module"))
    base = str(edge.get("base_variant", "unknown_background"))
    target_system = str(edge.get("system", CANONICAL_TARGET_SYSTEM))
    source_refs = [str(edge["evidence_id"])] if edge.get("evidence_id") else []
    operator_args = {"base_variant": base, "module": module, "system": target_system}
    required_endpoints = ["matched pH7.4 endpoint", "matched pH6.0 endpoint"]
    candidate_id = f"complete_edge_{module}_{base}"
    return Candidate(
        candidate_id=candidate_id,
        variant_id=candidate_id,
        design_space_id=design_space_id,
        operator="complete_missing_edge",
        category="lattice_repair",
        target_system=target_system,
        background=base,
        modules=[module],
        rationale=str(edge.get("reason", "Create an exact primary matched edge.")),
        evidence_refs=source_refs,
        score_components={"lattice_repair": 0.85, "contrast_value": 0.65, "decision_value": 0.25},
        required_measurements=list(required_endpoints),
        required_endpoints=required_endpoints,
        operator_args=operator_args,
        lineage=_lineage("complete_missing_edge", operator_args, [base], source_refs),
        parent_ids=[base],
        source_refs=source_refs,
        cost=1.0,
    )


def complete_square(
    background: str,
    module_a: str,
    module_b: str,
    *,
    design_space_id: str | None = None,
    target_system: str = CANONICAL_TARGET_SYSTEM,
) -> Candidate:
    background = str(_canonicalize_identifier(background))
    module_a = str(_canonicalize_identifier(module_a))
    module_b = str(_canonicalize_identifier(module_b))
    target_system = str(_canonicalize_identifier(target_system))
    operator_args = {"background": background, "module_a": module_a, "module_b": module_b}
    required_endpoints = ["all square corners at pH7.4 and pH6.0"]
    candidate_id = f"complete_square_{background}_{module_a}_{module_b}"
    return Candidate(
        candidate_id=candidate_id,
        variant_id=candidate_id,
        design_space_id=design_space_id,
        operator="complete_square",
        category="interaction_square",
        target_system=target_system,
        background=background,
        modules=[module_a, module_b],
        rationale="Test whether two modules interact instead of assuming additivity.",
        score_components={"decision_value": 0.45, "contrast_value": 0.6, "lattice_repair": 0.55},
        risk_flags=["interaction_unknown"],
        required_measurements=list(required_endpoints),
        required_endpoints=required_endpoints,
        operator_args=operator_args,
        lineage=_lineage("complete_square", operator_args, [background], []),
        parent_ids=[background],
        cost=2.0,
    )


def validate_genotype_coverage(
    champion: str,
    *,
    design_space_id: str | None = None,
    target_system: str = CANONICAL_TARGET_SYSTEM,
) -> Candidate:
    champion = str(_canonicalize_identifier(champion))
    target_system = str(_canonicalize_identifier(target_system))
    operator_args = {"champion": champion}
    required_endpoints = ["Ae pH7.4", "B pH7.4", "D1 pH7.4", "pH6.0 paired checks"]
    candidate_id = f"coverage_{champion}"
    return Candidate(
        candidate_id=candidate_id,
        variant_id=candidate_id,
        design_space_id=design_space_id,
        operator="validate_genotype_coverage",
        category="control",
        target_system=target_system,
        background=champion,
        rationale="Validate neutral-pH binding coverage before advancing a pH-switch champion.",
        score_components={"coverage_qc": 0.95, "decision_value": 0.45, "performance": 0.2},
        required_measurements=list(required_endpoints),
        required_endpoints=required_endpoints,
        operator_args=operator_args,
        lineage=_lineage("validate_genotype_coverage", operator_args, [champion], []),
        parent_ids=[champion],
        cost=1.0,
    )


def repeat_or_control(
    edge_or_variant: str,
    *,
    design_space_id: str | None = None,
    target_system: str = CANONICAL_TARGET_SYSTEM,
) -> Candidate:
    edge_or_variant = str(_canonicalize_identifier(edge_or_variant))
    target_system = str(_canonicalize_identifier(target_system))
    operator_args = {"edge_or_variant": edge_or_variant}
    required_endpoints = ["repeat pH7.4", "repeat pH6.0", "QC flag"]
    candidate_id = f"repeat_{edge_or_variant}"
    return Candidate(
        candidate_id=candidate_id,
        variant_id=candidate_id,
        design_space_id=design_space_id,
        operator="repeat_or_control",
        category="repeat",
        target_system=target_system,
        background=edge_or_variant,
        rationale="Add repeat/control measurement for risk reduction and QC.",
        score_components={"coverage_qc": 0.55, "decision_value": 0.25},
        required_measurements=list(required_endpoints),
        required_endpoints=required_endpoints,
        operator_args=operator_args,
        lineage=_lineage("repeat_or_control", operator_args, [edge_or_variant], []),
        parent_ids=[edge_or_variant],
        cost=1.0,
    )


def candidate_pool_diagnostics(
    pool: CandidatePool | list[Candidate],
    *,
    min_pool_size: int = 8,
) -> dict[str, Any]:
    candidates = pool.candidates if isinstance(pool, CandidatePool) else list(pool)
    category_counts = Counter(candidate.category for candidate in candidates)
    feasibility_counts = Counter(candidate.feasibility_status for candidate in candidates)
    reasons: list[str] = []
    if len(candidates) < min_pool_size:
        reasons.append("small_candidate_pool")

    return {
        "candidate_count": len(candidates),
        "category_counts": dict(sorted(category_counts.items())),
        "feasibility_counts": dict(sorted(feasibility_counts.items())),
        "generator_limited_run": bool(reasons),
        "reasons": reasons,
    }


def _backgrounds(design_space: DesignSpace) -> tuple[str, str]:
    target = design_space.target_systems[0] if design_space.target_systems else CANONICAL_TARGET_SYSTEM
    background = f"{target}_background_candidate"
    champion = f"{target}_champion_candidate"
    for candidate_background in design_space.backgrounds:
        if str(candidate_background).endswith("_background_candidate"):
            background = str(candidate_background)
        elif str(candidate_background).endswith("_champion_candidate"):
            champion = str(candidate_background)
    return background, champion


def _sort_candidates(candidates: list[Candidate], strategy: str) -> None:
    if strategy == "pure_lattice_repair":
        candidates.sort(key=lambda c: (-c.score_components.get("lattice_repair", 0.0), c.candidate_id))
    elif strategy == "fixed_mix":
        preferred_categories = {"champion", "champion_contrast"}
        candidates.sort(key=lambda c: (c.category not in preferred_categories, c.candidate_id))
    else:
        candidates.sort(
            key=lambda c: (
                -(
                    c.score_components.get("performance", 0.0)
                    + c.score_components.get("decision_value", 0.0)
                    + c.score_components.get("contrast_value", 0.0)
                    + c.score_components.get("lattice_repair", 0.0)
                ),
                c.candidate_id,
            )
        )


def generate_candidate_pool(
    design_space: DesignSpace,
    state: dict[str, Any],
    evidence: list[dict[str, Any]] | list[Any],
    strategy: str = "mechanism_aware",
) -> CandidatePool:
    canonical_state = _canonicalize_data(state)
    canonical_evidence = _canonicalize_data(evidence)
    target_system = design_space.target_systems[0] if design_space.target_systems else CANONICAL_TARGET_SYSTEM
    background, champion = _backgrounds(design_space)
    modules = design_space.modules or _module_ids(canonical_state)
    preferred_modules = sorted(modules, key=_module_sort_key)[:4]
    evidence_refs = set(_evidence_refs(canonical_evidence))
    add_module_evidence = "role_split_sdab_1E62" if "role_split_sdab_1E62" in evidence_refs else None
    candidates: list[Candidate] = []

    for module in preferred_modules[:4]:
        candidates.append(
            add_module(
                background,
                module,
                add_module_evidence,
                design_space_id=design_space.design_space_id,
                target_system=target_system,
            )
        )
        candidates.append(
            remove_module(
                champion,
                module,
                design_space_id=design_space.design_space_id,
                target_system=target_system,
            )
        )

    unresolved_edges = [
        edge
        for edge in canonical_state.get("unresolved_edges", [])
        if isinstance(edge, dict)
    ]
    unresolved_edges.sort(
        key=lambda edge: (
            str(edge.get("module_id", "")),
            str(edge.get("base_variant", "")),
            str(edge.get("system", "")),
        )
    )
    for edge in unresolved_edges[:24]:
        candidates.append(complete_missing_edge(edge, design_space_id=design_space.design_space_id))

    if len(preferred_modules) >= 2:
        candidates.append(
            complete_square(
                background,
                preferred_modules[0],
                preferred_modules[1],
                design_space_id=design_space.design_space_id,
                target_system=target_system,
            )
        )

    candidates.append(
        validate_genotype_coverage(
            champion,
            design_space_id=design_space.design_space_id,
            target_system=target_system,
        )
    )
    candidates.append(
        repeat_or_control(
            "primary_edge_control",
            design_space_id=design_space.design_space_id,
            target_system=target_system,
        )
    )

    _sort_candidates(candidates, strategy)
    pool = CandidatePool(
        candidate_pool_id=f"{design_space.design_space_id}:{strategy}",
        design_space_id=design_space.design_space_id,
        strategy=strategy,
        candidates=candidates,
    )
    pool.diagnostics = candidate_pool_diagnostics(pool)
    return pool


def generate_candidates(
    state: dict[str, Any],
    evidence: list[dict[str, Any]] | list[Any],
    strategy: str = "mechanism_aware",
) -> list[Candidate]:
    design_space = build_design_space(state, evidence)
    return generate_candidate_pool(design_space, state, evidence, strategy=strategy).candidates
