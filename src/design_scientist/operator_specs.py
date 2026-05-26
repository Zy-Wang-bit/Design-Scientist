"""Compile literature mechanism cards into algorithm operator specs."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Iterable

from design_scientist.io import ensure_dir, write_json


SUPPORTED_EVIDENCE_STRENGTHS = {"candidate_from_text", "needs_manual_review"}

OPERATOR_GAP_FIELDNAMES = (
    "operator_id",
    "mechanism_id",
    "missing_source_paper_ids",
    "missing_problem_setting",
    "missing_state_model",
    "missing_candidate_generation",
    "missing_acquisition_objective",
    "missing_uncertainty_model",
    "missing_transfer_model",
    "missing_stress_tests",
    "unsupported_fields",
    "evidence_strength",
    "claim_limits",
)

CRITICAL_CARD_FIELDS = (
    "source_paper_ids",
    "problem_setting",
    "state_model",
    "candidate_generation",
    "acquisition_objective",
    "uncertainty_model",
    "transfer_model",
    "stress_tests",
)

ACTIONABLE_OPERATOR_FIELDS = (
    "source_paper_ids",
    "problem_setting",
    "state_model",
    "candidate_generation",
    "acquisition_objective",
)


def compile_operator_spec_artifacts(mechanism_cards: list[dict[str, Any]]) -> dict[str, Any]:
    """Compile mechanism cards into operator spec artifacts.

    The compiler preserves the evidence boundary from Literature Engine cards:
    output specs are algorithm candidates, not validated claims.
    """

    specs = [_operator_spec_from_card(card) for card in mechanism_cards]
    return {
        "operator_specs": specs,
        "operator_gap_matrix": [_gap_row(spec) for spec in specs],
        "operator_evidence_map": _operator_evidence_map(specs),
        "operator_negative_controls": {
            "source": "operator_specs",
            "negative_controls": {
                spec["operator_id"]: spec["negative_controls"] for spec in specs
            },
        },
    }


def write_operator_spec_artifacts(
    framework_dir: str | Path,
    mechanism_cards: list[dict[str, Any]],
) -> dict[str, Path]:
    """Write operator spec artifacts into a framework directory."""

    base = ensure_dir(framework_dir)
    artifacts = compile_operator_spec_artifacts(mechanism_cards)
    paths = {
        "operator_specs": base / "operator_specs.json",
        "operator_gap_matrix": base / "operator_gap_matrix.csv",
        "operator_evidence_map": base / "operator_evidence_map.json",
        "operator_negative_controls": base / "operator_negative_controls.json",
    }
    write_json(paths["operator_specs"], artifacts["operator_specs"])
    _write_gap_matrix(paths["operator_gap_matrix"], artifacts["operator_gap_matrix"])
    write_json(paths["operator_evidence_map"], artifacts["operator_evidence_map"])
    write_json(paths["operator_negative_controls"], artifacts["operator_negative_controls"])
    return paths


def _operator_spec_from_card(card: dict[str, Any]) -> dict[str, Any]:
    mechanism_id = _clean_text(card.get("mechanism_id")) or "mechanism_manual_review_queue"
    operator_id = f"operator_{_slugify(mechanism_id) or 'manual_review_queue'}"
    mechanism_name = _clean_text(card.get("mechanism_name") or card.get("name")) or mechanism_id
    source_paper_ids = _as_list(card.get("source_paper_ids"))
    reusable_components = _as_list(card.get("reusable_components") or card.get("components"))
    constraints = _as_list(card.get("constraints"))
    assumptions = _as_list(card.get("assumptions"))
    failure_modes = _as_list(card.get("failure_modes"))
    stress_tests = _as_list(card.get("stress_tests"))

    missing_fields = _missing_fields(card)
    evidence_strength = _bounded_evidence_strength(
        _evidence_strength(card.get("evidence_strength")),
        missing_fields,
    )
    acquisition_objective = _clean_text(card.get("acquisition_objective"))
    state_model = _clean_text(card.get("state_model"))
    candidate_generation = _clean_text(card.get("candidate_generation"))
    uncertainty_model = _clean_text(card.get("uncertainty_model"))
    transfer_model = _clean_text(card.get("transfer_model"))
    claim_limits = _claim_limits(evidence_strength, missing_fields, failure_modes)
    acquisition_formula = _acquisition_formula(
        acquisition_objective=acquisition_objective,
        uncertainty_model=uncertainty_model,
        transfer_model=transfer_model,
        constraints=constraints,
    )
    design_state_update = _design_state_update(state_model)
    field_support = _field_support(missing_fields)

    return {
        "operator_id": operator_id,
        "mechanism_id": mechanism_id,
        "mechanism_name": mechanism_name,
        "source_paper_ids": source_paper_ids,
        "evidence_strength": evidence_strength,
        "state_model": _structure_field(
            field="state_model",
            description=state_model,
            missing_fields=missing_fields,
        ),
        "objective": {
            "problem_setting": _clean_text(card.get("problem_setting")),
            "description": acquisition_objective,
            "acquisition_formula": acquisition_formula,
            "missing_acquisition_objective": "acquisition_objective" in missing_fields,
            "support": field_support["acquisition_objective"],
        },
        "update_rule": {
            "state_model": state_model,
            "design_state_update": design_state_update,
            "missing_state_model": "state_model" in missing_fields,
            "missing_update_rule": not bool(design_state_update),
            "required_observations": [
                "observed_candidates",
                "assay_measurements",
                "provenance_labels",
            ],
        },
        "candidate_generation": {
            "description": candidate_generation,
            "constraints": constraints,
            "missing": "candidate_generation" in missing_fields,
            "support": field_support["candidate_generation"],
        },
        "acquisition_objective": {
            "description": acquisition_objective,
            "formula": acquisition_formula,
            "missing": "acquisition_objective" in missing_fields,
            "support": field_support["acquisition_objective"],
        },
        "uncertainty": {
            "description": uncertainty_model,
            "model": uncertainty_model,
            "missing": "uncertainty_model" in missing_fields,
            "support": field_support["uncertainty_model"],
        },
        "transfer": {
            "description": transfer_model,
            "model": transfer_model,
            "missing": "transfer_model" in missing_fields,
            "support": field_support["transfer_model"],
        },
        "required_baselines": _required_baselines(
            card,
            uncertainty_model=uncertainty_model,
            transfer_model=transfer_model,
            constraints=constraints,
            reusable_components=reusable_components,
        ),
        "negative_controls": _negative_controls(
            operator_id,
            uncertainty_model=uncertainty_model,
            transfer_model=transfer_model,
            constraints=constraints,
            reusable_components=reusable_components,
        ),
        "ablation_hypotheses": _ablation_hypotheses(
            operator_id,
            reusable_components=reusable_components,
            uncertainty_model=uncertainty_model,
            transfer_model=transfer_model,
        ),
        "implementation_tests": _implementation_tests(
            operator_id,
            required_baselines_hint=True,
            missing_fields=missing_fields,
        ),
        "claim_limits": claim_limits,
        "evidence_map": {
            "supported_fields": [
                field for field in CRITICAL_CARD_FIELDS if field not in missing_fields
            ],
            "unsupported_fields": missing_fields,
            "field_support": field_support,
            "assumptions": assumptions,
            "failure_modes": failure_modes,
        },
    }


def _missing_fields(card: dict[str, Any]) -> list[str]:
    return [field for field in CRITICAL_CARD_FIELDS if _is_missing(card.get(field))]


def _bounded_evidence_strength(evidence_strength: str, missing_fields: list[str]) -> str:
    if evidence_strength != "candidate_from_text":
        return evidence_strength
    if any(field in missing_fields for field in ACTIONABLE_OPERATOR_FIELDS):
        return "needs_manual_review"
    return evidence_strength


def _structure_field(
    *,
    field: str,
    description: str,
    missing_fields: list[str],
) -> dict[str, Any]:
    missing = field in missing_fields
    return {
        "description": description,
        "missing": missing,
        "support": "unsupported" if missing else "supported",
    }


def _field_support(missing_fields: list[str]) -> dict[str, str]:
    missing = set(missing_fields)
    return {
        field: "unsupported" if field in missing else "supported"
        for field in CRITICAL_CARD_FIELDS
    }


def _acquisition_formula(
    *,
    acquisition_objective: str,
    uncertainty_model: str,
    transfer_model: str,
    constraints: list[str],
) -> str:
    if not acquisition_objective:
        return ""
    terms = ["objective_value(candidate; literature_objective)"]
    if uncertainty_model:
        terms.append("uncertainty_value(candidate; design_state)")
    if transfer_model:
        terms.append("transfer_value(candidate; source_target_state)")
    if constraints:
        terms.append("- constraint_penalty(candidate; constraints)")
    return "score(candidate | design_state) = " + " + ".join(terms).replace("+ -", "-")


def _design_state_update(state_model: str) -> str:
    if not state_model:
        return ""
    return (
        "design_state_{t+1} = update("
        "design_state_t, observations_t; evidence_tier='literature_candidate')"
    )


def _required_baselines(
    card: dict[str, Any],
    *,
    uncertainty_model: str,
    transfer_model: str,
    constraints: list[str],
    reusable_components: list[str],
) -> list[str]:
    text = _joined_text(
        [
            card.get("mechanism_id"),
            card.get("mechanism_name") or card.get("name"),
            card.get("problem_setting"),
            card.get("acquisition_objective"),
            *reusable_components,
            *constraints,
        ]
    )
    baselines = ["random_feasible", "fixed_mix_policy"]
    if not _is_missing(card.get("acquisition_objective")):
        baselines.append("top_predicted_utility")
    if uncertainty_model:
        baselines.append("pure_uncertainty_sampling")
    if transfer_model or "transfer" in text:
        baselines.extend(["target_only_baseline", "source_ablation"])
    if any(term in text for term in ("lattice", "contrast", "matched")):
        baselines.append("pure_lattice_repair")
    if any(term in text for term in ("guardrail", "endpoint", "coverage")):
        baselines.append("guardrail_blind_policy")
    return _unique(baselines)


def _negative_controls(
    operator_id: str,
    *,
    uncertainty_model: str,
    transfer_model: str,
    constraints: list[str],
    reusable_components: list[str],
) -> list[dict[str, str]]:
    controls = [
        {
            "control_id": f"{operator_id}_label_permutation",
            "description": "Shuffle outcome labels while preserving candidate metadata.",
            "expected_result": "Operator advantage should collapse under label permutation.",
        },
        {
            "control_id": f"{operator_id}_paper_holdout",
            "description": "Hold out source-paper evidence before compiling the operator.",
            "expected_result": "Unsupported formula or update terms should be removed or flagged.",
        },
    ]
    text = _joined_text([*constraints, *reusable_components])
    if uncertainty_model:
        controls.append(
            {
                "control_id": f"{operator_id}_uncertainty_shuffle",
                "description": "Randomize uncertainty estimates independently of candidates.",
                "expected_result": "Uncertainty-specific advantage should disappear.",
            }
        )
    if transfer_model or "transfer" in text:
        controls.append(
            {
                "control_id": f"{operator_id}_incompatible_source",
                "description": "Use an intentionally incompatible source task for transfer.",
                "expected_result": "Transfer terms should not improve target-task decisions.",
            }
        )
    if any(term in text for term in ("guardrail", "endpoint", "coverage")):
        controls.append(
            {
                "control_id": f"{operator_id}_pooled_endpoint",
                "description": "Pool mechanism and guardrail endpoints before scoring.",
                "expected_result": "Claim-limit checks should flag endpoint pooling.",
            }
        )
    return controls


def _ablation_hypotheses(
    operator_id: str,
    *,
    reusable_components: list[str],
    uncertainty_model: str,
    transfer_model: str,
) -> list[dict[str, str]]:
    hypotheses: list[dict[str, str]] = []
    for component in reusable_components:
        component_id = _slugify(component)
        if not component_id:
            continue
        hypotheses.append(
            {
                "ablation_id": f"{operator_id}_remove_{component_id}",
                "removed_component": component,
                "hypothesis": f"Removing {component} reduces operator value if the mechanism card is actionable.",
            }
        )
    if uncertainty_model:
        hypotheses.append(
            {
                "ablation_id": f"{operator_id}_remove_uncertainty_model",
                "removed_component": "uncertainty_model",
                "hypothesis": "Removing uncertainty terms reduces exploration value.",
            }
        )
    if transfer_model:
        hypotheses.append(
            {
                "ablation_id": f"{operator_id}_remove_transfer_model",
                "removed_component": "transfer_model",
                "hypothesis": "Removing transfer terms reduces sparse-target performance only when transfer is valid.",
            }
        )
    if not hypotheses:
        hypotheses.append(
            {
                "ablation_id": f"{operator_id}_manual_review_required",
                "removed_component": "none",
                "hypothesis": "No component ablation is justified until missing card fields are resolved.",
            }
        )
    return hypotheses


def _implementation_tests(
    operator_id: str,
    *,
    required_baselines_hint: bool,
    missing_fields: list[str],
) -> list[dict[str, str]]:
    tests = [
        {
            "test_id": f"{operator_id}_schema_contract",
            "assertion": "OperatorSpec contains objective, update_rule, baselines, controls, ablations, tests, and claim limits.",
        },
        {
            "test_id": f"{operator_id}_evidence_boundary",
            "assertion": "Compiled evidence strength is candidate_from_text or needs_manual_review, never a strong validation claim.",
        },
    ]
    if required_baselines_hint:
        tests.append(
            {
                "test_id": f"{operator_id}_baseline_comparison",
                "assertion": "Offline replay compares the operator against all required baselines before claims are made.",
            }
        )
    if missing_fields:
        tests.append(
            {
                "test_id": f"{operator_id}_missing_field_gap",
                "assertion": "Missing mechanism-card fields remain visible in operator_gap_matrix.csv.",
            }
        )
    return tests


def _claim_limits(
    evidence_strength: str,
    missing_fields: list[str],
    failure_modes: list[str],
) -> list[str]:
    limits = [
        "This operator is a literature candidate and not prospective wet-lab validation.",
        "Do not claim superiority without offline replay, required baselines, and ablation evidence.",
    ]
    if evidence_strength == "needs_manual_review":
        limits.append("Evidence requires manual review before executable mechanism claims.")
    if missing_fields:
        limits.append("Missing card fields require manual review: " + ", ".join(missing_fields) + ".")
    for failure_mode in failure_modes:
        limits.append("Known failure mode: " + failure_mode)
    return _unique(limits)


def _gap_row(spec: dict[str, Any]) -> dict[str, str]:
    missing_fields = set(spec["evidence_map"]["unsupported_fields"])
    unsupported_fields = spec["evidence_map"]["unsupported_fields"]
    return {
        "operator_id": spec["operator_id"],
        "mechanism_id": spec["mechanism_id"],
        "missing_source_paper_ids": _bool_text("source_paper_ids" in missing_fields),
        "missing_problem_setting": _bool_text("problem_setting" in missing_fields),
        "missing_state_model": _bool_text("state_model" in missing_fields),
        "missing_candidate_generation": _bool_text("candidate_generation" in missing_fields),
        "missing_acquisition_objective": _bool_text("acquisition_objective" in missing_fields),
        "missing_uncertainty_model": _bool_text("uncertainty_model" in missing_fields),
        "missing_transfer_model": _bool_text("transfer_model" in missing_fields),
        "missing_stress_tests": _bool_text("stress_tests" in missing_fields),
        "unsupported_fields": "; ".join(unsupported_fields),
        "evidence_strength": spec["evidence_strength"],
        "claim_limits": "; ".join(spec["claim_limits"]),
    }


def _operator_evidence_map(specs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "source": "mechanism_cards",
        "operators": {
            spec["operator_id"]: {
                "mechanism_id": spec["mechanism_id"],
                "source_paper_ids": spec["source_paper_ids"],
                "evidence_strength": spec["evidence_strength"],
                "supported_fields": spec["evidence_map"]["supported_fields"],
                "unsupported_fields": spec["evidence_map"]["unsupported_fields"],
                "field_support": spec["evidence_map"]["field_support"],
                "claim_limits": spec["claim_limits"],
            }
            for spec in specs
        },
    }


def _write_gap_matrix(path: Path, rows: list[dict[str, str]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OPERATOR_GAP_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _evidence_strength(value: Any) -> str:
    text = _clean_text(value)
    if text in SUPPORTED_EVIDENCE_STRENGTHS:
        return text
    return "needs_manual_review"


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return _unique(_clean_text(item) for item in value)
    return _unique([_clean_text(value)])


def _joined_text(values: Iterable[Any]) -> str:
    return " ".join(_clean_text(value).lower() for value in values if _clean_text(value))


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, list):
        return not any(not _is_missing(item) for item in value)
    return False


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _slugify(value: Any) -> str:
    text = _clean_text(value).lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = _clean_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _bool_text(value: bool) -> str:
    return "true" if value else "false"
