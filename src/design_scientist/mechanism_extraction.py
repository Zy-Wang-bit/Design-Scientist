"""Extract reusable mechanism cards from the literature corpus.

This module is standalone by design: it consumes
``framework/literature_corpus.jsonl`` and writes mechanism artifacts without
depending on the older method-extraction pipeline. A model backend can supply
structured cards, but malformed model output falls back to deterministic,
conservative text heuristics.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

from design_scientist.io import ensure_dir, write_json
from design_scientist.schemas import ModelRequest


REQUIRED_CARD_FIELDS = (
    "mechanism_id",
    "source_paper_ids",
    "mechanism_name",
    "problem_setting",
    "state_model",
    "candidate_generation",
    "acquisition_objective",
    "uncertainty_model",
    "transfer_model",
    "constraints",
    "assumptions",
    "failure_modes",
    "reusable_components",
    "stress_tests",
    "evidence_strength",
)

LIST_FIELDS = {
    "source_paper_ids",
    "constraints",
    "assumptions",
    "failure_modes",
    "reusable_components",
    "stress_tests",
}
TEXT_FIELDS = set(REQUIRED_CARD_FIELDS) - LIST_FIELDS
EVIDENCE_STRENGTHS = {"candidate_from_text", "needs_manual_review"}
GAP_FIELDNAMES = (
    "mechanism_id",
    "missing_state_model",
    "missing_candidate_generation",
    "missing_uncertainty_model",
    "missing_transfer_model",
    "missing_stress_tests",
    "reusable_components",
    "failure_modes",
)

HEURISTIC_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "mechanism_id": "active_learning_acquisition",
        "mechanism_name": "Active Learning Acquisition",
        "keywords": (
            "active learning",
            "batch active",
            "bayesian optimization",
            "acquisition",
            "expected improvement",
            "uncertainty sampling",
            "surrogate",
            "posterior",
            "protein engineering",
            "variant design",
        ),
        "min_score": 2,
        "problem_setting": (
            "Select experimental candidates in a batch-limited design loop using "
            "observed sequence-function measurements."
        ),
        "state_model": "Surrogate or design-state model updated from observed variants.",
        "candidate_generation": "Generate feasible candidate variants or batches before scoring.",
        "acquisition_objective": (
            "Rank candidates by expected improvement, utility, or information gain "
            "under feasibility constraints."
        ),
        "uncertainty_model": "Posterior or surrogate uncertainty used to guide exploration.",
        "transfer_model": "",
        "constraints": ("batch_budget", "feasibility_filters"),
        "assumptions": (
            "Assay observations are comparable within the endpoint strata described by the source text.",
        ),
        "failure_modes": (
            "Uncertainty may be miscalibrated outside observed regions.",
            "Batch diversity can displace high-utility candidates.",
        ),
        "reusable_components": ("surrogate_model", "acquisition_policy", "batch_design"),
        "stress_tests": ("retrospective_round_masking", "uncertainty_calibration_check"),
        "field_gates": {
            "uncertainty_model": ("uncertainty", "posterior", "gaussian process", "variance"),
            "transfer_model": ("transfer", "multi-task", "multitask", "domain adaptation", "pretrain"),
            "stress_tests": ("retrospective", "masking", "benchmark", "replay", "calibration"),
        },
    },
    {
        "mechanism_id": "matched_contrast_lattice",
        "mechanism_name": "Matched Contrast Lattice",
        "keywords": (
            "matched contrast",
            "matched contrasts",
            "contrast lattice",
            "lattice repair",
            "confounding",
            "interaction square",
            "evidence tier",
            "causal",
        ),
        "min_score": 2,
        "problem_setting": (
            "Separate supported mechanism claims from confounded descriptive rankings "
            "in small variant panels."
        ),
        "state_model": "Matched-edge or contrast-lattice state over modules and backgrounds.",
        "candidate_generation": "Generate missing matched edges or interaction squares around important variants.",
        "acquisition_objective": "Allocate budget to repair unsupported contrasts while preserving design value.",
        "uncertainty_model": "",
        "transfer_model": "",
        "constraints": ("matched_comparison_required", "avoid_confounded_claims"),
        "assumptions": (
            "Mechanism claims require matched or explicitly labeled near-matched evidence.",
        ),
        "failure_modes": (
            "Descriptive comparisons can be mistaken for primary mechanism evidence.",
            "Pure lattice repair can over-spend budget on interpretability.",
        ),
        "reusable_components": ("matched_contrast", "lattice_repair", "evidence_tiers"),
        "stress_tests": ("false_positive_module_claim_check",),
        "field_gates": {
            "uncertainty_model": ("uncertainty", "posterior", "calibration"),
            "transfer_model": ("transfer", "background transfer", "cross-system"),
            "stress_tests": ("false positive", "stress test", "retrospective", "benchmark"),
        },
    },
    {
        "mechanism_id": "adaptive_decision_value",
        "mechanism_name": "Adaptive Decision-Value Allocation",
        "keywords": (
            "decision value",
            "decision-value",
            "adaptive allocation",
            "optimal design",
            "state-dependent",
            "champion",
            "control allocation",
            "wet lab design",
        ),
        "min_score": 2,
        "problem_setting": (
            "Allocate limited experimental budget to actions that improve the next "
            "design decision."
        ),
        "state_model": "Round state tracking champions, controls, open decisions, and evidence gaps.",
        "candidate_generation": "Generate champion expansions, contrasts, controls, repeats, and exploration slots.",
        "acquisition_objective": "Maximize marginal decision value under batch and guardrail constraints.",
        "uncertainty_model": "",
        "transfer_model": "",
        "constraints": ("batch_budget", "required_controls", "decision_relevance"),
        "assumptions": ("Decision-value categories are fixed before evaluating a prospective panel.",),
        "failure_modes": (
            "Adaptive allocation can overreact to noisy early observations.",
            "Control slots can be squeezed out by short-term utility pressure.",
        ),
        "reusable_components": ("decision_value", "adaptive_allocation", "control_allocation"),
        "stress_tests": ("fixed_mix_ablation",),
        "field_gates": {
            "uncertainty_model": ("uncertainty", "posterior", "confidence"),
            "transfer_model": ("transfer", "generalize", "cross-system"),
            "stress_tests": ("ablation", "fixed mix", "retrospective", "benchmark"),
        },
    },
    {
        "mechanism_id": "mechanism_guardrail_design",
        "mechanism_name": "Mechanism-Aware Guardrail Design",
        "keywords": (
            "mechanism-aware",
            "mechanism aware",
            "guardrail",
            "neutral binding",
            "acidic release",
            "ph-dependent",
            "ph dependent",
            "developability",
            "coverage",
            "antibody",
        ),
        "min_score": 2,
        "problem_setting": (
            "Improve a target mechanism while preserving required assay, coverage, "
            "or developability guardrails."
        ),
        "state_model": "Endpoint-stratified design state with mechanism and guardrail observations kept separate.",
        "candidate_generation": "Generate mechanism candidates plus guardrail-preserving controls or contrasts.",
        "acquisition_objective": "Prioritize expected feasible performance subject to required guardrail checks.",
        "uncertainty_model": "",
        "transfer_model": "",
        "constraints": ("guardrail_endpoints", "coverage_or_qc_requirements"),
        "assumptions": ("Mechanism and guardrail endpoints are not silently pooled.",),
        "failure_modes": (
            "Mechanism gains can hide losses on required guardrail endpoints.",
            "Rationale may not transfer across backgrounds without validation.",
        ),
        "reusable_components": ("guardrail_constraints", "endpoint_stratification", "mechanism_contrasts"),
        "stress_tests": ("guardrail_retention_check",),
        "field_gates": {
            "uncertainty_model": ("uncertainty", "confidence", "calibration"),
            "transfer_model": ("transfer", "background", "cross-system", "generalize"),
            "stress_tests": ("validation", "stress test", "retrospective", "retention"),
        },
    },
    {
        "mechanism_id": "transfer_prior_design",
        "mechanism_name": "Transfer Prior Design",
        "keywords": (
            "transfer learning",
            "transfer",
            "multi-task",
            "multitask",
            "domain adaptation",
            "pretrained",
            "prior",
            "few-shot",
            "cross-domain",
            "cross domain",
        ),
        "min_score": 2,
        "problem_setting": "Use related systems or historical tasks to improve a sparse target design problem.",
        "state_model": "Target-task state augmented with source-task priors or representations.",
        "candidate_generation": "Generate target candidates scored by target evidence and transferable priors.",
        "acquisition_objective": "Balance target-task utility with transfer uncertainty and negative-transfer risk.",
        "uncertainty_model": "Separate target uncertainty from transfer confidence where the source text supports it.",
        "transfer_model": "Source-to-target prior, representation, or multi-task model.",
        "constraints": ("source_target_compatibility",),
        "assumptions": ("Source and target systems share transferable structure that must be validated.",),
        "failure_modes": ("Negative transfer can bias candidate ranking in the target system.",),
        "reusable_components": ("transfer_prior", "source_target_validation"),
        "stress_tests": ("source_ablation", "target_only_baseline"),
        "field_gates": {
            "uncertainty_model": ("uncertainty", "confidence", "posterior", "calibration"),
            "stress_tests": ("ablation", "baseline", "benchmark", "validation"),
        },
    },
)


def extract_mechanisms(
    project_dir: str | Path,
    backend: Any | None = None,
    max_chunks: int = 20,
) -> dict[str, Any]:
    """Extract mechanism cards from ``framework/literature_corpus.jsonl``.

    A supplied backend is asked for structured mechanism cards per corpus chunk.
    If no backend is supplied, or the backend output is malformed, extraction
    uses deterministic text heuristics and labels evidence conservatively.
    """

    if max_chunks <= 0:
        raise ValueError("max_chunks must be positive")

    root = Path(project_dir).expanduser().resolve()
    framework_dir = ensure_dir(root / "framework")
    corpus_path = framework_dir / "literature_corpus.jsonl"
    records, load_warnings, failure_reason = _load_literature_corpus(corpus_path)
    if failure_reason is not None:
        return {
            "status": "failure",
            "reason": failure_reason,
            "paper_count": 0,
            "mechanism_card_count": 0,
            "mechanism_cards": [],
            "warnings": load_warnings,
            "artifacts": {},
        }

    warnings = list(load_warnings)
    fallback_used = backend is None
    extraction_source = "heuristic"
    cards: list[dict[str, Any]]

    if backend is not None:
        backend_cards, backend_warnings = _extract_with_backend(backend, records, max_chunks=max_chunks)
        warnings.extend(backend_warnings)
        if backend_cards is not None:
            cards = backend_cards
            fallback_used = False
            extraction_source = "backend"
        else:
            cards = _heuristic_cards(records)
            fallback_used = True
    else:
        cards = _heuristic_cards(records)

    if not cards:
        cards = [_manual_review_card(records)]
        fallback_used = True

    cards = _dedupe_cards(cards)
    artifact_paths = {
        "mechanism_cards": framework_dir / "mechanism_cards.json",
        "mechanism_library": framework_dir / "mechanism_library.json",
        "mechanism_gap_matrix": framework_dir / "mechanism_gap_matrix.csv",
    }
    write_json(artifact_paths["mechanism_cards"], cards)
    library = _build_mechanism_library(cards)
    write_json(artifact_paths["mechanism_library"], library)
    _write_gap_matrix(artifact_paths["mechanism_gap_matrix"], cards)

    status = "ok_with_warnings" if warnings else "ok"
    return {
        "status": status,
        "paper_count": len(records),
        "mechanism_card_count": len(cards),
        "mechanism_cards": cards,
        "fallback_used": fallback_used,
        "extraction_source": extraction_source if not fallback_used else "heuristic",
        "warnings": warnings,
        "artifacts": {name: str(path) for name, path in artifact_paths.items()},
    }


def _load_literature_corpus(path: Path) -> tuple[list[dict[str, Any]], list[str], str | None]:
    if not path.exists():
        return [], [], "missing_literature_corpus"

    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [], [f"corpus_read_error: {exc}"], "invalid_literature_corpus"

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            warnings.append(f"corpus_jsonl_line_{line_number}_invalid: {exc.msg}")
            continue
        if not isinstance(raw, dict):
            warnings.append(f"corpus_jsonl_line_{line_number}_not_object")
            continue
        record = _normalize_corpus_record(raw, index=len(records) + 1)
        if record["text"]:
            records.append(record)

    if not records:
        return [], warnings, "empty_literature_corpus"
    return records, warnings, None


def _normalize_corpus_record(raw: dict[str, Any], *, index: int) -> dict[str, Any]:
    paper_id = _clean_text(
        _first_present(raw, ("paper_id", "source_paper_id", "id", "doi", "pmid", "arxiv_id"))
    )
    if not paper_id:
        paper_id = f"paper_{index}"
    title = _clean_text(raw.get("title"))
    text_parts = [title]
    for field in ("abstract", "summary", "full_text", "text", "body", "content", "methods"):
        text_parts.append(_text_from_value(raw.get(field)))
    text_parts.append(_text_from_value(raw.get("chunks")))
    text = "\n".join(part for part in text_parts if part)
    return {
        "paper_id": paper_id,
        "title": title,
        "text": text,
        "metadata": {key: value for key, value in raw.items() if key not in {"full_text", "text", "body", "content"}},
    }


def _extract_with_backend(
    backend: Any,
    records: list[dict[str, Any]],
    *,
    max_chunks: int,
) -> tuple[list[dict[str, Any]] | None, list[str]]:
    schema = _response_schema()
    cards: list[dict[str, Any]] = []
    warnings: list[str] = []
    chunks = _chunk_records(records, max_chunks=max_chunks)
    for chunk_index, chunk in enumerate(chunks, start=1):
        request = ModelRequest(
            purpose="mechanism_extraction",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract reusable active-design mechanism cards from literature text. "
                        "Return only cards supported as candidates by the supplied chunk; "
                        "do not label evidence as strong."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "chunk_index": chunk_index,
                            "chunk_count": len(chunks),
                            "records": [
                                {
                                    "paper_id": record["paper_id"],
                                    "title": record["title"],
                                    "text": _truncate_text(record["text"], limit=9000),
                                }
                                for record in chunk
                            ],
                        },
                        sort_keys=True,
                    ),
                },
            ],
            response_schema=schema,
            temperature=0.0,
            max_tokens=4000,
            context_refs=["framework/literature_corpus.jsonl"],
        )
        try:
            response = backend.complete(request)
        except Exception as exc:  # pragma: no cover - defensive around external backends.
            return None, [f"backend_exception: {type(exc).__name__}: {exc}"]

        payload = _structured_payload(response)
        chunk_cards = _cards_from_backend_payload(payload)
        if chunk_cards is None:
            return None, [f"backend_malformed_output: chunk_{chunk_index}"]
        cards.extend(chunk_cards)

    if not cards:
        warnings.append("backend_malformed_output: empty_mechanism_cards")
        return None, warnings
    return cards, warnings


def _response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["mechanism_cards"],
        "properties": {
            "mechanism_cards": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": sorted(REQUIRED_CARD_FIELDS),
                    "properties": {
                        "mechanism_id": {"type": "string"},
                        "source_paper_ids": {"type": "array", "items": {"type": "string"}},
                        "mechanism_name": {"type": "string"},
                        "problem_setting": {"type": "string"},
                        "state_model": {"type": "string"},
                        "candidate_generation": {"type": "string"},
                        "acquisition_objective": {"type": "string"},
                        "uncertainty_model": {"type": "string"},
                        "transfer_model": {"type": "string"},
                        "constraints": {"type": "array", "items": {"type": "string"}},
                        "assumptions": {"type": "array", "items": {"type": "string"}},
                        "failure_modes": {"type": "array", "items": {"type": "string"}},
                        "reusable_components": {"type": "array", "items": {"type": "string"}},
                        "stress_tests": {"type": "array", "items": {"type": "string"}},
                        "evidence_strength": {
                            "type": "string",
                            "enum": sorted(EVIDENCE_STRENGTHS),
                        },
                    },
                },
            }
        },
    }


def _structured_payload(response: Any) -> Any:
    structured = getattr(response, "structured", None)
    if isinstance(structured, dict):
        return structured
    text = getattr(response, "text", "")
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _cards_from_backend_payload(payload: Any) -> list[dict[str, Any]] | None:
    if not isinstance(payload, dict):
        return None
    raw_cards = payload.get("mechanism_cards")
    if not isinstance(raw_cards, list) or not raw_cards:
        return None

    cards: list[dict[str, Any]] = []
    for raw_card in raw_cards:
        if not isinstance(raw_card, dict):
            return None
        if any(field not in raw_card for field in REQUIRED_CARD_FIELDS):
            return None
        normalized = _normalize_mechanism_card(raw_card)
        if normalized is None:
            return None
        cards.append(normalized)
    return cards


def _normalize_mechanism_card(raw_card: dict[str, Any]) -> dict[str, Any] | None:
    mechanism_id = _clean_text(raw_card.get("mechanism_id"))
    if not mechanism_id:
        return None

    card: dict[str, Any] = {"mechanism_id": mechanism_id}
    for field in REQUIRED_CARD_FIELDS:
        if field == "mechanism_id":
            continue
        value = raw_card.get(field)
        if field in LIST_FIELDS:
            if not isinstance(value, list):
                return None
            card[field] = _unique(_clean_text(item) for item in value)
        else:
            card[field] = _clean_text(value)

    if card["evidence_strength"] not in EVIDENCE_STRENGTHS:
        card["evidence_strength"] = "needs_manual_review"
    if not card["mechanism_name"]:
        card["mechanism_name"] = mechanism_id.replace("_", " ").title()
    return card


def _heuristic_cards(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    corpus_text = "\n".join(record["text"] for record in records)
    for template in HEURISTIC_TEMPLATES:
        score = _keyword_score(corpus_text, template["keywords"])
        if score < int(template["min_score"]):
            continue
        source_records = [
            record for record in records if _keyword_score(record["text"], template["keywords"]) > 0
        ]
        cards.append(_card_from_template(template, source_records or records, corpus_text))

    if not cards:
        return [_manual_review_card(records)]
    return cards


def _card_from_template(
    template: dict[str, Any],
    source_records: list[dict[str, Any]],
    corpus_text: str,
) -> dict[str, Any]:
    gates = template.get("field_gates", {})
    uncertainty_model = (
        template["uncertainty_model"]
        if _keyword_score(corpus_text, gates.get("uncertainty_model", ())) > 0
        else ""
    )
    transfer_model = (
        template["transfer_model"]
        if _keyword_score(corpus_text, gates.get("transfer_model", ())) > 0
        else ""
    )
    stress_tests = (
        list(template["stress_tests"])
        if _keyword_score(corpus_text, gates.get("stress_tests", ())) > 0
        else []
    )
    return {
        "mechanism_id": template["mechanism_id"],
        "source_paper_ids": _unique(record["paper_id"] for record in source_records),
        "mechanism_name": template["mechanism_name"],
        "problem_setting": template["problem_setting"],
        "state_model": template["state_model"],
        "candidate_generation": template["candidate_generation"],
        "acquisition_objective": template["acquisition_objective"],
        "uncertainty_model": uncertainty_model,
        "transfer_model": transfer_model,
        "constraints": list(template["constraints"]),
        "assumptions": list(template["assumptions"]),
        "failure_modes": list(template["failure_modes"]),
        "reusable_components": list(template["reusable_components"]),
        "stress_tests": stress_tests,
        "evidence_strength": "candidate_from_text",
    }


def _manual_review_card(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mechanism_id": "mechanism_manual_review_queue",
        "source_paper_ids": _unique(record["paper_id"] for record in records),
        "mechanism_name": "Mechanism Manual Review Queue",
        "problem_setting": "The corpus contains text, but no supported mechanism family was extracted heuristically.",
        "state_model": "",
        "candidate_generation": "",
        "acquisition_objective": "",
        "uncertainty_model": "",
        "transfer_model": "",
        "constraints": [],
        "assumptions": ["Manual review is required before treating this text as an executable mechanism."],
        "failure_modes": ["The source text may not describe an active-design mechanism."],
        "reusable_components": ["manual_review_queue"],
        "stress_tests": [],
        "evidence_strength": "needs_manual_review",
    }


def _dedupe_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for card in cards:
        normalized = _normalize_mechanism_card(card)
        if normalized is None:
            continue
        mechanism_id = normalized["mechanism_id"]
        if mechanism_id not in deduped:
            deduped[mechanism_id] = normalized
            continue
        existing = deduped[mechanism_id]
        for field in LIST_FIELDS:
            existing[field] = _unique([*existing[field], *normalized[field]])
    return list(deduped.values())


def _build_mechanism_library(cards: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for card in cards:
        components = card.get("reusable_components") or ["manual_review_queue"]
        for component in components:
            clean_component = _slugify(component) or "manual_review_queue"
            group = groups.setdefault(
                clean_component,
                {
                    "mechanism_ids": [],
                    "source_paper_ids": [],
                    "mechanism_cards": [],
                },
            )
            group["mechanism_ids"].append(card["mechanism_id"])
            group["source_paper_ids"].extend(card["source_paper_ids"])
            group["mechanism_cards"].append(card)

    for group in groups.values():
        group["mechanism_ids"] = _unique(group["mechanism_ids"])
        group["source_paper_ids"] = _unique(group["source_paper_ids"])

    return {
        "source": "literature_engine_v3",
        "mechanism_count": len(cards),
        "component_groups": {key: groups[key] for key in sorted(groups)},
    }


def _write_gap_matrix(path: Path, cards: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GAP_FIELDNAMES)
        writer.writeheader()
        for card in cards:
            writer.writerow(
                {
                    "mechanism_id": card["mechanism_id"],
                    "missing_state_model": _bool_text(_is_missing(card.get("state_model"))),
                    "missing_candidate_generation": _bool_text(_is_missing(card.get("candidate_generation"))),
                    "missing_uncertainty_model": _bool_text(_is_missing(card.get("uncertainty_model"))),
                    "missing_transfer_model": _bool_text(_is_missing(card.get("transfer_model"))),
                    "missing_stress_tests": _bool_text(_is_missing(card.get("stress_tests"))),
                    "reusable_components": "; ".join(card.get("reusable_components") or []),
                    "failure_modes": "; ".join(card.get("failure_modes") or []),
                }
            )


def _chunk_records(records: list[dict[str, Any]], *, max_chunks: int) -> list[list[dict[str, Any]]]:
    chunk_count = min(max_chunks, len(records))
    chunk_size = max(1, math.ceil(len(records) / chunk_count))
    return [records[index : index + chunk_size] for index in range(0, len(records), chunk_size)]


def _keyword_score(text: str, keywords: Iterable[str]) -> int:
    haystack = text.lower()
    return sum(1 for keyword in keywords if keyword.lower() in haystack)


def _first_present(data: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            return value
    return ""


def _text_from_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _clean_text(value)
    if isinstance(value, list):
        parts = []
        for item in value:
            part = _text_from_value(item)
            if part:
                parts.append(part)
        return "\n".join(parts)
    if isinstance(value, dict):
        parts = []
        for key in ("title", "heading", "abstract", "summary", "text", "content", "body"):
            part = _text_from_value(value.get(key))
            if part:
                parts.append(part)
        return "\n".join(parts)
    return _clean_text(value)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _slugify(value: Any) -> str:
    text = _clean_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or ""


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


def _truncate_text(text: str, *, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, list):
        return not any(not _is_missing(item) for item in value)
    return False


def _bool_text(value: bool) -> str:
    return "true" if value else "false"
