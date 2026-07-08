"""Minimal mechanism graph utilities for V4 literature mining."""

from __future__ import annotations

import re
from typing import Any


GAP_FIELDS = (
    "state_model",
    "candidate_generation",
    "acquisition_objective",
    "uncertainty_model",
    "transfer_model",
    "stress_tests",
)


def build_mechanism_graph(cards: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    components: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, str]] = []
    gaps: list[dict[str, Any]] = []

    for card in sorted(cards, key=lambda item: str(item.get("mechanism_id", ""))):
        mechanism_id = str(card.get("mechanism_id", "")).strip()
        if not mechanism_id:
            continue
        missing = [field for field in GAP_FIELDS if _missing(card.get(field))]
        if missing:
            gaps.append({"mechanism_id": mechanism_id, "missing_fields": missing})

        for component_name in card.get("reusable_components") or ["manual_review_queue"]:
            component_id = _slugify(component_name) or "manual_review_queue"
            component = components.setdefault(
                component_id,
                {
                    "component_id": component_id,
                    "name": str(component_name).strip() or component_id,
                    "mechanism_ids": [],
                    "source_paper_ids": [],
                },
            )
            component["mechanism_ids"].append(mechanism_id)
            component["source_paper_ids"].extend(str(item) for item in card.get("source_paper_ids") or [])
            edges.append(
                {
                    "source": mechanism_id,
                    "target": component_id,
                    "edge_type": "uses_component",
                }
            )

    for component in components.values():
        component["mechanism_ids"] = sorted(set(component["mechanism_ids"]))
        component["source_paper_ids"] = sorted(set(component["source_paper_ids"]))

    return {
        "components": [components[key] for key in sorted(components)],
        "gaps": sorted(gaps, key=lambda item: item["mechanism_id"]),
        "edges": sorted(edges, key=lambda item: (item["source"], item["target"])),
    }


def _missing(value: Any) -> bool:
    return value in (None, "", []) or (isinstance(value, str) and not value.strip())


def _slugify(value: Any) -> str:
    text = str(value).strip().lower()
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", text)).strip("_")
