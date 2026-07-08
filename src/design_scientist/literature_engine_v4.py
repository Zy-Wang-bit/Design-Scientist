"""Deterministic V4 literature mechanism mining."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from design_scientist.io import ensure_dir, write_json
from design_scientist.mechanism_extraction import (
    _dedupe_cards,
    _heuristic_cards,
    _load_literature_corpus,
)
from design_scientist.mechanism_graph import GAP_FIELDS, build_mechanism_graph


def mine_mechanisms_from_corpus(project_dir: str | Path) -> dict[str, Any]:
    root = Path(project_dir).expanduser().resolve()
    framework = ensure_dir(root / "framework")
    artifacts = {
        "mechanism_cards": framework / "mechanism_cards_v4.json",
        "mechanism_library": framework / "mechanism_library_v4.json",
        "mechanism_gap_matrix": framework / "mechanism_gap_matrix_v4.csv",
        "literature_mine_trace": framework / "literature_mine_trace_v4.json",
        "mechanism_graph": framework / "mechanism_graph_v4.json",
    }

    records, warnings, reason = _load_literature_corpus(framework / "literature_corpus.jsonl")
    if reason is not None:
        trace = _trace("failure", records, [], artifacts, warnings, reason=reason)
        write_json(artifacts["literature_mine_trace"], trace)
        return trace

    cards = _v4_cards(records)
    graph = build_mechanism_graph(cards)
    write_json(artifacts["mechanism_cards"], cards)
    write_json(artifacts["mechanism_library"], _library(cards, graph))
    write_json(artifacts["mechanism_graph"], graph)
    _write_gap_matrix(artifacts["mechanism_gap_matrix"], cards)
    trace = _trace("ok", records, cards, artifacts, warnings)
    write_json(artifacts["literature_mine_trace"], trace)
    return trace


def _library(cards: list[dict[str, Any]], graph: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "literature_engine_v4",
        "mechanism_count": len(cards),
        "mechanism_ids": [card["mechanism_id"] for card in cards],
        "component_groups": {
            component["component_id"]: {
                "mechanism_ids": component["mechanism_ids"],
                "source_paper_ids": component["source_paper_ids"],
            }
            for component in graph["components"]
        },
        "graph": graph,
    }


def _v4_cards(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cards = _dedupe_cards(_heuristic_cards(records))
    for card in cards:
        if card["mechanism_id"] == "transfer_prior_design" and _missing(card.get("transfer_model")):
            card["transfer_model"] = "Source-to-target prior, representation, or multi-task model."
    return cards


def _trace(
    status: str,
    records: list[dict[str, Any]],
    cards: list[dict[str, Any]],
    artifacts: dict[str, Path],
    warnings: list[str],
    *,
    reason: str | None = None,
) -> dict[str, Any]:
    trace = {
        "source": "literature_engine_v4",
        "status": status,
        "paper_count": len(records),
        "mechanism_card_count": len(cards),
        "mechanism_ids": [card["mechanism_id"] for card in cards],
        "warnings": warnings,
        "artifacts": {key: str(path) for key, path in artifacts.items()},
    }
    if reason:
        trace["reason"] = reason
    return trace


def _write_gap_matrix(path: Path, cards: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "mechanism_id",
        "missing_state_model",
        "missing_candidate_generation",
        "missing_acquisition_objective",
        "missing_uncertainty_model",
        "missing_transfer_model",
        "missing_stress_tests",
        "reusable_components",
        "failure_modes",
        "source_paper_ids",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for card in cards:
            row = {"mechanism_id": card["mechanism_id"]}
            row.update({f"missing_{field}": str(_missing(card.get(field))).lower() for field in GAP_FIELDS})
            row["reusable_components"] = "; ".join(card.get("reusable_components") or [])
            row["failure_modes"] = "; ".join(card.get("failure_modes") or [])
            row["source_paper_ids"] = "; ".join(card.get("source_paper_ids") or [])
            writer.writerow(row)


def _missing(value: Any) -> bool:
    return value in (None, "", []) or (isinstance(value, str) and not value.strip())
