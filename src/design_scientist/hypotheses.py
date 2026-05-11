"""Hypothesis board construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from design_scientist.io import read_json, write_json
from design_scientist.schemas import to_plain_data


HypothesisStatus = Literal["supported", "contradicted", "unresolved", "needs_data", "rejected"]


@dataclass
class Hypothesis:
    hypothesis_id: str
    claim: str
    status: HypothesisStatus
    evidence_cards: list[str]
    failure_modes: list[str]
    next_analysis: str
    next_experiment: str
    confidence: float


def _status_from_evidence(card: dict[str, Any]) -> HypothesisStatus:
    claim = str(card.get("claim", "")).lower()
    actionability = str(card.get("actionability", "")).lower()
    tags = {str(tag).lower() for tag in card.get("tags", [])}
    if "unsupported" in tags or "do not support" in claim or "cannot" in claim:
        return "needs_data"
    if actionability == "needs_data":
        return "needs_data"
    if actionability in {"advance", "validate", "monitor"}:
        return "supported"
    return "unresolved"


def build_hypothesis_board(project_dir: str | Path) -> list[Hypothesis]:
    root = Path(project_dir).expanduser().resolve()
    state = read_json(root / "state" / "design_state.json")
    evidence_cards = read_json(root / "state" / "evidence_cards.json")
    board: list[Hypothesis] = []

    for idx, card in enumerate(evidence_cards, start=1):
        status = _status_from_evidence(card)
        board.append(
            Hypothesis(
                hypothesis_id=f"evidence_hypothesis_{idx}",
                claim=str(card.get("claim", "")),
                status=status,
                evidence_cards=[str(card.get("evidence_id", f"card_{idx}"))],
                failure_modes=list(card.get("limitations", [])),
                next_analysis="Rebuild evidence with updated standardized tables.",
                next_experiment=(
                    "Design next measurement only after evidence tier and provenance are validated."
                    if status == "supported"
                    else "Collect exact matched or clearly stratified data before advancing this claim."
                ),
                confidence=0.65 if status == "supported" else 0.25,
            )
        )

    for idx, claim in enumerate(state.get("unsupported_claims", []), start=1):
        lowered = claim.lower()
        status: HypothesisStatus = "rejected" if "cannot" in lowered or "without validation" in lowered else "needs_data"
        board.append(
            Hypothesis(
                hypothesis_id=f"unsupported_claim_{idx}",
                claim=claim,
                status=status,
                evidence_cards=[],
                failure_modes=["Unsupported by current evidence state."],
                next_analysis="Identify which matched contrasts or controls would make the claim testable.",
                next_experiment="Add exact matched edge, control, repeat, or target-system validation as appropriate.",
                confidence=0.1,
            )
        )

    write_json(root / "state" / "hypothesis_board.json", board)
    md_lines = ["# Hypothesis Board", ""]
    for hypothesis in board:
        md_lines.extend(
            [
                f"## {hypothesis.hypothesis_id}",
                f"- status: {hypothesis.status}",
                f"- confidence: {hypothesis.confidence}",
                f"- claim: {hypothesis.claim}",
                f"- next_analysis: {hypothesis.next_analysis}",
                f"- next_experiment: {hypothesis.next_experiment}",
                "",
            ]
        )
    (root / "state" / "hypothesis_board.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote hypothesis board under {root / 'state'}")
    return board

