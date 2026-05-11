from __future__ import annotations

from pathlib import Path

from design_scientist.hypotheses import build_hypothesis_board
from design_scientist.io import read_json, write_json


def test_hypothesis_board_never_supports_unsupported_1e62_causality(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "state").mkdir(parents=True)
    write_json(
        project / "state" / "design_state.json",
        {
            "project_id": "anti_hbsag",
            "unsupported_claims": [
                "Current 1E62 combo data cannot support module causal claims.",
            ],
        },
    )
    write_json(
        project / "state" / "evidence_cards.json",
        [
            {
                "evidence_id": "onee62_no_module_causality",
                "claim": "Current 1E62 combo data do not support module-level causal claims.",
                "actionability": "needs_data",
                "limitations": ["Exact matched edges are missing."],
                "tags": ["unsupported_claim"],
            }
        ],
    )

    board = build_hypothesis_board(project)
    persisted = read_json(project / "state" / "hypothesis_board.json")

    assert (project / "state" / "hypothesis_board.md").exists()
    assert persisted
    statuses = {h.status for h in board if "1E62" in h.claim or "module causal" in h.claim}
    assert statuses
    assert "supported" not in statuses
    assert statuses <= {"needs_data", "rejected"}

