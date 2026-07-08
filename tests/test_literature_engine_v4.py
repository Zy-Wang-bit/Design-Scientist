from __future__ import annotations

import csv
import json
from pathlib import Path

from design_scientist.io import read_json
from design_scientist.literature_engine_v4 import mine_mechanisms_from_corpus


def _write_corpus(project: Path, records: list[dict[str, str]]) -> None:
    framework = project / "framework"
    framework.mkdir(parents=True, exist_ok=True)
    with (framework / "literature_corpus.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def test_mine_mechanisms_from_corpus_writes_v4_artifacts(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write_corpus(
        project,
        [
            {
                "paper_id": "active_source",
                "title": "Batch active learning acquisition for protein engineering",
                "abstract": (
                    "Active learning uses a surrogate model, posterior uncertainty, "
                    "acquisition function, feasible candidate batches, constraints, "
                    "and retrospective masking benchmark stress tests."
                ),
            },
            {
                "paper_id": "transfer_source",
                "title": "Transfer learning prior for few-shot variant design",
                "abstract": (
                    "Transfer learning uses a source-to-target prior, multi-task model, "
                    "target candidates, transfer uncertainty, and source ablation baseline."
                ),
            },
        ],
    )

    result = mine_mechanisms_from_corpus(project)

    assert result["status"] == "ok"
    assert result["paper_count"] == 2
    assert result["mechanism_card_count"] >= 2

    cards = read_json(project / "framework" / "mechanism_cards_v4.json")
    assert [card["mechanism_id"] for card in cards] == result["mechanism_ids"]
    assert {"active_learning_acquisition", "transfer_prior_design"} <= set(result["mechanism_ids"])

    library = read_json(project / "framework" / "mechanism_library_v4.json")
    assert library["source"] == "literature_engine_v4"
    assert library["mechanism_count"] == len(cards)
    assert "graph" in library
    assert "surrogate_model" in {component["component_id"] for component in library["graph"]["components"]}

    with (project / "framework" / "mechanism_gap_matrix_v4.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [row["mechanism_id"] for row in rows] == result["mechanism_ids"]
    assert any(row["missing_transfer_model"] == "false" for row in rows)

    trace = read_json(project / "framework" / "literature_mine_trace_v4.json")
    assert trace["source"] == "literature_engine_v4"
    assert trace["mechanism_ids"] == result["mechanism_ids"]
    assert trace["artifacts"]["mechanism_cards"].endswith("mechanism_cards_v4.json")
