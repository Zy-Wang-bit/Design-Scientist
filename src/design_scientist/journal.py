"""Journal persistence for controlled Design Scientist runs."""

from __future__ import annotations

from pathlib import Path

from design_scientist.io import read_json, write_json
from design_scientist.schemas import DesignJournal, JournalNode


def save_journal(path: str | Path, journal: DesignJournal) -> None:
    write_json(path, journal)


def load_journal(path: str | Path) -> DesignJournal:
    data = read_json(path)
    return DesignJournal(
        run_id=data["run_id"],
        project_id=data["project_id"],
        selected_node_id=data.get("selected_node_id"),
        nodes=[JournalNode(**node) for node in data.get("nodes", [])],
    )

