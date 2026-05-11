"""Project history utilities."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from design_scientist.io import read_json, write_json


def _history_path(project_dir: str | Path) -> Path:
    return Path(project_dir).expanduser().resolve() / "state" / "project_history.json"


def load_project_history(project_dir: str | Path) -> list[dict[str, Any]]:
    path = _history_path(project_dir)
    if not path.exists():
        return []
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected project history list in {path}")
    return data


def append_history_event(project_dir: str | Path, event: dict[str, Any]) -> dict[str, Any]:
    history = load_project_history(project_dir)
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **event,
    }
    history.append(record)
    write_json(_history_path(project_dir), history)
    return record

