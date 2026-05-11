"""Small file IO helpers for Design Scientist artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from design_scientist.schemas import to_plain_data


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def write_yaml(path: str | Path, data: Any) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(to_plain_data(data), handle, sort_keys=False, allow_unicode=False)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: Any) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(to_plain_data(data), handle, indent=2, sort_keys=False)
        handle.write("\n")


def project_path(project_dir: str | Path, *parts: str) -> Path:
    return Path(project_dir).expanduser().resolve().joinpath(*parts)

