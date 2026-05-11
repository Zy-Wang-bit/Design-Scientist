"""New-data intake without bypassing schema audit."""

from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path
from typing import Any

from design_scientist.io import ensure_dir, write_json
from design_scientist.project_history import append_history_event


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_for_file(source: Path, dest: Path) -> dict[str, Any]:
    stat = dest.stat()
    return {
        "source_path": str(source),
        "stored_path": str(dest),
        "name": dest.name,
        "size_bytes": stat.st_size,
        "sha256": _file_sha256(dest),
        "status": "pending_schema_audit",
    }


def merge_new_data(
    project_dir: str | Path,
    new_data_dir: str | Path,
    round_label: str | None = None,
) -> Path:
    """Copy new files into incoming/<round> and mark them pending schema audit.

    This function intentionally does not update standardized tables, evidence cards,
    or design state. A schema-audit/update stage must promote new data first.
    """
    root = Path(project_dir).expanduser().resolve()
    source_root = Path(new_data_dir).expanduser().resolve()
    if not source_root.exists() or not source_root.is_dir():
        raise ValueError(f"new_data_dir must be an existing directory: {source_root}")

    label = round_label or time.strftime("round_%Y%m%d_%H%M%S")
    incoming_dir = ensure_dir(root / "incoming" / label)
    files: list[dict[str, Any]] = []
    for source in sorted(p for p in source_root.rglob("*") if p.is_file()):
        rel = source.relative_to(source_root)
        dest = incoming_dir / rel
        ensure_dir(dest.parent)
        shutil.copy2(source, dest)
        files.append(_manifest_for_file(source, dest))

    manifest = {
        "round_label": label,
        "source_dir": str(source_root),
        "status": "pending_schema_audit",
        "files": files,
        "note": "New data are not evidence until schema audit promotes standardized tables.",
    }
    manifest_path = incoming_dir / "manifest.json"
    write_json(manifest_path, manifest)
    append_history_event(
        root,
        {
            "event_type": "new_data_intake",
            "round_label": label,
            "manifest": str(manifest_path),
            "status": "pending_schema_audit",
            "file_count": len(files),
        },
    )
    print(f"Merged new data into {incoming_dir}; status=pending_schema_audit")
    return manifest_path

