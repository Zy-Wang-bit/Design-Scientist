from __future__ import annotations

from pathlib import Path

from design_scientist.data_merge import merge_new_data
from design_scientist.io import read_json, write_json
from design_scientist.project_history import load_project_history


def test_merge_new_data_marks_pending_schema_audit(tmp_path: Path) -> None:
    project = tmp_path / "project"
    new_data = tmp_path / "new_data"
    (project / "state").mkdir(parents=True)
    new_data.mkdir()
    write_json(project / "state" / "evidence_cards.json", [{"evidence_id": "old"}])
    (new_data / "round.csv").write_text("id,value\nv1,1\n", encoding="utf-8")

    manifest_path = merge_new_data(project, new_data, round_label="r1")

    manifest = read_json(manifest_path)
    history = load_project_history(project)
    evidence = read_json(project / "state" / "evidence_cards.json")
    assert manifest["status"] == "pending_schema_audit"
    assert manifest["files"][0]["sha256"]
    assert history[-1]["event_type"] == "new_data_intake"
    assert history[-1]["status"] == "pending_schema_audit"
    assert evidence == [{"evidence_id": "old"}]

