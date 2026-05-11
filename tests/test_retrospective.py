from __future__ import annotations

from pathlib import Path

from design_scientist.io import read_json, write_json
from design_scientist.retrospective import run_retrospective_eval


def test_retrospective_eval_writes_framework_artifacts(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "state").mkdir(parents=True)
    write_json(
        project / "state" / "design_state.json",
        {
            "project_id": "anti_hbsag",
            "module_status": [{"module_id": "HD110H"}, {"module_id": "HG56H"}],
            "unresolved_edges": [{"system": "1E62", "module_id": "HA23K", "base_variant": "com1"}],
            "unsupported_claims": ["Current 1E62 combo data cannot support module causal claims."],
        },
    )
    write_json(project / "state" / "evidence_cards.json", [])

    benchmark, ablation, protocol = run_retrospective_eval(project, budget=3)

    assert benchmark.exists()
    assert ablation.exists()
    assert protocol.exists()
    assert "unsupported_claim_penalty" in benchmark.read_text(encoding="utf-8")
    assert "removed_component" in ablation.read_text(encoding="utf-8")

