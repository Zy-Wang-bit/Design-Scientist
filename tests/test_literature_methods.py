from __future__ import annotations

from pathlib import Path

import yaml

from design_scientist.io import read_json
from design_scientist.literature import build_research_gap_matrix, create_seed_paper_cards
from design_scientist.methods import propose_method_hypotheses


def test_literature_and_methods_artifacts(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "project.yaml").write_text("goal: Improve variants over rounds\n", encoding="utf-8")

    cards_path = create_seed_paper_cards(project)
    gap_path = build_research_gap_matrix(project)
    hypotheses_path, registry_path = propose_method_hypotheses(project)

    cards = read_json(cards_path)
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    assert len(cards) >= 4
    assert "acquisition" in hypotheses_path.read_text(encoding="utf-8")
    assert "matched_contrast_lattice" in gap_path.read_text(encoding="utf-8")
    assert registry["policies"][0]["name"] == "mechanism_aware"
    assert "fixed mix" in registry["policies"][0]["baselines"]

