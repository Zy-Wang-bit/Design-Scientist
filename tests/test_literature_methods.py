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


def test_legacy_seed_paper_cards_do_not_overwrite_framework_v2_literature_artifacts(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    framework = project / "framework"
    framework.mkdir(parents=True)
    (project / "project.yaml").write_text("goal: Improve variants over rounds\n", encoding="utf-8")
    v2_cards = framework / "paper_cards.json"
    v2_trace = framework / "literature_search_trace.json"
    v2_cards.write_text('[{"paper_id": "v2_card"}]\n', encoding="utf-8")
    v2_trace.write_text('{"trace": "v2"}\n', encoding="utf-8")

    cards_path = create_seed_paper_cards(project)

    assert cards_path == framework / "legacy_seed_paper_cards.json"
    assert read_json(cards_path)[0]["review_status"] == "seed_unverified"
    assert v2_cards.read_text(encoding="utf-8") == '[{"paper_id": "v2_card"}]\n'
    assert v2_trace.read_text(encoding="utf-8") == '{"trace": "v2"}\n'
