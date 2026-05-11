from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from design_scientist.io import read_json, write_json
from design_scientist.literature_pipeline import run_literature_search
from design_scientist.method_extraction import REQUIRED_POLICY_NAMES, extract_methods


FIXTURES = Path(__file__).parent / "fixtures" / "literature"


def test_extract_methods_from_offline_paper_cards_writes_framework_artifacts(tmp_path: Path) -> None:
    project = tmp_path / "project"
    framework = project / "framework"
    framework.mkdir(parents=True)
    write_json(
        framework / "paper_cards.json",
        [
            {
                "paper_id": "active_learning_antibody",
                "source": "pubmed",
                "title": "Batch active learning for antibody protein engineering",
                "abstract": "Surrogate models and acquisition functions improve batch protein design under sparse measurements.",
                "problem": "Choose informative and high-value variants across repeated wet-lab rounds.",
                "data_regime": "small, noisy, batch-limited biological measurements",
                "algorithm": "surrogate model plus acquisition policy",
                "acquisition_or_policy": "expected improvement, uncertainty sampling, constrained batch selection",
                "baselines": ["random feasible", "top observed", "uncertainty sampling"],
                "evaluation": "retrospective masking and prospective hit-rate tracking",
                "what_to_reuse": "batch-aware acquisition and retrospective replay",
            },
            {
                "paper_id": "matched_contrast_panel",
                "source": "pubmed",
                "title": "Matched contrast design for protein variant panels",
                "abstract": "Matched contrasts reduce confounding in small protein engineering panels.",
                "problem": "Avoid false module claims from confounded variant combinations.",
                "algorithm": "contrast lattice and evidence-tier tracking",
                "acquisition_or_policy": "lattice repair value plus champion-centered contrasts",
                "baselines": ["top ratio", "descriptive ranking", "pure lattice repair"],
                "evaluation": "false-positive advancement rate and matched-edge creation",
                "what_not_to_copy": "claiming causal module effects from descriptive comparisons",
            },
            {
                "paper_id": "decision_value",
                "source": "semantic_scholar",
                "title": "Decision value acquisition for wet lab protein design",
                "abstract": "Decision value acquisition for constrained wet-lab design with adaptive allocation.",
                "algorithm": "decision-value acquisition and adaptive allocation",
                "acquisition_or_policy": "state-dependent champion contrast and control allocation",
                "baselines": ["fixed allocation", "pure exploration"],
            },
            {
                "paper_id": "ph_antibody",
                "source": "biorxiv",
                "title": "pH-dependent antibody binding design with constrained optimization",
                "abstract": "Mechanism-aware antibody design preserves neutral binding, acidic release, genotype coverage, and QC guardrails.",
                "evaluation": "best feasible variant, coverage retention, validation contrasts",
            },
        ],
    )

    result = extract_methods(project)

    assert result["status"] == "ok"
    assert result["paper_count"] == 4
    assert result["method_module_count"] >= 4

    method_modules = read_json(framework / "method_modules.json")
    modules = method_modules["method_modules"]
    module_ids = {module["module_id"] for module in modules}
    assert "active_learning_batch_design" in module_ids
    assert "matched_contrast_lattice" in module_ids
    assert "adaptive_decision_value_allocation" in module_ids
    assert "mechanism_aware_antibody_design" in module_ids

    required_fields = {
        "problem_setting",
        "data_regime",
        "algorithm_family",
        "acquisition_policy_mechanism",
        "baselines",
        "evaluation_protocol",
        "failure_modes",
        "reusable_ideas",
        "source_papers",
    }
    assert all(required_fields <= set(module) for module in modules)
    assert any("active_learning_antibody" in module["source_papers"] for module in modules)
    assert method_modules["unassigned_papers"] == []

    registry = yaml.safe_load((framework / "method_registry.yaml").read_text(encoding="utf-8"))
    policy_names = {policy["name"] for policy in registry["policies"]}
    assert set(REQUIRED_POLICY_NAMES) <= policy_names
    codex_policy = next(policy for policy in registry["policies"] if policy["name"] == "codex_generated")
    assert codex_policy["status"] == "not_implemented"

    algorithm_spec = (framework / "algorithm_spec.md").read_text(encoding="utf-8")
    for heading in [
        "## Design Space",
        "## Observation Model",
        "## Objective",
        "## Constraints",
        "## Round Update",
        "## Policy API",
    ]:
        assert heading in algorithm_spec

    gap_matrix = (framework / "research_gap_matrix.csv").read_text(encoding="utf-8")
    assert "method_module_id" in gap_matrix
    assert "gap_for_design_scientist" in gap_matrix
    assert "matched_contrast_lattice" in gap_matrix

    literature_map = (framework / "literature_map.md").read_text(encoding="utf-8")
    assert "Batch active learning for antibody protein engineering" in literature_map
    assert "Matched Contrast Lattice" in literature_map

    hypotheses = (framework / "method_hypotheses.md").read_text(encoding="utf-8")
    assert "mechanism_aware" in hypotheses
    assert "pure_lattice_repair" in hypotheses


def test_extract_methods_empty_paper_cards_writes_failure_without_fabricating_methods(tmp_path: Path) -> None:
    project = tmp_path / "project"
    framework = project / "framework"
    framework.mkdir(parents=True)
    write_json(framework / "paper_cards.json", [])

    result = extract_methods(project)

    assert result["status"] == "failure"
    assert result["reason"] == "empty_paper_cards"
    assert result["method_modules"] == []
    failure_report = framework / "method_extraction_failure.md"
    assert failure_report.exists()
    assert "without source paper evidence" in failure_report.read_text(encoding="utf-8")
    assert not (framework / "method_modules.json").exists()
    assert not (framework / "method_registry.yaml").exists()


def test_extract_methods_consumes_literature_pipeline_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DESIGN_SCIENTIST_LITERATURE_FIXTURES", str(FIXTURES))
    project = tmp_path / "project"
    project.mkdir()
    (project / "project.yaml").write_text(
        "name: anti-HBsAg pH-dependent antibody design\n"
        "goal: Improve antibody protein engineering over rounds\n",
        encoding="utf-8",
    )
    run_literature_search(project, max_papers=5, offline_fixtures=True)

    result = extract_methods(project)

    assert result["status"] == "ok"
    assert result["paper_count"] == 5
    modules = read_json(project / "framework" / "method_modules.json")["method_modules"]
    module_ids = {module["module_id"] for module in modules}
    assert "active_learning_batch_design" in module_ids
    assert "matched_contrast_lattice" in module_ids
    assert "mechanism_aware_antibody_design" in module_ids
