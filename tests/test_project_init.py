from __future__ import annotations

from pathlib import Path

import yaml

from design_scientist.projects.anti_hbsag import initialize_project


def test_initialize_anti_hbsag_project(tmp_path: Path) -> None:
    project_dir = tmp_path / "projects" / "anti_hbsag"
    initialize_project(project_dir)

    assert (project_dir / "project.yaml").exists()
    assert (project_dir / "data_contract.yaml").exists()
    assert (project_dir / "estimands.yaml").exists()
    assert (project_dir / "standardized" / "README.md").exists()
    assert (project_dir / "state").is_dir()
    assert (project_dir / "runs").is_dir()

    project = yaml.safe_load((project_dir / "project.yaml").read_text())
    assert project["constraints"]["system_roles"]["sdAb"] == "module_learning"
    assert project["constraints"]["system_roles"]["1E62"] == "target_system_design_genotype_coverage"

    contract = yaml.safe_load((project_dir / "data_contract.yaml").read_text())
    assert contract["raw_access_policy"] == "schema_audit_only"
    assert any("1E62 module-level causality" in claim for claim in contract["forbidden_claims"])

