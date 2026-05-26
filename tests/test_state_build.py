from __future__ import annotations

from pathlib import Path

from design_scientist.io import read_json, write_yaml
from design_scientist.state import build_state


def _write_csv(path: Path, header: str, rows: list[str]) -> None:
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")


def test_build_state_from_allowlisted_tables(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    primary = data / "primary.csv"
    long = data / "long.csv"
    auxiliary = data / "auxiliary.csv"
    modules = data / "module_summary.csv"
    missing = data / "missing.csv"

    _write_csv(primary, "id,value", ["a,1", "b,2"])
    _write_csv(long, "id,value", ["a,1", "b,2", "c,3"])
    _write_csv(auxiliary, "id,value", ["a,ok"])
    _write_csv(
        modules,
        "system,module_id,classification,n_primary_edges,primary_median_tau",
        ["sdAb,HD110H,candidate_acid_release_module,11,-0.131"],
    )
    _write_csv(
        missing,
        "system,module_id,base_variant,reason",
        ["1E62,HA23K,com1,would create exact primary edge"],
    )

    project = tmp_path / "project"
    (project / "state").mkdir(parents=True)
    write_yaml(
        project / "data_contract.yaml",
        {
            "version": "test",
            "project_id": "anti_hbsag",
            "raw_access_policy": "schema_audit_only",
            "allowed_tables": [
                {"name": "standardized_binding_primary_table", "path": str(primary), "role": "primary_endpoint"},
                {"name": "standardized_binding_long_table", "path": str(long), "role": "binding_endpoint"},
                {"name": "standardized_auxiliary_table", "path": str(auxiliary), "role": "auxiliary_qc"},
                {"name": "module_summary", "path": str(modules), "role": "module_summary"},
                {"name": "missing_primary_edge_candidates", "path": str(missing), "role": "lattice_repair"},
            ],
        },
    )

    state = build_state(project)

    assert state.row_counts["standardized_binding_primary_table"] == 2
    assert state.system_roles["sdAb"] == "module_learning"
    assert state.system_roles["1E62"] == "target_system_design_genotype_coverage"
    assert any(module["module_id"] == "HD110H" for module in state.module_status)
    persisted = read_json(project / "state" / "design_state.json")
    assert "Current 1E62 combo data cannot support module causal claims" in persisted["unsupported_claims"][0]


def test_build_state_canonicalizes_onee62_in_missing_edges(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    missing = data / "missing.csv"
    _write_csv(
        missing,
        "system,module_id,base_variant,reason",
        ["1e+62,HA23K,com1,would create exact primary edge"],
    )

    project = tmp_path / "project"
    (project / "state").mkdir(parents=True)
    write_yaml(
        project / "data_contract.yaml",
        {
            "version": "test",
            "project_id": "anti_hbsag",
            "raw_access_policy": "schema_audit_only",
            "allowed_tables": [
                {"name": "missing_primary_edge_candidates", "path": str(missing), "role": "lattice_repair"},
            ],
        },
    )

    build_state(project)

    persisted = read_json(project / "state" / "design_state.json")
    assert persisted["unresolved_edges"][0]["system"] == "1E62"


def test_build_state_from_startup_allowed_sources_counts_files_and_project_metadata(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    wet_lab = project / "raw" / "wet_lab"
    sequences = project / "raw" / "base_sequences"
    structures = project / "raw" / "base_structures_optional"
    (project / "state").mkdir(parents=True)
    wet_lab.mkdir(parents=True)
    sequences.mkdir(parents=True)
    structures.mkdir(parents=True)

    _write_csv(
        wet_lab / "ae_kd_ratio.csv",
        "variant_id,antigen_genotype,endpoint,value",
        ["1e+62,Ae,KD_ratio,2.0", "HK52H,Ae,KD_ratio,1.2"],
    )
    (sequences / "antigen_genotypes.fasta").write_text(
        ">AeS\nAAAA\n>BaS\nBBBB\n",
        encoding="utf-8",
    )
    (structures / "ab_wt.pdb").write_text(
        "ATOM      1  N   GLU A   1       0.000   0.000   0.000\n"
        "HETATM    2  O   HOH A   2       1.000   1.000   1.000\n"
        "REMARK not a coordinate row\n",
        encoding="utf-8",
    )
    write_yaml(
        project / "project.yaml",
        {
            "project_id": "1E62_pH_sensitive_design_startup",
            "system": "1E62",
            "domain": "antibody_variant_design",
            "objective": "Design 1E62 variants with retained neutral binding.",
        },
    )
    write_yaml(
        project / "data_contract.yaml",
        {
            "contract_id": "1E62_startup_data_contract",
            "local_root": str(project),
            "allowed_sources": {
                "base_sequences": {
                    "evidence_tier": "sequence_reference",
                    "allowed_as_experimental_evidence": False,
                    "files": ["raw/base_sequences/antigen_genotypes.fasta"],
                },
                "optional_base_structures": {
                    "evidence_tier": "structure_reference_optional",
                    "allowed_as_experimental_evidence": False,
                    "files": ["raw/base_structures_optional/ab_wt.pdb"],
                },
                "wet_lab": {
                    "evidence_tier": "primary_wet_lab",
                    "allowed_as_experimental_evidence": True,
                    "files": ["raw/wet_lab/ae_kd_ratio.csv"],
                },
            },
            "identifier_columns": ["variant_id", "antigen_genotype"],
        },
    )

    state = build_state(project)

    assert state.project_id == "1E62_pH_sensitive_design_startup"
    assert state.row_counts["raw/wet_lab/ae_kd_ratio.csv"] == 2
    assert state.row_counts["raw/base_sequences/antigen_genotypes.fasta"] == 2
    assert state.row_counts["raw/base_structures_optional/ab_wt.pdb"] == 2
    assert "sdAb" not in state.system_roles
    assert state.module_status == []
    persisted = read_json(project / "state" / "design_state.json")
    assert persisted["project_id"] == "1E62_pH_sensitive_design_startup"
    assert persisted["domain"] == "antibody_variant_design"
    assert persisted["objective"] == "Design 1E62 variants with retained neutral binding."


def test_build_state_rejects_allowed_source_csv_with_empty_header(tmp_path: Path) -> None:
    project = tmp_path / "project"
    wet_lab = project / "raw" / "wet_lab"
    (project / "state").mkdir(parents=True)
    wet_lab.mkdir(parents=True)
    _write_csv(wet_lab / "bad.csv", "variant_id,,value", ["com1,,1.0"])
    write_yaml(
        project / "project.yaml",
        {
            "project_id": "startup_with_bad_header",
            "domain": "antibody_variant_design",
            "objective": "Keep invalid source files out of counted evidence.",
        },
    )
    write_yaml(
        project / "data_contract.yaml",
        {
            "contract_id": "bad_header_contract",
            "local_root": str(project),
            "allowed_sources": {
                "wet_lab": {
                    "evidence_tier": "primary_wet_lab",
                    "allowed_as_experimental_evidence": True,
                    "files": ["raw/wet_lab/bad.csv"],
                },
            },
        },
    )

    state = build_state(project)

    assert state.row_counts["raw/wet_lab/bad.csv"] == 0
    evidence = read_json(project / "state" / "evidence_cards.json")
    assert any(
        card["evidence_id"].startswith("invalid_csv_header_")
        and card["source_tables"] == ["raw/wet_lab/bad.csv"]
        and card["actionability"] == "needs_data"
        for card in evidence
    )
