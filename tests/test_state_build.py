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
