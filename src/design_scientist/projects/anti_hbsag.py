"""anti-HBsAg project initializer."""

from __future__ import annotations

from pathlib import Path

from design_scientist.io import ensure_dir, write_yaml


SCHEMA_AUDIT_DIR = Path("/Users/ziyang/Documents/Playground/outputs/anti_hbsag_schema_audit")
LATTICE_DIR = Path(
    "/Users/ziyang/Documents/Playground/tmp/anti_hbsag/results/"
    "anti_hbsag_estimand_lattice_20260506_160435"
)


def _table(name: str, path: Path, role: str, measurement_types: list[str] | None = None) -> dict:
    return {
        "name": name,
        "path": str(path),
        "role": role,
        "required": True,
        "allow_raw_access": False,
        "measurement_types": measurement_types or [],
        "provenance_columns": [
            "source_file",
            "source_row_csv",
            "source_column",
            "source_cell_id",
        ],
    }


def initialize_project(project_dir: str | Path, project_id: str = "anti_hbsag") -> Path:
    """Create the local anti-HBsAg Design Scientist project skeleton."""
    root = ensure_dir(project_dir)
    ensure_dir(root / "standardized")
    ensure_dir(root / "state")
    ensure_dir(root / "runs")

    project = {
        "project_id": project_id,
        "name": "anti-HBsAg pH-dependent antibody design",
        "goal": (
            "Prioritize anti-HBsAg variants that retain pH 7.4 binding, reduce pH 6.0 "
            "binding, preserve genotype coverage where relevant, and avoid obvious QC failures."
        ),
        "mode": "project_execution",
        "target_systems": ["sdAb", "1E62"],
        "constraints": {
            "performance_first": True,
            "interpretability_role": "support decision quality, not replace performance",
            "default_dry_run_budget": 24,
            "system_roles": {
                "sdAb": "module_learning",
                "1E62": "target_system_design_genotype_coverage",
            },
        },
        "notes": [
            "sdAb data are used for module learning.",
            "1E62 data are used for target-system design and genotype coverage.",
            "Current 1E62 combo data must not be used to claim module-level causality.",
        ],
    }

    data_contract = {
        "version": "2026-05-09",
        "project_id": project_id,
        "raw_access_policy": "schema_audit_only",
        "allowed_tables": [
            _table(
                "standardized_binding_long_table",
                SCHEMA_AUDIT_DIR / "standardized_binding_long_table.csv",
                "binding_endpoint",
                ["ELISA", "clean_response", "combo_summary"],
            ),
            _table(
                "standardized_binding_primary_table",
                SCHEMA_AUDIT_DIR / "standardized_binding_primary_table.csv",
                "primary_endpoint",
                ["ELISA", "clean_response", "combo_summary"],
            ),
            _table(
                "standardized_auxiliary_table",
                SCHEMA_AUDIT_DIR / "standardized_auxiliary_table.csv",
                "auxiliary_qc",
                ["expression", "recovery", "activity_call"],
            ),
            _table("schema_inventory", SCHEMA_AUDIT_DIR / "schema_inventory.csv", "schema_audit"),
            _table("validation_summary", SCHEMA_AUDIT_DIR / "validation_summary.json", "schema_audit"),
            _table("sdab_endpoint_table", LATTICE_DIR / "sdab_endpoint_table.csv", "lattice_endpoint"),
            _table("1e62_endpoint_table", LATTICE_DIR / "1e62_endpoint_table.csv", "lattice_endpoint"),
            _table("module_summary", LATTICE_DIR / "module_summary.csv", "module_summary"),
            _table("valid_primary_edges_summary", LATTICE_DIR / "valid_primary_edges_summary.csv", "matched_contrast"),
            _table("missing_primary_edge_candidates", LATTICE_DIR / "missing_primary_edge_candidates.csv", "lattice_repair"),
        ],
        "forbidden_claims": [
            "1E62 module-level causality from current combo-only data",
            "sdAb module effects transfer to 1E62 without validation",
            "synthetic or model-derived records are wet-lab evidence",
            "top pH ratio alone is sufficient for final design selection",
        ],
        "forbidden_table_patterns": ["raw", "synthetic"],
    }

    estimands = {
        "project_id": project_id,
        "primary": [
            {
                "estimand_id": "neutral_binding_retention",
                "name": "pH 7.4 binding retention",
                "description": "Maintain neutral-pH HBsAg binding.",
                "endpoint": "pH7.4 binding",
                "evidence_tier": "primary_matched",
                "success_direction": "higher_is_better",
                "thresholds": {},
            },
            {
                "estimand_id": "acid_release",
                "name": "pH 6.0 binding reduction",
                "description": "Reduce acidic-pH binding relative to neutral-pH binding.",
                "endpoint": "pH6.0 vs pH7.4 paired response",
                "evidence_tier": "primary_matched",
                "success_direction": "lower_acid_relative_to_neutral_is_better",
                "thresholds": {},
            },
        ],
        "secondary": [
            {
                "estimand_id": "module_matched_contrast",
                "name": "matched module contrast",
                "description": "Estimate module effect only from exact or explicitly labeled near-matched contrasts.",
                "endpoint": "matched contrast tau",
                "evidence_tier": "primary_matched",
                "success_direction": "more_negative_tau_is_better_if_neutral_binding_retained",
                "thresholds": {},
            }
        ],
        "guardrails": [
            {
                "estimand_id": "genotype_coverage",
                "name": "genotype coverage",
                "description": "Preserve Ae/B/D1 or relevant antigen/genotype coverage.",
                "endpoint": "multi-genotype pH7.4 binding",
                "evidence_tier": "descriptive",
                "success_direction": "coverage_retained",
                "thresholds": {},
            }
        ],
    }

    write_yaml(root / "project.yaml", project)
    write_yaml(root / "data_contract.yaml", data_contract)
    write_yaml(root / "estimands.yaml", estimands)
    (root / "standardized" / "README.md").write_text(
        "This project reads allowlisted standardized tables from data_contract.yaml.\n"
        "Raw CSV guessing is forbidden outside an explicit schema-audit stage.\n",
        encoding="utf-8",
    )
    print(f"Initialized project at {root}")
    return root

