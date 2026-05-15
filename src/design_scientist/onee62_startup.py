"""Startup ingestion for 1E62 pH-sensitive antibody design projects."""

from __future__ import annotations

import ast
import csv
import hashlib
import time
from collections import Counter
from pathlib import Path
from typing import Any

from design_scientist.io import ensure_dir, read_json, read_yaml, write_json, write_yaml
from design_scientist.project_history import append_history_event


PROJECT_ID = "1E62_pH_sensitive_design"
PROJECT_FAMILY = "1e62_startup"

WET_LAB_FILES = {
    "ae_kd_ratio": Path("raw/wet_lab/ae_kd_ratio.csv"),
    "elisa_dilution": Path("raw/wet_lab/elisa_dilution_measurements.csv"),
    "elisa_summary": Path("raw/wet_lab/elisa_summary.csv"),
    "expression": Path("raw/wet_lab/expression.csv"),
    "variant_sequences": Path("raw/wet_lab/variant_sequences.csv"),
}

REQUIRED_COLUMNS = {
    WET_LAB_FILES["ae_kd_ratio"]: [
        "variant_id",
        "antigen_genotype",
        "endpoint",
        "value",
    ],
    WET_LAB_FILES["elisa_dilution"]: [
        "variant_id",
        "antigen_genotype",
        "pH",
        "concentration_ng_ml",
        "replicate_id",
        "elisa_signal",
    ],
    WET_LAB_FILES["elisa_summary"]: [
        "variant",
        "Ae_pH6.0",
        "Ae_pH7.4",
        "B_pH6.0",
        "B_pH7.4",
        "D1_pH6.0",
        "D1_pH7.4",
        "Ae_ratio",
        "B_ratio",
        "D1_ratio",
        "H_chain",
        "L_chain",
        "mutations",
    ],
    WET_LAB_FILES["expression"]: [
        "编号",
        "浓度ug/ul",
        "首孔500ng所需体积ul",
    ],
    WET_LAB_FILES["variant_sequences"]: [
        "variant_id",
        "heavy_source_id",
        "heavy_variant_label",
        "heavy_construct",
        "heavy_chain_seq",
        "light_variant_label",
        "light_construct",
        "light_chain_seq",
    ],
}

STANDARDIZED_TABLES = {
    "standardized_binding_long_table": Path("standardized/binding_long.csv"),
    "standardized_binding_primary_table": Path("standardized/binding_primary.csv"),
    "ae_kd_ratio_table": Path("standardized/ae_kd_ratio.csv"),
    "standardized_auxiliary_table": Path("standardized/expression_qc.csv"),
    "variant_sequence_provenance": Path("standardized/variant_sequence_provenance.csv"),
    "schema_inventory": Path("standardized/schema_inventory.csv"),
    "validation_summary": Path("standardized/validation_summary.json"),
}

PROVENANCE_COLUMNS = [
    "source_file",
    "source_row_csv",
    "source_column",
    "source_cell_id",
]


def start_1e62_project(startup_dir: str | Path, project_dir: str | Path) -> dict[str, Any]:
    """Create a first-stage 1E62 execution project from a startup data package."""
    source_root = Path(startup_dir).expanduser().resolve()
    target_root = Path(project_dir).expanduser().resolve()
    if not source_root.is_dir():
        raise ValueError(f"startup_dir must be an existing directory: {source_root}")

    ensure_dir(target_root / "standardized")
    ensure_dir(target_root / "state")
    ensure_dir(target_root / "runs")

    audit = audit_1e62_startup_package(source_root)
    standardized = write_1e62_standardized_tables(source_root, target_root, audit)
    write_1e62_project_artifacts(source_root, target_root, audit)
    state = write_1e62_state_artifacts(target_root, audit)

    append_history_event(
        target_root,
        {
            "event_type": "onee62_startup",
            "status": "schema_audit_state_only",
            "startup_source": str(source_root),
            "standardized_tables": {
                name: str(path) for name, path in standardized.items()
            },
        },
    )

    from design_scientist.validators import validate_project

    validation_report = validate_project(target_root)
    print(f"Started 1E62 project at {target_root}")
    print(f"Wrote standardized tables under {target_root / 'standardized'}")
    print(f"Wrote design state under {target_root / 'state'}")
    return {
        "project_dir": target_root,
        "startup_dir": source_root,
        "standardized": standardized,
        "state": state,
        "validation_report": validation_report,
    }


def advance_1e62_project(
    project_dir: str | Path,
    *,
    budget: int = 24,
    use_codex: bool = False,
    max_papers: int = 30,
    offline_fixtures: bool = False,
) -> str:
    """Advance a started 1E62 project into the dry-panel draft stage."""
    root = Path(project_dir).expanduser().resolve()
    if not _has_onee62_literature_stage(root):
        run_1e62_literature_stage(
            root,
            max_papers=max_papers,
            offline_fixtures=offline_fixtures,
            allow_degraded=True,
        )
    _prepare_1e62_panel_stage(root)

    from design_scientist.pipeline import run_dry_panel

    run_id = run_dry_panel(root, budget=budget, use_codex=use_codex)
    append_history_event(
        root,
        {
            "event_type": "onee62_panel_stage",
            "status": "dry_panel_created",
            "run_id": run_id,
            "budget": budget,
            "use_codex": use_codex,
        },
    )
    return run_id


def review_1e62_panel(project_dir: str | Path, run_id: str | None = None) -> dict[str, Any]:
    """Write a 1E62 panel review gate report for a dry-panel run."""
    root = Path(project_dir).expanduser().resolve()
    selected_run = _select_run_dir(root, run_id)
    panel_rows = _read_standardized_csv(selected_run / "panel_recommendation.csv")
    candidate_rows = _read_standardized_csv(selected_run / "candidate_pool.csv")
    metrics = read_json(selected_run / "policy_metrics.json")
    state = read_json(root / "state" / "design_state.json")
    validation = read_json(selected_run / "validation_report.json")
    literature_stage = (
        read_json(root / "framework" / "1e62_literature_stage.json")
        if (root / "framework" / "1e62_literature_stage.json").exists()
        else {}
    )
    mechanism_cards = (
        read_json(root / "framework" / "mechanism_cards.json")
        if (root / "framework" / "mechanism_cards.json").exists()
        else []
    )

    panel_summary = _panel_summary(panel_rows)
    checklist = _panel_review_checklist(
        panel_rows=panel_rows,
        candidate_rows=candidate_rows,
        metrics=metrics,
        validation=validation,
        literature_stage=literature_stage,
        state=state,
    )
    status = "human_review_required"
    review = {
        "project_id": state.get("project_id", PROJECT_ID),
        "project_family": state.get("project_family", PROJECT_FAMILY),
        "run_id": selected_run.name,
        "status": status,
        "not_wet_lab_final": True,
        "panel_summary": panel_summary,
        "checklist": checklist,
        "baseline_comparison": metrics.get("baseline_comparison", []),
        "unsupported_claims": metrics.get("unsupported_claims", []),
        "mechanism_cards_used": [
            {
                "mechanism_id": card.get("mechanism_id"),
                "mechanism_name": card.get("mechanism_name"),
                "evidence_strength": card.get("evidence_strength"),
            }
            for card in mechanism_cards
            if isinstance(card, dict)
        ],
        "required_human_decisions": [
            "Confirm endpoint semantics before treating this as a wet-lab submission panel.",
            "Confirm that Ae KD_ratio single-point/module evidence may be used only as design rationale.",
            "Confirm selected variants/modules are feasible to construct and assay.",
            "Confirm budget and assay layout before wet-lab submission.",
        ],
    }
    write_json(selected_run / "panel_review.json", review)
    (selected_run / "panel_review_report.md").write_text(
        _panel_review_markdown(review, panel_rows),
        encoding="utf-8",
    )
    append_history_event(
        root,
        {
            "event_type": "onee62_panel_review",
            "status": status,
            "run_id": selected_run.name,
            "panel_count": panel_summary["panel_count"],
        },
    )
    print(f"Wrote panel review: {selected_run / 'panel_review_report.md'}")
    return review


def finalize_1e62_result(project_dir: str | Path, run_id: str | None = None) -> dict[str, Any]:
    """Finalize the current 1E62 computational result package for a dry-panel run."""
    root = Path(project_dir).expanduser().resolve()
    selected_run = _select_run_dir(root, run_id)
    review_path = selected_run / "panel_review.json"
    review = (
        read_json(review_path)
        if review_path.exists()
        else review_1e62_panel(root, run_id=selected_run.name)
    )
    panel_rows = _read_standardized_csv(selected_run / "panel_recommendation.csv")
    metrics = read_json(selected_run / "policy_metrics.json")
    validation = read_json(selected_run / "validation_report.json")
    project = read_yaml(root / "project.yaml")
    state = read_json(root / "state" / "design_state.json")
    literature_stage = (
        read_json(root / "framework" / "1e62_literature_stage.json")
        if (root / "framework" / "1e62_literature_stage.json").exists()
        else {}
    )

    final_status = (
        "computational_result_complete"
        if bool(validation.get("valid"))
        and all(item.get("passed") for item in review.get("checklist", []))
        and literature_stage.get("status") == "completed"
        else "computational_result_blocked"
    )
    assay_plan_path = selected_run / "assay_plan_draft.csv"
    _write_assay_plan_draft(assay_plan_path, panel_rows)
    final_result = {
        "project_id": state.get("project_id", PROJECT_ID),
        "project_family": state.get("project_family", PROJECT_FAMILY),
        "run_id": selected_run.name,
        "status": final_status,
        "result_type": "dry_run_panel_result",
        "wet_lab_final": False,
        "selected_panel_path": str((selected_run / "panel_recommendation.csv").resolve()),
        "assay_plan_draft_path": str(assay_plan_path.resolve()),
        "panel_review_path": str(review_path.resolve()),
        "validation_report_path": str((selected_run / "validation_report.json").resolve()),
        "decision": (
            "Use this as the current best computational dry-run panel for 1E62 "
            "experimental planning; do not treat selected variants as wet-lab evidence "
            "until assayed."
        ),
        "panel_summary": review.get("panel_summary", _panel_summary(panel_rows)),
        "validation": {
            "valid": bool(validation.get("valid")),
            "error_count": _count_findings(validation, "error"),
            "warning_count": _count_findings(validation, "warning"),
        },
        "review": {
            "status": review.get("status"),
            "all_checks_passed": all(item.get("passed") for item in review.get("checklist", [])),
            "failed_checks": [
                item.get("check")
                for item in review.get("checklist", [])
                if not item.get("passed")
            ],
            "required_human_decisions": review.get("required_human_decisions", []),
        },
        "literature_stage": {
            "status": literature_stage.get("status"),
            "mechanism_card_count": literature_stage.get("mechanism_card_count"),
        },
        "baseline_comparison": metrics.get("baseline_comparison", []),
        "unsupported_claims": metrics.get("unsupported_claims", []),
        "selected_candidates": [
            {
                "candidate_id": row.get("candidate_id"),
                "operator": row.get("operator"),
                "category": row.get("category"),
                "background": row.get("background"),
                "modules": _parse_listish(row.get("modules", "")),
                "required_measurements": _parse_listish(row.get("required_measurements", "")),
                "risk_flags": _parse_listish(row.get("risk_flags", "")),
                "rationale": row.get("rationale"),
            }
            for row in panel_rows
        ],
    }
    write_json(selected_run / "final_result.json", final_result)
    (selected_run / "final_result_report.md").write_text(
        _final_result_markdown(final_result),
        encoding="utf-8",
    )

    state["latest_result"] = {
        "run_id": selected_run.name,
        "status": final_status,
        "result_type": "dry_run_panel_result",
        "wet_lab_final": False,
        "artifacts": {
            "final_result": str((selected_run / "final_result.json").resolve()),
            "final_result_report": str((selected_run / "final_result_report.md").resolve()),
            "assay_plan_draft": str(assay_plan_path.resolve()),
        },
    }
    write_json(root / "state" / "design_state.json", state)
    project["stage"] = final_status
    project["latest_run_id"] = selected_run.name
    project["latest_result_artifacts"] = state["latest_result"]["artifacts"]
    write_yaml(root / "project.yaml", project)
    append_history_event(
        root,
        {
            "event_type": "onee62_final_result",
            "status": final_status,
            "run_id": selected_run.name,
            "panel_count": final_result["panel_summary"]["panel_count"],
        },
    )
    print(f"Wrote final result: {selected_run / 'final_result_report.md'}")
    print(f"Wrote assay plan draft: {assay_plan_path}")
    return final_result


def run_1e62_literature_stage(
    project_dir: str | Path,
    *,
    max_papers: int = 30,
    offline_fixtures: bool = False,
    allow_degraded: bool = True,
) -> dict[str, Any]:
    """Run the explicit 1E62 literature retrieval/reading/mechanism stage."""
    root = Path(project_dir).expanduser().resolve()
    framework_dir = ensure_dir(root / "framework")
    _write_onee62_literature_queries(root)
    stage: dict[str, Any] = {
        "project_id": PROJECT_ID,
        "project_family": PROJECT_FAMILY,
        "stage": "literature",
        "max_papers": max_papers,
        "offline_fixtures": offline_fixtures,
        "allow_degraded": allow_degraded,
        "steps": [],
        "status": "started",
        "artifacts": {},
    }

    try:
        from design_scientist.literature_pipeline import run_literature_search

        paper_cards = run_literature_search(
            root,
            max_papers=max_papers,
            offline_fixtures=offline_fixtures,
        )
        stage["steps"].append({"name": "literature_retrieval", "status": "completed"})
        stage["artifacts"]["paper_cards"] = str(paper_cards)
    except Exception as exc:
        _record_stage_exception(stage, "literature_retrieval", exc)
        if not allow_degraded:
            raise

    try:
        from design_scientist.literature_fulltext import build_literature_corpus

        corpus = build_literature_corpus(root, offline_fixtures=offline_fixtures)
        stage["steps"].append({"name": "fulltext_reading", "status": "completed"})
        stage["artifacts"]["literature_corpus"] = str(corpus)
        stage["artifacts"]["literature_reading_trace"] = str(framework_dir / "literature_reading_trace.json")
    except Exception as exc:
        _record_stage_exception(stage, "fulltext_reading", exc)
        if not allow_degraded:
            raise

    try:
        from design_scientist.mechanism_extraction import extract_mechanisms

        extraction = extract_mechanisms(root)
        status = str(extraction.get("status", "unknown")) if isinstance(extraction, dict) else "unknown"
        if not status.startswith("ok"):
            raise RuntimeError(str(extraction.get("reason") or status))
        stage["steps"].append({"name": "mechanism_extraction", "status": "completed"})
        stage["artifacts"].update(extraction.get("artifacts", {}))
        stage["mechanism_card_count"] = extraction.get("mechanism_card_count")
    except Exception as exc:
        _record_stage_exception(stage, "mechanism_extraction", exc)
        if not allow_degraded:
            raise

    stage["status"] = (
        "completed"
        if all(step["status"] == "completed" for step in stage["steps"])
        and _has_onee62_literature_stage(root)
        else "degraded"
    )
    write_json(framework_dir / "1e62_literature_stage.json", stage)
    _write_onee62_literature_report(root, stage)
    append_history_event(
        root,
        {
            "event_type": "onee62_literature_stage",
            "status": stage["status"],
            "max_papers": max_papers,
            "offline_fixtures": offline_fixtures,
            "allow_degraded": allow_degraded,
        },
    )
    return stage


def audit_1e62_startup_package(startup_dir: str | Path) -> dict[str, Any]:
    """Validate startup package integrity and source schemas without writing outputs."""
    root = Path(startup_dir).expanduser().resolve()
    startup_contract = read_yaml(root / "data_contract.yaml")
    checksums = _validate_checksums(root)
    csv_tables = {
        key: _read_csv_records(root, rel, REQUIRED_COLUMNS[rel])
        for key, rel in WET_LAB_FILES.items()
    }

    summary_variants = {row["variant"] for row in csv_tables["elisa_summary"]["records"]}
    expression_variants = {row["编号"] for row in csv_tables["expression"]["records"]}
    sequence_variants = {row["variant_id"] for row in csv_tables["variant_sequences"]["records"]}
    dilution_variants = {row["variant_id"] for row in csv_tables["elisa_dilution"]["records"]}
    kd_variants = {row["variant_id"] for row in csv_tables["ae_kd_ratio"]["records"]}

    aligned_variant_sets = {
        "elisa_summary": sorted(summary_variants),
        "expression": sorted(expression_variants),
        "variant_sequences": sorted(sequence_variants),
        "elisa_dilution": sorted(dilution_variants),
    }
    all_com_variants = set().union(*[set(value) for value in aligned_variant_sets.values()])
    variant_set_diffs = {
        name: sorted(all_com_variants ^ set(variants))
        for name, variants in aligned_variant_sets.items()
    }
    com_variant_tables_aligned = not any(variant_set_diffs.values())

    dilution_records = csv_tables["elisa_dilution"]["records"]
    dimensions = {
        "variants": sorted(dilution_variants),
        "antigen_genotypes": sorted({row["antigen_genotype"] for row in dilution_records}),
        "pH_values": sorted({row["pH"] for row in dilution_records}),
        "concentrations_ng_ml": sorted(
            {float(row["concentration_ng_ml"]) for row in dilution_records},
            reverse=True,
        ),
        "replicate_ids": sorted({row["replicate_id"] for row in dilution_records}),
    }
    expected_dilution_rows = (
        len(dimensions["variants"])
        * len(dimensions["antigen_genotypes"])
        * len(dimensions["pH_values"])
        * len(dimensions["concentrations_ng_ml"])
        * len(dimensions["replicate_ids"])
    )

    if not com_variant_tables_aligned:
        raise ValueError(f"1E62 startup variant IDs are not aligned: {variant_set_diffs}")
    if len(dilution_records) != expected_dilution_rows:
        raise ValueError(
            "elisa_dilution_measurements.csv row count does not match its "
            f"variant/genotype/pH/concentration/replicate dimensions: "
            f"{len(dilution_records)} != {expected_dilution_rows}"
        )

    return {
        "startup_contract": startup_contract,
        "checksums": checksums,
        "csv_tables": csv_tables,
        "id_alignment": {
            "com_variant_tables_aligned": com_variant_tables_aligned,
            "variant_ids": sorted(all_com_variants),
            "variant_set_diffs": variant_set_diffs,
            "table_counts": {
                name: len(variants) for name, variants in aligned_variant_sets.items()
            },
        },
        "dilution_dimensions": {
            **dimensions,
            "observed_rows": len(dilution_records),
            "expected_rows": expected_dilution_rows,
            "matches_expected": len(dilution_records) == expected_dilution_rows,
        },
        "kd_ratio": {
            "row_count": len(csv_tables["ae_kd_ratio"]["records"]),
            "variant_ids": sorted(kd_variants),
            "overlap_with_com_variants": sorted(kd_variants & all_com_variants),
            "overlaps_com_variants": bool(kd_variants & all_com_variants),
            "evidence_scope": "Ae-specific single-point/module evidence",
        },
    }


def write_1e62_standardized_tables(
    startup_dir: str | Path,
    project_dir: str | Path,
    audit: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write standardized first-stage project tables."""
    source_root = Path(startup_dir).expanduser().resolve()
    target_root = Path(project_dir).expanduser().resolve()
    audit = audit or audit_1e62_startup_package(source_root)
    tables = audit["csv_tables"]

    standardized_paths = {
        name: target_root / rel for name, rel in STANDARDIZED_TABLES.items()
    }

    _write_binding_long(
        standardized_paths["standardized_binding_long_table"],
        tables["elisa_dilution"]["records"],
    )
    _write_binding_primary(
        standardized_paths["standardized_binding_primary_table"],
        tables["elisa_summary"]["records"],
    )
    _write_ae_kd_ratio(
        standardized_paths["ae_kd_ratio_table"],
        tables["ae_kd_ratio"]["records"],
    )
    _write_expression_qc(
        standardized_paths["standardized_auxiliary_table"],
        tables["expression"]["records"],
    )
    _write_variant_sequence_provenance(
        standardized_paths["variant_sequence_provenance"],
        tables["variant_sequences"]["records"],
    )
    _write_schema_inventory(
        standardized_paths["schema_inventory"],
        source_root,
        audit,
    )
    write_json(
        standardized_paths["validation_summary"],
        _validation_summary(source_root, audit),
    )
    return standardized_paths


def write_1e62_project_artifacts(
    startup_dir: str | Path,
    project_dir: str | Path,
    audit: dict[str, Any],
) -> None:
    source_root = Path(startup_dir).expanduser().resolve()
    target_root = Path(project_dir).expanduser().resolve()
    table = _table_spec
    write_yaml(
        target_root / "project.yaml",
        {
            "project_id": PROJECT_ID,
            "project_family": PROJECT_FAMILY,
            "name": "1E62 pH-sensitive antibody design",
            "goal": (
                "Design 1E62 antibody variants that retain pH 7.4 binding "
                "and reduce pH 6.0 binding or promote dissociation."
            ),
            "mode": "project_execution",
            "stage": "schema_audit_state_only",
            "target_systems": ["1E62"],
            "startup_source": str(source_root),
            "constraints": {
                "performance_first": True,
                "no_panel_recommendation_in_startup_stage": True,
                "unbounded_sequence_exploration_allowed_after_human_gate": True,
                "untested_variants_are_model_derived_only": True,
            },
            "human_checkpoints": [
                "confirm_endpoint_semantics_before_modeling",
                "approve_design_space_before_candidate_generation",
                "review_panel_before_wet_lab_submission",
            ],
        },
    )
    write_yaml(
        target_root / "data_contract.yaml",
        {
            "version": time.strftime("%Y-%m-%d"),
            "project_id": PROJECT_ID,
            "project_family": PROJECT_FAMILY,
            "raw_access_policy": "schema_audit_only",
            "startup_source_root": str(source_root),
            "allowed_startup_sources": [
                str(path) for path in _allowed_source_paths(audit["startup_contract"])
            ],
            "allowed_tables": [
                table("standardized_binding_long_table", target_root, "binding_endpoint", ["ELISA", "replicate"]),
                table("standardized_binding_primary_table", target_root, "primary_endpoint", ["ELISA", "summary"]),
                table("ae_kd_ratio_table", target_root, "ae_specific_kd_ratio", ["KD_ratio"]),
                table("standardized_auxiliary_table", target_root, "auxiliary_qc", ["expression"]),
                table("variant_sequence_provenance", target_root, "sequence_provenance", ["sequence"]),
                table("schema_inventory", target_root, "schema_audit", ["schema"]),
                table("validation_summary", target_root, "schema_audit", ["validation"]),
            ],
            "forbidden_claims": [
                "Untested 1E62 variants or model-designed sequences are wet-lab evidence.",
                "Ae KD_ratio single-point/module evidence directly proves com combination behavior.",
                "Sequence or structure references are experimental binding evidence.",
                "pH-sensitive release claims without matched pH 6.0 and pH 7.4 evidence.",
            ],
            "forbidden_table_patterns": ["raw", "synthetic"],
        },
    )
    write_yaml(
        target_root / "estimands.yaml",
        {
            "project_id": PROJECT_ID,
            "primary": [
                {
                    "estimand_id": "neutral_binding_retention",
                    "name": "pH 7.4 binding retention",
                    "description": "Retain neutral-pH HBsAg binding across measured genotypes.",
                    "endpoint": "pH7.4 ELISA summary signal",
                    "evidence_tier": "primary_matched",
                    "success_direction": "higher_is_better",
                    "thresholds": {},
                },
                {
                    "estimand_id": "acid_release",
                    "name": "pH 6.0 binding reduction",
                    "description": "Reduce acidic-pH binding while preserving pH 7.4 binding.",
                    "endpoint": "pH6.0 vs pH7.4 paired ELISA summary",
                    "evidence_tier": "primary_matched",
                    "success_direction": "lower_acid_relative_to_neutral_is_better",
                    "thresholds": {},
                },
            ],
            "secondary": [
                {
                    "estimand_id": "ae_kd_ratio",
                    "name": "Ae KD_ratio",
                    "description": "Ae-specific KD_ratio evidence from startup data.",
                    "endpoint": "KD_ratio",
                    "evidence_tier": "primary_matched",
                    "success_direction": "requires_endpoint_semantics_confirmation",
                    "thresholds": {},
                }
            ],
            "guardrails": [
                {
                    "estimand_id": "genotype_coverage",
                    "name": "genotype coverage",
                    "description": "Preserve Ae, B, and D1 neutral-pH coverage.",
                    "endpoint": "Ae/B/D1 pH7.4 ELISA",
                    "evidence_tier": "descriptive",
                    "success_direction": "coverage_retained",
                    "thresholds": {},
                },
                {
                    "estimand_id": "expression_feasibility",
                    "name": "expression feasibility",
                    "description": "Use expression/concentration as QC evidence.",
                    "endpoint": "expression concentration and dosing volume",
                    "evidence_tier": "descriptive",
                    "success_direction": "feasible_expression",
                    "thresholds": {},
                },
            ],
        },
    )


def write_1e62_state_artifacts(project_dir: str | Path, audit: dict[str, Any]) -> dict[str, Any]:
    root = Path(project_dir).expanduser().resolve()
    rows = {
        name: _count_csv_rows(root / rel)
        for name, rel in STANDARDIZED_TABLES.items()
        if rel.suffix == ".csv"
    }
    primary_records = audit["csv_tables"]["elisa_summary"]["records"]
    kd_records = audit["csv_tables"]["ae_kd_ratio"]["records"]
    expression_records = audit["csv_tables"]["expression"]["records"]

    state: dict[str, Any] = {
        "project_id": PROJECT_ID,
        "project_family": PROJECT_FAMILY,
        "data_version": time.strftime("startup_%Y-%m-%d"),
        "current_round": 0,
        "row_counts": rows,
        "system_roles": {"1E62": "target_system_design"},
        "known_champions": [],
        "tested_variants": [
            {
                "variant_id": row["variant"],
                "source": "elisa_summary",
                "mutation_count": _mutation_count(row.get("mutations", "")),
            }
            for row in primary_records
        ],
        "module_status": [
            {
                "system": "1E62",
                "module_id": row["variant_id"],
                "antigen_genotype": row["antigen_genotype"],
                "endpoint": row["endpoint"],
                "ae_kd_ratio": _to_float(row["value"]),
                "status": "observed_ae_kd_single_point_or_module_evidence",
            }
            for row in kd_records
        ],
        "expression_qc": [
            {
                "variant_id": row["编号"],
                "concentration_ug_ul": _to_float(row["浓度ug/ul"]),
                "first_well_500ng_volume_ul": _to_float(row["首孔500ng所需体积ul"]),
            }
            for row in expression_records
        ],
        "genotype_coverage": audit["dilution_dimensions"]["antigen_genotypes"],
        "design_space": {
            "sequence_exploration": "unbounded_full_length",
            "stage": "registered_only_no_candidate_generation",
            "future_candidate_generation_allowed_after_human_gate": True,
            "untested_sequences_evidence_tier": "model_derived",
            "risk_gate": "untested or model-designed variants must not be treated as wet-lab evidence",
        },
        "endpoint_policy": {
            "primary_binding_source": "raw/wet_lab/elisa_summary.csv",
            "replicate_provenance_source": "raw/wet_lab/elisa_dilution_measurements.csv",
            "status": "pending_human_confirmation",
        },
        "unsupported_claims": [
            "Untested 1E62 variants or model-designed sequences are not wet-lab evidence.",
            "Ae KD_ratio single-point/module evidence cannot be directly merged as com combination evidence.",
            "Sequence and structure references are not experimental binding evidence.",
            "pH-sensitive release claims require matched pH 6.0 and pH 7.4 evidence by variant and genotype.",
        ],
        "policy_history": [],
    }
    evidence_cards = _evidence_cards(audit)
    write_json(root / "state" / "design_state.json", state)
    write_json(root / "state" / "evidence_cards.json", evidence_cards)
    return state


def _has_onee62_literature_stage(root: Path) -> bool:
    required = [
        root / "framework" / "paper_cards.json",
        root / "framework" / "literature_search_trace.json",
        root / "framework" / "literature_corpus.jsonl",
        root / "framework" / "literature_reading_trace.json",
        root / "framework" / "mechanism_cards.json",
        root / "framework" / "mechanism_library.json",
        root / "framework" / "mechanism_gap_matrix.csv",
    ]
    return all(path.exists() for path in required)


def _select_run_dir(root: Path, run_id: str | None) -> Path:
    runs_dir = root / "runs"
    if run_id:
        run_dir = runs_dir / run_id
    else:
        run_dirs = sorted(path for path in runs_dir.iterdir() if path.is_dir())
        if not run_dirs:
            raise ValueError(f"No run directories found under {runs_dir}")
        run_dir = run_dirs[-1]
    if not run_dir.is_dir():
        raise ValueError(f"Run directory does not exist: {run_dir}")
    for required in ("candidate_pool.csv", "panel_recommendation.csv", "policy_metrics.json", "validation_report.json"):
        if not (run_dir / required).exists():
            raise ValueError(f"Run {run_dir.name} is missing {required}")
    return run_dir


def _panel_summary(panel_rows: list[dict[str, str]]) -> dict[str, Any]:
    categories = Counter(row.get("category", "") for row in panel_rows)
    operators = Counter(row.get("operator", "") for row in panel_rows)
    backgrounds = Counter(row.get("background", "") for row in panel_rows)
    modules = Counter(
        module
        for row in panel_rows
        for module in _parse_listish(row.get("modules", ""))
        if module
    )
    risk_flags = Counter(
        risk
        for row in panel_rows
        for risk in _parse_listish(row.get("risk_flags", ""))
        if risk
    )
    cost = sum(_to_float(row.get("cost")) or 0.0 for row in panel_rows)
    return {
        "panel_count": len(panel_rows),
        "total_cost": round(cost, 6),
        "category_counts": dict(sorted(categories.items())),
        "operator_counts": dict(sorted(operators.items())),
        "background_counts": dict(sorted(backgrounds.items())),
        "module_counts": dict(sorted(modules.items())),
        "risk_flag_counts": dict(sorted(risk_flags.items())),
    }


def _panel_review_checklist(
    *,
    panel_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    metrics: dict[str, Any],
    validation: dict[str, Any],
    literature_stage: dict[str, Any],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    panel_ids = [row.get("candidate_id", "") for row in panel_rows]
    candidate_ids = {row.get("candidate_id", "") for row in candidate_rows}
    category_counts = Counter(row.get("category", "") for row in panel_rows)
    checklist = [
        {
            "check": "validation_passed",
            "passed": bool(validation.get("valid")),
            "detail": "Run validation report is valid.",
        },
        {
            "check": "literature_stage_completed",
            "passed": literature_stage.get("status") == "completed",
            "detail": "Explicit 1E62 literature/mechanism stage completed before panel review.",
        },
        {
            "check": "panel_subset_of_candidate_pool",
            "passed": set(panel_ids) <= candidate_ids,
            "detail": "Every panel row appears in candidate_pool.csv.",
        },
        {
            "check": "no_duplicate_panel_candidates",
            "passed": len(panel_ids) == len(set(panel_ids)),
            "detail": "Panel candidate IDs are unique.",
        },
        {
            "check": "no_infeasible_panel_candidates",
            "passed": all(row.get("feasibility_status") != "infeasible" for row in panel_rows),
            "detail": "No selected candidate is marked infeasible.",
        },
        {
            "check": "unsupported_claims_visible",
            "passed": bool(metrics.get("unsupported_claims")),
            "detail": "Unsupported claims are carried into policy metrics.",
        },
        {
            "check": "has_control_or_repeat",
            "passed": bool(category_counts.get("control") or category_counts.get("repeat")),
            "detail": "Panel includes at least one control or repeat category.",
        },
        {
            "check": "unbounded_design_space_not_evidence",
            "passed": "model_derived" in str(state.get("design_space", {})) or "model_derived" in str(state.get("panel_stage", {})),
            "detail": "Untested sequence exploration remains separated from wet-lab evidence.",
        },
    ]
    return checklist


def _panel_review_markdown(review: dict[str, Any], panel_rows: list[dict[str, str]]) -> str:
    summary = review["panel_summary"]
    lines = [
        f"# 1E62 Panel Review: {review['run_id']}",
        "",
        f"Status: `{review['status']}`",
        "",
        "This is not a wet-lab-final recommendation. It is a dry-run panel ready for human review.",
        "",
        "## Summary",
        "",
        f"- Panel count: {summary['panel_count']}",
        f"- Total cost: {summary['total_cost']}",
        f"- Categories: {summary['category_counts']}",
        f"- Operators: {summary['operator_counts']}",
        f"- Backgrounds: {summary['background_counts']}",
        f"- Risk flags: {summary['risk_flag_counts']}",
        "",
        "## Review Checklist",
        "",
    ]
    for item in review["checklist"]:
        mark = "PASS" if item["passed"] else "FAIL"
        lines.append(f"- {mark} `{item['check']}`: {item['detail']}")
    lines.extend(["", "## Required Human Decisions", ""])
    for decision in review["required_human_decisions"]:
        lines.append(f"- {decision}")
    lines.extend(["", "## First Panel Rows", ""])
    for row in panel_rows[:12]:
        lines.append(
            f"- `{row.get('candidate_id')}`: {row.get('operator')} on "
            f"`{row.get('background')}` modules={row.get('modules')} "
            f"measurements={row.get('required_measurements')}"
        )
    lines.extend(["", "## Unsupported Claims", ""])
    for claim in review["unsupported_claims"]:
        lines.append(f"- {claim}")
    lines.append("")
    return "\n".join(lines)


def _write_assay_plan_draft(path: Path, panel_rows: list[dict[str, str]]) -> None:
    rows = [
        {
            "candidate_id": row.get("candidate_id"),
            "operator": row.get("operator"),
            "category": row.get("category"),
            "background": row.get("background"),
            "modules": ";".join(_parse_listish(row.get("modules", ""))),
            "required_measurements": ";".join(_parse_listish(row.get("required_measurements", ""))),
            "evidence_status": "model_derived_until_tested",
            "wet_lab_evidence": "false",
            "risk_flags": ";".join(_parse_listish(row.get("risk_flags", ""))),
            "rationale": row.get("rationale"),
        }
        for row in panel_rows
    ]
    _write_csv(
        path,
        rows,
        [
            "candidate_id",
            "operator",
            "category",
            "background",
            "modules",
            "required_measurements",
            "evidence_status",
            "wet_lab_evidence",
            "risk_flags",
            "rationale",
        ],
    )


def _count_findings(validation: dict[str, Any], severity: str) -> int:
    findings = validation.get("findings", [])
    if not isinstance(findings, list):
        return 0
    return sum(
        1
        for finding in findings
        if isinstance(finding, dict) and finding.get("severity") == severity
    )


def _final_result_markdown(final_result: dict[str, Any]) -> str:
    summary = final_result["panel_summary"]
    validation = final_result["validation"]
    review = final_result["review"]
    lines = [
        f"# 1E62 Final Computational Result: {final_result['run_id']}",
        "",
        f"Status: `{final_result['status']}`",
        "",
        "This is the completed computational result for the current 1E62 task. It is not a wet-lab-final result and does not claim that any untested candidate works experimentally.",
        "",
        "## Decision",
        "",
        final_result["decision"],
        "",
        "## Artifacts",
        "",
        f"- Selected panel: `{final_result['selected_panel_path']}`",
        f"- Assay plan draft: `{final_result['assay_plan_draft_path']}`",
        f"- Panel review: `{final_result['panel_review_path']}`",
        f"- Validation report: `{final_result['validation_report_path']}`",
        "",
        "## Panel Summary",
        "",
        f"- Panel count: {summary['panel_count']}",
        f"- Total cost: {summary['total_cost']}",
        f"- Categories: {summary['category_counts']}",
        f"- Operators: {summary['operator_counts']}",
        f"- Backgrounds: {summary['background_counts']}",
        f"- Risk flags: {summary['risk_flag_counts']}",
        "",
        "## Validation And Review",
        "",
        f"- Validation valid: {validation['valid']}",
        f"- Validation errors: {validation['error_count']}",
        f"- Validation warnings: {validation['warning_count']}",
        f"- Panel review status: `{review['status']}`",
        f"- Panel review checks passed: {review['all_checks_passed']}",
        "",
        "## Selected Candidates",
        "",
    ]
    for candidate in final_result["selected_candidates"]:
        modules = ", ".join(candidate["modules"]) or "none"
        measurements = ", ".join(candidate["required_measurements"]) or "not specified"
        risk = ", ".join(candidate["risk_flags"]) or "none"
        lines.append(
            f"- `{candidate['candidate_id']}`: {candidate['operator']} / "
            f"{candidate['category']} on `{candidate['background']}`; "
            f"modules={modules}; measurements={measurements}; risks={risk}"
        )
    lines.extend(["", "## Unsupported Claims", ""])
    for claim in final_result["unsupported_claims"]:
        lines.append(f"- {claim}")
    lines.extend(["", "## Required Human Decisions", ""])
    for decision in review["required_human_decisions"]:
        lines.append(f"- {decision}")
    lines.append("")
    return "\n".join(lines)


def _parse_listish(value: str) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [text]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def _write_onee62_literature_queries(root: Path) -> None:
    query_path = root / "framework" / "literature_queries.yaml"
    if query_path.exists():
        return
    write_yaml(
        query_path,
        {
            "queries": [
                {
                    "query_id": "onee62_ph_sensitive_antibody_design",
                    "query": (
                        "pH-dependent antibody antigen binding design histidine engineering "
                        "HBsAg antibody variant"
                    ),
                    "domain": "antibody_variant_design",
                    "status": "planned",
                    "sources": ["pubmed", "biorxiv", "arxiv", "semantic_scholar"],
                },
                {
                    "query_id": "active_learning_antibody_variant_design",
                    "query": (
                        "active learning Bayesian optimization antibody protein engineering "
                        "variant design wet lab"
                    ),
                    "domain": "computational_design_method",
                    "status": "planned",
                    "sources": ["pubmed", "biorxiv", "arxiv", "semantic_scholar"],
                },
                {
                    "query_id": "batch_experimental_design_protein_engineering",
                    "query": (
                        "batch experimental design protein engineering acquisition function "
                        "retrospective evaluation"
                    ),
                    "domain": "adaptive_experimental_design",
                    "status": "planned",
                    "sources": ["pubmed", "biorxiv", "arxiv", "semantic_scholar"],
                },
            ]
        },
    )


def _record_stage_exception(stage: dict[str, Any], name: str, exc: Exception) -> None:
    stage["steps"].append(
        {
            "name": name,
            "status": "degraded",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    )


def _write_onee62_literature_report(root: Path, stage: dict[str, Any]) -> None:
    lines = [
        "# 1E62 Literature Stage",
        "",
        f"Status: `{stage['status']}`",
        f"Max papers: `{stage['max_papers']}`",
        f"Offline fixtures: `{stage['offline_fixtures']}`",
        "",
        "## Steps",
        "",
    ]
    for step in stage["steps"]:
        line = f"- {step['name']}: {step['status']}"
        if step.get("error"):
            line += f" ({step['error_type']}: {step['error']})"
        lines.append(line)
    lines.extend(["", "## Artifacts", ""])
    for name, artifact in sorted(stage["artifacts"].items()):
        lines.append(f"- {name}: `{artifact}`")
    lines.extend(
        [
            "",
            "## Gate",
            "",
            "This stage runs before dry-panel drafting. Missing or degraded literature artifacts must be visible here, not hidden behind panel generation.",
            "",
        ]
    )
    (root / "framework" / "1e62_literature_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _prepare_1e62_panel_stage(root: Path) -> None:
    project_path = root / "project.yaml"
    state_path = root / "state" / "design_state.json"
    project = read_yaml(project_path)
    state = read_json(state_path)

    primary_rows = _read_standardized_csv(root / STANDARDIZED_TABLES["standardized_binding_primary_table"])
    kd_rows = _read_standardized_csv(root / STANDARDIZED_TABLES["ae_kd_ratio_table"])
    champions = _rank_observed_champions(primary_rows)[:3]
    module_priorities = _rank_kd_modules(kd_rows)[:12]
    primary_evidence_refs = [
        "onee62_elisa_primary_summary",
        "onee62_ae_kd_ratio_single_point_evidence",
    ]

    state["known_champions"] = [
        {
            "variant_id": item["variant_id"],
            "selection_basis": "observed_elisa_summary_proxy",
            "score": item["score"],
            "mean_pH74_signal": item["mean_pH74_signal"],
            "mean_pH6_signal": item["mean_pH6_signal"],
            "status": "observed_candidate_for_dry_panel_contrast",
        }
        for item in champions
    ]
    state["backgrounds"] = [item["variant_id"] for item in champions] or [
        "full_length_1E62_sequence_space"
    ]
    state["target_backgrounds"] = list(state["backgrounds"])
    state["module_status"] = [
        {
            **module,
            "priority_rank": rank,
            "selection_basis": "Ae KD_ratio absolute log-deviation; endpoint direction pending confirmation",
        }
        for rank, module in enumerate(module_priorities, start=1)
    ] + [
        module
        for module in state.get("module_status", [])
        if module.get("module_id") not in {item["module_id"] for item in module_priorities}
    ]
    state["unresolved_edges"] = [
        {
            "system": "1E62",
            "module_id": module["module_id"],
            "base_variant": champion["variant_id"],
            "evidence_id": "onee62_ae_kd_ratio_single_point_evidence",
            "priority_rank": edge_rank,
            "reason": (
                "Dry-run exact contrast candidate linking Ae KD_ratio single-point/module "
                "evidence to an observed com background; not wet-lab evidence until tested."
            ),
        }
        for edge_rank, (module, champion) in enumerate(
            (
                (module, champion)
                for champion in champions[:3]
                for module in module_priorities[:4]
            ),
            start=1,
        )
    ]
    state["panel_stage"] = {
        "status": "dry_panel_draft_allowed",
        "candidate_generation_evidence_refs": primary_evidence_refs,
        "untested_sequences_evidence_tier": "model_derived",
        "wet_lab_submission_requires_human_review": True,
    }
    state.setdefault("policy_history", []).append(
        {
            "event_type": "advance_to_dry_panel",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "champion_count": len(champions),
            "module_priority_count": len(module_priorities),
        }
    )

    project["stage"] = "dry_panel_draft"
    project.setdefault("constraints", {})["no_panel_recommendation_in_startup_stage"] = False
    project.setdefault("constraints", {})["wet_lab_submission_requires_human_review"] = True
    write_yaml(project_path, project)
    write_json(state_path, state)


def _read_standardized_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            {key: ("" if value is None else str(value)) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _rank_observed_champions(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_variant: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_variant.setdefault(row["variant_id"], []).append(row)

    ranked: list[dict[str, Any]] = []
    for variant_id, variant_rows in by_variant.items():
        p_h74 = [_to_float(row.get("pH74_signal")) or 0.0 for row in variant_rows]
        p_h6 = [_to_float(row.get("pH6_signal")) or 0.0 for row in variant_rows]
        mean_p_h74 = sum(p_h74) / len(p_h74)
        mean_p_h6 = sum(p_h6) / len(p_h6)
        score = mean_p_h74 - mean_p_h6
        ranked.append(
            {
                "variant_id": variant_id,
                "score": round(score, 6),
                "mean_pH74_signal": round(mean_p_h74, 6),
                "mean_pH6_signal": round(mean_p_h6, 6),
            }
        )
    return sorted(ranked, key=lambda item: (-item["score"], item["variant_id"]))


def _rank_kd_modules(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    modules: list[dict[str, Any]] = []
    for row in rows:
        value = _to_float(row.get("KD_ratio"))
        if value is None or value <= 0:
            deviation = 0.0
        else:
            deviation = abs(value - 1.0)
        modules.append(
            {
                "system": "1E62",
                "module_id": row["variant_id"],
                "antigen_genotype": row["antigen_genotype"],
                "endpoint": row["endpoint"],
                "ae_kd_ratio": value,
                "status": "observed_ae_kd_single_point_or_module_evidence",
                "priority_score": round(deviation, 6),
            }
        )
    return sorted(modules, key=lambda item: (-item["priority_score"], item["module_id"]))


def _read_csv_records(root: Path, rel: Path, required_columns: list[str]) -> dict[str, Any]:
    path = root / rel
    if not path.is_file():
        raise ValueError(f"Missing required startup CSV: {rel}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{rel} is empty")
        _validate_header(rel, header, required_columns)
        records = [
            dict(zip(header, row, strict=False))
            for row in reader
        ]
    return {
        "path": path,
        "relative_path": str(rel),
        "header": header,
        "records": records,
        "row_count": len(records),
        "sha256": _file_sha256(path),
    }


def _validate_header(rel: Path, header: list[str], required_columns: list[str]) -> None:
    empty = [idx + 1 for idx, column in enumerate(header) if not column.strip()]
    if empty:
        raise ValueError(f"{rel} has empty CSV header columns at positions {empty}")
    duplicates = sorted(
        column for column, count in Counter(header).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"{rel} has duplicate CSV header columns: {duplicates}")
    missing = [column for column in required_columns if column not in header]
    if missing:
        raise ValueError(f"{rel} is missing required columns: {missing}")


def _validate_checksums(root: Path) -> dict[str, Any]:
    checksum_file = root / "metadata" / "checksums.sha256"
    if not checksum_file.is_file():
        raise ValueError(f"Missing checksum manifest: {checksum_file}")
    entries: list[dict[str, Any]] = []
    failures: list[str] = []
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, rel_text = line.split(maxsplit=1)
        rel = Path(rel_text.removeprefix("./"))
        path = root / rel
        if not path.is_file():
            failures.append(str(rel))
            entries.append({"path": str(rel), "expected": expected, "actual": None, "ok": False})
            continue
        actual = _file_sha256(path)
        ok = actual == expected
        if not ok:
            failures.append(str(rel))
        entries.append({"path": str(rel), "expected": expected, "actual": actual, "ok": ok})
    if failures:
        raise ValueError(f"Checksum validation failed for: {failures}")
    return {"status": "passed", "entries": entries}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _allowed_source_paths(contract: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    allowed_sources = contract.get("allowed_sources", {})
    if not isinstance(allowed_sources, dict):
        return paths
    for group in allowed_sources.values():
        if isinstance(group, dict):
            paths.extend(Path(path) for path in group.get("files", []))
    return sorted(paths, key=str)


def _table_spec(
    name: str,
    root: Path,
    role: str,
    measurement_types: list[str],
) -> dict[str, Any]:
    return {
        "name": name,
        "path": str((root / STANDARDIZED_TABLES[name]).resolve()),
        "role": role,
        "required": True,
        "allow_raw_access": False,
        "measurement_types": measurement_types,
        "provenance_columns": list(PROVENANCE_COLUMNS),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_binding_long(path: Path, records: list[dict[str, str]]) -> None:
    fieldnames = [
        "variant_id",
        "antigen_genotype",
        "pH",
        "concentration_ng_ml",
        "replicate_id",
        "elisa_signal",
        "measurement_type",
        *PROVENANCE_COLUMNS,
    ]
    rows = []
    rel = "raw/wet_lab/elisa_dilution_measurements.csv"
    for idx, record in enumerate(records, start=2):
        rows.append(
            {
                **{key: record[key] for key in fieldnames[:6]},
                "measurement_type": "ELISA_dilution_replicate",
                "source_file": rel,
                "source_row_csv": idx,
                "source_column": "elisa_signal",
                "source_cell_id": f"{rel}:R{idx}:elisa_signal",
            }
        )
    _write_csv(path, rows, fieldnames)


def _write_binding_primary(path: Path, records: list[dict[str, str]]) -> None:
    fieldnames = [
        "variant_id",
        "antigen_genotype",
        "pH6_signal",
        "pH74_signal",
        "pH_sensitive_ratio",
        "summary_ratio_raw",
        "endpoint_source",
        "endpoint_semantics_status",
        "heavy_chain_seq",
        "light_chain_seq",
        "mutations",
        *PROVENANCE_COLUMNS,
    ]
    rows = []
    rel = "raw/wet_lab/elisa_summary.csv"
    for idx, record in enumerate(records, start=2):
        for genotype in ("Ae", "B", "D1"):
            source_columns = f"{genotype}_pH6.0|{genotype}_pH7.4|{genotype}_ratio"
            rows.append(
                {
                    "variant_id": record["variant"],
                    "antigen_genotype": genotype,
                    "pH6_signal": record[f"{genotype}_pH6.0"],
                    "pH74_signal": record[f"{genotype}_pH7.4"],
                    "pH_sensitive_ratio": record[f"{genotype}_ratio"],
                    "summary_ratio_raw": record[f"{genotype}_ratio"],
                    "endpoint_source": "elisa_summary",
                    "endpoint_semantics_status": "pending_human_confirmation",
                    "heavy_chain_seq": record["H_chain"],
                    "light_chain_seq": record["L_chain"],
                    "mutations": record["mutations"],
                    "source_file": rel,
                    "source_row_csv": idx,
                    "source_column": source_columns,
                    "source_cell_id": f"{rel}:R{idx}:{source_columns}",
                }
            )
    _write_csv(path, rows, fieldnames)


def _write_ae_kd_ratio(path: Path, records: list[dict[str, str]]) -> None:
    fieldnames = [
        "variant_id",
        "antigen_genotype",
        "endpoint",
        "KD_ratio",
        "evidence_scope",
        *PROVENANCE_COLUMNS,
    ]
    rel = "raw/wet_lab/ae_kd_ratio.csv"
    rows = [
        {
            "variant_id": record["variant_id"],
            "antigen_genotype": record["antigen_genotype"],
            "endpoint": record["endpoint"],
            "KD_ratio": record["value"],
            "evidence_scope": "Ae-specific single-point/module evidence",
            "source_file": rel,
            "source_row_csv": idx,
            "source_column": "value",
            "source_cell_id": f"{rel}:R{idx}:value",
        }
        for idx, record in enumerate(records, start=2)
    ]
    _write_csv(path, rows, fieldnames)


def _write_expression_qc(path: Path, records: list[dict[str, str]]) -> None:
    fieldnames = [
        "variant_id",
        "expression_concentration_ug_ul",
        "first_well_500ng_volume_ul",
        "qc_role",
        *PROVENANCE_COLUMNS,
    ]
    rel = "raw/wet_lab/expression.csv"
    rows = [
        {
            "variant_id": record["编号"],
            "expression_concentration_ug_ul": record["浓度ug/ul"],
            "first_well_500ng_volume_ul": record["首孔500ng所需体积ul"],
            "qc_role": "feasibility_qc",
            "source_file": rel,
            "source_row_csv": idx,
            "source_column": "浓度ug/ul|首孔500ng所需体积ul",
            "source_cell_id": f"{rel}:R{idx}:expression_qc",
        }
        for idx, record in enumerate(records, start=2)
    ]
    _write_csv(path, rows, fieldnames)


def _write_variant_sequence_provenance(path: Path, records: list[dict[str, str]]) -> None:
    fieldnames = [
        "variant_id",
        "heavy_source_id",
        "heavy_variant_label",
        "heavy_construct",
        "heavy_chain_seq",
        "light_variant_label",
        "light_construct",
        "light_chain_seq",
        "evidence_role",
        *PROVENANCE_COLUMNS,
    ]
    rel = "raw/wet_lab/variant_sequences.csv"
    rows = [
        {
            **record,
            "evidence_role": "tested_variant_sequence_provenance",
            "source_file": rel,
            "source_row_csv": idx,
            "source_column": "heavy_chain_seq|light_chain_seq",
            "source_cell_id": f"{rel}:R{idx}:sequence",
        }
        for idx, record in enumerate(records, start=2)
    ]
    _write_csv(path, rows, fieldnames)


def _write_schema_inventory(path: Path, source_root: Path, audit: dict[str, Any]) -> None:
    fieldnames = [
        "source_file",
        "role",
        "evidence_tier",
        "allowed_as_experimental_evidence",
        "row_count",
        "column_count",
        "columns",
        "empty_header_count",
        "duplicate_header_count",
        "sha256",
        "status",
    ]
    source_metadata = _source_metadata(audit["startup_contract"])
    rows = []
    for table in audit["csv_tables"].values():
        rel = table["relative_path"]
        header = table["header"]
        meta = source_metadata.get(rel, {})
        rows.append(
            {
                "source_file": rel,
                "role": meta.get("role", "startup_csv"),
                "evidence_tier": meta.get("evidence_tier", "primary_wet_lab"),
                "allowed_as_experimental_evidence": meta.get("allowed_as_experimental_evidence", True),
                "row_count": table["row_count"],
                "column_count": len(header),
                "columns": ";".join(header),
                "empty_header_count": sum(1 for column in header if not column.strip()),
                "duplicate_header_count": sum(
                    count - 1 for count in Counter(header).values() if count > 1
                ),
                "sha256": table["sha256"],
                "status": "schema_audit_passed",
            }
        )
    for rel in _allowed_source_paths(audit["startup_contract"]):
        rel_text = str(rel)
        if rel_text in {row["source_file"] for row in rows}:
            continue
        source = source_root / rel
        meta = source_metadata.get(rel_text, {})
        rows.append(
            {
                "source_file": rel_text,
                "role": meta.get("role", "reference"),
                "evidence_tier": meta.get("evidence_tier", "reference"),
                "allowed_as_experimental_evidence": meta.get("allowed_as_experimental_evidence", False),
                "row_count": "",
                "column_count": "",
                "columns": "",
                "empty_header_count": "",
                "duplicate_header_count": "",
                "sha256": _file_sha256(source) if source.is_file() else "",
                "status": "reference_recorded",
            }
        )
    _write_csv(path, rows, fieldnames)


def _source_metadata(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for role, group in contract.get("allowed_sources", {}).items():
        if not isinstance(group, dict):
            continue
        for file_name in group.get("files", []):
            metadata[str(file_name)] = {
                "role": role,
                "evidence_tier": group.get("evidence_tier", ""),
                "allowed_as_experimental_evidence": bool(group.get("allowed_as_experimental_evidence", False)),
            }
    return metadata


def _validation_summary(source_root: Path, audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "project_id": PROJECT_ID,
        "project_family": PROJECT_FAMILY,
        "startup_source_root": str(source_root),
        "checksum_status": audit["checksums"]["status"],
        "csv_header_status": "passed",
        "id_alignment": audit["id_alignment"],
        "dilution_dimensions": audit["dilution_dimensions"],
        "kd_ratio": audit["kd_ratio"],
        "endpoint_semantics": {
            "primary_binding_source": "raw/wet_lab/elisa_summary.csv",
            "replicate_provenance_source": "raw/wet_lab/elisa_dilution_measurements.csv",
            "status": "pending_human_confirmation",
        },
        "warnings": [
            "endpoint semantics pending confirmation",
            "unbounded design-space exploration registered only; no untested sequence is evidence",
        ],
    }


def _evidence_cards(audit: dict[str, Any]) -> list[dict[str, Any]]:
    row_counts = {
        "binding_long": audit["dilution_dimensions"]["observed_rows"],
        "binding_primary": len(audit["csv_tables"]["elisa_summary"]["records"]) * 3,
        "ae_kd_ratio": audit["kd_ratio"]["row_count"],
        "expression_qc": len(audit["csv_tables"]["expression"]["records"]),
        "variant_sequence_provenance": len(audit["csv_tables"]["variant_sequences"]["records"]),
    }
    return [
        {
            "evidence_id": "onee62_startup_schema_audit",
            "claim": "The 1E62 startup package passed checksum, explicit-header, and ID-alignment checks.",
            "evidence_tier": "descriptive",
            "source_tables": ["schema_inventory", "validation_summary"],
            "effect": row_counts,
            "limitations": ["Audit does not resolve endpoint semantics."],
            "actionability": "advance_to_endpoint_review",
            "tags": ["schema_audit", "startup"],
        },
        {
            "evidence_id": "onee62_elisa_primary_summary",
            "claim": "ELISA summary provides matched pH 6.0 and pH 7.4 evidence for Ae, B, and D1 across tested com variants.",
            "evidence_tier": "primary_matched",
            "source_tables": ["standardized_binding_primary_table"],
            "effect": {"variant_count": len(audit["id_alignment"]["variant_ids"]), "genotypes": audit["dilution_dimensions"]["antigen_genotypes"]},
            "limitations": ["elisa_summary is treated as primary pending human endpoint-semantics confirmation."],
            "actionability": "needs_endpoint_confirmation",
            "tags": ["ELISA", "primary_summary"],
        },
        {
            "evidence_id": "onee62_elisa_replicate_provenance",
            "claim": "Dilution ELISA records preserve replicate-level provenance for each measured variant, genotype, pH, and concentration.",
            "evidence_tier": "descriptive",
            "source_tables": ["standardized_binding_long_table"],
            "effect": audit["dilution_dimensions"],
            "limitations": ["Replicate table is provenance until endpoint aggregation rules are confirmed."],
            "actionability": "support_audit",
            "tags": ["ELISA", "replicate_provenance"],
        },
        {
            "evidence_id": "onee62_ae_kd_ratio_single_point_evidence",
            "claim": "Ae KD_ratio records are separate single-point/module evidence and do not directly identify com combination behavior.",
            "evidence_tier": "primary_matched",
            "source_tables": ["ae_kd_ratio_table"],
            "effect": audit["kd_ratio"],
            "limitations": ["KD_ratio variant IDs do not overlap tested com combination IDs."],
            "actionability": "mechanism_prior_only",
            "tags": ["KD_ratio", "Ae", "scope_guard"],
        },
        {
            "evidence_id": "onee62_expression_qc",
            "claim": "Expression data provide feasibility and QC evidence for tested com variants.",
            "evidence_tier": "descriptive",
            "source_tables": ["standardized_auxiliary_table"],
            "effect": {"variant_count": len(audit["id_alignment"]["variant_ids"])},
            "limitations": ["Expression is not direct binding efficacy evidence."],
            "actionability": "feasibility_guardrail",
            "tags": ["expression", "QC"],
        },
        {
            "evidence_id": "onee62_sequence_and_structure_reference_guard",
            "claim": "Sequence and optional structure inputs are references, not experimental binding evidence.",
            "evidence_tier": "descriptive",
            "source_tables": ["variant_sequence_provenance", "schema_inventory"],
            "effect": {"tested_variant_sequence_count": len(audit["id_alignment"]["variant_ids"])},
            "limitations": ["Unbounded future sequence exploration must be labeled model-derived until tested."],
            "actionability": "provenance_guardrail",
            "tags": ["sequence", "structure", "unsupported_claim"],
        },
    ]


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _mutation_count(mutations: str) -> int:
    return len([item for item in mutations.split(";") if item.strip()])


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "PROJECT_FAMILY",
    "PROJECT_ID",
    "audit_1e62_startup_package",
    "start_1e62_project",
    "write_1e62_project_artifacts",
    "write_1e62_standardized_tables",
    "write_1e62_state_artifacts",
]
