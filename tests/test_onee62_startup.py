from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from design_scientist.cli import main
from design_scientist.io import read_json
from design_scientist.onee62_startup import (
    advance_1e62_project,
    audit_1e62_startup_package,
    finalize_1e62_result,
    review_1e62_panel,
    run_1e62_literature_stage,
    start_1e62_project,
)
from design_scientist.validators import validate_project


def test_start_1e62_project_creates_standardized_state_without_panel(tmp_path: Path) -> None:
    startup = _write_startup_fixture(tmp_path / "startup")
    project = tmp_path / "projects" / "1e62_ph_design"

    result = start_1e62_project(startup, project)
    report = validate_project(project, write_report=False)

    assert result["project_dir"] == project.resolve()
    assert report.valid
    assert not list((project / "runs").glob("*/panel_recommendation.csv"))

    assert _csv_row_count(project / "standardized" / "binding_long.csv") == 2160
    assert _csv_row_count(project / "standardized" / "binding_primary.csv") == 60
    assert _csv_row_count(project / "standardized" / "ae_kd_ratio.csv") == 52
    assert _csv_row_count(project / "standardized" / "expression_qc.csv") == 20
    assert _csv_row_count(project / "standardized" / "variant_sequence_provenance.csv") == 20

    state = read_json(project / "state" / "design_state.json")
    assert state["project_family"] == "1e62_startup"
    assert state["system_roles"] == {"1E62": "target_system_design"}
    assert state["design_space"]["sequence_exploration"] == "unbounded_full_length"
    assert state["endpoint_policy"]["status"] == "pending_human_confirmation"
    assert len(state["tested_variants"]) == 20
    assert len(state["module_status"]) == 52

    evidence = read_json(project / "state" / "evidence_cards.json")
    assert any(card["evidence_id"] == "onee62_ae_kd_ratio_single_point_evidence" for card in evidence)


def test_1e62_startup_audit_rejects_empty_headers(tmp_path: Path) -> None:
    startup = _write_startup_fixture(tmp_path / "startup")
    _rewrite_first_header(startup / "raw" / "wet_lab" / "variant_sequences.csv", "")
    _write_checksums(startup)

    with pytest.raises(ValueError, match="empty CSV header"):
        audit_1e62_startup_package(startup)


def test_1e62_startup_audit_rejects_duplicate_headers(tmp_path: Path) -> None:
    startup = _write_startup_fixture(tmp_path / "startup")
    _rewrite_first_header(startup / "raw" / "wet_lab" / "variant_sequences.csv", "heavy_source_id")
    _write_checksums(startup)

    with pytest.raises(ValueError, match="duplicate CSV header"):
        audit_1e62_startup_package(startup)


def test_1e62_startup_audit_records_dimensions_alignment_and_kd_scope(tmp_path: Path) -> None:
    startup = _write_startup_fixture(tmp_path / "startup")

    audit = audit_1e62_startup_package(startup)

    assert audit["id_alignment"]["com_variant_tables_aligned"] is True
    assert audit["dilution_dimensions"]["observed_rows"] == 2160
    assert audit["dilution_dimensions"]["expected_rows"] == 2160
    assert audit["dilution_dimensions"]["matches_expected"] is True
    assert audit["kd_ratio"]["row_count"] == 52
    assert audit["kd_ratio"]["overlaps_com_variants"] is False
    assert audit["kd_ratio"]["overlap_with_com_variants"] == []


def test_start_1e62_cli_creates_project(tmp_path: Path) -> None:
    startup = _write_startup_fixture(tmp_path / "startup")
    project = tmp_path / "projects" / "1e62_ph_design"

    assert main(["start-1e62", str(startup), str(project)]) == 0

    assert (project / "project.yaml").exists()
    assert (project / "standardized" / "validation_summary.json").exists()
    assert (project / "state" / "design_state.json").exists()
    assert (project / "state" / "validation_report.json").exists()


def test_advance_1e62_project_runs_dry_panel_after_stage_bridge(tmp_path: Path) -> None:
    startup = _write_startup_fixture(tmp_path / "startup")
    project = tmp_path / "projects" / "1e62_ph_design"
    start_1e62_project(startup, project)

    run_id = advance_1e62_project(project, budget=24, offline_fixtures=True)
    report = validate_project(project, run_id=run_id, write_report=False)

    run_dir = project / "runs" / run_id
    assert report.valid
    assert (run_dir / "candidate_pool.csv").exists()
    assert (run_dir / "panel_recommendation.csv").exists()
    assert _csv_row_count(run_dir / "candidate_pool.csv") > 0
    assert _csv_row_count(run_dir / "panel_recommendation.csv") > 0

    state = read_json(project / "state" / "design_state.json")
    assert state["panel_stage"]["status"] == "dry_panel_draft_allowed"
    assert state["known_champions"]
    assert state["unresolved_edges"]


def test_review_1e62_panel_writes_human_review_gate(tmp_path: Path) -> None:
    startup = _write_startup_fixture(tmp_path / "startup")
    project = tmp_path / "projects" / "1e62_ph_design"
    start_1e62_project(startup, project)
    run_id = advance_1e62_project(project, budget=24, offline_fixtures=True)

    review = review_1e62_panel(project, run_id=run_id)
    run_dir = project / "runs" / run_id

    assert review["status"] == "human_review_required"
    assert review["not_wet_lab_final"] is True
    assert review["panel_summary"]["panel_count"] > 0
    assert all(item["passed"] for item in review["checklist"])
    assert (run_dir / "panel_review.json").exists()
    assert (run_dir / "panel_review_report.md").exists()


def test_finalize_1e62_result_writes_complete_result_package(tmp_path: Path) -> None:
    startup = _write_startup_fixture(tmp_path / "startup")
    project = tmp_path / "projects" / "1e62_ph_design"
    start_1e62_project(startup, project)
    run_id = advance_1e62_project(project, budget=24, offline_fixtures=True)

    result = finalize_1e62_result(project, run_id=run_id)
    report = validate_project(project, run_id=run_id, write_report=False)
    state = read_json(project / "state" / "design_state.json")

    assert result["status"] == "computational_result_complete"
    assert result["wet_lab_final"] is False
    assert result["panel_summary"]["panel_count"] > 0
    assert result["review"]["all_checks_passed"] is True
    assert report.valid
    assert state["latest_result"]["run_id"] == run_id
    assert (project / "runs" / run_id / "final_result.json").exists()
    assert (project / "runs" / run_id / "final_result_report.md").exists()
    assert (project / "runs" / run_id / "assay_plan_draft.csv").exists()


def test_advance_1e62_cli_runs_dry_panel(tmp_path: Path) -> None:
    startup = _write_startup_fixture(tmp_path / "startup")
    project = tmp_path / "projects" / "1e62_ph_design"

    assert main(["start-1e62", str(startup), str(project)]) == 0
    assert main(["advance-1e62", str(project), "--budget", "24", "--offline-fixtures"]) == 0

    run_dirs = [path for path in (project / "runs").iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "panel_recommendation.csv").exists()


def test_review_1e62_panel_cli_writes_latest_run_review(tmp_path: Path) -> None:
    startup = _write_startup_fixture(tmp_path / "startup")
    project = tmp_path / "projects" / "1e62_ph_design"

    assert main(["start-1e62", str(startup), str(project)]) == 0
    assert main(["advance-1e62", str(project), "--budget", "24", "--offline-fixtures"]) == 0
    assert main(["review-1e62-panel", str(project)]) == 0

    run_dirs = [path for path in (project / "runs").iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "panel_review.json").exists()
    assert (run_dirs[0] / "panel_review_report.md").exists()


def test_finalize_1e62_result_cli_writes_latest_run_result(tmp_path: Path) -> None:
    startup = _write_startup_fixture(tmp_path / "startup")
    project = tmp_path / "projects" / "1e62_ph_design"

    assert main(["start-1e62", str(startup), str(project)]) == 0
    assert main(["advance-1e62", str(project), "--budget", "24", "--offline-fixtures"]) == 0
    assert main(["finalize-1e62-result", str(project)]) == 0

    run_dirs = [path for path in (project / "runs").iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "final_result.json").exists()
    assert (run_dirs[0] / "final_result_report.md").exists()
    assert (run_dirs[0] / "assay_plan_draft.csv").exists()


def test_literature_1e62_stage_writes_v3_artifacts_before_panel(tmp_path: Path) -> None:
    startup = _write_startup_fixture(tmp_path / "startup")
    project = tmp_path / "projects" / "1e62_ph_design"
    start_1e62_project(startup, project)

    stage = run_1e62_literature_stage(project, max_papers=5, offline_fixtures=True)

    assert stage["status"] == "completed"
    assert (project / "framework" / "literature_queries.yaml").exists()
    assert (project / "framework" / "paper_cards.json").exists()
    assert (project / "framework" / "literature_search_trace.json").exists()
    assert (project / "framework" / "literature_corpus.jsonl").exists()
    assert (project / "framework" / "literature_reading_trace.json").exists()
    assert (project / "framework" / "mechanism_cards.json").exists()
    assert (project / "framework" / "mechanism_library.json").exists()
    assert (project / "framework" / "mechanism_gap_matrix.csv").exists()
    assert (project / "framework" / "1e62_literature_report.md").exists()
    assert not list((project / "runs").glob("*/panel_recommendation.csv"))


def test_literature_1e62_cli_runs_stage(tmp_path: Path) -> None:
    startup = _write_startup_fixture(tmp_path / "startup")
    project = tmp_path / "projects" / "1e62_ph_design"
    assert main(["start-1e62", str(startup), str(project)]) == 0

    assert main(["literature-1e62", str(project), "--max-papers", "5", "--offline-fixtures"]) == 0

    assert (project / "framework" / "1e62_literature_stage.json").exists()
    assert (project / "framework" / "mechanism_cards.json").exists()


def _write_startup_fixture(root: Path) -> Path:
    (root / "metadata").mkdir(parents=True)
    (root / "raw" / "wet_lab").mkdir(parents=True)
    (root / "raw" / "base_sequences").mkdir(parents=True)
    (root / "raw" / "base_structures_optional").mkdir(parents=True)

    (root / "README.md").write_text("1E62 startup fixture\n", encoding="utf-8")
    (root / "project.yaml").write_text(
        "project_id: 1E62_pH_sensitive_design_startup\nsystem: 1E62\n",
        encoding="utf-8",
    )
    (root / "estimands.yaml").write_text(
        "estimand_set_id: 1E62_pH_sensitive_binding\n",
        encoding="utf-8",
    )
    (root / "data_contract.yaml").write_text(
        "\n".join(
            [
                "contract_id: 1E62_startup_data_contract",
                "allowed_sources:",
                "  base_sequences:",
                "    evidence_tier: sequence_reference",
                "    allowed_as_experimental_evidence: false",
                "    files:",
                "      - raw/base_sequences/antigen_genotypes.fasta",
                "      - raw/base_sequences/heavy.fasta",
                "      - raw/base_sequences/light.fasta",
                "      - raw/base_sequences/original.fasta",
                "  optional_base_structures:",
                "    evidence_tier: structure_reference_optional",
                "    allowed_as_experimental_evidence: false",
                "    files:",
                "      - raw/base_structures_optional/ab_wt.pdb",
                "      - raw/base_structures_optional/AF3-1E62-AeS-1.pdb",
                "      - raw/base_structures_optional/AF3-1E62-BaS-1.pdb",
                "      - raw/base_structures_optional/AF3-1E62-CeS-1.pdb",
                "      - raw/base_structures_optional/AF3-1E62-D1S-1.pdb",
                "  wet_lab:",
                "    evidence_tier: primary_wet_lab",
                "    allowed_as_experimental_evidence: true",
                "    files:",
                "      - raw/wet_lab/ae_kd_ratio.csv",
                "      - raw/wet_lab/elisa_dilution_measurements.csv",
                "      - raw/wet_lab/elisa_summary.csv",
                "      - raw/wet_lab/expression.csv",
                "      - raw/wet_lab/variant_sequences.csv",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (root / "metadata" / "source_manifest.tsv").write_text(
        "local_path\trole\tallowed_as_experimental_evidence\tnotes\n",
        encoding="utf-8",
    )
    for rel in [
        "raw/base_sequences/antigen_genotypes.fasta",
        "raw/base_sequences/heavy.fasta",
        "raw/base_sequences/light.fasta",
        "raw/base_sequences/original.fasta",
    ]:
        (root / rel).write_text(">seq\nAAAA\n", encoding="utf-8")
    for rel in [
        "raw/base_structures_optional/ab_wt.pdb",
        "raw/base_structures_optional/AF3-1E62-AeS-1.pdb",
        "raw/base_structures_optional/AF3-1E62-BaS-1.pdb",
        "raw/base_structures_optional/AF3-1E62-CeS-1.pdb",
        "raw/base_structures_optional/AF3-1E62-D1S-1.pdb",
    ]:
        (root / rel).write_text("HEADER 1E62 fixture\n", encoding="utf-8")

    variants = [f"com{i}" for i in range(1, 21)]
    genotypes = ["Ae", "B", "D1"]
    p_h_values = ["6.0", "7.4"]
    concentrations = ["500", "100", "20", "4", "0.8", "0.16"]
    replicates = ["1", "2", "3"]

    _write_csv(
        root / "raw" / "wet_lab" / "elisa_dilution_measurements.csv",
        ["variant_id", "antigen_genotype", "pH", "concentration_ng_ml", "replicate_id", "elisa_signal"],
        [
            {
                "variant_id": variant,
                "antigen_genotype": genotype,
                "pH": p_h,
                "concentration_ng_ml": concentration,
                "replicate_id": replicate,
                "elisa_signal": f"{0.001 * (idx + 1):.4f}",
            }
            for idx, (variant, genotype, p_h, concentration, replicate) in enumerate(
                (variant, genotype, p_h, concentration, replicate)
                for variant in variants
                for genotype in genotypes
                for p_h in p_h_values
                for concentration in concentrations
                for replicate in replicates
            )
        ],
    )
    _write_csv(
        root / "raw" / "wet_lab" / "elisa_summary.csv",
        [
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
        [
            {
                "variant": variant,
                "Ae_pH6.0": "0.01",
                "Ae_pH7.4": "0.02",
                "B_pH6.0": "0.01",
                "B_pH7.4": "0.02",
                "D1_pH6.0": "0.01",
                "D1_pH7.4": "0.02",
                "Ae_ratio": "0.5",
                "B_ratio": "0.5",
                "D1_ratio": "0.5",
                "H_chain": "AAAA",
                "L_chain": "BBBB",
                "mutations": "HA1B; LB2C",
            }
            for variant in variants
        ],
    )
    _write_csv(
        root / "raw" / "wet_lab" / "expression.csv",
        ["编号", "浓度ug/ul", "首孔500ng所需体积ul"],
        [
            {"编号": variant, "浓度ug/ul": "1.0", "首孔500ng所需体积ul": "500.0"}
            for variant in variants
        ],
    )
    _write_csv(
        root / "raw" / "wet_lab" / "variant_sequences.csv",
        [
            "variant_id",
            "heavy_source_id",
            "heavy_variant_label",
            "heavy_construct",
            "heavy_chain_seq",
            "light_variant_label",
            "light_construct",
            "light_chain_seq",
        ],
        [
            {
                "variant_id": variant,
                "heavy_source_id": f"mut_{idx:06d}",
                "heavy_variant_label": f"Hmt{idx}",
                "heavy_construct": "1E62vHm3",
                "heavy_chain_seq": "AAAA",
                "light_variant_label": f"Lmt{idx}",
                "light_construct": "1E62L",
                "light_chain_seq": "BBBB",
            }
            for idx, variant in enumerate(variants, start=1)
        ],
    )
    _write_csv(
        root / "raw" / "wet_lab" / "ae_kd_ratio.csv",
        ["variant_id", "antigen_genotype", "endpoint", "value"],
        [
            {
                "variant_id": f"KD{i:03d}H",
                "antigen_genotype": "Ae",
                "endpoint": "KD_ratio",
                "value": f"{1.0 + i / 100:.3f}",
            }
            for i in range(1, 53)
        ],
    )
    _write_checksums(root)
    return root


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_checksums(root: Path) -> None:
    checksum_path = root / "metadata" / "checksums.sha256"
    lines = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p != checksum_path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rel = path.relative_to(root).as_posix()
        lines.append(f"{digest}  ./{rel}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rewrite_first_header(path: Path, value: str) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    rows[0][0] = value
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def _csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)
