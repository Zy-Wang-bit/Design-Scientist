from __future__ import annotations

import csv
import json
from pathlib import Path

from design_scientist.external_antibody_benchmark import (
    run_external_antibody_benchmark,
    run_external_ph_switch_benchmark,
)


def _write_fixture(path: Path) -> None:
    rows = [
        {
            "Antigen": "fixture",
            "heavy": "QVQLVESG",
            "light": "DIQMTQSP",
            "Round": "alpha",
            "Chain mutated": "WT",
            "Design": "WT",
            "fitness": "0.50",
        },
        {
            "Antigen": "fixture",
            "heavy": "QVHLVESG",
            "light": "DIQMTQSP",
            "Round": "alpha",
            "Chain mutated": "HC",
            "Design": "Q3H",
            "fitness": "0.71",
        },
        {
            "Antigen": "fixture",
            "heavy": "QVQLVESG",
            "light": "DIHMTQSP",
            "Round": "alpha",
            "Chain mutated": "LC",
            "Design": "Q3H",
            "fitness": "0.66",
        },
        {
            "Antigen": "fixture",
            "heavy": "QVHLVESG",
            "light": "DIHMTQSP",
            "Round": "beta",
            "Chain mutated": "HC + LC",
            "Design": "Q3H/Q3H",
            "fitness": "0.91",
        },
        {
            "Antigen": "fixture",
            "heavy": "QVQLVESG",
            "light": "DIQMTYSP",
            "Round": "beta",
            "Chain mutated": "LC",
            "Design": "Q6Y",
            "fitness": "0.43",
        },
        {
            "Antigen": "fixture",
            "heavy": "QVKLVEYG",
            "light": "DIQMTQSP",
            "Round": "beta",
            "Chain mutated": "HC",
            "Design": "Q3K/S7Y",
            "fitness": "0.57",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_external_antibody_benchmark_writes_independent_replay_artifacts(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "flab_fixture.csv"
    _write_fixture(fixture)

    result = run_external_antibody_benchmark(
        tmp_path,
        run_id="external_unit",
        datasets=(
            {
                "dataset_id": "flab_fixture",
                "source": "FLAb_fixture",
                "path": str(fixture),
                "split_column": "Round",
                "train_values": ["alpha"],
                "test_values": ["beta"],
            },
        ),
        seeds=(0, 1),
        budget=2,
    )

    assert result["status"] == "completed"
    assert result["dataset_count"] == 1
    assert Path(result["benchmark_results_path"]).exists()
    assert Path(result["summary_results_path"]).exists()
    assert Path(result["config_path"]).exists()
    assert Path(result["source_trace_path"]).exists()

    rows = _csv_rows(result["benchmark_results_path"])
    assert {
        "pcig_external_replay",
        "observed_edit_effect_transfer",
        "fewest_edits",
        "random_measured",
        "oracle_top_measured",
    } <= {row["method"] for row in rows}
    assert {row["dataset_id"] for row in rows} == {"flab_fixture"}
    pcig_rows = [row for row in rows if row["method"] == "pcig_external_replay"]
    assert pcig_rows
    assert all(row["selector_uses_oracle_truth"] == "False" for row in pcig_rows)
    assert all(float(row["best_selected_fitness"]) >= 0.0 for row in pcig_rows)

    config = json.loads(Path(result["config_path"]).read_text(encoding="utf-8"))
    assert config["benchmark"] == "external_antibody_replay"
    assert config["data_sources"][0]["source"] == "FLAb_fixture"
    assert config["leakage_controls"]["selector_inputs_exclude_external_fitness"] is True
    assert config["evidence_scope"] == "external_retrospective_antibody_replay"

    summary_rows = _csv_rows(result["summary_results_path"])
    pcig_summary = next(row for row in summary_rows if row["method"] == "pcig_external_replay")
    assert pcig_summary["dataset_count"] == "1"
    assert float(pcig_summary["mean_best_selected_fitness"]) > 0.0


def test_external_antibody_benchmark_rejects_unusable_dataset(tmp_path: Path) -> None:
    fixture = tmp_path / "bad.csv"
    fixture.write_text("heavy,light,fitness\nAAAA,BBBB,\n", encoding="utf-8")

    result = run_external_antibody_benchmark(
        tmp_path,
        run_id="external_bad",
        datasets=({"dataset_id": "bad", "path": str(fixture), "source": "fixture"},),
        seeds=(0,),
    )

    assert result["status"] == "failed"
    assert result["dataset_count"] == 0
    rows = _csv_rows(result["benchmark_results_path"])
    assert rows[0]["status"] == "failed"


def test_external_ph_switch_benchmark_writes_public_literature_table_replay(
    tmp_path: Path,
) -> None:
    result = run_external_ph_switch_benchmark(tmp_path, run_id="ph_external", budget=2)

    assert result["status"] == "completed"
    assert result["dataset_count"] >= 5
    assert Path(result["benchmark_results_path"]).exists()
    assert Path(result["summary_results_path"]).exists()
    assert Path(result["literature_transfer_model_path"]).exists()
    assert Path(result["curated_records_path"]).exists()
    assert Path(result["variant_replay_results_path"]).exists()
    assert Path(result["variant_replay_summary_path"]).exists()

    rows = _csv_rows(result["benchmark_results_path"])
    assert {
        "ph_switch_residue_prior",
        "transition_context_prior",
        "leave_one_study_transition_calibration",
        "histidine_count",
        "ionizable_count",
        "fewest_edits",
        "parent_reference",
        "oracle_top_pH_ratio",
    } <= {row["method"] for row in rows}
    prior = next(row for row in rows if row["method"] == "ph_switch_residue_prior")
    transition = next(row for row in rows if row["method"] == "transition_context_prior")
    parent = next(row for row in rows if row["method"] == "parent_reference")
    assert prior["selector_uses_oracle_truth"] == "False"
    assert transition["selector_uses_oracle_truth"] == "False"
    assert float(prior["best_selected_ph_ratio"]) > float(parent["best_selected_ph_ratio"])
    assert float(transition["best_selected_ph_ratio"]) > float(parent["best_selected_ph_ratio"])
    transfer_rows = [
        row for row in rows if row["method"] == "leave_one_study_transition_calibration"
    ]
    assert transfer_rows
    assert all(row["selector_uses_oracle_truth"] == "False" for row in transfer_rows)

    curated = _csv_rows(result["curated_records_path"])
    assert any(row["dataset_id"] == "adalimumab_tnf_release_2015" for row in curated)
    assert any(row["dataset_id"] == "her2_bh1_fab_release_2019" for row in curated)
    adalimumab_psv2 = next(
        row
        for row in curated
        if row["dataset_id"] == "adalimumab_tnf_release_2015"
        and row["variant_id"] == "PSV#2"
    )
    assert adalimumab_psv2["mutation_signature"] == "S100bH;S100cH;Q89H;R90H;N92H"
    assert adalimumab_psv2["selectivity_direction"] == "KD_pH6.0_over_KD_pH7.4_release"
    assert float(adalimumab_psv2["ph74_over_ph60_kd_ratio"]) == 785.0

    trace = json.loads(Path(result["source_trace_path"]).read_text(encoding="utf-8"))
    assert any(
        source["dataset_id"] == "adalimumab_tnf_release_2015"
        and source["status"] == "curated_public_table"
        for source in trace["sources"]
    )
    her2_transition = next(
        row
        for row in rows
        if row["dataset_id"] == "her2_bh1_fab_release_2019"
        and row["method"] == "transition_context_prior"
    )
    her2_parent = next(
        row
        for row in rows
        if row["dataset_id"] == "her2_bh1_fab_release_2019"
        and row["method"] == "parent_reference"
    )
    assert float(her2_transition["best_selected_ph_ratio"]) < float(her2_parent["best_selected_ph_ratio"])

    claims = json.loads(Path(result["claims_path"]).read_text(encoding="utf-8"))
    assert "validated 1E62 pH 6.0 dissociation" in claims["unsupported_scope"]
    config = json.loads(Path(result["config_path"]).read_text(encoding="utf-8"))
    assert config["leakage_controls"]["selector_inputs_exclude_reported_pH_ratio"] is True
    assert "selectivity_direction" in config["interpretation_note"]
    model = json.loads(Path(result["literature_transfer_model_path"]).read_text(encoding="utf-8"))
    assert model["method"] == "leave_one_study_transition_calibration"
    assert model["selector_inputs_exclude_heldout_reported_pH_ratio"] is True
    assert model["heldout_models"]

    variant_rows = _csv_rows(result["variant_replay_results_path"])
    assert len(variant_rows) == len(curated)
    assert {
        "predicted_score",
        "observed_normalized_log_ratio",
        "actual_top_tertile",
        "model_training_row_count",
    } <= set(variant_rows[0])
    assert {row["actual_top_tertile"] for row in variant_rows} <= {"True", "False"}
    assert all(int(row["model_training_row_count"]) == len(curated) - 1 for row in variant_rows)
    variant_summary = json.loads(Path(result["variant_replay_summary_path"]).read_text(encoding="utf-8"))
    assert variant_summary["method"] == "leave_one_variant_transition_calibration"
    assert variant_summary["selector_inputs_exclude_heldout_reported_pH_ratio"] is True
    assert variant_summary["variant_count"] == len(curated)
    assert variant_summary["dataset_count"] >= 5
    assert "predicted_vs_observed_normalized_log_ratio_pearson" in variant_summary
