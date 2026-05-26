from __future__ import annotations

import csv
from pathlib import Path

import pytest

from design_scientist.project_replay import (
    ProjectReplayDataset,
    dataset_is_available,
    load_project_replay_dataset,
)


OBSERVATION_COLUMNS = [
    "observation_id",
    "source_file",
    "source_row",
    "id_namespace",
    "variant_id",
    "antigen_genotype",
    "endpoint",
    "value",
    "value_status",
    "unit",
    "pH",
    "concentration_ng_ml",
    "replicate_id",
    "evidence_role",
]


def write_project_fixture(root: Path) -> Path:
    standardized = root / "standardized"
    standardized.mkdir(parents=True)

    variants = [
        ("com1", "HA10G; LB20C", 1.20, 0.18, 1.05, 0.28, 3.0),
        ("com2", "HA10G; LC30D", 0.88, 0.22, 0.74, 0.20, 2.0),
        ("com3", "HD40E", 0.42, 0.30, 0.34, 0.28, 1.5),
        ("com4", "LE50F", 1.10, 0.92, 0.95, 0.82, 3.2),
        ("com5", "HA10G; LB20C; LF60Y", 0.95, 0.12, 0.80, 0.15, 0.25),
    ]

    observation_rows: list[dict[str, object]] = []
    source_row = 1
    for variant_id, _mutations, ae74, ae6, b74, b6, expression in variants:
        for genotype, ph74, ph6 in [("Ae", ae74, ae6), ("B", b74, b6)]:
            for ph, value in [(7.4, ph74), (6.0, ph6)]:
                observation_rows.append(
                    {
                        "observation_id": f"summary:{variant_id}:{genotype}:{ph}",
                        "source_file": "raw/wet_lab/elisa_summary.csv",
                        "source_row": source_row,
                        "id_namespace": "com",
                        "variant_id": variant_id,
                        "antigen_genotype": genotype,
                        "endpoint": "summary_elisa_signal",
                        "value": value,
                        "value_status": "observed",
                        "unit": "summary_value",
                        "pH": ph,
                        "concentration_ng_ml": "",
                        "replicate_id": "",
                        "evidence_role": "derived_summary",
                    }
                )
            ratio = ph6 / ph74
            observation_rows.append(
                {
                    "observation_id": f"summary-ratio:{variant_id}:{genotype}",
                    "source_file": "raw/wet_lab/elisa_summary.csv",
                    "source_row": source_row,
                    "id_namespace": "com",
                    "variant_id": variant_id,
                    "antigen_genotype": genotype,
                    "endpoint": "summary_pH74_over_pH60_ratio",
                    "value": ratio,
                    "value_status": "observed",
                    "unit": "summary_value",
                    "pH": "",
                    "concentration_ng_ml": "",
                    "replicate_id": "",
                    "evidence_role": "derived_summary",
                }
            )
            for ph, value in [(7.4, ph74 * 0.9), (6.0, ph6 * 1.1)]:
                observation_rows.append(
                    {
                        "observation_id": f"dilution:{variant_id}:{genotype}:{ph}",
                        "source_file": "raw/wet_lab/elisa_dilution_measurements.csv",
                        "source_row": source_row,
                        "id_namespace": "com",
                        "variant_id": variant_id,
                        "antigen_genotype": genotype,
                        "endpoint": "elisa_signal",
                        "value": value,
                        "value_status": "observed",
                        "unit": "signal",
                        "pH": ph,
                        "concentration_ng_ml": 500,
                        "replicate_id": 1,
                        "evidence_role": "primary_observation",
                    }
                )
        observation_rows.append(
            {
                "observation_id": f"expression:{variant_id}",
                "source_file": "raw/wet_lab/expression.csv",
                "source_row": source_row,
                "id_namespace": "com",
                "variant_id": variant_id,
                "antigen_genotype": "",
                "endpoint": "expression_concentration",
                "value": expression,
                "value_status": "observed",
                "unit": "ug/ul",
                "pH": "",
                "concentration_ng_ml": "",
                "replicate_id": "",
                "evidence_role": "qc_observation",
            }
        )
        source_row += 1

    observation_rows.append(
        {
            "observation_id": "kd:KD_ONLY_A",
            "source_file": "raw/wet_lab/ae_kd_ratio.csv",
            "source_row": 99,
            "id_namespace": "kd",
            "variant_id": "KD_ONLY_A",
            "antigen_genotype": "Ae",
            "endpoint": "KD_ratio",
            "value": 1.75,
            "value_status": "observed",
            "unit": "",
            "pH": "",
            "concentration_ng_ml": "",
            "replicate_id": "",
            "evidence_role": "primary_observation",
        }
    )

    with (standardized / "observations_long.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OBSERVATION_COLUMNS)
        writer.writeheader()
        writer.writerows(observation_rows)

    with (standardized / "variant_sequences.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant_id",
                "heavy_chain_seq",
                "light_chain_seq",
                "mutations",
                "source_file",
                "source_row",
            ],
        )
        writer.writeheader()
        for index, (variant_id, mutations, *_rest) in enumerate(variants, start=1):
            writer.writerow(
                {
                    "variant_id": variant_id,
                    "heavy_chain_seq": f"EVQLVESGGG{index}AST",
                    "light_chain_seq": f"DIVMSQSPSS{index}QQ",
                    "mutations": mutations,
                    "source_file": "raw/wet_lab/variant_sequences.csv",
                    "source_row": index,
                }
            )

    (root / "data_contract.yaml").write_text(
        """
contract_id: fixture_contract
allowed_sources:
  wet_lab:
    allowed_as_experimental_evidence: true
    files:
      - raw/wet_lab/elisa_summary.csv
      - raw/wet_lab/elisa_dilution_measurements.csv
      - raw/wet_lab/expression.csv
      - raw/wet_lab/variant_sequences.csv
      - raw/wet_lab/ae_kd_ratio.csv
validation_rules:
  - Keep pH measurements paired by variant and genotype.
""".strip(),
        encoding="utf-8",
    )
    (root / "estimands.yaml").write_text(
        """
estimand_set_id: fixture_estimands
primary_goal: retained_pH74_and_reduced_pH60_binding
primary_endpoints:
  - name: elisa_signal_pH6
  - name: elisa_signal_pH74
secondary_endpoints:
  - name: expression_feasibility
constraints:
  - pH7.4 binding must not be optimized away.
""".strip(),
        encoding="utf-8",
    )
    return root


def test_load_project_replay_dataset_uses_standardized_records_and_keeps_kd_separate(
    tmp_path: Path,
) -> None:
    project = write_project_fixture(tmp_path / "project")

    dataset = load_project_replay_dataset(project)

    assert isinstance(dataset, ProjectReplayDataset)
    assert dataset_is_available(project) is True
    assert dataset.maskable_variant_ids == ("com1", "com2", "com3", "com4", "com5")
    assert set(dataset.variant_by_id) == set(dataset.maskable_variant_ids)
    assert dataset.variant_by_id["com1"].modules == ("HA10G", "LB20C")
    assert dataset.variant_by_id["com5"].modules == ("HA10G", "LB20C", "LF60Y")
    assert dataset.variant_by_id["com1"].dilution_record_count > 0
    assert dataset.variant_by_id["com1"].expression_concentration == pytest.approx(3.0)
    assert "KD_ONLY_A" not in dataset.variant_by_id
    assert dataset.supplementary_tracks["KD_ratio"]["KD_ONLY_A"] == pytest.approx(1.75)
    assert all("KD_ratio" not in record.endpoint_values for record in dataset.variants)
    assert dataset.context["estimands"]["estimand_set_id"] == "fixture_estimands"


def test_project_replay_dataset_unavailable_when_standardized_files_are_missing(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    (project / "standardized").mkdir(parents=True)

    assert dataset_is_available(project) is False
    with pytest.raises(FileNotFoundError, match="standardized/observations_long.csv"):
        load_project_replay_dataset(project)
