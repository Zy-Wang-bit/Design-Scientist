from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from design_scientist.cli import main
from design_scientist.startup_standardization import audit_schema, standardize_startup_data


def test_standardize_startup_data_writes_generic_project_artifacts(tmp_path: Path) -> None:
    project = _write_startup_project(tmp_path)

    result = standardize_startup_data(project)

    assert result["status"] == "ok"
    assert result["summary"]["observations"] == 9
    assert result["summary"]["sequences"] == 2

    observations = _read_csv(project / "standardized" / "observations_long.csv")
    assert any(row["endpoint"] == "KD_ratio" and row["variant_id"] == "HK52H" for row in observations)
    assert any(
        row["endpoint"] == "elisa_signal"
        and row["pH"] == "6.0"
        and row["concentration_ng_ml"] == "100"
        for row in observations
    )
    assert any(
        row["endpoint"] == "summary_pH74_over_pH60_ratio"
        and row["value_status"] == "missing_sentinel"
        for row in observations
    )
    assert any(row["endpoint"] == "expression_concentration" for row in observations)

    context = json.loads((project / "standardized" / "project_context.json").read_text())
    assert context["project_id"] == "startup_test"
    assert context["objective"].startswith("Design pH-sensitive")
    assert context["observed_row_counts"]["raw/wet_lab/ae_kd_ratio.csv"] == 2
    assert any(item["source_file"].endswith("ae_kd_ratio.csv") for item in context["identifier_namespaces"])


def test_audit_schema_records_disjoint_identifier_namespaces(tmp_path: Path) -> None:
    project = _write_startup_project(tmp_path)

    report = audit_schema(project)

    assert report["status"] == "ok"
    assert any(item["code"] == "disjoint_identifier_namespaces" for item in report["diagnostics"])


def test_standardize_startup_cli(tmp_path: Path, capsys) -> None:
    project = _write_startup_project(tmp_path)

    assert main(["standardize-startup-data", str(project)]) == 0
    output = capsys.readouterr().out

    assert "Startup standardization ok" in output
    assert (project / "standardized" / "project_context.json").exists()


def _write_startup_project(tmp_path: Path) -> Path:
    project = tmp_path / "startup"
    wet = project / "raw" / "wet_lab"
    wet.mkdir(parents=True)
    (project / "standardized").mkdir()
    (project / "project.yaml").write_text(
        yaml.safe_dump(
            {
                "project_id": "startup_test",
                "domain": "antibody_variant_design",
                "objective": "Design pH-sensitive antibody variants.",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (project / "estimands.yaml").write_text(
        yaml.safe_dump(
            {
                "primary_endpoints": [
                    {"name": "elisa_signal_pH6", "pH": 6.0},
                    {"name": "elisa_signal_pH74", "pH": 7.4},
                ],
                "constraints": ["retain neutral binding"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (project / "data_contract.yaml").write_text(
        yaml.safe_dump(
            {
                "contract_id": "startup_contract",
                "local_root": str(project),
                "identifier_columns": ["variant_id", "variant", "编号"],
                "allowed_sources": {
                    "wet_lab": {
                        "evidence_tier": "primary_wet_lab",
                        "allowed_as_experimental_evidence": True,
                        "files": [
                            "raw/wet_lab/ae_kd_ratio.csv",
                            "raw/wet_lab/elisa_dilution_measurements.csv",
                            "raw/wet_lab/elisa_summary.csv",
                            "raw/wet_lab/expression.csv",
                            "raw/wet_lab/variant_sequences.csv",
                        ],
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_text(
        wet / "ae_kd_ratio.csv",
        "variant_id,antigen_genotype,endpoint,value\nHK52H,Ae,KD_ratio,2.0\nHE50H,Ae,KD_ratio,1.5\n",
    )
    _write_text(
        wet / "elisa_dilution_measurements.csv",
        "variant_id,antigen_genotype,pH,concentration_ng_ml,replicate_id,elisa_signal\n"
        "com1,Ae,6.0,100,1,0.1\n"
        "com1,Ae,7.4,100,1,1.0\n",
    )
    _write_text(
        wet / "elisa_summary.csv",
        "variant,Ae_pH6.0,Ae_pH7.4,Ae_ratio,H_chain,L_chain,mutations\n"
        "com1,0.1,1.0,-1.0,HHH,LLL,HA1K\n",
    )
    _write_text(
        wet / "expression.csv",
        "编号,浓度ug/ul,首孔500ng所需体积ul\ncom1,1.0,700\n",
    )
    _write_text(
        wet / "variant_sequences.csv",
        "variant_id,heavy_construct,light_construct,heavy_chain_seq,light_chain_seq\n"
        "com1,H1,L1,HHH,LLL\ncom2,H2,L2,HHK,LLK\n",
    )
    return project


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
