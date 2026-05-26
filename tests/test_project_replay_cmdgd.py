from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from design_scientist.algorithms import cmdgd
from design_scientist.project_replay import run_project_cmdgd_design, run_project_masking_benchmark
from test_project_replay import write_project_fixture


def _rows(path: str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_cmdgd_design_fixture(root: Path) -> Path:
    standardized = root / "standardized"
    base_sequences = root / "raw" / "base_sequences"
    standardized.mkdir(parents=True)
    base_sequences.mkdir(parents=True)

    base_heavy = "AAAAAAAAAAAA"
    base_light = "SSSSSSSSSSSS"
    variants = [
        ("base", base_heavy, base_light, "", 0.72, 0.21, 0.75),
        ("h3y", "AAYAAAAAAAAA", base_light, "H3Y", 0.83, 0.18, 0.82),
        ("l5t", base_heavy, "SSSSTSSSSSSS", "L5T", 0.79, 0.20, 0.81),
        ("h7f", "AAAAAAFAAAAA", base_light, "H7F", 0.81, 0.19, 0.80),
    ]

    observation_rows: list[dict[str, object]] = []
    source_row = 1
    for variant_id, _heavy, _light, _mutations, ph74, ph6, expression in variants:
        for ph, value in [(7.4, ph74), (6.0, ph6)]:
            observation_rows.append(
                {
                    "observation_id": f"{variant_id}:Ae:{ph}",
                    "source_file": "raw/wet_lab/elisa_summary.csv",
                    "source_row": source_row,
                    "id_namespace": "synthetic",
                    "variant_id": variant_id,
                    "antigen_genotype": "Ae",
                    "endpoint": "summary_elisa_signal",
                    "value": value,
                    "value_status": "observed",
                    "unit": "signal",
                    "pH": ph,
                    "concentration_ng_ml": "",
                    "replicate_id": "",
                    "evidence_role": "derived_summary",
                }
            )
        observation_rows.append(
            {
                "observation_id": f"{variant_id}:expression",
                "source_file": "raw/wet_lab/expression.csv",
                "source_row": source_row,
                "id_namespace": "synthetic",
                "variant_id": variant_id,
                "antigen_genotype": "",
                "endpoint": "expression_concentration",
                "value": expression,
                "value_status": "observed",
                "unit": "relative",
                "pH": "",
                "concentration_ng_ml": "",
                "replicate_id": "",
                "evidence_role": "qc_observation",
            }
        )
        source_row += 1

    observation_fields = [
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
    with (standardized / "observations_long.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=observation_fields)
        writer.writeheader()
        writer.writerows(observation_rows)

    with (standardized / "variant_sequences.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["variant_id", "heavy_chain_seq", "light_chain_seq", "mutations", "source_file", "source_row"],
        )
        writer.writeheader()
        for index, (variant_id, heavy, light, mutations, *_rest) in enumerate(variants, start=1):
            writer.writerow(
                {
                    "variant_id": variant_id,
                    "heavy_chain_seq": heavy,
                    "light_chain_seq": light,
                    "mutations": mutations,
                    "source_file": "raw/wet_lab/variant_sequences.csv",
                    "source_row": index,
                }
            )

    (base_sequences / "heavy.fasta").write_text(f">synthetic_heavy\n{base_heavy}\n", encoding="utf-8")
    (base_sequences / "light.fasta").write_text(f">synthetic_light\n{base_light}\n", encoding="utf-8")
    (standardized / "project_context.json").write_text(
        json.dumps(
            {
                "candidate_edit_vocabulary": [
                    {
                        "token": "H10W",
                        "chain": "H",
                        "position": 10,
                        "residue": "W",
                        "annotation": "contrast",
                        "scope": "non_project_specific",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (root / "data_contract.yaml").write_text(
        """
contract_id: synthetic_cmdgd_design
allowed_sources:
  base_sequences:
    evidence_tier: sequence_reference
    allowed_as_experimental_evidence: false
    files:
      - raw/base_sequences/heavy.fasta
      - raw/base_sequences/light.fasta
  wet_lab:
    evidence_tier: primary_wet_lab
    allowed_as_experimental_evidence: true
    files:
      - raw/wet_lab/elisa_summary.csv
      - raw/wet_lab/expression.csv
      - raw/wet_lab/variant_sequences.csv
validation_rules:
  - Do not use untested variants as evidence.
""".strip(),
        encoding="utf-8",
    )
    (root / "estimands.yaml").write_text(
        """
estimand_set_id: synthetic_cmdgd_design
primary_endpoints:
  - name: elisa_signal_pH74
  - name: elisa_signal_pH6
secondary_endpoints:
  - name: expression_feasibility
""".strip(),
        encoding="utf-8",
    )
    return root


def test_project_cmdgd_design_passes_startup_inputs_and_allowed_priors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = _write_cmdgd_design_fixture(tmp_path / "project")
    captured: dict[str, Any] = {}

    def fake_fit_state(
        base_heavy_chain_seq,
        base_light_chain_seq,
        observed_variants,
        observed_endpoints,
        candidate_edit_vocabulary=None,
        **_kwargs,
    ):
        captured["base_heavy_chain_seq"] = base_heavy_chain_seq
        captured["base_light_chain_seq"] = base_light_chain_seq
        captured["observed_variants"] = list(observed_variants)
        captured["observed_endpoints"] = list(observed_endpoints)
        captured["candidate_edit_vocabulary"] = list(candidate_edit_vocabulary or [])
        return {"observed_ids": {record["variant_id"] for record in observed_variants}}

    def fake_generate_candidates(_state, **_kwargs):
        return [
            {
                "candidate_id": "cmdgd_H10W",
                "variant_id": "cmdgd_H10W",
                "heavy_chain_seq": "AAAAAAAAAWAA",
                "light_chain_seq": "SSSSSSSSSSSS",
                "edits": [{"id": "H10W", "chain": "H", "position": 10, "from": "A", "to": "W"}],
                "modules": ["H10W"],
                "parent_ids": ["design_vocabulary"],
                "operator": "beam_recombine_modules",
                "source_refs": [{"kind": "mechanism_prior"}],
                "feasibility_status": "review",
                "cost": 1.0,
            }
        ]

    def fake_score_candidates(_state, candidates):
        return [{**candidate, "score": 1.0} for candidate in candidates]

    def fake_select_panel(_state, candidates, budget, _rng):
        return [candidate["candidate_id"] for candidate in candidates[: int(budget)]]

    monkeypatch.setattr(cmdgd, "fit_state", fake_fit_state)
    monkeypatch.setattr(cmdgd, "generate_candidates", fake_generate_candidates)
    monkeypatch.setattr(cmdgd, "score_candidates", fake_score_candidates)
    monkeypatch.setattr(cmdgd, "select_panel", fake_select_panel)

    result = run_project_cmdgd_design(project, run_id="cmdgd_design_inputs", budget=1, generation_budget=8)

    assert result["status"] == "completed"
    assert captured["base_heavy_chain_seq"] == "AAAAAAAAAAAA"
    assert captured["base_light_chain_seq"] == "SSSSSSSSSSSS"
    assert {record["variant_id"] for record in captured["observed_variants"]} == {"base", "h3y", "l5t", "h7f"}
    assert all(record["endpoint_values"] for record in captured["observed_variants"])
    assert {record["variant_id"] for record in captured["observed_endpoints"]} == {"base", "h3y", "l5t", "h7f"}
    assert captured["candidate_edit_vocabulary"] == [
        {
            "token": "H10W",
            "chain": "H",
            "position": 10,
            "residue": "W",
            "annotation": "contrast",
            "scope": "non_project_specific",
        }
    ]


def test_project_cmdgd_design_reports_new_site_generated_candidates_when_enabled(tmp_path: Path) -> None:
    project = _write_cmdgd_design_fixture(tmp_path / "project")

    result = run_project_cmdgd_design(project, run_id="cmdgd_design_new_site", budget=2, generation_budget=32)

    rows = _rows(result["generated_candidates_path"])
    assert result["status"] == "completed"
    assert rows
    assert any(row["uses_new_mutation_site"] == "true" for row in rows)
    assert any(row["uses_new_position"] == "true" for row in rows)
    new_site_rows = [row for row in rows if row["uses_new_position"] == "true"]
    assert any("H10W" in row["new_mutation_sites"] for row in new_site_rows)
    assert all(row["observed_duplicate"] == "false" for row in new_site_rows)

    summary = json.loads(Path(result["design_summary_path"]).read_text(encoding="utf-8"))
    assert summary["generated_new_mutation_site_count"] >= 1
    assert summary["generated_new_position_count"] >= 1
    assert summary["observed_duplicate_selected_count"] == 0


def test_project_masking_registry_runs_cmdgd_observed_pool_aliases(tmp_path: Path) -> None:
    project = write_project_fixture(tmp_path / "project")

    result = run_project_masking_benchmark(
        project,
        run_id="cmdgd_aliases",
        budget=1,
        folds="kfold_2",
        mechanisms=["cmdgd", "cmdgd_generative"],
    )

    rows = _rows(result["benchmark_results_path"])
    by_mechanism = {row["mechanism"] for row in rows}
    assert by_mechanism == {"cmdgd", "cmdgd_generative"}
    assert {row["status"] for row in rows} == {"completed"}
    assert all(int(row["selected_count"]) <= 1 for row in rows)
    assert all(
        set(json.loads(row["selected_ids"])) <= set(json.loads(row["candidate_ids"]))
        for row in rows
    )

    summary_rows = _rows(result["summary_results_path"])
    assert {row["mechanism"] for row in summary_rows} == {"cmdgd", "cmdgd_generative"}


def test_cmdgd_project_selector_does_not_receive_heldout_endpoint_values(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = write_project_fixture(tmp_path / "project")
    checked_folds: list[dict[str, Any]] = []

    def fake_fit_state(
        base_heavy_chain_seq,
        base_light_chain_seq,
        observed_variants,
        observed_endpoints,
        **_kwargs,
    ):
        observed_ids = {record["variant_id"] for record in observed_variants}
        endpoint_ids = {record["variant_id"] for record in observed_endpoints}
        assert observed_ids == endpoint_ids
        assert base_heavy_chain_seq
        assert base_light_chain_seq
        assert all(record.get("endpoint_values") for record in observed_variants)
        return {"observed_ids": observed_ids}

    def fake_score_candidates(state, candidates):
        scored = []
        candidate_ids = set()
        for candidate in candidates:
            candidate_ids.add(candidate["candidate_id"])
            assert candidate["variant_id"] == candidate["candidate_id"]
            assert candidate.get("endpoint_values") in ({}, None)
            assert "observed_utility" not in candidate
            assert "true_feasible" not in candidate
            assert "heavy_chain_seq" in candidate
            assert "light_chain_seq" in candidate
            assert candidate.get("edits") is not None
            scored.append({**candidate, "score": 1.0})
        assert candidate_ids.isdisjoint(state["observed_ids"])
        checked_folds.append(
            {"observed_ids": state["observed_ids"], "candidate_ids": candidate_ids}
        )
        return scored

    def fake_select_panel(_state, candidates, budget, _rng):
        return [candidate["candidate_id"] for candidate in candidates[: int(budget)]]

    monkeypatch.setattr(cmdgd, "fit_state", fake_fit_state)
    monkeypatch.setattr(cmdgd, "score_candidates", fake_score_candidates)
    monkeypatch.setattr(cmdgd, "select_panel", fake_select_panel)

    result = run_project_masking_benchmark(
        project,
        run_id="cmdgd_no_leak",
        budget=2,
        folds="kfold_2",
        mechanisms=["cmdgd"],
    )

    rows = _rows(result["benchmark_results_path"])
    assert {row["status"] for row in rows} == {"completed"}
    assert len(checked_folds) == 2
    assert all(item["candidate_ids"].isdisjoint(item["observed_ids"]) for item in checked_folds)


def test_project_masking_failed_mechanism_does_not_interrupt_cmdgd(
    tmp_path: Path,
) -> None:
    project = write_project_fixture(tmp_path / "project")

    def exploding_selector(_observed, _candidates, _budget, _round_index, _rng):
        raise RuntimeError("selector exploded")

    result = run_project_masking_benchmark(
        project,
        run_id="cmdgd_with_failure",
        budget=1,
        folds="leave_one_variant",
        mechanisms=[exploding_selector, "cmdgd"],
    )

    rows = _rows(result["benchmark_results_path"])
    statuses = {(row["mechanism"], row["fold_id"]): row["status"] for row in rows}
    assert {
        status
        for (mechanism, _fold), status in statuses.items()
        if mechanism == "exploding_selector"
    } == {"failed"}
    assert {status for (mechanism, _fold), status in statuses.items() if mechanism == "cmdgd"} == {
        "completed"
    }
    assert all(
        "selector exploded" in row["error"]
        for row in rows
        if row["mechanism"] == "exploding_selector"
    )
