from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from design_scientist.io import read_json
from design_scientist.mechanism_extraction import extract_mechanisms
from design_scientist.operator_specs import (
    OPERATOR_GAP_FIELDNAMES,
    compile_operator_spec_artifacts,
    write_operator_spec_artifacts,
)
from design_scientist.schemas import ModelRequest, ModelResponse


REQUIRED_CARD_FIELDS = {
    "mechanism_id",
    "source_paper_ids",
    "mechanism_name",
    "problem_setting",
    "state_model",
    "candidate_generation",
    "acquisition_objective",
    "uncertainty_model",
    "transfer_model",
    "constraints",
    "assumptions",
    "failure_modes",
    "reusable_components",
    "stress_tests",
    "evidence_strength",
}


class FakeBackend:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("FakeBackend received an unexpected request")
        return self.responses.pop(0)


def _write_corpus(project: Path, records: list[dict[str, Any]]) -> None:
    framework = project / "framework"
    framework.mkdir(parents=True, exist_ok=True)
    with (framework / "literature_corpus.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def _complete_card(**updates: Any) -> dict[str, Any]:
    card = {
        "mechanism_id": "backend_active_learning",
        "source_paper_ids": ["paper_1"],
        "mechanism_name": "Backend Active Learning Mechanism",
        "problem_setting": "Small-batch protein variant design.",
        "state_model": "Round-aware surrogate state over observed variants.",
        "candidate_generation": "Generate feasible variant batches from mutation operators.",
        "acquisition_objective": "Expected improvement with diversity and feasibility terms.",
        "uncertainty_model": "Posterior variance from the surrogate model.",
        "transfer_model": "Reuse priors across related protein systems.",
        "constraints": ["batch_budget", "feasibility_filters"],
        "assumptions": ["Observed assay labels are comparable within endpoint strata."],
        "failure_modes": ["Surrogate uncertainty may be miscalibrated."],
        "reusable_components": ["surrogate_acquisition", "batch_design"],
        "stress_tests": ["Retrospective round masking"],
        "evidence_strength": "candidate_from_text",
    }
    card.update(updates)
    return card


def _operator_spec_card(**updates: Any) -> dict[str, Any]:
    card = {
        "mechanism_id": "active_learning_acquisition",
        "source_paper_ids": ["paper_active"],
        "mechanism_name": "Active Learning Acquisition",
        "problem_setting": "Batch-limited variant design.",
        "state_model": "Posterior state over observed variants.",
        "candidate_generation": "Generate feasible mutation candidates.",
        "acquisition_objective": "Expected improvement with uncertainty and feasibility.",
        "uncertainty_model": "Posterior variance from a surrogate model.",
        "transfer_model": "",
        "constraints": ["batch_budget", "feasibility_filters"],
        "assumptions": ["Assays are comparable within endpoint strata."],
        "failure_modes": ["Uncertainty may be miscalibrated."],
        "reusable_components": ["surrogate_model", "acquisition_policy"],
        "stress_tests": ["retrospective_round_masking"],
        "evidence_strength": "candidate_from_text",
    }
    card.update(updates)
    return card


def test_fake_backend_structured_cards_are_written_and_grouped(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write_corpus(
        project,
        [
            {
                "paper_id": "paper_1",
                "title": "Batch active learning for protein engineering",
                "abstract": "Surrogate models and acquisition functions guide feasible batches.",
            }
        ],
    )
    card = _complete_card()
    backend = FakeBackend([ModelResponse(text="", structured={"mechanism_cards": [card]})])

    result = extract_mechanisms(project, backend=backend)

    assert result["status"] == "ok"
    assert result["mechanism_card_count"] == 1
    assert result["fallback_used"] is False
    assert len(backend.requests) == 1
    request = backend.requests[0]
    assert request.purpose == "mechanism_extraction"
    assert request.response_schema is not None
    assert request.response_schema["additionalProperties"] is False
    assert request.response_schema["properties"]["mechanism_cards"]["type"] == "array"
    assert request.response_schema["properties"]["mechanism_cards"]["items"]["required"] == sorted(
        REQUIRED_CARD_FIELDS
    )

    cards = read_json(project / "framework" / "mechanism_cards.json")
    assert cards == [card]

    library = read_json(project / "framework" / "mechanism_library.json")
    assert library["source"] == "literature_engine_v3"
    assert library["component_groups"]["surrogate_acquisition"]["mechanism_ids"] == [
        "backend_active_learning"
    ]
    assert library["component_groups"]["surrogate_acquisition"]["mechanism_cards"] == [card]
    assert library["component_groups"]["batch_design"]["source_paper_ids"] == ["paper_1"]

    operator_specs = read_json(project / "framework" / "operator_specs.json")
    assert operator_specs[0]["mechanism_id"] == "backend_active_learning"
    assert set(operator_specs[0]) >= {
        "objective",
        "update_rule",
        "required_baselines",
        "negative_controls",
        "ablation_hypotheses",
        "implementation_tests",
        "claim_limits",
    }
    assert operator_specs[0]["objective"]["acquisition_formula"].startswith(
        "score(candidate | design_state) ="
    )
    assert operator_specs[0]["update_rule"]["design_state_update"].startswith(
        "design_state_{t+1} = update"
    )
    assert operator_specs[0]["evidence_strength"] == "candidate_from_text"

    operator_evidence_map = read_json(project / "framework" / "operator_evidence_map.json")
    assert operator_evidence_map["operators"]["operator_backend_active_learning"][
        "source_paper_ids"
    ] == ["paper_1"]

    operator_negative_controls = read_json(project / "framework" / "operator_negative_controls.json")
    assert "operator_backend_active_learning" in operator_negative_controls["negative_controls"]

    with (project / "framework" / "operator_gap_matrix.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        operator_gap_rows = list(csv.DictReader(handle))
    assert operator_gap_rows[0]["missing_acquisition_objective"] == "false"


def test_compile_operator_spec_artifacts_preserves_evidence_boundaries() -> None:
    artifacts = compile_operator_spec_artifacts([_operator_spec_card()])

    specs = artifacts["operator_specs"]
    assert len(specs) == 1
    spec = specs[0]
    assert spec["operator_id"] == "operator_active_learning_acquisition"
    assert spec["mechanism_id"] == "active_learning_acquisition"
    assert spec["evidence_strength"] == "candidate_from_text"
    assert spec["objective"]["acquisition_formula"].startswith(
        "score(candidate | design_state) ="
    )
    assert spec["update_rule"]["design_state_update"].startswith(
        "design_state_{t+1} = update"
    )
    assert spec["state_model"]["description"] == "Posterior state over observed variants."
    assert spec["state_model"]["missing"] is False
    assert spec["acquisition_objective"]["description"] == (
        "Expected improvement with uncertainty and feasibility."
    )
    assert spec["acquisition_objective"]["formula"].startswith(
        "score(candidate | design_state) ="
    )
    assert spec["uncertainty"]["description"] == "Posterior variance from a surrogate model."
    assert spec["uncertainty"]["missing"] is False
    assert spec["transfer"]["description"] == ""
    assert spec["transfer"]["missing"] is True
    assert set(spec) >= {
        "state_model",
        "objective",
        "update_rule",
        "required_baselines",
        "negative_controls",
        "acquisition_objective",
        "uncertainty",
        "transfer",
        "ablation_hypotheses",
        "implementation_tests",
        "claim_limits",
    }
    assert {"random_feasible", "fixed_mix_policy", "pure_uncertainty_sampling"} <= set(
        spec["required_baselines"]
    )
    assert "validated_wet_lab_improvement" not in spec["claim_limits"]
    assert any("literature candidate" in limit for limit in spec["claim_limits"])
    assert all(
        evidence["evidence_strength"] in {"candidate_from_text", "needs_manual_review"}
        for evidence in artifacts["operator_evidence_map"]["operators"].values()
    )


def test_operator_gap_matrix_records_missing_card_fields_without_fabricating_support() -> None:
    artifacts = compile_operator_spec_artifacts(
        [
            _operator_spec_card(
                mechanism_id="sparse_operator",
                source_paper_ids=[],
                state_model="",
                candidate_generation="",
                acquisition_objective="",
                uncertainty_model="",
                transfer_model="",
                stress_tests=[],
                evidence_strength="candidate_from_text",
            )
        ]
    )

    spec = artifacts["operator_specs"][0]
    gap_row = artifacts["operator_gap_matrix"][0]

    assert spec["evidence_strength"] == "needs_manual_review"
    assert spec["objective"]["acquisition_formula"] == ""
    assert spec["update_rule"]["design_state_update"] == ""
    assert gap_row["operator_id"] == "operator_sparse_operator"
    assert gap_row["missing_source_paper_ids"] == "true"
    assert gap_row["missing_state_model"] == "true"
    assert gap_row["missing_candidate_generation"] == "true"
    assert gap_row["missing_acquisition_objective"] == "true"
    assert gap_row["missing_uncertainty_model"] == "true"
    assert gap_row["missing_transfer_model"] == "true"
    assert gap_row["missing_stress_tests"] == "true"
    assert gap_row["evidence_strength"] == "needs_manual_review"
    assert gap_row["unsupported_fields"] == (
        "source_paper_ids; state_model; candidate_generation; "
        "acquisition_objective; uncertainty_model; transfer_model; stress_tests"
    )
    assert "manual review" in gap_row["claim_limits"]
    evidence = artifacts["operator_evidence_map"]["operators"]["operator_sparse_operator"]
    assert evidence["unsupported_fields"] == [
        "source_paper_ids",
        "state_model",
        "candidate_generation",
        "acquisition_objective",
        "uncertainty_model",
        "transfer_model",
        "stress_tests",
    ]
    assert evidence["field_support"]["state_model"] == "unsupported"
    assert evidence["field_support"]["problem_setting"] == "supported"


def test_negative_controls_cover_label_holdout_uncertainty_and_transfer() -> None:
    artifacts = compile_operator_spec_artifacts(
        [
            _operator_spec_card(
                transfer_model="Multi-task source-to-target prior.",
                reusable_components=["surrogate_model", "transfer_prior"],
            )
        ]
    )

    spec = artifacts["operator_specs"][0]
    control_ids = {control["control_id"] for control in spec["negative_controls"]}

    assert f'{spec["operator_id"]}_label_permutation' in control_ids
    assert f'{spec["operator_id"]}_paper_holdout' in control_ids
    assert f'{spec["operator_id"]}_uncertainty_shuffle' in control_ids
    assert f'{spec["operator_id"]}_incompatible_source' in control_ids
    assert artifacts["operator_negative_controls"]["negative_controls"][spec["operator_id"]] == (
        spec["negative_controls"]
    )


def test_write_operator_spec_artifacts_writes_json_and_csv_contracts(tmp_path: Path) -> None:
    framework_dir = tmp_path / "framework"

    paths = write_operator_spec_artifacts(framework_dir, [_operator_spec_card()])

    assert set(paths) == {
        "operator_specs",
        "operator_gap_matrix",
        "operator_evidence_map",
        "operator_negative_controls",
    }
    specs = read_json(framework_dir / "operator_specs.json")
    assert specs[0]["operator_id"] == "operator_active_learning_acquisition"

    evidence_map = read_json(framework_dir / "operator_evidence_map.json")
    assert evidence_map["source"] == "mechanism_cards"
    assert sorted(evidence_map["operators"]) == ["operator_active_learning_acquisition"]

    negative_controls = read_json(framework_dir / "operator_negative_controls.json")
    assert sorted(negative_controls["negative_controls"]) == [
        "operator_active_learning_acquisition"
    ]

    with (framework_dir / "operator_gap_matrix.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(OPERATOR_GAP_FIELDNAMES)
        rows = list(reader)
    assert rows[0]["operator_id"] == "operator_active_learning_acquisition"
    assert rows[0]["missing_acquisition_objective"] == "false"

    json.dumps(specs)


def test_malformed_backend_output_falls_back_and_records_warning(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write_corpus(
        project,
        [
            {
                "paper_id": "active_learning_paper",
                "title": "Active learning for antibody design",
                "abstract": (
                    "Batch active learning uses a surrogate model, acquisition function, "
                    "uncertainty sampling, constraints, and retrospective stress tests."
                ),
            }
        ],
    )
    backend = FakeBackend([ModelResponse(text="{}", structured={"not_mechanism_cards": []})])

    result = extract_mechanisms(project, backend=backend)

    assert result["status"] == "ok_with_warnings"
    assert result["fallback_used"] is True
    assert any("backend_malformed_output" in warning for warning in result["warnings"])

    cards = read_json(project / "framework" / "mechanism_cards.json")
    assert cards
    assert {card["evidence_strength"] for card in cards} <= {
        "candidate_from_text",
        "needs_manual_review",
    }
    assert all(REQUIRED_CARD_FIELDS <= set(card) for card in cards)


def test_backend_accepts_empty_chunk_cards_and_preserves_other_chunk_cards(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write_corpus(
        project,
        [
            {
                "paper_id": "paper_without_mechanism",
                "title": "Protein assay report",
                "abstract": "This chunk has measurements but no reusable active design mechanism.",
            },
            {
                "paper_id": "paper_with_mechanism",
                "title": "Batch active learning for protein engineering",
                "abstract": "Surrogate acquisition functions guide feasible variant batches.",
            },
        ],
    )
    card = _complete_card(source_paper_ids=["paper_with_mechanism"])
    backend = FakeBackend(
        [
            ModelResponse(text="", structured={"mechanism_cards": []}),
            ModelResponse(text="", structured={"mechanism_cards": [card]}),
        ]
    )

    result = extract_mechanisms(project, backend=backend, max_chunks=2)

    assert result["status"] == "ok"
    assert result["fallback_used"] is False
    assert result["mechanism_cards"] == [card]
    assert len(backend.requests) == 2
    assert read_json(project / "framework" / "mechanism_cards.json") == [card]


def test_empty_corpus_returns_failure_without_fabricating_library(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write_corpus(project, [])

    result = extract_mechanisms(project)

    assert result["status"] == "failure"
    assert result["reason"] == "empty_literature_corpus"
    assert result["mechanism_cards"] == []
    assert not (project / "framework" / "mechanism_library.json").exists()
    assert not (project / "framework" / "mechanism_cards.json").exists()
    assert not (project / "framework" / "operator_specs.json").exists()
    assert not (project / "framework" / "operator_evidence_map.json").exists()


def test_shallow_keyword_hit_is_manual_review_not_literature_backed_algorithm(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    _write_corpus(
        project,
        [
            {
                "paper_id": "shallow_active_learning_hit",
                "title": "Active learning acquisition for protein engineering",
                "abstract": (
                    "This short summary says active learning and acquisition can help "
                    "protein engineering."
                ),
            }
        ],
    )

    result = extract_mechanisms(project)

    assert result["status"] == "ok"
    cards = read_json(project / "framework" / "mechanism_cards.json")
    active_card = {card["mechanism_id"]: card for card in cards}["active_learning_acquisition"]
    assert active_card["evidence_strength"] == "needs_manual_review"
    assert active_card["state_model"] == ""
    assert active_card["candidate_generation"] == ""
    assert active_card["acquisition_objective"] == ""

    operator_specs = read_json(project / "framework" / "operator_specs.json")
    spec = {item["operator_id"]: item for item in operator_specs}[
        "operator_active_learning_acquisition"
    ]
    assert spec["evidence_strength"] == "needs_manual_review"
    assert spec["update_rule"]["design_state_update"] == ""
    assert spec["candidate_generation"]["missing"] is True
    assert spec["acquisition_objective"]["formula"] == ""
    assert spec["evidence_map"]["unsupported_fields"] == [
        "state_model",
        "candidate_generation",
        "acquisition_objective",
        "uncertainty_model",
        "transfer_model",
        "stress_tests",
    ]

    evidence_map = read_json(project / "framework" / "operator_evidence_map.json")
    evidence = evidence_map["operators"]["operator_active_learning_acquisition"]
    assert evidence["evidence_strength"] == "needs_manual_review"
    assert evidence["unsupported_fields"] == spec["evidence_map"]["unsupported_fields"]

    with (project / "framework" / "operator_gap_matrix.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        gap_rows = list(csv.DictReader(handle))
    gap_row = {
        row["operator_id"]: row for row in gap_rows
    }["operator_active_learning_acquisition"]
    assert gap_row["unsupported_fields"] == (
        "state_model; candidate_generation; acquisition_objective; "
        "uncertainty_model; transfer_model; stress_tests"
    )
    assert "Missing card fields require manual review" in gap_row["claim_limits"]


def test_heuristic_active_learning_provenance_excludes_generic_protein_engineering_records(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    _write_corpus(
        project,
        [
            {
                "paper_id": "generic_protein_engineering",
                "title": "General protein engineering survey",
                "abstract": "Protein engineering studies variant libraries and assay measurements.",
            },
            {
                "paper_id": "active_learning_source",
                "title": "Batch active learning acquisition for protein engineering",
                "abstract": (
                    "Active learning uses surrogate uncertainty and acquisition functions "
                    "for feasible variant design."
                ),
            },
        ],
    )

    result = extract_mechanisms(project)

    by_id = {card["mechanism_id"]: card for card in result["mechanism_cards"]}
    active_card = by_id["active_learning_acquisition"]
    assert "active_learning_source" in active_card["source_paper_ids"]
    assert "generic_protein_engineering" not in active_card["source_paper_ids"]


def test_heuristic_optional_fields_ignore_unrelated_records_outside_provenance(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    _write_corpus(
        project,
        [
            {
                "paper_id": "active_learning_source",
                "title": "Batch active learning acquisition for protein engineering",
                "abstract": "Active learning acquisition ranks feasible variant batches.",
            },
            {
                "paper_id": "unrelated_benchmark_source",
                "title": "Retrospective uncertainty calibration benchmark",
                "abstract": (
                    "This unrelated benchmark discusses posterior variance, calibration, "
                    "masking, and replay for a separate evaluation workflow."
                ),
            },
        ],
    )

    result = extract_mechanisms(project)

    by_id = {card["mechanism_id"]: card for card in result["mechanism_cards"]}
    active_card = by_id["active_learning_acquisition"]
    assert active_card["source_paper_ids"] == ["active_learning_source"]
    assert active_card["uncertainty_model"] == ""
    assert active_card["stress_tests"] == []


def test_heuristic_guardrail_provenance_excludes_generic_antibody_developability(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    _write_corpus(
        project,
        [
            {
                "paper_id": "generic_antibody_developability",
                "title": "Antibody developability and coverage review",
                "abstract": "Antibody developability programs track coverage and assay liabilities.",
            },
            {
                "paper_id": "mechanism_guardrail_source",
                "title": "Mechanism-aware guardrail design for pH-dependent antibodies",
                "abstract": (
                    "A mechanism-aware design separates guardrail endpoints from "
                    "acidic release and neutral binding claims."
                ),
            },
        ],
    )

    result = extract_mechanisms(project)

    by_id = {card["mechanism_id"]: card for card in result["mechanism_cards"]}
    guardrail_card = by_id["mechanism_guardrail_design"]
    assert guardrail_card["source_paper_ids"] == ["mechanism_guardrail_source"]


def test_normalized_backend_request_text_does_not_duplicate_literature_corpus_chunks(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    evidence_sentence = "Unique active learning acquisition evidence."
    _write_corpus(
        project,
        [
            {
                "paper_id": "paper_1",
                "title": "Chunked paper",
                "text": evidence_sentence,
                "chunks": [
                    {
                        "chunk_id": "paper_1::chunk_1",
                        "text": evidence_sentence,
                    }
                ],
            }
        ],
    )
    backend = FakeBackend([ModelResponse(text="", structured={"mechanism_cards": [_complete_card()]})])

    extract_mechanisms(project, backend=backend)

    request_payload = json.loads(backend.requests[0].messages[1]["content"])
    request_text = request_payload["records"][0]["text"]
    assert request_text.count(evidence_sentence) == 1


def test_gap_matrix_flags_missing_mechanism_fields(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write_corpus(
        project,
        [
            {
                "paper_id": "paper_1",
                "title": "Sparse mechanism paper",
                "abstract": "Mentions acquisition but omits uncertainty, transfer, and stress tests.",
            }
        ],
    )
    sparse_card = _complete_card(
        mechanism_id="sparse_card",
        state_model="",
        candidate_generation="",
        uncertainty_model="",
        transfer_model="",
        stress_tests=[],
        reusable_components=["acquisition_policy"],
        failure_modes=["Insufficient detail in source text."],
    )
    backend = FakeBackend([ModelResponse(text="", structured={"mechanism_cards": [sparse_card]})])

    result = extract_mechanisms(project, backend=backend)

    assert result["status"] == "ok"
    with (project / "framework" / "mechanism_gap_matrix.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [
        {
            "mechanism_id": "sparse_card",
            "missing_state_model": "true",
            "missing_candidate_generation": "true",
            "missing_uncertainty_model": "true",
            "missing_transfer_model": "true",
            "missing_stress_tests": "true",
            "reusable_components": "acquisition_policy",
            "failure_modes": "Insufficient detail in source text.",
        }
    ]

    with (project / "framework" / "operator_gap_matrix.csv").open(encoding="utf-8", newline="") as handle:
        operator_gap_rows = list(csv.DictReader(handle))

    assert operator_gap_rows[0]["operator_id"] == "operator_sparse_card"
    assert operator_gap_rows[0]["missing_state_model"] == "true"
    assert operator_gap_rows[0]["missing_candidate_generation"] == "true"
    assert operator_gap_rows[0]["missing_uncertainty_model"] == "true"
    assert operator_gap_rows[0]["missing_transfer_model"] == "true"
    assert operator_gap_rows[0]["missing_stress_tests"] == "true"
    assert "literature candidate" in operator_gap_rows[0]["claim_limits"]
