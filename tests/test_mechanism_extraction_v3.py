from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from design_scientist.io import read_json
from design_scientist.mechanism_extraction import extract_mechanisms
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
