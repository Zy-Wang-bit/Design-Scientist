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


def test_empty_corpus_returns_failure_without_fabricating_library(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write_corpus(project, [])

    result = extract_mechanisms(project)

    assert result["status"] == "failure"
    assert result["reason"] == "empty_literature_corpus"
    assert result["mechanism_cards"] == []
    assert not (project / "framework" / "mechanism_library.json").exists()
    assert not (project / "framework" / "mechanism_cards.json").exists()


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
