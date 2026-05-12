from __future__ import annotations

from dataclasses import is_dataclass

import pytest

from design_scientist import artifacts, schemas
from design_scientist.cli import build_parser
from design_scientist.schemas import to_plain_data


def test_v3_mechanism_dataclasses_serialize_to_plain_data() -> None:
    component = schemas.MechanismComponent(
        component_id="state_update",
        component_type="state_model",
        description="Update posterior state from observed assays.",
        inputs=["observed"],
        outputs=["state"],
        literature_refs=["paper:active_design"],
    )
    spec = schemas.MechanismSpec(
        mechanism_id="mechanism_active_design",
        name="Mechanism-aware active design",
        hypothesis="Explicit mechanisms improve multi-round design decisions.",
        components=[component],
        state_model={"component_id": "state_update"},
        claims=["reduces false positives"],
    )
    request = schemas.MechanismRunRequest(
        observed=[{"candidate_id": "obs_1", "endpoint": 1.0}],
        candidates=[{"candidate_id": "cand_1"}],
        budget=8,
        round_index=2,
        context={"domain": "protein design"},
    )
    decision = schemas.MechanismDecision(
        candidate_ids=["cand_1"],
        scores={"cand_1": 0.82},
        rationale="Highest expected improvement under constraints.",
        diagnostics={"budget_used": 1},
        trace_id="trace_1",
    )
    trace = schemas.MechanismTrace(
        trace_id="trace_1",
        mechanism_id=spec.mechanism_id,
        stages=["state_model", "acquisition_objective"],
        component_outputs={"state_update": {"posterior": "updated"}},
        warnings=["low evidence"],
    )
    stress_plan = schemas.StressTestPlan(
        plan_id="stress_mechanism_active_design",
        worlds=[{"world_id": "missing_not_at_random"}],
        claim_mapping={"reduces false positives": ["missing_not_at_random"]},
        required_baselines=["random_feasible", "fixed_mix"],
        acceptance_criteria={"max_false_claim_rate": 0.0},
    )
    ablation = schemas.MechanismAblationResult(
        mechanism_id=spec.mechanism_id,
        ablation_id="remove_uncertainty",
        removed_components=["uncertainty_model"],
        metrics={"best_observed": 1.4},
        delta_vs_full={"best_observed": -0.2},
        passed=False,
    )

    plain = to_plain_data(
        {
            "spec": spec,
            "request": request,
            "decision": decision,
            "trace": trace,
            "stress_plan": stress_plan,
            "ablation": ablation,
        }
    )

    assert is_dataclass(component)
    assert plain["spec"]["version"] == "v3"
    assert plain["spec"]["components"][0]["inputs"] == ["observed"]
    assert plain["request"]["context"]["domain"] == "protein design"
    assert plain["decision"]["scores"]["cand_1"] == 0.82
    assert plain["trace"]["component_outputs"]["state_update"]["posterior"] == "updated"
    assert plain["stress_plan"]["required_baselines"] == ["random_feasible", "fixed_mix"]
    assert plain["ablation"]["passed"] is False


def test_v3_mechanism_dataclass_defaults_are_independent() -> None:
    first = schemas.MechanismRunRequest()
    second = schemas.MechanismRunRequest()

    first.observed.append({"candidate_id": "obs_1"})
    first.context["round"] = 1

    assert second.observed == []
    assert second.context == {}


def test_v3_artifact_constants_freeze_framework_and_node_contracts() -> None:
    assert set(artifacts.V3_FRAMEWORK_ARTIFACTS) >= {
        "framework/literature_corpus.jsonl",
        "framework/literature_reading_trace.json",
        "framework/mechanism_cards.json",
        "framework/mechanism_library.json",
        "framework/mechanism_gap_matrix.csv",
    }
    assert artifacts.V3_MECHANISM_NODE_ARTIFACTS == (
        "mechanism_spec.json",
        "mechanism.py",
        "proposal.json",
        "ablation_plan.json",
        "stress_test_plan.json",
        "mechanism_metrics.json",
        "validation_report.json",
    )


def test_v3_stage_sequence_matches_scientist_search_v3() -> None:
    from design_scientist.scientist_search_v3 import SCIENTIST_V3_STAGES

    assert artifacts.V3_STAGE_SEQUENCE == SCIENTIST_V3_STAGES


def test_cli_help_exposes_v3_mechanism_commands(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "V3 mechanism scientist loop" in output
    assert "read-literature" in output
    assert "extract-mechanisms" in output
    assert "develop-method" not in output
    assert "benchmark-methods" not in output


def test_run_scientist_help_hides_legacy_v2_flag(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["run-scientist", "--help"])

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "--legacy-v2" not in output
