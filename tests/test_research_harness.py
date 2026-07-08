from pathlib import Path

from design_scientist.research_harness import (
    ClaimRecord,
    EvidenceRecord,
    MechanismRecord,
    ResearchHarness,
)


def test_claim_without_evidence_is_not_supported(tmp_path: Path):
    harness = ResearchHarness(tmp_path)
    harness.add_claim(
        ClaimRecord(
            claim_id="claim_001",
            subject="mechanism_x",
            claim_type="algorithmic",
            text="mechanism_x improves sparse round design",
            support_status="unsupported",
            evidence_ids=[],
        )
    )

    summary = harness.validate()

    assert summary["valid"] is False
    assert "claim_001" in summary["unsupported_claim_ids"]


def test_supported_claim_requires_existing_evidence(tmp_path: Path):
    harness = ResearchHarness(tmp_path)
    harness.add_evidence(
        EvidenceRecord(
            evidence_id="ev_001",
            source_type="benchmark",
            path="runs/run_001/mechanism_benchmark_summary.csv",
            summary="majority worlds improved against fixed_mix",
            strength="moderate",
        )
    )
    harness.add_claim(
        ClaimRecord(
            claim_id="claim_002",
            subject="mechanism_x",
            claim_type="algorithmic",
            text="mechanism_x improves over fixed_mix in replay",
            support_status="supported",
            evidence_ids=["ev_001"],
        )
    )

    summary = harness.validate()

    assert summary["valid"] is True


def test_supported_claim_with_missing_evidence_is_rejected(tmp_path: Path):
    harness = ResearchHarness(tmp_path)
    harness.add_claim(
        ClaimRecord(
            claim_id="claim_003",
            subject="mechanism_x",
            claim_type="algorithmic",
            text="mechanism_x improves over fixed_mix in replay",
            support_status="supported",
            evidence_ids=["missing_ev"],
        )
    )

    summary = harness.validate()

    assert summary["valid"] is False
    assert summary["missing_evidence_by_claim"] == {"claim_003": ["missing_ev"]}


def test_empty_mechanism_components_are_rejected(tmp_path: Path):
    harness = ResearchHarness(tmp_path)
    harness.add_mechanism(
        MechanismRecord(
            mechanism_id="mechanism_001",
            name="empty mechanism",
            components=[],
        )
    )

    summary = harness.validate()

    assert summary["valid"] is False
    assert "mechanism_001" in summary["empty_component_mechanism_ids"]


def test_ledgers_are_written_as_jsonl_and_summary(tmp_path: Path):
    harness = ResearchHarness(tmp_path)
    harness.add_evidence(
        EvidenceRecord(
            evidence_id="ev_001",
            source_type="benchmark",
            path="runs/run_001/mechanism_benchmark_summary.csv",
            summary="benchmark summary",
            strength="moderate",
        )
    )
    harness.add_claim(
        ClaimRecord(
            claim_id="claim_001",
            subject="mechanism_x",
            claim_type="algorithmic",
            text="supported claim",
            support_status="supported",
            evidence_ids=["ev_001"],
        )
    )
    harness.add_mechanism(
        MechanismRecord(
            mechanism_id="mechanism_001",
            name="mechanism_x",
            components=["state_model"],
        )
    )

    summary = harness.write_summary()

    assert summary["valid"] is True
    assert (tmp_path / "framework" / "claim_ledger.jsonl").read_text(encoding="utf-8")
    assert (tmp_path / "framework" / "evidence_ledger.jsonl").read_text(encoding="utf-8")
    assert (tmp_path / "framework" / "mechanism_ledger.jsonl").read_text(encoding="utf-8")
    assert (tmp_path / "framework" / "research_harness_summary.json").is_file()
