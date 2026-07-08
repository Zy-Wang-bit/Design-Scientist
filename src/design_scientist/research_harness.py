"""Claim, evidence, and mechanism ledgers for framework R&D runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from design_scientist.artifacts import (
    V4_CLAIM_LEDGER,
    V4_EVIDENCE_LEDGER,
    V4_MECHANISM_LEDGER,
    V4_RESEARCH_HARNESS_SUMMARY,
)
from design_scientist.io import ensure_dir, write_json
from design_scientist.schemas import to_plain_data


SUPPORTED_STATUS = "supported"
UNSUPPORTED_STATUS = "unsupported"


@dataclass
class ClaimRecord:
    claim_id: str
    subject: str
    claim_type: str
    text: str
    support_status: str
    evidence_ids: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)


@dataclass
class EvidenceRecord:
    evidence_id: str
    source_type: str
    path: str
    summary: str
    strength: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MechanismRecord:
    mechanism_id: str
    name: str
    components: list[str]
    literature_refs: list[str] = field(default_factory=list)
    implementation_path: str | None = None
    status: str = "proposed"


class ResearchHarness:
    """Append-only ledgers plus validation for method claims."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.claims: list[ClaimRecord] = []
        self.evidence: list[EvidenceRecord] = []
        self.mechanisms: list[MechanismRecord] = []

    @property
    def claim_ledger_path(self) -> Path:
        return self.root / V4_CLAIM_LEDGER

    @property
    def evidence_ledger_path(self) -> Path:
        return self.root / V4_EVIDENCE_LEDGER

    @property
    def mechanism_ledger_path(self) -> Path:
        return self.root / V4_MECHANISM_LEDGER

    @property
    def summary_path(self) -> Path:
        return self.root / V4_RESEARCH_HARNESS_SUMMARY

    def add_claim(self, record: ClaimRecord) -> None:
        self.claims.append(record)
        self._append_jsonl(self.claim_ledger_path, record)

    def add_evidence(self, record: EvidenceRecord) -> None:
        self.evidence.append(record)
        self._append_jsonl(self.evidence_ledger_path, record)

    def add_mechanism(self, record: MechanismRecord) -> None:
        self.mechanisms.append(record)
        self._append_jsonl(self.mechanism_ledger_path, record)

    def validate(self, selected_claim_ids: Iterable[str] | None = None) -> dict[str, Any]:
        evidence_ids = {record.evidence_id for record in self.evidence}
        unsupported = [
            record.claim_id
            for record in self.claims
            if record.support_status == UNSUPPORTED_STATUS
        ]
        missing_evidence = {
            record.claim_id: [
                evidence_id
                for evidence_id in record.evidence_ids
                if evidence_id not in evidence_ids
            ]
            for record in self.claims
            if record.support_status == SUPPORTED_STATUS
        }
        missing_evidence = {
            claim_id: ids for claim_id, ids in missing_evidence.items() if ids
        }
        empty_components = [
            record.mechanism_id for record in self.mechanisms if not record.components
        ]
        selected_unsupported = sorted(set(selected_claim_ids or []) & set(unsupported))
        errors = []
        if unsupported:
            errors.append("unsupported_claims_present")
        if missing_evidence:
            errors.append("supported_claims_missing_evidence")
        if empty_components:
            errors.append("mechanisms_missing_components")
        if selected_unsupported:
            errors.append("selected_mechanism_uses_unsupported_claims")

        return {
            "valid": not errors,
            "errors": errors,
            "claim_count": len(self.claims),
            "evidence_count": len(self.evidence),
            "mechanism_count": len(self.mechanisms),
            "unsupported_claim_ids": unsupported,
            "missing_evidence_by_claim": missing_evidence,
            "empty_component_mechanism_ids": empty_components,
            "selected_unsupported_claim_ids": selected_unsupported,
        }

    def write_summary(self, selected_claim_ids: Iterable[str] | None = None) -> dict[str, Any]:
        summary = self.validate(selected_claim_ids=selected_claim_ids)
        write_json(self.summary_path, summary)
        return summary

    @staticmethod
    def _append_jsonl(path: Path, record: Any) -> None:
        import json

        ensure_dir(path.parent)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(to_plain_data(record), sort_keys=False))
            handle.write("\n")
