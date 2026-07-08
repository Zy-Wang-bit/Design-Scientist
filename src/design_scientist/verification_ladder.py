"""Verification ladder for V4 mechanism claims."""

from __future__ import annotations

from typing import Any, Mapping


STAGES = (
    "contract",
    "toy_invariant",
    "synthetic_replay",
    "retrospective_masking",
    "structure_proxy",
    "human_review",
)


def run_verification_ladder(
    mechanism_name: str,
    checks: Mapping[str, Any],
    strong_claims: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] | None = None,
) -> dict[str, Any]:
    """Run ordered verification gates and claim-level support checks."""

    stages: list[dict[str, Any]] = []
    blocking_stage: str | None = None
    for stage_name in STAGES:
        if blocking_stage is not None:
            stages.append({"name": stage_name, "status": "skipped", "passed": False})
            continue
        passed, reason = _check_passed(checks.get(stage_name))
        status = "passed" if passed else "failed"
        stages.append({"name": stage_name, "status": status, "passed": passed, "reason": reason})
        if not passed:
            blocking_stage = stage_name

    claim_findings = _claim_findings(checks, strong_claims or [])
    return {
        "schema_version": 1,
        "mechanism_name": mechanism_name,
        "passed": blocking_stage is None and not claim_findings,
        "blocking_stage": blocking_stage,
        "stages": stages,
        "claim_findings": claim_findings,
    }


def _check_passed(check: Any) -> tuple[bool, str | None]:
    if check is True:
        return True, None
    if check is False or check is None:
        return False, "missing or failed check"
    if isinstance(check, Mapping):
        if "passed" in check:
            return bool(check["passed"]), _reason(check)
        if "valid" in check:
            return bool(check["valid"]), _reason(check)
        if check.get("status") in {"passed", "pass", "ok"}:
            return True, _reason(check)
        if check.get("status") in {"failed", "fail", "error"}:
            return False, _reason(check)
    return bool(check), None


def _claim_findings(
    checks: Mapping[str, Any],
    claims: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for claim in claims:
        claim_id = str(claim.get("claim_id") or claim.get("id") or "claim")
        claim_type = str(claim.get("claim_type") or claim.get("type") or "").lower()
        strength = str(claim.get("strength") or "").lower()
        if claim_type == "algorithm" and strength == "strong":
            replay_passed, _ = _check_passed(checks.get("synthetic_replay"))
            ablation_passed = _has_ablation_support(checks)
            if not replay_passed or not ablation_passed:
                findings.append(
                    {
                        "claim_id": claim_id,
                        "reason": "strong algorithm claim requires synthetic_replay and ablation support",
                    }
                )
        if claim_type in {"biology", "biological"} and _supported_only_by_structure_proxy(claim):
            findings.append(
                {
                    "claim_id": claim_id,
                    "reason": "biological claim cannot be supported only by structure_proxy",
                }
            )
    return findings


def _has_ablation_support(checks: Mapping[str, Any]) -> bool:
    synthetic_replay = checks.get("synthetic_replay")
    if isinstance(synthetic_replay, Mapping) and synthetic_replay.get("ablation") is True:
        return True
    ablation = checks.get("ablation") or checks.get("ablation_support")
    passed, _ = _check_passed(ablation)
    return passed


def _supported_only_by_structure_proxy(claim: Mapping[str, Any]) -> bool:
    supported_by = claim.get("supported_by") or claim.get("evidence_stages") or []
    if isinstance(supported_by, str):
        supported_by = [supported_by]
    if not isinstance(supported_by, (list, tuple, set)):
        return False
    stages = {str(stage) for stage in supported_by if str(stage)}
    return stages == {"structure_proxy"}


def _reason(check: Mapping[str, Any]) -> str | None:
    reason = check.get("reason") or check.get("message")
    return str(reason) if reason is not None else None
