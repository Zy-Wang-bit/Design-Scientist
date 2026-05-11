"""Deterministic project and run validators."""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Any

from design_scientist.artifacts import (
    FRAMEWORK_VALIDATION_ARTIFACTS,
    PROJECT_ARTIFACTS,
    RUN_ARTIFACTS,
)
from design_scientist.io import read_json, read_yaml, write_json
from design_scientist.schemas import ValidationFinding, ValidationReport
from design_scientist.table_io import UnsafeIdentifierIssue, find_unsafe_identifier_values


REQUIRED_PROJECT_FILES = tuple(rel for rel in PROJECT_ARTIFACTS if not rel.startswith("state/"))
REQUIRED_STATE_FILES = tuple(rel for rel in PROJECT_ARTIFACTS if rel.startswith("state/"))
REQUIRED_RUN_FILES = RUN_ARTIFACTS
FRAMEWORK_WARNING_FILES = FRAMEWORK_VALIDATION_ARTIFACTS


def _finding(
    severity: str, code: str, message: str, artifact: str | None = None
) -> ValidationFinding:
    return ValidationFinding(  # type: ignore[arg-type]
        severity=severity, code=code, message=message, artifact=artifact
    )


def _load_optional_json(path: Path, findings: list[ValidationFinding]) -> Any:
    if not path.exists():
        findings.append(_finding("error", "missing_artifact", f"Missing {path}", str(path)))
        return None
    try:
        return read_json(path)
    except Exception as exc:  # pragma: no cover - defensive message path
        findings.append(
            _finding("error", "invalid_json", f"Cannot read JSON {path}: {exc}", str(path))
        )
        return None


def _load_csv_records(path: Path, findings: list[ValidationFinding]) -> list[dict[str, str]] | None:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            return [
                {column: ("" if row.get(column) is None else str(row.get(column))) for column in fieldnames}
                for row in reader
            ]
    except Exception as exc:  # pragma: no cover - defensive message path
        findings.append(
            _finding("error", "invalid_csv", f"Cannot read CSV {path}: {exc}", str(path))
        )
        return None


def _check_identifier_issues(
    data: Any,
    *,
    artifact: str,
    findings: list[ValidationFinding],
) -> None:
    for issue in find_unsafe_identifier_values(data, artifact=artifact):
        findings.append(
            _finding("error", issue.code, _identifier_issue_message(issue), issue.artifact)
        )


def _identifier_issue_message(issue: UnsafeIdentifierIssue) -> str:
    location: list[str] = []
    if issue.row_number is not None:
        location.append(f"row {issue.row_number}")
    if issue.column:
        location.append(f"column {issue.column}")
    where = f" at {', '.join(location)}" if location else ""
    if issue.code == "stale_onee62_identifier":
        return f"{issue.artifact}{where} contains noncanonical 1E62 identifier notation."
    return f"{issue.artifact}{where} contains an unknown scientific-notation-like identifier."


def _check_json_artifact_if_exists(path: Path, artifact: str, findings: list[ValidationFinding]) -> None:
    if not path.exists():
        return
    data = _load_optional_json(path, findings)
    if data is not None:
        _check_identifier_issues(data, artifact=artifact, findings=findings)


def _check_csv_artifact_if_exists(path: Path, artifact: str, findings: list[ValidationFinding]) -> None:
    if not path.exists():
        return
    records = _load_csv_records(path, findings)
    if records is not None:
        _check_identifier_issues(records, artifact=artifact, findings=findings)


def _run_dir(project_dir: Path, run_id: str | None) -> Path | None:
    if run_id:
        return project_dir / "runs" / run_id
    runs_dir = project_dir / "runs"
    if not runs_dir.exists():
        return None
    run_dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir()])
    return run_dirs[-1] if run_dirs else None


def validate_project(
    project_dir: str | Path,
    run_id: str | None = None,
    write_report: bool = True,
) -> ValidationReport:
    """Validate project state and optionally the latest or named run."""
    root = Path(project_dir).expanduser().resolve()
    findings: list[ValidationFinding] = []
    project_id = root.name

    for rel in REQUIRED_PROJECT_FILES:
        path = root / rel
        if not path.exists():
            findings.append(_finding("error", "missing_project_file", f"Missing {rel}", rel))

    data_contract: dict[str, Any] = {}
    contract_path = root / "data_contract.yaml"
    if contract_path.exists():
        try:
            data_contract = read_yaml(contract_path)
            project_id = str(data_contract.get("project_id") or project_id)
        except Exception as exc:
            findings.append(
                _finding(
                    "error",
                    "invalid_data_contract",
                    f"Cannot read data_contract.yaml: {exc}",
                    "data_contract.yaml",
                )
            )

    if data_contract:
        if data_contract.get("raw_access_policy") != "schema_audit_only":
            findings.append(
                _finding(
                    "error",
                    "raw_access_policy",
                    "Raw data access must be limited to schema_audit_only.",
                    "data_contract.yaml",
                )
            )
        allowed_tables = data_contract.get("allowed_tables", [])
        for table in allowed_tables:
            if table.get("allow_raw_access") and table.get("role") != "schema_audit":
                findings.append(
                    _finding(
                        "error",
                        "raw_access_outside_audit",
                        f"Table {table.get('name')} allows raw access outside schema audit.",
                        "data_contract.yaml",
                    )
                )

    for rel in REQUIRED_STATE_FILES:
        if not (root / rel).exists():
            findings.append(_finding("error", "missing_state_file", f"Missing {rel}", rel))

    for rel in FRAMEWORK_WARNING_FILES:
        if not (root / rel).exists():
            findings.append(
                _finding(
                    "warning",
                    "missing_framework_artifact",
                    f"Missing optional framework artifact: {rel}",
                    rel,
                )
            )

    state = _load_optional_json(root / "state" / "design_state.json", findings)
    evidence = _load_optional_json(root / "state" / "evidence_cards.json", findings)

    if isinstance(state, dict):
        _check_identifier_issues(state, artifact="state/design_state.json", findings=findings)
        roles = state.get("system_roles", {})
        if roles.get("sdAb") != "module_learning":
            findings.append(
                _finding(
                    "error",
                    "sdab_role_missing",
                    "sdAb must be recorded as module_learning data.",
                    "state/design_state.json",
                )
            )
        if roles.get("1E62") != "target_system_design_genotype_coverage":
            findings.append(
                _finding(
                    "error",
                    "onee62_role_missing",
                    "1E62 must be recorded as target_system_design_genotype_coverage data.",
                    "state/design_state.json",
                )
            )
        unsupported = " ".join(state.get("unsupported_claims", []))
        if "1E62" not in unsupported or "module" not in unsupported or "causal" not in unsupported:
            findings.append(
                _finding(
                    "error",
                    "unsupported_claim_missing",
                    "State must explicitly forbid 1E62 module-causality claims from current combo data.",
                    "state/design_state.json",
                )
            )

    if isinstance(evidence, list):
        for idx, card in enumerate(evidence):
            tier = card.get("evidence_tier")
            if tier not in {
                "primary_matched",
                "secondary_near_matched",
                "descriptive",
                "model_derived",
            }:
                findings.append(
                    _finding(
                        "error",
                        "evidence_tier_missing",
                        f"Evidence card {idx} has invalid evidence_tier.",
                        "state/evidence_cards.json",
                    )
                )
            joined = " ".join(str(x) for x in card.get("source_tables", []))
            if "synthetic" in joined.lower() and tier != "model_derived":
                findings.append(
                    _finding(
                        "error",
                        "synthetic_as_real_evidence",
                        f"Evidence card {idx} uses synthetic source as real evidence.",
                        "state/evidence_cards.json",
                    )
                )

    _check_json_artifact_if_exists(
        root / "state" / "validation_report.json",
        "state/validation_report.json",
        findings,
    )

    selected_run_dir = _run_dir(root, run_id)
    selected_run_id = selected_run_dir.name if selected_run_dir else run_id
    if selected_run_dir:
        for rel in REQUIRED_RUN_FILES:
            if not (selected_run_dir / rel).exists():
                findings.append(
                    _finding("error", "missing_run_artifact", f"Missing {rel}", str(selected_run_dir / rel))
                )
        metrics = selected_run_dir / "policy_metrics.json"
        if metrics.exists():
            metrics_data = read_json(metrics)
            if not metrics_data.get("baseline_comparison"):
                findings.append(
                    _finding(
                        "error",
                        "missing_baseline_comparison",
                        "Run must include baseline comparison.",
                        str(metrics),
                    )
                )
            if not metrics_data.get("unsupported_claims"):
                findings.append(
                    _finding(
                        "error",
                        "missing_unsupported_claims",
                        "Run must include unsupported claims.",
                        str(metrics),
                    )
                )
        _check_csv_artifact_if_exists(
            selected_run_dir / "candidate_pool.csv",
            str(selected_run_dir / "candidate_pool.csv"),
            findings,
        )
        _check_csv_artifact_if_exists(
            selected_run_dir / "panel_recommendation.csv",
            str(selected_run_dir / "panel_recommendation.csv"),
            findings,
        )
        _check_json_artifact_if_exists(
            selected_run_dir / "validation_report.json",
            str(selected_run_dir / "validation_report.json"),
            findings,
        )

    valid = not any(f.severity == "error" for f in findings)
    report = ValidationReport(project_id=project_id, run_id=selected_run_id, valid=valid, findings=findings)

    if write_report:
        out_path = (
            selected_run_dir / "validation_report.json"
            if selected_run_dir
            else root / "state" / "validation_report.json"
        )
        write_json(out_path, report)
    if findings:
        for finding in findings:
            print(f"{finding.severity.upper()} {finding.code}: {finding.message}")
    else:
        print("Validation passed")
    return report


def report_to_dict(report: ValidationReport) -> dict[str, Any]:
    return asdict(report)
