"""Human-readable reports for Design Scientist runs."""

from __future__ import annotations

from pathlib import Path

from design_scientist.io import read_json
from design_scientist.schemas import ValidationReport
from design_scientist.validators import FRAMEWORK_WARNING_FILES


def write_human_review_packet(
    project_dir: str | Path,
    run_id: str,
    validation_report: ValidationReport | None = None,
) -> Path:
    root = Path(project_dir).expanduser().resolve()
    run_dir = root / "runs" / run_id
    metrics = read_json(run_dir / "policy_metrics.json") if (run_dir / "policy_metrics.json").exists() else {}

    lines = [
        f"# Human Review Packet: {run_id}",
        "",
        "## Decision Status",
        "This dry-run is not a wet-lab-final recommendation until human review passes.",
        "",
        "## Required Artifacts",
        "- candidate_pool.csv",
        "- panel_recommendation.csv",
        "- policy_comparison.csv",
        "- policy_metrics.json",
        "- validation_report.json",
        "- decision_report.md",
        "",
        "## Framework Artifact Summary",
    ]
    for rel in FRAMEWORK_WARNING_FILES:
        status = "present" if (root / rel).exists() else "missing"
        lines.append(f"- {rel}: {status}")

    lines.extend([
        "",
        "## Baseline Comparison",
    ])
    for item in metrics.get("baseline_comparison", []):
        lines.append(f"- {item}")

    lines.extend(["", "## Unsupported Claims"])
    for claim in metrics.get("unsupported_claims", []):
        lines.append(f"- {claim}")

    lines.extend(["", "## Validation Findings"])
    if validation_report and validation_report.findings:
        for finding in validation_report.findings:
            lines.append(f"- {finding.severity.upper()} {finding.code}: {finding.message}")
    else:
        lines.append("- No validation findings recorded.")

    out = run_dir / "human_review_packet.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
