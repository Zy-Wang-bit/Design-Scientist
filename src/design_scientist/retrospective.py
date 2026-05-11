"""Retrospective evaluation scaffolds for method development."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from design_scientist.acquisition import default_policy, score_panel
from design_scientist.candidates import generate_candidates
from design_scientist.io import ensure_dir, read_json
from design_scientist.panel import select_panel


DEFAULT_POLICIES = ("mechanism_aware", "fixed_mix", "pure_lattice_repair")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _category_count(panel: list[Any], category: str) -> int:
    return sum(1 for candidate in panel if getattr(candidate, "category", None) == category)


def run_retrospective_eval(
    project_dir: str | Path,
    policies: list[str] | None = None,
    budget: int = 24,
) -> tuple[Path, Path, Path]:
    """Generate deterministic retrospective/proxy benchmark artifacts."""
    root = Path(project_dir).expanduser().resolve()
    framework_dir = ensure_dir(root / "framework")
    state = read_json(root / "state" / "design_state.json")
    evidence = read_json(root / "state" / "evidence_cards.json")
    policies = policies or list(DEFAULT_POLICIES)

    benchmark_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    for policy_name in policies:
        policy = default_policy(policy_name)
        candidates = generate_candidates(state, evidence, strategy=policy_name)
        panel, score = select_panel(candidates, policy=policy, budget=budget)
        unsupported_penalty = len(state.get("unsupported_claims", []))
        benchmark_rows.append(
            {
                "policy": policy_name,
                "budget": budget,
                "candidate_count": len(candidates),
                "panel_count": len(panel),
                "best_score": score,
                "lattice_repair_count": _category_count(panel, "lattice_repair"),
                "champion_count": _category_count(panel, "champion"),
                "control_count": _category_count(panel, "control"),
                "unsupported_claim_penalty": unsupported_penalty,
            }
        )
        for removed_component in ("performance", "contrast_value", "lattice_repair", "coverage_qc"):
            ablated_policy = default_policy(policy_name)
            ablated_policy.weights[removed_component] = 0.0
            ablated_panel, ablated_score = select_panel(candidates, policy=ablated_policy, budget=budget)
            ablation_rows.append(
                {
                    "policy": policy_name,
                    "removed_component": removed_component,
                    "score": ablated_score,
                    "delta_from_full": round(score - score_panel(ablated_panel, ablated_policy), 6),
                    "panel_count": len(ablated_panel),
                }
            )

    benchmark_path = framework_dir / "benchmark_results.csv"
    ablation_path = framework_dir / "ablation_results.csv"
    protocol_path = framework_dir / "evaluation_protocol.md"
    _write_csv(benchmark_path, benchmark_rows)
    _write_csv(ablation_path, ablation_rows)
    protocol_path.write_text(
        "# Evaluation Protocol\n\n"
        "This is a deterministic retrospective scaffold. It uses current standardized "
        "state and candidate-generation outputs to compare policy behavior under a fixed "
        "budget. It is not a substitute for prospective wet-lab validation.\n\n"
        "Metrics: best_score, lattice_repair_count, champion_count, control_count, "
        "unsupported_claim_penalty.\n",
        encoding="utf-8",
    )
    print(f"Wrote retrospective evaluation artifacts under {framework_dir}")
    return benchmark_path, ablation_path, protocol_path

