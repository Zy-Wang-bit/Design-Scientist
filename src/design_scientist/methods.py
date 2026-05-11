"""Method hypothesis and policy registry generation."""

from __future__ import annotations

from pathlib import Path

from design_scientist.io import ensure_dir, write_yaml


METHOD_HYPOTHESES = [
    {
        "method_id": "performance_first_mechanism_aware_acquisition",
        "claim": (
            "If panel selection combines expected feasible performance with champion-centered "
            "contrast value and QC guardrails, it should find better usable variants than pure "
            "ratio ranking or pure lattice repair."
        ),
        "must_beat": ["top observed ratio", "pure lattice repair", "fixed allocation"],
    },
    {
        "method_id": "adaptive_batch_allocation",
        "claim": (
            "If batch allocation shifts between champion expansion, contrasts, lattice repair, "
            "and controls based on state maturity, it should outperform fixed allocation."
        ),
        "must_beat": ["fixed_mix"],
    },
    {
        "method_id": "matched_contrast_guardrail",
        "claim": (
            "If module claims require exact or explicitly labeled near-matched evidence, false "
            "positive advancement should drop relative to descriptive module ranking."
        ),
        "must_beat": ["top predicted utility without evidence tiers"],
    },
    {
        "method_id": "retrospective_policy_replay",
        "claim": (
            "If policy changes are evaluated by retrospective masking/replay before prospective "
            "panel selection, method revisions should be auditable and less anecdotal."
        ),
        "must_beat": ["single-run anecdotal selection"],
    },
]


def propose_method_hypotheses(project_dir: str | Path) -> tuple[Path, Path]:
    root = Path(project_dir).expanduser().resolve()
    framework_dir = ensure_dir(root / "framework")
    hypothesis_path = framework_dir / "method_hypotheses.md"
    registry_path = framework_dir / "policy_registry.yaml"

    lines = ["# Method Hypotheses", ""]
    for item in METHOD_HYPOTHESES:
        lines.extend(
            [
                f"## {item['method_id']}",
                item["claim"],
                "",
                "Must beat:",
                *[f"- {baseline}" for baseline in item["must_beat"]],
                "",
            ]
        )
    hypothesis_path.write_text("\n".join(lines), encoding="utf-8")

    registry = {
        "policies": [
            {
                "name": "mechanism_aware",
                "stage": "default",
                "weights": {
                    "performance": 1.0,
                    "decision_value": 0.7,
                    "contrast_value": 0.45,
                    "lattice_repair": 0.35,
                    "coverage_qc": 0.25,
                    "risk": -0.7,
                },
                "batch_allocation": {
                    "champion": "50-80%",
                    "champion_contrast": "10-35%",
                    "lattice_repair_or_interaction": "5-30%",
                    "controls": "5-15%",
                },
                "baselines": [
                    "random feasible",
                    "top observed ratio",
                    "top predicted utility",
                    "pure uncertainty",
                    "pure lattice repair",
                    "fixed mix",
                ],
                "ablations": [
                    "remove performance",
                    "remove contrast value",
                    "remove lattice repair",
                    "remove coverage QC",
                ],
                "evaluation": "retrospective replay plus prospective validation report",
            }
        ]
    }
    write_yaml(registry_path, registry)
    print(f"Wrote method hypotheses: {hypothesis_path}")
    print(f"Wrote policy registry: {registry_path}")
    return hypothesis_path, registry_path

