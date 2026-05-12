#!/usr/bin/env python3
"""Lightweight validator for design-scientist project artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def check(condition: bool, ok: str, bad: str, results: list[dict]):
    results.append({"ok": bool(condition), "message": ok if condition else bad})


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: validate_project.py <project_dir>", file=sys.stderr)
        return 2

    project_dir = Path(argv[1]).expanduser().resolve()
    results: list[dict] = []

    check(project_dir.exists(), f"project exists: {project_dir}", f"missing project: {project_dir}", results)
    if not project_dir.exists():
        print(json.dumps({"ok": False, "checks": results}, indent=2))
        return 1

    for name in ["project.yaml", "data_contract.yaml", "estimands.yaml"]:
        path = project_dir / name
        check(path.exists(), f"found {name}", f"missing {name}", results)

    state_path = project_dir / "state" / "design_state.json"
    evidence_path = project_dir / "state" / "evidence_cards.json"
    check(state_path.exists(), "found state/design_state.json", "missing state/design_state.json", results)
    check(evidence_path.exists(), "found state/evidence_cards.json", "missing state/evidence_cards.json", results)

    if state_path.exists():
        try:
            state = load_json(state_path)
            check(isinstance(state, dict), "design_state.json is an object", "design_state.json is not an object", results)
            for key in ["round", "champions", "modules", "policy_state"]:
                check(key in state, f"design_state has {key}", f"design_state missing {key}", results)
        except Exception as exc:
            check(False, "", f"could not parse design_state.json: {exc}", results)

    if evidence_path.exists():
        try:
            evidence = load_json(evidence_path)
            check(isinstance(evidence, list), "evidence_cards.json is a list", "evidence_cards.json is not a list", results)
            if isinstance(evidence, list):
                missing_tier = [
                    idx for idx, item in enumerate(evidence)
                    if not isinstance(item, dict) or "evidence_tier" not in item
                ]
                check(
                    not missing_tier,
                    "all evidence cards include evidence_tier",
                    f"evidence cards missing evidence_tier at indices {missing_tier[:10]}",
                    results,
                )
        except Exception as exc:
            check(False, "", f"could not parse evidence_cards.json: {exc}", results)

    run_dirs = sorted((project_dir / "runs").glob("*")) if (project_dir / "runs").exists() else []
    if run_dirs:
        latest = run_dirs[-1]
        for name in ["candidate_pool.csv", "panel_recommendation.csv", "policy_comparison.csv", "decision_report.md"]:
            path = latest / name
            check(path.exists(), f"latest run has {name}", f"latest run missing {name}", results)
            if path.suffix == ".csv" and path.exists():
                check(count_csv_rows(path) > 0, f"{name} has rows", f"{name} has no data rows", results)

    overall_ok = all(item["ok"] for item in results)
    print(json.dumps({"ok": overall_ok, "project_dir": str(project_dir), "checks": results}, indent=2))
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
