from __future__ import annotations

import csv
import json
from pathlib import Path

from design_scientist.framework_validation import review_framework_run
from design_scientist.method_report import write_method_report


def test_complete_v3_framework_run_validates(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)

    write_method_report(project, run_id="v3_unit")
    report = review_framework_run(project, run_id="v3_unit")

    assert report["valid"]
    assert report["summary"]["errors"] == 0


def test_v3_validation_requires_literature_corpus_and_mechanism_library(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)
    (project / "framework" / "literature_corpus.jsonl").unlink()
    (project / "framework" / "mechanism_library.json").unlink()

    write_method_report(project, run_id="v3_unit")
    report = review_framework_run(project, run_id="v3_unit")

    assert not report["valid"]
    assert {"missing_literature_corpus", "missing_mechanism_library"} <= _error_codes(report)


def test_v3_validation_rejects_selected_node_missing_mechanism_spec(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)
    (_selected_workspace(project) / "mechanism_spec.json").unlink()

    write_method_report(project, run_id="v3_unit")
    report = review_framework_run(project, run_id="v3_unit")

    assert not report["valid"]
    assert "missing_selected_node_mechanism_spec" in _error_codes(report)


def test_v3_validation_rejects_selected_architecture_clone(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)
    metrics_path = _selected_workspace(project) / "mechanism_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["architecture_clone"] = True
    _write_json(metrics_path, metrics)
    _rewrite_summary_row(project, {"architecture_clone": "true", "selected_eligible": "false"})

    write_method_report(project, run_id="v3_unit")
    report = review_framework_run(project, run_id="v3_unit")

    assert not report["valid"]
    assert "selected_node_architecture_clone" in _error_codes(report)


def test_v3_validation_rejects_selected_eligible_false(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)
    metrics_path = _selected_workspace(project) / "mechanism_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["selected_eligible"] = False
    _write_json(metrics_path, metrics)
    _rewrite_summary_row(project, {"selected_eligible": "false"})

    write_method_report(project, run_id="v3_unit")
    report = review_framework_run(project, run_id="v3_unit")

    assert not report["valid"]
    assert "selected_mechanism_not_eligible" in _error_codes(report)


def test_v3_validation_requires_required_baseline_summary_rows(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)
    summary_path = project / "runs" / "v3_unit" / "mechanism_benchmark_summary.csv"
    rows = [
        row
        for row in _read_csv_rows(summary_path)
        if row["mechanism"] != "fixed_mix"
    ]
    _write_csv(summary_path, rows)

    write_method_report(project, run_id="v3_unit")
    report = review_framework_run(project, run_id="v3_unit")

    assert not report["valid"]
    assert "mechanism_benchmark_summary_missing_required_baselines" in _error_codes(report)


def _build_v3_framework_run(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    framework_dir = project / "framework"
    run_dir = project / "runs" / "v3_unit"
    workspace = run_dir / "nodes" / "node_literature_kernel"
    framework_dir.mkdir(parents=True)
    workspace.mkdir(parents=True)

    (framework_dir / "framework_spec.yaml").write_text(
        "framework_id: v3_unit\n"
        "domain: protein variant design\n"
        "artifact_root: framework\n",
        encoding="utf-8",
    )
    (framework_dir / "literature_queries.yaml").write_text(
        "queries:\n"
        "  - query_id: q1\n"
        "    domain: protein variant design\n"
        "    query: mechanism-aware active design\n",
        encoding="utf-8",
    )
    (framework_dir / "quest.yaml").write_text(
        "schema_version: 1\n"
        "quest_id: v3_unit\n"
        "domain: protein variant design\n"
        "status: active\n",
        encoding="utf-8",
    )
    _write_json(
        framework_dir / "research_map.json",
        {
            "loops": [{"loop_id": "mechanism_spec_kernel", "status": "active"}],
            "memory": {
                "findings": "framework/findings_memory.jsonl",
                "failures": "framework/failure_memory.jsonl",
            },
        },
    )
    (framework_dir / "findings_memory.jsonl").write_text(
        json.dumps({"summary": "Literature Engine V3 found a reusable mechanism pattern."}) + "\n",
        encoding="utf-8",
    )
    (framework_dir / "failure_memory.jsonl").write_text(
        json.dumps({"summary": "A clone-like mechanism was rejected."}) + "\n",
        encoding="utf-8",
    )
    _write_json(
        framework_dir / "paper_cards.json",
        [
            {
                "paper_id": "paper_active_design",
                "title": "Mechanism-aware active design",
                "source": "fixture",
                "year": 2026,
            }
        ],
    )
    _write_json(
        framework_dir / "literature_search_plan.json",
        {
            "queries": [
                {
                    "query_id": "q1",
                    "query": "mechanism-aware active design",
                    "purpose": "method discovery",
                }
            ]
        },
    )
    _write_json(
        framework_dir / "literature_search_trace.json",
        {"events": [{"event": "offline_fixture_loaded", "paper_id": "paper_active_design"}]},
    )
    _write_json(
        framework_dir / "citation_graph.json",
        {"nodes": [{"paper_id": "paper_active_design"}], "edges": []},
    )
    _write_csv(
        framework_dir / "paper_scores.csv",
        [{"paper_id": "paper_active_design", "total_score": "1.0"}],
    )
    (framework_dir / "literature_corpus.jsonl").write_text(
        json.dumps(
            {
                "paper_id": "paper_active_design",
                "title": "Mechanism-aware active design",
                "chunks": ["Explicit mechanism components support stress-test mapping."],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        framework_dir / "literature_reading_trace.json",
        {"events": [{"event": "read_fulltext", "paper_id": "paper_active_design"}]},
    )
    _write_json(
        framework_dir / "mechanism_cards.json",
        [
            {
                "mechanism_id": "literature_kernel",
                "name": "Literature-guided mechanism kernel",
                "components": ["state_update"],
                "claims": ["reduces unsupported transfer claims"],
            }
        ],
    )
    _write_json(
        framework_dir / "mechanism_library.json",
        {
            "mechanisms": [
                {
                    "mechanism_id": "literature_kernel",
                    "name": "Literature-guided mechanism kernel",
                    "components": [{"component_id": "state_update"}],
                    "claims": ["reduces unsupported transfer claims"],
                }
            ]
        },
    )
    _write_csv(
        framework_dir / "mechanism_gap_matrix.csv",
        [{"mechanism_id": "literature_kernel", "gap": "needs sparse early-round stress test"}],
    )

    _write_json(
        workspace / "mechanism_spec.json",
        {
            "mechanism_id": "literature_kernel",
            "name": "Literature-guided MechanismSpec",
            "version": "v3",
            "components": [
                {
                    "component_id": "state_update",
                    "component_type": "state_model",
                    "description": "Update state from observed assay evidence.",
                }
            ],
            "claims": ["reduces unsupported transfer claims"],
            "literature_basis": ["paper_active_design"],
            "stress_test_requirements": ["sparse_early_round"],
            "ablation_targets": ["state_update"],
        },
    )
    (workspace / "mechanism.py").write_text(
        "def fit_state(context=None):\n"
        "    return context or {}\n",
        encoding="utf-8",
    )
    _write_json(
        workspace / "proposal.json",
        {
            "mechanism_id": "literature_kernel",
            "hypothesis": "MechanismSpec components can map claims to stress tests.",
            "claims": ["reduces unsupported transfer claims"],
        },
    )
    _write_json(
        workspace / "ablation_plan.json",
        {"ablations": [{"ablation_id": "remove_state_update", "removed_components": ["state_update"]}]},
    )
    _write_json(
        workspace / "stress_test_plan.json",
        {
            "worlds": [{"world_id": "sparse_early_round"}],
            "required_baselines": ["random_feasible", "fixed_mix"],
        },
    )
    _write_json(
        workspace / "mechanism_metrics.json",
        {
            "mechanism": "literature_kernel",
            "selected_eligible": True,
            "architecture_clone": False,
            "metrics": {"mean_best_feasible_utility": 0.875},
            "unsupported_claims": ["Synthetic replay does not prove wet-lab superiority."],
        },
    )
    _write_json(
        workspace / "validation_report.json",
        {
            "valid": True,
            "caveats": ["Synthetic replay is not prospective wet-lab evidence."],
        },
    )

    _write_json(
        run_dir / "stage_progress.json",
        {
            "stages": [
                {"stage_id": "read_literature", "status": "completed"},
                {"stage_id": "validate_mechanisms", "status": "completed"},
            ]
        },
    )
    _write_json(
        run_dir / "route_tree.json",
        {
            "selected_path": ["node_literature_kernel"],
            "nodes": [{"node_id": "node_literature_kernel", "status": "selected"}],
        },
    )
    _write_csv(
        run_dir / "mechanism_benchmark_results.csv",
        [
            {
                "world_id": "sparse_early_round",
                "mechanism": "literature_kernel",
                "best_feasible_utility": "0.90",
                "false_claim_rate": "0.02",
            },
            {
                "world_id": "sparse_early_round",
                "mechanism": "random_feasible",
                "best_feasible_utility": "0.40",
                "false_claim_rate": "0.05",
            },
            {
                "world_id": "sparse_early_round",
                "mechanism": "fixed_mix",
                "best_feasible_utility": "0.60",
                "false_claim_rate": "0.06",
            },
            {
                "world_id": "noisy_endpoint",
                "mechanism": "literature_kernel",
                "best_feasible_utility": "0.85",
                "false_claim_rate": "0.03",
            },
            {
                "world_id": "noisy_endpoint",
                "mechanism": "random_feasible",
                "best_feasible_utility": "0.50",
                "false_claim_rate": "0.04",
            },
            {
                "world_id": "noisy_endpoint",
                "mechanism": "fixed_mix",
                "best_feasible_utility": "0.70",
                "false_claim_rate": "0.05",
            },
        ],
    )
    _write_csv(
        run_dir / "mechanism_benchmark_summary.csv",
        [
            {
                "mechanism": "literature_kernel",
                "rank": "1",
                "worlds_tested": "2",
                "majority_win_vs_random_feasible": "true",
                "majority_win_vs_fixed_mix": "true",
                "mean_best_feasible_utility": "0.875",
                "mean_false_claim_rate": "0.025",
                "key_ablation_delta": "0.12",
                "architecture_clone": "false",
                "selected_eligible": "true",
            },
            {
                "mechanism": "random_feasible",
                "rank": "2",
                "worlds_tested": "2",
                "majority_win_vs_random_feasible": "false",
                "majority_win_vs_fixed_mix": "false",
                "mean_best_feasible_utility": "0.45",
                "mean_false_claim_rate": "0.045",
                "key_ablation_delta": "0.0",
                "architecture_clone": "false",
                "selected_eligible": "false",
            },
            {
                "mechanism": "fixed_mix",
                "rank": "3",
                "worlds_tested": "2",
                "majority_win_vs_random_feasible": "true",
                "majority_win_vs_fixed_mix": "false",
                "mean_best_feasible_utility": "0.65",
                "mean_false_claim_rate": "0.055",
                "key_ablation_delta": "0.0",
                "architecture_clone": "false",
                "selected_eligible": "false",
            },
        ],
    )
    _write_csv(
        run_dir / "mechanism_ablation_results.csv",
        [
            {
                "world_id": "sparse_early_round",
                "mechanism": "literature_kernel",
                "ablation": "none",
                "best_feasible_utility": "0.90",
                "delta_from_full_best_feasible_utility": "0.0",
                "false_claim_rate": "0.02",
            },
            {
                "world_id": "sparse_early_round",
                "mechanism": "literature_kernel",
                "ablation": "key_component_removed",
                "best_feasible_utility": "0.78",
                "delta_from_full_best_feasible_utility": "-0.12",
                "false_claim_rate": "0.03",
            },
        ],
    )
    _write_json(
        run_dir / "scientist_journal.json",
        {
            "version": "v3",
            "run_id": "v3_unit",
            "selected_node_id": "node_literature_kernel",
            "selected_node": {
                "node_id": "node_literature_kernel",
                "mechanism": "literature_kernel",
            },
            "nodes": [
                {
                    "node_id": "node_literature_kernel",
                    "status": "completed",
                    "valid": True,
                    "contract": {"valid": True},
                    "mechanism": "literature_kernel",
                    "workspace": str(workspace),
                    "artifacts": {
                        "mechanism_spec": str(workspace / "mechanism_spec.json"),
                        "mechanism": str(workspace / "mechanism.py"),
                        "proposal": str(workspace / "proposal.json"),
                        "ablation_plan": str(workspace / "ablation_plan.json"),
                        "stress_test_plan": str(workspace / "stress_test_plan.json"),
                        "mechanism_metrics": str(workspace / "mechanism_metrics.json"),
                        "validation_report": str(workspace / "validation_report.json"),
                    },
                },
                {
                    "node_id": "node_failed",
                    "status": "failed",
                    "valid": False,
                    "reasons": ["contract missing plan_ablations"],
                },
            ],
            "unsupported_claims": ["Synthetic replay does not prove wet-lab superiority."],
        },
    )
    return project


def _selected_workspace(project: Path) -> Path:
    journal = json.loads((project / "runs" / "v3_unit" / "scientist_journal.json").read_text(encoding="utf-8"))
    selected_id = journal["selected_node_id"]
    selected = next(node for node in journal["nodes"] if node["node_id"] == selected_id)
    return Path(selected["workspace"])


def _rewrite_summary_row(project: Path, updates: dict[str, str]) -> None:
    summary_path = project / "runs" / "v3_unit" / "mechanism_benchmark_summary.csv"
    rows = _read_csv_rows(summary_path)
    for row in rows:
        if row["mechanism"] == "literature_kernel":
            row.update(updates)
    _write_csv(summary_path, rows)


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    assert rows
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _error_codes(report: dict[str, object]) -> set[str]:
    findings = report["findings"]
    assert isinstance(findings, list)
    return {
        finding["code"]
        for finding in findings
        if isinstance(finding, dict) and finding.get("severity") == "error"
    }
