from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from design_scientist.artifacts import FRAMEWORK_ARTIFACTS
from design_scientist.cli import build_parser, main
from design_scientist.framework_validation import CRITICAL_FRAMEWORK_ARTIFACTS
from design_scientist.framework import init_framework


def test_init_framework_creates_bootstrap_artifacts(tmp_path: Path) -> None:
    spec = init_framework(tmp_path, domain="antibody design")

    assert spec.framework_id == "antibody_design"
    assert (tmp_path / "framework" / "framework_spec.yaml").exists()
    assert (tmp_path / "framework" / "literature_queries.yaml").exists()
    assert (tmp_path / "framework" / "cache").is_dir()
    assert (tmp_path / "benchmarks" / "synthetic_replay").is_dir()
    assert (tmp_path / "runs").is_dir()

    spec_data = yaml.safe_load((tmp_path / "framework" / "framework_spec.yaml").read_text())
    queries_data = yaml.safe_load((tmp_path / "framework" / "literature_queries.yaml").read_text())
    assert spec_data["domain"] == "antibody design"
    assert queries_data["queries"][0]["status"] == "planned"


def test_cli_help_includes_framework_commands(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "init-framework" in output
    assert "review-framework" in output


def test_cli_help_marks_run_scientist_as_recommended_and_staged_commands_as_debug(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "recommended full-chain" in output
    assert "debug/development" in output


def test_framework_artifacts_follow_current_validation_contract() -> None:
    assert set(FRAMEWORK_ARTIFACTS) == set(CRITICAL_FRAMEWORK_ARTIFACTS.values())
    assert "framework/benchmark_results.csv" not in FRAMEWORK_ARTIFACTS


def test_benchmark_methods_default_run_does_not_overwrite_scientist_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / "framework_cli"
    scientist_run = project / "runs" / "synthetic_replay_seed_1729"
    scientist_run.mkdir(parents=True)
    (scientist_run / "scientist_journal.json").write_text("{}", encoding="utf-8")
    sentinel = scientist_run / "benchmark_results.csv"
    sentinel.write_text("existing scientist benchmark\n", encoding="utf-8")

    assert main(["benchmark-methods", str(project), "--rounds", "1", "--budget", "1"]) == 0

    output = capsys.readouterr().out
    assert "baseline_synthetic_replay_seed_1729" in output
    assert sentinel.read_text(encoding="utf-8") == "existing scientist benchmark\n"
    assert (project / "runs" / "baseline_synthetic_replay_seed_1729" / "benchmark_results.csv").exists()


def test_benchmark_methods_rejects_existing_scientist_run_id(tmp_path: Path) -> None:
    project = tmp_path / "framework_cli"
    scientist_run = project / "runs" / "scientist_run"
    scientist_run.mkdir(parents=True)
    (scientist_run / "scientist_journal.json").write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        main(["benchmark-methods", str(project), "--run-id", "scientist_run"])

    assert exc.value.code == 2


def test_readme_and_framework_markdown_do_not_document_failing_staged_review_workflow() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    framework_markdown = Path("output/pdf/design_scientist_framework_explanation.md").read_text(
        encoding="utf-8"
    )

    assert "`run-scientist` is the recommended full-chain CLI path" in readme
    assert "Staged framework commands are debug/development entry points" in readme
    assert "--legacy-v2" in readme
    assert (
        "uv run design-scientist benchmark-methods /tmp/my_design_scientist_project --rounds 3\n"
        "uv run design-scientist review-framework /tmp/my_design_scientist_project"
    ) not in framework_markdown


def test_review_framework_missing_artifacts_reports_without_crashing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["review-framework", str(tmp_path)])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "Framework validation failed:" in output
    assert "framework/framework_spec.yaml" in output


def test_cli_offline_run_scientist_generates_report_and_validates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture_dir = Path(__file__).parent / "fixtures" / "literature"
    monkeypatch.setenv("DESIGN_SCIENTIST_LITERATURE_FIXTURES", str(fixture_dir))
    project = tmp_path / "framework_cli"

    assert main(["init-framework", str(project), "--domain", "protein variant design"]) == 0
    assert main(
        [
            "run-scientist",
            str(project),
            "--max-papers",
            "5",
            "--nodes",
            "2",
            "--rounds",
            "2",
            "--offline-fixtures",
            "--legacy-v2",
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "Wrote scientist journal:" in output
    assert "Wrote method report:" in output
    assert main(["review-framework", str(project)]) == 0
    review_output = capsys.readouterr().out
    assert "Framework validation passed:" in review_output
    assert main(["review-framework", str(project), "--json"]) == 0
    review_report = json.loads(capsys.readouterr().out)
    assert review_report["valid"]

    run_dirs = [
        path
        for path in (project / "runs").iterdir()
        if (path / "scientist_journal.json").exists()
    ]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    journal = json.loads((run_dir / "scientist_journal.json").read_text(encoding="utf-8"))
    selected_record = next(
        node for node in journal["nodes"] if node["node_id"] == journal["selected_node_id"]
    )
    artifacts = selected_record["artifacts"]

    expected_full_chain_artifacts = {
        "design_space": "design_space.json",
        "candidate_pool_csv": "candidate_pool.csv",
        "candidate_pool_jsonl": "candidate_pool.jsonl",
        "candidate_pool_diagnostics": "candidate_pool_diagnostics.json",
    }
    for key, filename in expected_full_chain_artifacts.items():
        assert f"selected_node_{key}" in review_report["artifacts"]
        path = Path(artifacts[key])
        assert path.name == filename
        assert path.exists()

    proposal = json.loads(Path(artifacts["proposal"]).read_text(encoding="utf-8"))
    novelty = json.loads(Path(artifacts["novelty_report"]).read_text(encoding="utf-8"))
    design_space = json.loads(Path(artifacts["design_space"]).read_text(encoding="utf-8"))
    diagnostics = json.loads(
        Path(artifacts["candidate_pool_diagnostics"]).read_text(encoding="utf-8")
    )
    with Path(artifacts["candidate_pool_csv"]).open("r", encoding="utf-8", newline="") as handle:
        candidate_rows = list(csv.DictReader(handle))
    jsonl_rows = [
        json.loads(line)
        for line in Path(artifacts["candidate_pool_jsonl"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert proposal["literature_basis"]
    assert proposal["literature_gap_ids"]
    assert novelty["baseline_clone"] is False
    assert design_space["operators"]
    assert design_space["target_systems"]
    assert candidate_rows
    assert jsonl_rows
    assert candidate_rows[0]["candidate_id"] == jsonl_rows[0]["candidate_id"]
    assert diagnostics["candidate_count"] == len(jsonl_rows)
    assert isinstance(diagnostics["generator_limited_run"], bool)

    method_report = (run_dir / "method_report.md").read_text(encoding="utf-8")
    assert "## Generator-Limited Diagnostics" in method_report
    assert "design_space.json" in method_report
    assert "candidate_pool.csv" in method_report
    assert "candidate_pool.jsonl" in method_report
    assert "candidate_pool_diagnostics.json" in method_report
    assert "## Literature Basis" in method_report
