from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from design_scientist import artifacts
from design_scientist.cli import build_parser, main
from design_scientist.framework import init_framework, review_framework


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
    assert "literature-search" in output
    assert "read-literature" in output
    assert "extract-mechanisms" in output
    assert "extract-methods" not in output
    assert "develop-method" not in output
    assert "benchmark-methods" not in output


def test_framework_artifacts_follow_v3_contract() -> None:
    assert artifacts.FRAMEWORK_ARTIFACTS == artifacts.V3_FRAMEWORK_ARTIFACTS
    assert "framework/method_modules.json" not in artifacts.V3_FRAMEWORK_ARTIFACTS
    assert "framework/algorithm_spec.md" not in artifacts.V3_FRAMEWORK_ARTIFACTS
    assert "framework/benchmark_results.csv" not in artifacts.V3_FRAMEWORK_ARTIFACTS


def test_review_framework_uses_v3_framework_artifact_contract(tmp_path: Path) -> None:
    init_framework(tmp_path, domain="protein variant design")

    missing = review_framework(tmp_path)["missing_artifacts"]
    expected_missing = artifacts.missing_artifacts(tmp_path, artifacts.V3_FRAMEWORK_ARTIFACTS)

    assert missing == expected_missing
    assert "framework/method_modules.json" not in missing
    assert "framework/algorithm_spec.md" not in missing


def test_v2_method_commands_are_not_cli_entrypoints(tmp_path: Path) -> None:
    project = tmp_path / "framework_cli"
    for command in ("extract-methods", "develop-method", "benchmark-methods"):
        with pytest.raises(SystemExit) as exc:
            main([command, str(project)])
        assert exc.value.code == 2


def test_readme_and_framework_markdown_do_not_document_failing_staged_review_workflow() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    framework_markdown = Path("output/pdf/design_scientist_framework_explanation.md").read_text(
        encoding="utf-8"
    )

    assert "`run-scientist` is the recommended full-chain CLI path" in readme
    assert "Staged framework commands are debug/development entry points" in readme
    assert "read-literature" in readme
    assert "extract-mechanisms" in readme
    for forbidden in ("--legacy-v2", "develop-method", "benchmark-methods", "novelty_report"):
        assert forbidden not in readme
        assert forbidden not in framework_markdown


def test_review_framework_missing_artifacts_reports_without_crashing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["review-framework", str(tmp_path)])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "Framework validation failed:" in output
    assert "framework/framework_spec.yaml" in output


def test_cli_review_framework_fresh_init_uses_v3_contract(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / "framework_cli"

    assert main(["init-framework", str(project), "--domain", "protein variant design"]) == 0
    exit_code = main(["review-framework", str(project)])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "Framework validation failed:" in output
    assert "framework/literature_corpus.jsonl" in output
    assert "framework/mechanism_library.json" in output
    assert "framework/method_modules.json" not in output
    assert "framework/algorithm_spec.md" not in output
    assert "framework/method_registry.yaml" not in output


def test_review_framework_rejects_unsafe_run_id_with_argparse_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["review-framework", str(tmp_path), "--run-id", "../escape"])

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "error: run_id must not contain path separators or '..'" in captured.err
    assert "Traceback" not in captured.err


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
    selected_artifacts = selected_record["artifacts"]

    assert journal["version"] == "v3"
    assert journal["stage_sequence"] == list(artifacts.V3_STAGE_SEQUENCE)
    assert "selected_node_novelty_report" not in review_report["artifacts"]
    expected_node_artifacts = {
        "mechanism_spec": "mechanism_spec.json",
        "mechanism": "mechanism.py",
        "proposal": "proposal.json",
        "ablation_plan": "ablation_plan.json",
        "stress_test_plan": "stress_test_plan.json",
        "mechanism_metrics": "mechanism_metrics.json",
        "validation_report": "validation_report.json",
    }
    for key, filename in expected_node_artifacts.items():
        assert f"selected_node_{key}" in review_report["artifacts"]
        path = Path(selected_artifacts[key])
        assert path.name == filename
        assert path.exists()

    proposal = json.loads(Path(selected_artifacts["proposal"]).read_text(encoding="utf-8"))
    mechanism_spec = json.loads(Path(selected_artifacts["mechanism_spec"]).read_text(encoding="utf-8"))
    mechanism_metrics = json.loads(Path(selected_artifacts["mechanism_metrics"]).read_text(encoding="utf-8"))
    assert proposal["literature_basis"]
    assert proposal["literature_gap_ids"]
    assert mechanism_spec["version"] == "v3"
    assert mechanism_spec["components"]
    assert mechanism_metrics["selected_eligible"] is True

    method_report = (run_dir / "method_report.md").read_text(encoding="utf-8")
    assert "## Literature Engine V3" in method_report
    assert "## MechanismSpec Kernel" in method_report
    assert "mechanism_spec.json" in method_report
    assert "novelty_report.json" not in method_report
