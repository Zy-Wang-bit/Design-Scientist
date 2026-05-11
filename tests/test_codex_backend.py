from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from design_scientist.backends.codex_cli import CodexCliBackend
from design_scientist.schemas import WorkspaceAgentTask


def test_read_only_plan_uses_read_only_sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append({"cmd": cmd, **kwargs})
        return subprocess.CompletedProcess(cmd, 0, stdout="plan\ntranscript_id: abc123\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    task = WorkspaceAgentTask(
        objective="Inspect the workspace",
        workspace=str(tmp_path),
        mode="read_only_plan",
        allowed_paths=[str(tmp_path)],
    )
    result = CodexCliBackend().run_task(task)

    assert result.returncode == 0
    assert result.transcript_id == "abc123"
    assert result.summary.startswith("stdout:\nplan")
    assert calls[0]["cmd"][:4] == ["codex", "exec", "-s", "read-only"]
    assert "--skip-git-repo-check" in calls[0]["cmd"]
    assert calls[0]["cwd"] == str(tmp_path)
    assert calls[0]["capture_output"] is True
    assert calls[0]["check"] is False


def test_verify_uses_read_only_sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: list[str] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.extend(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="verified")

    monkeypatch.setattr(subprocess, "run", fake_run)

    task = WorkspaceAgentTask(
        objective="Run checks",
        workspace=str(tmp_path),
        mode="verify",
        allowed_paths=[str(tmp_path)],
    )
    result = CodexCliBackend().run_task(task)

    assert captured[:4] == ["codex", "exec", "-s", "read-only"]
    assert result.summary == "stderr:\nverified"


def test_patch_uses_workspace_write_and_requires_workspace_within_allowed_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[str] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.extend(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout='{"transcript_id": "tid-1", "ok": true}', stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    task = WorkspaceAgentTask(
        objective="Patch files",
        workspace=str(tmp_path),
        mode="patch",
        allowed_paths=[str(tmp_path.parent)],
    )
    result = CodexCliBackend().run_task(task)

    assert captured[:4] == ["codex", "exec", "-s", "workspace-write"]
    assert result.structured == {"transcript_id": "tid-1", "ok": True}
    assert result.transcript_id == "tid-1"


def test_patch_rejects_workspace_outside_allowed_paths(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    task = WorkspaceAgentTask(
        objective="Patch files",
        workspace=str(outside),
        mode="patch",
        allowed_paths=[str(allowed)],
    )

    with pytest.raises(ValueError, match="Patch mode workspace must be within allowed_paths"):
        CodexCliBackend().run_task(task)


def test_output_schema_is_written_to_temporary_file_and_removed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen_schema_paths: list[Path] = []
    seen_schema_payloads: list[dict[str, object]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        schema_path = Path(cmd[cmd.index("--output-schema") + 1])
        seen_schema_paths.append(schema_path)
        seen_schema_payloads.append(json.loads(schema_path.read_text()))
        return subprocess.CompletedProcess(cmd, 7, stdout="{}", stderr="warning")

    monkeypatch.setattr(subprocess, "run", fake_run)

    schema = {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
    }
    task = WorkspaceAgentTask(
        objective="Return JSON",
        workspace=str(tmp_path),
        mode="read_only_plan",
        allowed_paths=[str(tmp_path)],
        output_schema=schema,
    )
    result = CodexCliBackend(codex_binary="codex-test").run_task(task)

    assert result.returncode == 7
    assert seen_schema_payloads == [schema]
    assert not seen_schema_paths[0].exists()
