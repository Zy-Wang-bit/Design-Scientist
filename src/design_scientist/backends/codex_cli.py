"""Local Codex CLI backend for controlled workspace-agent execution."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from design_scientist.backends.base import ModelBackend, WorkspaceAgentBackend
from design_scientist.schemas import (
    ModelRequest,
    ModelResponse,
    WorkspaceAgentResult,
    WorkspaceAgentTask,
    to_plain_data,
)


_TRANSCRIPT_PATTERNS = (
    re.compile(r"\btranscript[_ -]?id\b\s*[:=]\s*([A-Za-z0-9_.:/@+-]+)", re.IGNORECASE),
    re.compile(r"\bconversation[_ -]?id\b\s*[:=]\s*([A-Za-z0-9_.:/@+-]+)", re.IGNORECASE),
)


class CodexCliBackend(ModelBackend, WorkspaceAgentBackend):
    """Run local Codex CLI commands behind the Design Scientist backend API."""

    def __init__(
        self,
        codex_binary: str = "codex",
        timeout_seconds: int | None = 300,
        extra_args: list[str] | None = None,
    ) -> None:
        self.codex_binary = codex_binary
        self.timeout_seconds = timeout_seconds
        self.extra_args = list(extra_args if extra_args is not None else ["--skip-git-repo-check"])

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Run a read-only Codex model request.

        This keeps model-style calls available through the same local backend,
        while reserving workspace writes for explicit ``WorkspaceAgentTask`` patch
        mode.
        """
        prompt = self._model_prompt(request)
        completed = self._run_codex(
            prompt=prompt,
            cwd=Path.cwd(),
            sandbox="read-only",
            output_schema=request.response_schema,
        )
        structured = _parse_json_object(completed.stdout)
        return ModelResponse(
            text=_summarize_streams(completed.stdout, completed.stderr),
            structured=structured,
            raw_response={
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "returncode": completed.returncode,
            },
            transcript_id=_extract_transcript_id(completed.stdout, completed.stderr),
        )

    def run_model(self, request: ModelRequest) -> ModelResponse:
        """Backward-compatible alias for early internal callers."""
        return self.complete(request)

    def run_task(self, task: WorkspaceAgentTask) -> WorkspaceAgentResult:
        workspace = Path(task.workspace).expanduser().resolve()
        if not workspace.exists():
            raise ValueError(f"Workspace does not exist: {workspace}")
        if not workspace.is_dir():
            raise ValueError(f"Workspace is not a directory: {workspace}")

        allowed_paths = _resolve_allowed_paths(task.allowed_paths, workspace)
        if task.mode == "patch" and not _path_is_within_any(workspace, allowed_paths):
            raise ValueError("Patch mode workspace must be within allowed_paths")

        sandbox = "workspace-write" if task.mode == "patch" else "read-only"
        prompt = self._task_prompt(task, workspace, allowed_paths)
        completed = self._run_codex(
            prompt=prompt,
            cwd=workspace,
            sandbox=sandbox,
            output_schema=task.output_schema,
        )
        structured = _parse_json_object(completed.stdout)
        return WorkspaceAgentResult(
            summary=_summarize_streams(completed.stdout, completed.stderr),
            structured=structured,
            transcript_id=_extract_transcript_id(completed.stdout, completed.stderr),
            returncode=completed.returncode,
            artifacts={
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
        )

    def _run_codex(
        self,
        *,
        prompt: str,
        cwd: Path,
        sandbox: str,
        output_schema: dict[str, Any] | None,
    ) -> subprocess.CompletedProcess[str]:
        cmd = [self.codex_binary, "exec", "-s", sandbox, *self.extra_args]

        with _temporary_schema_file(output_schema) as schema_path:
            if schema_path is not None:
                cmd.extend(["--output-schema", str(schema_path)])
            cmd.append(prompt)
            return subprocess.run(
                cmd,
                cwd=str(cwd),
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )

    def _task_prompt(
        self,
        task: WorkspaceAgentTask,
        workspace: Path,
        allowed_paths: list[Path],
    ) -> str:
        payload = {
            "objective": task.objective,
            "mode": task.mode,
            "workspace": str(workspace),
            "allowed_paths": [str(path) for path in allowed_paths],
            "write_policy": task.write_policy,
            "commands_allowed": task.commands_allowed,
            "output_schema": task.output_schema,
        }
        return (
            "You are running as the local Design Scientist workspace agent.\n"
            "Respect the mode, workspace, allowed paths, and command policy below.\n"
            "Return a concise summary and structured JSON when an output schema is provided.\n\n"
            f"{json.dumps(to_plain_data(payload), indent=2, sort_keys=True)}"
        )

    def _model_prompt(self, request: ModelRequest) -> str:
        payload = {
            "purpose": request.purpose,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "context_refs": request.context_refs,
            "response_schema": request.response_schema,
        }
        return (
            "You are the local Design Scientist model backend.\n"
            "Answer the request below. Return structured JSON when a response schema is provided.\n\n"
            f"{json.dumps(to_plain_data(payload), indent=2, sort_keys=True)}"
        )


class _temporary_schema_file:
    def __init__(self, schema: dict[str, Any] | None) -> None:
        self.schema = schema
        self.path: Path | None = None

    def __enter__(self) -> Path | None:
        if self.schema is None:
            return None
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            prefix="design-scientist-codex-schema-",
            delete=False,
        ) as handle:
            json.dump(to_plain_data(self.schema), handle, indent=2, sort_keys=True)
            handle.write("\n")
            self.path = Path(handle.name)
        return self.path

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.path is not None:
            self.path.unlink(missing_ok=True)


def _resolve_allowed_paths(paths: list[str], workspace: Path) -> list[Path]:
    if not paths:
        return []
    resolved: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = workspace / path
        resolved.append(path.resolve())
    return resolved


def _path_is_within_any(path: Path, allowed_paths: list[Path]) -> bool:
    resolved = path.resolve()
    return any(resolved == allowed or resolved.is_relative_to(allowed) for allowed in allowed_paths)


def _parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _extract_transcript_id(stdout: str, stderr: str) -> str | None:
    for stream in (stdout, stderr):
        structured = _parse_json_object(stream)
        if structured is not None:
            transcript_id = structured.get("transcript_id") or structured.get("conversation_id")
            if transcript_id is not None:
                return str(transcript_id)
        for pattern in _TRANSCRIPT_PATTERNS:
            match = pattern.search(stream)
            if match:
                return match.group(1)
    return None


def _summarize_streams(stdout: str, stderr: str, max_chars: int = 4000) -> str:
    parts: list[str] = []
    if stdout.strip():
        parts.append(f"stdout:\n{stdout.strip()}")
    if stderr.strip():
        parts.append(f"stderr:\n{stderr.strip()}")
    summary = "\n\n".join(parts)
    if len(summary) <= max_chars:
        return summary
    return f"{summary[: max_chars - 15].rstrip()}\n...[truncated]"
