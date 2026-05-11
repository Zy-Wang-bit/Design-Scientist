"""V3 mechanism-node contract, execution harness, and path guard."""

from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from design_scientist.artifacts import V3_MECHANISM_NODE_ARTIFACTS
from design_scientist.method_nodes import (
    FilesystemChange,
    diff_filesystem_snapshots,
    find_escaping_symlinks,
    snapshot_filesystem,
)
from design_scientist.method_nodes import (
    _coerce_subprocess_output,
    _is_relative_to,
    _normalize_guard_roots,
    _read_json_object,
    _resolve_workspace,
    _subprocess_env,
)


V3_MECHANISM_NODE_REQUIRED_ARTIFACTS = V3_MECHANISM_NODE_ARTIFACTS
V3_MECHANISM_NODE_JSON_ARTIFACTS = tuple(
    artifact for artifact in V3_MECHANISM_NODE_REQUIRED_ARTIFACTS if artifact.endswith(".json")
)
V3_MECHANISM_NODE_GENERATED_ARTIFACTS = tuple(
    artifact for artifact in V3_MECHANISM_NODE_REQUIRED_ARTIFACTS if artifact != "mechanism.py"
)
MECHANISM_NODE_LIFECYCLE_CALLABLES = (
    "fit_state",
    "generate_candidates",
    "score_candidates",
    "select_panel",
    "plan_ablations",
)
DEFAULT_EXECUTION_TIMEOUT_SECONDS = 30.0

FAILURE_CONTRACT = "contract"
FAILURE_SUBPROCESS_NONZERO = "subprocess_nonzero"
FAILURE_TIMEOUT = "timeout"
FAILURE_MISSING_ARTIFACTS = "missing_artifacts"
FAILURE_MALFORMED_ARTIFACTS = "malformed_artifacts"
FAILURE_PATH_GUARD = "path_guard"
FAILURE_CALLABLE_IMPORT = "callable_import"

_SUBPROCESS_CODE = (
    "from design_scientist.mechanism_nodes import _mechanism_node_subprocess_main\n"
    "_mechanism_node_subprocess_main()\n"
)


class MechanismNodeError(RuntimeError):
    """Base exception for V3 mechanism-node harness failures."""


class MechanismNodeContractError(MechanismNodeError):
    """Raised when a node cannot be loaded under the V3 mechanism contract."""


@dataclass(frozen=True)
class MechanismNode:
    """Loaded V3 mechanism-node metadata."""

    workspace: Path
    mechanism_path: Path


@dataclass(frozen=True)
class MechanismNodeValidationResult:
    """Artifact and lifecycle validation result for a V3 mechanism-node workspace."""

    workspace: Path
    valid: bool
    missing_artifacts: list[str] = field(default_factory=list)
    malformed_json: dict[str, str] = field(default_factory=dict)
    escaping_symlinks: list[str] = field(default_factory=list)
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MechanismNodeExecutionResult:
    """Result of importing, executing, validating, and loading a V3 mechanism node."""

    workspace: Path
    valid: bool
    executed: bool
    failure_kinds: list[str] = field(default_factory=list)
    missing_artifacts: list[str] = field(default_factory=list)
    malformed_json: dict[str, str] = field(default_factory=dict)
    out_of_bounds_writes: list[str] = field(default_factory=list)
    escaping_symlinks: list[str] = field(default_factory=list)
    exception: str | None = None
    subprocess_returncode: int | None = None
    subprocess_stdout: str = ""
    subprocess_stderr: str = ""
    timed_out: bool = False
    timeout_seconds: float | None = None
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    filesystem_changes: list[FilesystemChange] = field(default_factory=list)
    snapshot_roots: list[Path] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    callables: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True)
class _MechanismNodeSubprocessResult:
    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


def load_mechanism_node(node_workspace: str | Path) -> MechanismNode:
    """Load V3 mechanism-node metadata without importing ``mechanism.py``."""

    workspace = _resolve_workspace(node_workspace)
    if not workspace.is_dir():
        raise MechanismNodeContractError(f"Node workspace does not exist: {workspace}")

    mechanism_path = workspace / "mechanism.py"
    if not mechanism_path.is_file():
        raise MechanismNodeContractError("Missing required artifact: mechanism.py")

    static_errors = _lifecycle_static_errors(mechanism_path)
    if static_errors:
        raise MechanismNodeContractError("; ".join(static_errors))

    return MechanismNode(workspace=workspace, mechanism_path=mechanism_path)


def validate_mechanism_node(
    node_workspace: str | Path,
    *,
    require_generated_artifacts: bool = True,
) -> MechanismNodeValidationResult:
    """Validate V3 mechanism-node artifacts, JSON shape, lifecycle API, and symlinks."""

    workspace = _resolve_workspace(node_workspace)
    required = (
        V3_MECHANISM_NODE_REQUIRED_ARTIFACTS
        if require_generated_artifacts
        else ("mechanism.py",)
    )
    missing_artifacts = [
        artifact
        for artifact in required
        if not (workspace / artifact).exists()
    ]
    malformed_json: dict[str, str] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    errors: list[str] = []

    mechanism_path = workspace / "mechanism.py"
    if mechanism_path.exists() and not mechanism_path.is_file():
        errors.append("mechanism.py must be a file")
    elif mechanism_path.is_file():
        errors.extend(_lifecycle_static_errors(mechanism_path))

    for artifact in V3_MECHANISM_NODE_JSON_ARTIFACTS:
        path = workspace / artifact
        if not path.exists():
            continue
        if not path.is_file():
            malformed_json[artifact] = f"JSON artifact must be a file: {artifact}"
            continue
        try:
            data = _read_json_object(path)
        except ValueError as exc:
            malformed_json[artifact] = str(exc)
            continue
        if not data:
            malformed_json[artifact] = f"Expected non-empty JSON object in {artifact}"
            continue
        artifacts[artifact] = data

    validation_report = artifacts.get("validation_report.json")
    if validation_report is not None and validation_report.get("valid") is not True:
        errors.append("validation_report.json must contain valid: true")

    escaping_symlinks = find_escaping_symlinks(workspace)
    valid = not missing_artifacts and not malformed_json and not escaping_symlinks and not errors
    return MechanismNodeValidationResult(
        workspace=workspace,
        valid=valid,
        missing_artifacts=missing_artifacts,
        malformed_json=malformed_json,
        escaping_symlinks=[str(path) for path in escaping_symlinks],
        artifacts=artifacts,
        errors=errors,
    )


def execute_mechanism_node(
    node_workspace: str | Path,
    *,
    guard_roots: Iterable[str | Path] | None = None,
    timeout_seconds: float | None = DEFAULT_EXECUTION_TIMEOUT_SECONDS,
) -> MechanismNodeExecutionResult:
    """Execute a V3 mechanism node in a subprocess and validate generated artifacts.

    The subprocess cwd is the node workspace. It imports ``mechanism.py``,
    verifies the V3 lifecycle callables, and calls ``run(workspace)`` only when
    that function is present. Legacy ``select_batch`` is never used as an
    execution contract.
    """

    workspace = _resolve_workspace(node_workspace)
    snapshot_roots = _normalize_guard_roots(workspace, guard_roots)
    before = snapshot_filesystem(snapshot_roots)
    executed = False
    exception: str | None = None
    errors: list[str] = []
    exported_callables: dict[str, Any] = {}
    subprocess_result = _MechanismNodeSubprocessResult(returncode=None)
    callable_import_failed = False
    contract_failed = False

    try:
        preflight = validate_mechanism_node(workspace, require_generated_artifacts=False)
        if preflight.escaping_symlinks:
            escaped = ", ".join(preflight.escaping_symlinks)
            raise MechanismNodeContractError(
                f"Node workspace has symlinks escaping workspace: {escaped}"
            )
        if preflight.missing_artifacts or preflight.malformed_json or preflight.errors:
            preflight_errors = [
                *(f"Missing required artifact: {artifact}" for artifact in preflight.missing_artifacts),
                *preflight.malformed_json.values(),
                *preflight.errors,
            ]
            raise MechanismNodeContractError("; ".join(preflight_errors))
        load_mechanism_node(workspace)
    except MechanismNodeContractError as exc:
        contract_failed = True
        errors.append(str(exc))
    except Exception as exc:
        contract_failed = True
        errors.append(f"Preflight failed: {exc}")

    if not errors:
        subprocess_result = _execute_mechanism_subprocess(
            workspace,
            timeout_seconds=timeout_seconds,
        )
        if subprocess_result.timed_out:
            exception = (
                subprocess_result.stderr.strip()
                or f"mechanism.py timed out after {timeout_seconds} seconds"
            )
        elif subprocess_result.returncode == 0:
            executed = True
        else:
            exception = (
                subprocess_result.stderr.strip()
                or f"mechanism.py exited with code {subprocess_result.returncode}"
            )

    after = snapshot_filesystem(snapshot_roots)
    filesystem_changes = diff_filesystem_snapshots(before, after)
    out_of_bounds_writes = _out_of_bounds_writes(filesystem_changes, workspace)
    validation = validate_mechanism_node(workspace)
    errors.extend(validation.errors)

    preliminary_valid = (
        executed
        and exception is None
        and validation.valid
        and not out_of_bounds_writes
        and not errors
    )
    if preliminary_valid:
        parent_import_before = snapshot_filesystem(snapshot_roots)
        try:
            exported_callables = load_mechanism_node_callables(workspace)
        except MechanismNodeContractError as exc:
            callable_import_failed = True
            errors.append(str(exc))
        except Exception:
            callable_import_failed = True
            exception = traceback.format_exc()
        parent_import_after = snapshot_filesystem(snapshot_roots)
        parent_import_changes = diff_filesystem_snapshots(
            parent_import_before,
            parent_import_after,
        )
        filesystem_changes.extend(parent_import_changes)
        out_of_bounds_writes.extend(
            change
            for change in _out_of_bounds_writes(parent_import_changes, workspace)
            if change not in out_of_bounds_writes
        )

    failure_kinds = _execution_failure_kinds(
        contract_failed=contract_failed,
        subprocess_result=subprocess_result,
        validation=validation,
        out_of_bounds_writes=out_of_bounds_writes,
        callable_import_failed=callable_import_failed,
    )
    valid = preliminary_valid and not callable_import_failed and not out_of_bounds_writes
    return MechanismNodeExecutionResult(
        workspace=workspace,
        valid=valid,
        executed=executed,
        failure_kinds=failure_kinds,
        missing_artifacts=validation.missing_artifacts,
        malformed_json=validation.malformed_json,
        out_of_bounds_writes=sorted(out_of_bounds_writes),
        escaping_symlinks=validation.escaping_symlinks,
        exception=exception,
        subprocess_returncode=subprocess_result.returncode,
        subprocess_stdout=subprocess_result.stdout,
        subprocess_stderr=subprocess_result.stderr,
        timed_out=subprocess_result.timed_out,
        timeout_seconds=timeout_seconds,
        artifacts=validation.artifacts,
        filesystem_changes=filesystem_changes,
        snapshot_roots=snapshot_roots,
        errors=errors,
        callables=exported_callables if valid else {},
    )


def load_mechanism_node_callables(node_workspace: str | Path | MechanismNode) -> dict[str, Any]:
    """Import ``mechanism.py`` and return V3 lifecycle callables without calling them."""

    node = (
        node_workspace
        if isinstance(node_workspace, MechanismNode)
        else load_mechanism_node(node_workspace)
    )
    module = _load_mechanism_module(node)
    runtime_errors = _lifecycle_runtime_errors(module)
    if runtime_errors:
        raise MechanismNodeContractError("; ".join(runtime_errors))
    return {
        name: getattr(module, name)
        for name in MECHANISM_NODE_LIFECYCLE_CALLABLES
    }


def _execute_mechanism_subprocess(
    workspace: Path,
    *,
    timeout_seconds: float | None,
) -> _MechanismNodeSubprocessResult:
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                _SUBPROCESS_CODE,
                os.fspath(workspace),
            ],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=_subprocess_env(),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stderr = _coerce_subprocess_output(exc.stderr)
        timeout_message = f"mechanism.py timed out after {timeout_seconds} seconds"
        if stderr:
            stderr = f"{stderr.rstrip()}\n{timeout_message}"
        else:
            stderr = timeout_message
        return _MechanismNodeSubprocessResult(
            returncode=None,
            stdout=_coerce_subprocess_output(exc.stdout),
            stderr=stderr,
            timed_out=True,
        )
    return _MechanismNodeSubprocessResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        timed_out=False,
    )


def _mechanism_node_subprocess_main() -> None:
    workspace = _resolve_workspace(sys.argv[1])
    node = load_mechanism_node(workspace)
    module = _load_mechanism_module(node)
    runtime_errors = _lifecycle_runtime_errors(module)
    if runtime_errors:
        raise MechanismNodeContractError("; ".join(runtime_errors))

    entrypoint = getattr(module, "run", None)
    if entrypoint is None:
        return
    if not callable(entrypoint):
        raise MechanismNodeContractError("mechanism.py run attribute must be callable when present")
    entrypoint(workspace)


def _load_mechanism_module(node: MechanismNode) -> Any:
    module_name = f"_design_scientist_mechanism_node_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, node.mechanism_path)
    if spec is None or spec.loader is None:
        raise MechanismNodeContractError(f"Cannot load mechanism.py from {node.mechanism_path}")

    module = importlib.util.module_from_spec(spec)
    previous_cwd = Path.cwd()
    previous_sys_path = list(sys.path)
    sys.modules[module_name] = module
    try:
        os.chdir(node.workspace)
        sys.path.insert(0, str(node.workspace))
        spec.loader.exec_module(module)
        return module
    finally:
        os.chdir(previous_cwd)
        sys.path[:] = previous_sys_path
        sys.modules.pop(module_name, None)


def _lifecycle_static_errors(mechanism_path: Path) -> list[str]:
    try:
        tree = ast.parse(mechanism_path.read_text(encoding="utf-8"), filename=os.fspath(mechanism_path))
    except SyntaxError as exc:
        return [f"Syntax error in mechanism.py: {exc.msg}"]
    except OSError as exc:
        return [f"Cannot read mechanism.py: {exc}"]

    defined_functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing = [
        name
        for name in MECHANISM_NODE_LIFECYCLE_CALLABLES
        if name not in defined_functions
    ]
    if not missing:
        return []
    if defined_functions == {"select_batch"}:
        return [
            "mechanism.py defines only legacy select_batch; V3 mechanism nodes must define "
            "lifecycle callables: "
            + ", ".join(MECHANISM_NODE_LIFECYCLE_CALLABLES)
        ]
    return [
        "mechanism.py missing lifecycle callables: "
        + ", ".join(missing)
    ]


def _lifecycle_runtime_errors(module: Any) -> list[str]:
    missing_or_not_callable = [
        name
        for name in MECHANISM_NODE_LIFECYCLE_CALLABLES
        if not callable(getattr(module, name, None))
    ]
    if not missing_or_not_callable:
        return []
    if callable(getattr(module, "select_batch", None)) and len(missing_or_not_callable) == len(
        MECHANISM_NODE_LIFECYCLE_CALLABLES
    ):
        return [
            "mechanism.py defines only legacy select_batch; V3 mechanism nodes must define "
            "lifecycle callables: "
            + ", ".join(MECHANISM_NODE_LIFECYCLE_CALLABLES)
        ]
    return [
        "mechanism.py missing callable lifecycle attributes: "
        + ", ".join(missing_or_not_callable)
    ]


def _out_of_bounds_writes(
    filesystem_changes: Iterable[FilesystemChange],
    workspace: Path,
) -> list[str]:
    return [
        str(change.path)
        for change in filesystem_changes
        if not _is_relative_to(change.path, workspace)
    ]


def _execution_failure_kinds(
    *,
    contract_failed: bool,
    subprocess_result: _MechanismNodeSubprocessResult,
    validation: MechanismNodeValidationResult,
    out_of_bounds_writes: list[str],
    callable_import_failed: bool,
) -> list[str]:
    kinds: list[str] = []
    if contract_failed or validation.errors:
        kinds.append(FAILURE_CONTRACT)
    if subprocess_result.timed_out:
        kinds.append(FAILURE_TIMEOUT)
    elif subprocess_result.returncode not in (None, 0):
        kinds.append(FAILURE_SUBPROCESS_NONZERO)
    if validation.missing_artifacts:
        kinds.append(FAILURE_MISSING_ARTIFACTS)
    if validation.malformed_json:
        kinds.append(FAILURE_MALFORMED_ARTIFACTS)
    if out_of_bounds_writes or validation.escaping_symlinks:
        kinds.append(FAILURE_PATH_GUARD)
    if callable_import_failed:
        kinds.append(FAILURE_CALLABLE_IMPORT)
    return kinds


__all__ = [
    "DEFAULT_EXECUTION_TIMEOUT_SECONDS",
    "FAILURE_CALLABLE_IMPORT",
    "FAILURE_CONTRACT",
    "FAILURE_MALFORMED_ARTIFACTS",
    "FAILURE_MISSING_ARTIFACTS",
    "FAILURE_PATH_GUARD",
    "FAILURE_SUBPROCESS_NONZERO",
    "FAILURE_TIMEOUT",
    "MECHANISM_NODE_LIFECYCLE_CALLABLES",
    "MechanismNode",
    "MechanismNodeContractError",
    "MechanismNodeError",
    "MechanismNodeExecutionResult",
    "MechanismNodeValidationResult",
    "V3_MECHANISM_NODE_GENERATED_ARTIFACTS",
    "V3_MECHANISM_NODE_JSON_ARTIFACTS",
    "V3_MECHANISM_NODE_REQUIRED_ARTIFACTS",
    "execute_mechanism_node",
    "load_mechanism_node",
    "load_mechanism_node_callables",
    "validate_mechanism_node",
]
