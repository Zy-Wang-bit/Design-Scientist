"""V3 mechanism-node contract, execution harness, and path guard."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import os
import subprocess
import sys
import tempfile
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
    _snapshot_entry,
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
FAILURE_STALE_ARTIFACTS = "stale_artifacts"
FAILURE_CALLABLE_IMPORT = "callable_import"

_SUBPROCESS_CODE = (
    "from design_scientist.mechanism_nodes import _mechanism_node_subprocess_main\n"
    "_mechanism_node_subprocess_main()\n"
)
_PATH_GUARD_ERROR_PREFIX = "path_guard: out_of_bounds_write:"
_STALE_ARTIFACTS_ERROR_PREFIX = (
    "stale_artifacts: generated artifacts not updated by run(workspace):"
)
_AUDIT_BLOCKED_PROCESS_EVENTS = {
    "os.exec",
    "os.fork",
    "os.forkpty",
    "os.system",
    "os.posix_spawn",
    "os.spawn",
    "subprocess.Popen",
}
_AUDIT_WRITE_PATH_INDEXES = {
    "os.remove": (0,),
    "os.unlink": (0,),
    "os.rmdir": (0,),
    "os.mkdir": (0,),
    "os.makedirs": (0,),
    "os.rename": (0, 1),
    "os.replace": (0, 1),
    "os.symlink": (0, 1),
    "os.link": (0, 1),
    "os.truncate": (0,),
    "os.chmod": (0,),
    "os.chown": (0,),
    "os.utime": (0,),
    "shutil.copyfile": (1,),
    "shutil.copymode": (1,),
    "shutil.copystat": (1,),
    "shutil.copytree": (1,),
    "shutil.move": (0, 1),
    "shutil.rmtree": (0,),
}


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
    stale_artifacts: list[str] = field(default_factory=list)
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
    snapshot_roots = _mechanism_snapshot_roots(workspace, guard_roots)
    before = _snapshot_mechanism_guard(snapshot_roots)
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

    after = _snapshot_mechanism_guard(snapshot_roots)
    filesystem_changes = diff_filesystem_snapshots(before, after)
    out_of_bounds_writes = _out_of_bounds_writes(filesystem_changes, workspace)
    subprocess_path_guard_errors = _subprocess_path_guard_errors(subprocess_result)
    out_of_bounds_writes = _merge_unique_strings(
        [
            *out_of_bounds_writes,
            *_subprocess_path_guard_violations(subprocess_result),
        ]
    )
    validation = validate_mechanism_node(workspace)
    parent_stale_artifacts = (
        _stale_generated_artifacts(before, after, workspace)
        if not validation.missing_artifacts and not validation.malformed_json
        else []
    )
    stale_artifacts = _merge_unique_strings(
        [
            *parent_stale_artifacts,
            *_subprocess_stale_artifacts(subprocess_result),
        ]
    )
    errors.extend(validation.errors)
    errors.extend(
        error for error in subprocess_path_guard_errors if error not in errors
    )
    if stale_artifacts:
        errors.append(
            "Generated V3 artifacts were not created or updated by run(workspace): "
            + ", ".join(stale_artifacts)
        )

    preliminary_valid = (
        executed
        and exception is None
        and validation.valid
        and not out_of_bounds_writes
        and not errors
        and not stale_artifacts
    )
    if preliminary_valid:
        parent_import_before = _snapshot_mechanism_guard(snapshot_roots)
        try:
            exported_callables = load_mechanism_node_callables(workspace)
        except MechanismNodeContractError as exc:
            callable_import_failed = True
            errors.append(str(exc))
        except Exception:
            callable_import_failed = True
            exception = traceback.format_exc()
        parent_import_after = _snapshot_mechanism_guard(snapshot_roots)
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

    out_of_bounds_writes = _merge_unique_strings(out_of_bounds_writes)
    if out_of_bounds_writes and not any(_PATH_GUARD_ERROR_PREFIX in error for error in errors):
        errors.append(
            f"{_PATH_GUARD_ERROR_PREFIX} filesystem writes detected: "
            + ", ".join(sorted(out_of_bounds_writes))
        )
    errors = _merge_unique_strings(errors)

    failure_kinds = _execution_failure_kinds(
        contract_failed=contract_failed,
        subprocess_result=subprocess_result,
        validation=validation,
        out_of_bounds_writes=out_of_bounds_writes,
        stale_artifacts=stale_artifacts,
        subprocess_path_guard_failed=bool(subprocess_path_guard_errors),
        callable_import_failed=callable_import_failed,
    )
    valid = (
        preliminary_valid
        and not callable_import_failed
        and not out_of_bounds_writes
        and not stale_artifacts
    )
    return MechanismNodeExecutionResult(
        workspace=workspace,
        valid=valid,
        executed=executed,
        failure_kinds=failure_kinds,
        missing_artifacts=validation.missing_artifacts,
        malformed_json=validation.malformed_json,
        stale_artifacts=stale_artifacts,
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
    env = _subprocess_env()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
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
            env=env,
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
    _install_mechanism_subprocess_path_guard(workspace)
    node = load_mechanism_node(workspace)
    module = _load_mechanism_module(node)
    runtime_errors = _lifecycle_runtime_errors(module)
    if runtime_errors:
        raise MechanismNodeContractError("; ".join(runtime_errors))

    run_before = snapshot_filesystem([workspace])
    _call_run_entrypoint(getattr(module, "run"), workspace)
    run_after = snapshot_filesystem([workspace])
    stale_artifacts = _stale_generated_artifacts(run_before, run_after, workspace)
    if stale_artifacts:
        print(
            _STALE_ARTIFACTS_ERROR_PREFIX + " " + ", ".join(stale_artifacts),
            file=sys.stderr,
        )


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
    errors: list[str] = []
    if defined_functions == {"select_batch"}:
        return [
            "mechanism.py defines only legacy select_batch; V3 mechanism nodes must define "
            "lifecycle callables: "
            + ", ".join(MECHANISM_NODE_LIFECYCLE_CALLABLES)
        ]
    if missing:
        errors.append(
            "mechanism.py missing lifecycle callables: "
            + ", ".join(missing)
        )
    if "run" not in defined_functions:
        errors.append("mechanism.py must define callable run(workspace)")
    return errors


def _lifecycle_runtime_errors(module: Any) -> list[str]:
    errors: list[str] = []
    missing_or_not_callable = [
        name
        for name in MECHANISM_NODE_LIFECYCLE_CALLABLES
        if not callable(getattr(module, name, None))
    ]
    if callable(getattr(module, "select_batch", None)) and len(missing_or_not_callable) == len(
        MECHANISM_NODE_LIFECYCLE_CALLABLES
    ):
        errors.append(
            "mechanism.py defines only legacy select_batch; V3 mechanism nodes must define "
            "lifecycle callables: "
            + ", ".join(MECHANISM_NODE_LIFECYCLE_CALLABLES)
        )
    elif missing_or_not_callable:
        errors.append(
            "mechanism.py missing callable lifecycle attributes: "
            + ", ".join(missing_or_not_callable)
        )

    entrypoint = getattr(module, "run", None)
    if not callable(entrypoint):
        errors.append("mechanism.py must define callable run(workspace)")
    else:
        errors.extend(_run_entrypoint_signature_errors(entrypoint))
    return errors


def _out_of_bounds_writes(
    filesystem_changes: Iterable[FilesystemChange],
    workspace: Path,
) -> list[str]:
    return [
        str(change.path)
        for change in filesystem_changes
        if not _is_relative_to(change.path, workspace)
    ]


def _stale_generated_artifacts(
    before: dict[Path, Any],
    after: dict[Path, Any],
    workspace: Path,
) -> list[str]:
    stale: list[str] = []
    for artifact in V3_MECHANISM_NODE_GENERATED_ARTIFACTS:
        path = _absolute_no_symlink_resolve(workspace / artifact)
        before_entry = before.get(path)
        after_entry = after.get(path)
        if before_entry is not None and after_entry is not None and before_entry == after_entry:
            stale.append(artifact)
    return stale


def _install_mechanism_subprocess_path_guard(workspace: Path) -> None:
    allowed_root = workspace.resolve()

    def guard(event: str, args: tuple[Any, ...]) -> None:
        if event in _AUDIT_BLOCKED_PROCESS_EVENTS:
            raise MechanismNodeContractError(
                f"{_PATH_GUARD_ERROR_PREFIX} {event} is outside workspace {allowed_root}"
            )
        for candidate in _audit_write_paths(event, args):
            path = _resolve_audit_path(candidate)
            if path is None or _is_relative_to(path, allowed_root):
                continue
            raise MechanismNodeContractError(
                f"{_PATH_GUARD_ERROR_PREFIX} {path} is outside workspace {allowed_root}"
            )

    sys.addaudithook(guard)


def _audit_write_paths(event: str, args: tuple[Any, ...]) -> list[Any]:
    if event == "open":
        if len(args) >= 3 and _audit_open_requests_write(args[1], args[2]):
            return [args[0]]
        return []
    indexes = _AUDIT_WRITE_PATH_INDEXES.get(event)
    if indexes is None:
        return []
    return [args[index] for index in indexes if index < len(args)]


def _audit_open_requests_write(mode: Any, flags: Any) -> bool:
    if isinstance(mode, str) and any(marker in mode for marker in ("w", "a", "x", "+")):
        return True
    if isinstance(flags, int):
        write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        return bool(flags & write_flags)
    return False


def _resolve_audit_path(candidate: Any) -> Path | None:
    if isinstance(candidate, int):
        return None
    try:
        path = Path(os.fsdecode(candidate))
    except (TypeError, ValueError):
        return None
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        return path.expanduser().resolve(strict=False)
    except OSError:
        return Path(os.path.abspath(os.fspath(path.expanduser())))


def _run_entrypoint_signature_errors(entrypoint: Any) -> list[str]:
    try:
        signature = inspect.signature(entrypoint)
    except (TypeError, ValueError):
        return []
    parameters = list(signature.parameters.values())
    if any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in parameters):
        return []
    if any(
        param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        for param in parameters
    ):
        return []
    keyword_workspace = signature.parameters.get("workspace")
    if keyword_workspace and keyword_workspace.kind == inspect.Parameter.KEYWORD_ONLY:
        return []
    return ["mechanism.py run(workspace) must accept the workspace argument"]


def _call_run_entrypoint(entrypoint: Any, workspace: Path) -> None:
    try:
        signature = inspect.signature(entrypoint)
    except (TypeError, ValueError):
        entrypoint(workspace)
        return
    parameters = list(signature.parameters.values())
    if any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in parameters):
        entrypoint(workspace)
        return
    if any(
        param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        for param in parameters
    ):
        entrypoint(workspace)
        return
    keyword_workspace = signature.parameters.get("workspace")
    if keyword_workspace and keyword_workspace.kind == inspect.Parameter.KEYWORD_ONLY:
        entrypoint(workspace=workspace)
        return
    raise MechanismNodeContractError("mechanism.py run(workspace) must accept the workspace argument")


def _subprocess_path_guard_errors(
    subprocess_result: _MechanismNodeSubprocessResult,
) -> list[str]:
    errors: list[str] = []
    for line in _subprocess_output_lines(subprocess_result):
        if _PATH_GUARD_ERROR_PREFIX not in line:
            continue
        errors.append(line[line.index(_PATH_GUARD_ERROR_PREFIX):].strip())
    return _merge_unique_strings(errors)


def _subprocess_path_guard_violations(
    subprocess_result: _MechanismNodeSubprocessResult,
) -> list[str]:
    violations: list[str] = []
    for error in _subprocess_path_guard_errors(subprocess_result):
        detail = error.split(_PATH_GUARD_ERROR_PREFIX, 1)[1].strip()
        path, _, _workspace_detail = detail.partition(" is outside workspace ")
        if path:
            violations.append(path)
    return _merge_unique_strings(violations)


def _subprocess_stale_artifacts(
    subprocess_result: _MechanismNodeSubprocessResult,
) -> list[str]:
    artifacts: list[str] = []
    for line in _subprocess_output_lines(subprocess_result):
        if _STALE_ARTIFACTS_ERROR_PREFIX not in line:
            continue
        detail = line[line.index(_STALE_ARTIFACTS_ERROR_PREFIX) :]
        detail = detail.removeprefix(_STALE_ARTIFACTS_ERROR_PREFIX).strip()
        artifacts.extend(
            artifact.strip()
            for artifact in detail.split(",")
            if artifact.strip()
        )
    return _merge_unique_strings(artifacts)


def _subprocess_output_lines(
    subprocess_result: _MechanismNodeSubprocessResult,
) -> list[str]:
    return [
        line
        for output in (subprocess_result.stderr, subprocess_result.stdout)
        for line in output.splitlines()
    ]


def _merge_unique_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        merged.append(value)
    return merged


def _absolute_no_symlink_resolve(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _mechanism_snapshot_roots(
    workspace: Path,
    guard_roots: Iterable[str | Path] | None,
) -> list[Path]:
    return _merge_unique_paths(
        [
            *_normalize_guard_roots(workspace, guard_roots),
            *_temporary_guard_roots(),
        ]
    )


def _snapshot_mechanism_guard(roots: Iterable[Path]) -> dict[Path, Any]:
    temp_roots = set(_temporary_guard_roots())
    recursive_roots = [root for root in roots if root not in temp_roots]
    snapshot = snapshot_filesystem(recursive_roots)
    snapshot.update(_snapshot_shallow_roots(temp_roots))
    return snapshot


def _snapshot_shallow_roots(roots: Iterable[Path]) -> dict[Path, Any]:
    snapshot: dict[Path, Any] = {}
    for root in _merge_unique_paths(roots):
        if not root.exists():
            continue
        paths = [root]
        if root.is_dir():
            try:
                paths.extend(root.iterdir())
            except OSError:
                pass
        for path in paths:
            entry = _snapshot_entry(path)
            if entry is not None:
                snapshot[_absolute_no_symlink_resolve(path)] = entry
    return snapshot


def _temporary_guard_roots() -> list[Path]:
    candidates: list[str | Path | None] = [
        tempfile.gettempdir(),
        os.environ.get("TMPDIR"),
        os.environ.get("TEMP"),
        os.environ.get("TMP"),
        Path("/tmp"),
    ]
    roots: list[Path] = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            root = _resolve_workspace(candidate)
        except OSError:
            continue
        if root.exists():
            roots.append(root)
    return _merge_unique_paths(roots)


def _merge_unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    merged: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        merged.append(path)
    return merged


def _execution_failure_kinds(
    *,
    contract_failed: bool,
    subprocess_result: _MechanismNodeSubprocessResult,
    validation: MechanismNodeValidationResult,
    out_of_bounds_writes: list[str],
    stale_artifacts: list[str],
    subprocess_path_guard_failed: bool,
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
    if out_of_bounds_writes or validation.escaping_symlinks or subprocess_path_guard_failed:
        kinds.append(FAILURE_PATH_GUARD)
    if stale_artifacts:
        kinds.append(FAILURE_STALE_ARTIFACTS)
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
    "FAILURE_STALE_ARTIFACTS",
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
