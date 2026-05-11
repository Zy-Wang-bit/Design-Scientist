"""Method-node contract, execution harness, and path guard."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import stat
import subprocess
import sys
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


METHOD_NODE_REQUIRED_ARTIFACTS = (
    "proposal.json",
    "method.py",
    "candidate_policy.json",
    "novelty_report.json",
    "benchmark_metrics.json",
    "validation_report.json",
)
METHOD_NODE_JSON_ARTIFACTS = (
    "proposal.json",
    "manifest.json",
    "candidate_policy.json",
    "novelty_report.json",
    "benchmark_metrics.json",
    "validation_report.json",
)
METHOD_NODE_GENERATED_ARTIFACTS = tuple(
    artifact for artifact in METHOD_NODE_REQUIRED_ARTIFACTS if artifact != "method.py"
)
NOVELTY_SELECTION_OVERLAP_THRESHOLD = 0.85
PROPOSAL_REQUIRED_FIELDS = (
    "method_hypothesis",
    "literature_basis",
    "literature_gap_ids",
    "reused_components",
    "architecture_delta",
    "new_mechanism_claim",
    "algorithm_mechanism",
    "expected_advantage",
    "failure_modes",
    "planned_ablation",
)
DEFAULT_ENTRYPOINT = "run"
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
    "from design_scientist.method_nodes import _method_node_subprocess_main\n"
    "_method_node_subprocess_main()\n"
)


class MethodNodeError(RuntimeError):
    """Base exception for method-node harness failures."""


class MethodNodeContractError(MethodNodeError):
    """Raised when a node cannot be loaded under the method-node contract."""


@dataclass(frozen=True)
class MethodNode:
    """Loaded method-node metadata."""

    workspace: Path
    method_path: Path
    manifest: dict[str, Any]
    entrypoint: str = DEFAULT_ENTRYPOINT


@dataclass(frozen=True)
class FilesystemEntry:
    """Stable file-system state used by the path guard."""

    kind: str
    mode: int | None = None
    size: int | None = None
    mtime_ns: int | None = None
    digest: str | None = None
    target: str | None = None


@dataclass(frozen=True)
class FilesystemChange:
    """A path created, deleted, or modified between two snapshots."""

    path: Path
    change_type: str


@dataclass(frozen=True)
class MethodNodeValidationResult:
    """Artifact validation result for a method-node workspace."""

    workspace: Path
    valid: bool
    missing_artifacts: list[str] = field(default_factory=list)
    malformed_json: dict[str, str] = field(default_factory=dict)
    escaping_symlinks: list[str] = field(default_factory=list)
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MethodNodeExecutionResult:
    """Result of loading, executing, and validating a method node."""

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
class _MethodNodeSubprocessResult:
    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


def load_method_node(node_workspace: str | Path) -> MethodNode:
    """Load node metadata without executing ``method.py``."""

    workspace = _resolve_workspace(node_workspace)
    if not workspace.is_dir():
        raise MethodNodeContractError(f"Node workspace does not exist: {workspace}")

    method_path = workspace / "method.py"
    if not method_path.is_file():
        raise MethodNodeContractError("Missing required artifact: method.py")

    manifest_path = workspace / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = _read_json_object(manifest_path)
        except ValueError as exc:
            raise MethodNodeContractError(str(exc)) from exc
    else:
        manifest = {"entrypoint": DEFAULT_ENTRYPOINT}
    entrypoint = manifest.get("entrypoint", DEFAULT_ENTRYPOINT)
    if not isinstance(entrypoint, str) or not entrypoint.strip():
        raise MethodNodeContractError("manifest.json entrypoint must be a non-empty string")

    return MethodNode(
        workspace=workspace,
        method_path=method_path,
        manifest=manifest,
        entrypoint=entrypoint,
    )


def execute_method_node(
    node_workspace: str | Path,
    *,
    guard_roots: Iterable[str | Path] | None = None,
    timeout_seconds: float | None = DEFAULT_EXECUTION_TIMEOUT_SECONDS,
) -> MethodNodeExecutionResult:
    """Execute a node and mark it invalid on contract or path-guard failure.

    The harness does not generate node code. It imports the workspace's
    ``method.py`` inside an isolated subprocess and calls the manifest-declared
    entrypoint, defaulting to ``run``. Relative writes land in the node
    workspace because the subprocess cwd is set to that workspace. Callable
    policies are imported in the parent only after artifacts validate, and the
    entrypoint is not called during that import.
    """

    workspace = _resolve_workspace(node_workspace)
    snapshot_roots = _normalize_guard_roots(workspace, guard_roots)
    before = snapshot_filesystem(snapshot_roots)
    executed = False
    exception: str | None = None
    errors: list[str] = []
    exported_callables: dict[str, Any] = {}
    node: MethodNode | None = None
    subprocess_result = _MethodNodeSubprocessResult(returncode=None)
    callable_import_failed = False
    contract_failed = False

    try:
        preflight = validate_method_node(workspace, require_generated_artifacts=False)
        if preflight.escaping_symlinks:
            escaped = ", ".join(preflight.escaping_symlinks)
            raise MethodNodeContractError(f"Node workspace has symlinks escaping workspace: {escaped}")
        node = load_method_node(workspace)
        _entrypoint_attribute(node.entrypoint)
    except MethodNodeContractError as exc:
        contract_failed = True
        errors.append(str(exc))
    except Exception as exc:
        contract_failed = True
        errors.append(f"Preflight failed: {exc}")

    if node is not None and not errors:
        subprocess_result = _execute_node_subprocess(
            node,
            timeout_seconds=timeout_seconds,
        )
        if subprocess_result.timed_out:
            exception = (
                subprocess_result.stderr.strip()
                or f"method.py timed out after {timeout_seconds} seconds"
            )
        elif subprocess_result.returncode == 0:
            executed = True
        else:
            exception = (
                subprocess_result.stderr.strip()
                or f"method.py exited with code {subprocess_result.returncode}"
            )

    after = snapshot_filesystem(snapshot_roots)
    filesystem_changes = diff_filesystem_snapshots(before, after)
    out_of_bounds_writes = [
        str(change.path)
        for change in filesystem_changes
        if not _is_relative_to(change.path, workspace)
    ]
    validation = validate_method_node(workspace)
    stale_artifacts = (
        _stale_generated_artifacts(before, after, workspace)
        if executed
        else []
    )
    errors.extend(validation.errors)

    preliminary_valid = (
        executed
        and exception is None
        and validation.valid
        and not out_of_bounds_writes
        and not errors
        and not stale_artifacts
    )
    if preliminary_valid and node is not None:
        try:
            exported_callables = load_method_node_callables(node)
        except MethodNodeContractError as exc:
            callable_import_failed = True
            errors.append(str(exc))
        except Exception:
            callable_import_failed = True
            exception = traceback.format_exc()

    failure_kinds = _execution_failure_kinds(
        contract_failed=contract_failed,
        subprocess_result=subprocess_result,
        validation=validation,
        out_of_bounds_writes=out_of_bounds_writes,
        stale_artifacts=stale_artifacts,
        callable_import_failed=callable_import_failed,
    )
    valid = preliminary_valid and not callable_import_failed
    return MethodNodeExecutionResult(
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
        callables=exported_callables,
    )


def validate_method_node(
    node_workspace: str | Path,
    *,
    require_generated_artifacts: bool = True,
) -> MethodNodeValidationResult:
    """Validate required method-node artifacts and JSON shape."""

    workspace = _resolve_workspace(node_workspace)
    required = (
        METHOD_NODE_REQUIRED_ARTIFACTS
        if require_generated_artifacts
        else ("method.py",)
    )
    missing_artifacts = [
        artifact
        for artifact in required
        if not (workspace / artifact).exists()
    ]
    malformed_json: dict[str, str] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    errors: list[str] = []

    for artifact in METHOD_NODE_JSON_ARTIFACTS:
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
        artifacts[artifact] = data

    manifest = artifacts.get("manifest.json")
    if manifest is not None:
        entrypoint = manifest.get("entrypoint", DEFAULT_ENTRYPOINT)
        if not isinstance(entrypoint, str) or not entrypoint.strip():
            errors.append("manifest.json entrypoint must be a non-empty string")

    proposal = artifacts.get("proposal.json")
    if proposal is not None:
        errors.extend(_proposal_contract_errors(proposal))

    novelty_report = artifacts.get("novelty_report.json")
    if novelty_report is not None:
        errors.extend(_baseline_clone_guard_errors(novelty_report))

    validation_report = artifacts.get("validation_report.json")
    if validation_report is not None and validation_report.get("valid") is not True:
        errors.append("validation_report.json must contain valid: true")

    escaping_symlinks = find_escaping_symlinks(workspace)
    valid = not missing_artifacts and not malformed_json and not escaping_symlinks and not errors
    return MethodNodeValidationResult(
        workspace=workspace,
        valid=valid,
        missing_artifacts=missing_artifacts,
        malformed_json=malformed_json,
        escaping_symlinks=[str(path) for path in escaping_symlinks],
        artifacts=artifacts,
        errors=errors,
    )


def snapshot_filesystem(roots: Iterable[str | Path]) -> dict[Path, FilesystemEntry]:
    """Return a before/after comparable filesystem snapshot for guard roots."""

    snapshot: dict[Path, FilesystemEntry] = {}
    for root in _unique_paths(_resolve_workspace(path) for path in roots):
        if not root.exists():
            continue
        for path in _iter_snapshot_paths(root):
            entry = _snapshot_entry(path)
            if entry is not None:
                snapshot[_absolute_no_symlink_resolve(path)] = entry
    return snapshot


def diff_filesystem_snapshots(
    before: dict[Path, FilesystemEntry],
    after: dict[Path, FilesystemEntry],
) -> list[FilesystemChange]:
    """Diff two snapshots into created, deleted, and modified paths."""

    changes: list[FilesystemChange] = []
    for path in sorted(set(before) | set(after)):
        if path not in before:
            changes.append(FilesystemChange(path=path, change_type="created"))
        elif path not in after:
            changes.append(FilesystemChange(path=path, change_type="deleted"))
        elif before[path] != after[path]:
            changes.append(FilesystemChange(path=path, change_type="modified"))
    return changes


def find_escaping_symlinks(node_workspace: str | Path) -> list[Path]:
    """Return symlinks inside the workspace whose targets resolve outside it."""

    workspace = _resolve_workspace(node_workspace)
    if not workspace.exists():
        return []
    escaping: list[Path] = []
    for path in _iter_snapshot_paths(workspace):
        try:
            if path.is_symlink() and not _is_relative_to(path.resolve(strict=False), workspace):
                escaping.append(_absolute_no_symlink_resolve(path))
        except OSError:
            escaping.append(_absolute_no_symlink_resolve(path))
    return sorted(escaping)


def load_method_node_callables(node_workspace: str | Path | MethodNode) -> dict[str, Any]:
    """Import ``method.py`` and return public callables without calling ``run``.

    This is intended for benchmarking policies after subprocess artifact
    execution and validation have succeeded.
    """

    node = (
        node_workspace
        if isinstance(node_workspace, MethodNode)
        else load_method_node(node_workspace)
    )
    module = _load_method_module(node)
    return {
        name: value
        for name, value in vars(module).items()
        if callable(value) and not name.startswith("_")
    }


def _execute_node_subprocess(
    node: MethodNode,
    *,
    timeout_seconds: float | None,
) -> _MethodNodeSubprocessResult:
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                _SUBPROCESS_CODE,
                os.fspath(node.workspace),
            ],
            cwd=node.workspace,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=_subprocess_env(),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stderr = _coerce_subprocess_output(exc.stderr)
        timeout_message = f"method.py timed out after {timeout_seconds} seconds"
        if stderr:
            stderr = f"{stderr.rstrip()}\n{timeout_message}"
        else:
            stderr = timeout_message
        return _MethodNodeSubprocessResult(
            returncode=None,
            stdout=_coerce_subprocess_output(exc.stdout),
            stderr=stderr,
            timed_out=True,
        )
    return _MethodNodeSubprocessResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        timed_out=False,
    )


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_entries = [
        os.fspath(Path.cwd()) if not entry else entry
        for entry in sys.path
    ]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return env


def _method_node_subprocess_main() -> None:
    workspace = _resolve_workspace(sys.argv[1])
    node = load_method_node(workspace)
    module = _load_method_module(node)
    entrypoint = getattr(module, _entrypoint_attribute(node.entrypoint), None)
    if not callable(entrypoint):
        raise MethodNodeContractError(
            f"method.py does not define callable entrypoint {node.entrypoint!r}"
        )
    _call_entrypoint(entrypoint, node.workspace)


def _load_method_module(node: MethodNode) -> Any:
    module_name = f"_design_scientist_method_node_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, node.method_path)
    if spec is None or spec.loader is None:
        raise MethodNodeContractError(f"Cannot load method.py from {node.method_path}")

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


def _coerce_subprocess_output(output: str | bytes | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


def _stale_generated_artifacts(
    before: dict[Path, FilesystemEntry],
    after: dict[Path, FilesystemEntry],
    workspace: Path,
) -> list[str]:
    stale: list[str] = []
    for artifact in METHOD_NODE_GENERATED_ARTIFACTS:
        path = _absolute_no_symlink_resolve(workspace / artifact)
        before_entry = before.get(path)
        after_entry = after.get(path)
        if before_entry is not None and after_entry is not None and before_entry == after_entry:
            stale.append(artifact)
    return stale


def _execution_failure_kinds(
    *,
    contract_failed: bool,
    subprocess_result: _MethodNodeSubprocessResult,
    validation: MethodNodeValidationResult,
    out_of_bounds_writes: list[str],
    stale_artifacts: list[str],
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
    if stale_artifacts:
        kinds.append(FAILURE_STALE_ARTIFACTS)
    if callable_import_failed:
        kinds.append(FAILURE_CALLABLE_IMPORT)
    return kinds


def _call_entrypoint(entrypoint: Any, workspace: Path) -> None:
    signature = inspect.signature(entrypoint)
    parameters = list(signature.parameters.values())
    accepts_args = any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in parameters)
    positional = [
        param
        for param in parameters
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    keyword_workspace = signature.parameters.get("workspace")
    workspace_names = {"workspace", "workspace_path", "node_workspace", "root"}

    if accepts_args:
        entrypoint(workspace)
    elif keyword_workspace and keyword_workspace.kind == inspect.Parameter.KEYWORD_ONLY:
        entrypoint(workspace=workspace)
    elif positional and positional[0].name in workspace_names:
        entrypoint(workspace)
    elif (
        len(positional) == 1
        and positional[0].default is inspect.Parameter.empty
    ):
        entrypoint(workspace)
    else:
        entrypoint()


def _entrypoint_attribute(entrypoint: str) -> str:
    """Return the Python attribute for either ``run`` or ``module:run`` syntax."""

    if ":" not in entrypoint:
        return entrypoint
    module_name, function_name = entrypoint.rsplit(":", 1)
    if module_name not in {"method", "method.py", ""}:
        raise MethodNodeContractError(
            f"Unsupported method-node entrypoint module {module_name!r}; expected 'method'."
        )
    if not function_name:
        raise MethodNodeContractError("manifest.json entrypoint function must be non-empty")
    return function_name


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(f"Missing JSON artifact: {path.name}") from exc
    except OSError as exc:
        raise ValueError(f"Invalid JSON artifact {path.name}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {path.name}: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path.name}")
    return data


def _proposal_contract_errors(proposal: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing_fields = [
        field
        for field in PROPOSAL_REQUIRED_FIELDS
        if field not in proposal
    ]
    if missing_fields:
        errors.append(
            "proposal.json missing required fields: "
            + ", ".join(missing_fields)
        )

    if "literature_gap_ids" in proposal and not _non_empty_string_list(
        proposal.get("literature_gap_ids")
    ):
        errors.append(
            "proposal.json literature_gap_ids must list at least one literature gap id"
        )
    if "reused_components" in proposal and not _string_list(proposal.get("reused_components")):
        errors.append("proposal.json reused_components must be a list of strings")
    for field in ("architecture_delta", "new_mechanism_claim", "planned_ablation"):
        if field in proposal and not _non_empty_proposal_value(proposal.get(field)):
            errors.append(
                f"proposal.json {field} must be a non-empty string, string list, or object"
            )
    return errors


def _baseline_clone_guard_errors(novelty_report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if novelty_report.get("baseline_clone") is True:
        errors.append("baseline clone guard failed: novelty_report.json marks baseline_clone true")

    overlap = _optional_float(novelty_report.get("selection_overlap_vs_baselines"))
    if overlap is not None and overlap > NOVELTY_SELECTION_OVERLAP_THRESHOLD:
        errors.append(
            "baseline clone guard failed: novelty_report.json selection_overlap_vs_baselines "
            f"{overlap:.3f} exceeds {NOVELTY_SELECTION_OVERLAP_THRESHOLD:.3f}"
        )
    wrapper_pattern = str(
        novelty_report.get("wrapper_pattern")
        or novelty_report.get("clone_pattern")
        or novelty_report.get("policy_pattern")
        or ""
    ).lower()
    if novelty_report.get("tail_swap_baseline_wrapper") is True or (
        "tail" in wrapper_pattern and "swap" in wrapper_pattern
    ):
        errors.append(
            "baseline clone guard failed: novelty_report.json marks a tail-swap baseline wrapper"
        )
    return errors


def _string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _non_empty_string_list(value: Any) -> bool:
    return _string_list(value) and any(item.strip() for item in value)


def _non_empty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _non_empty_proposal_value(value: Any) -> bool:
    if _non_empty_text(value):
        return True
    if isinstance(value, list):
        return any(_non_empty_proposal_value(item) for item in value)
    if isinstance(value, dict):
        return any(_non_empty_proposal_value(item) for item in value.values())
    return False


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _snapshot_entry(path: Path) -> FilesystemEntry | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None

    mode = stat.S_IMODE(metadata.st_mode)
    if stat.S_ISLNK(metadata.st_mode):
        try:
            target = os.readlink(path)
        except OSError as exc:
            target = f"<unreadable:{type(exc).__name__}>"
        return FilesystemEntry(kind="symlink", mode=mode, target=target)
    if stat.S_ISDIR(metadata.st_mode):
        return FilesystemEntry(kind="dir", mode=mode)
    if stat.S_ISREG(metadata.st_mode):
        return FilesystemEntry(
            kind="file",
            mode=mode,
            size=metadata.st_size,
            mtime_ns=metadata.st_mtime_ns,
            digest=_sha256_file(path),
        )
    return FilesystemEntry(
        kind="other",
        mode=mode,
        size=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        return f"<unreadable:{type(exc).__name__}>"
    return digest.hexdigest()


def _iter_snapshot_paths(root: Path) -> Iterable[Path]:
    yield root
    if root.is_dir():
        yield from root.rglob("*")


def _normalize_guard_roots(
    workspace: Path,
    guard_roots: Iterable[str | Path] | None,
) -> list[Path]:
    roots = list(guard_roots) if guard_roots is not None else [workspace.parent]
    if workspace.parent not in roots:
        roots.append(workspace.parent)
    project_root = _infer_project_root(workspace)
    if project_root is not None and project_root not in roots:
        roots.append(project_root)
    return _unique_paths(_resolve_workspace(path) for path in roots)


def _infer_project_root(workspace: Path) -> Path | None:
    parents = workspace.parents
    if len(parents) >= 4 and parents[0].name == "nodes" and parents[2].name == "runs":
        return parents[3]
    return None


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _resolve_workspace(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _absolute_no_symlink_resolve(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
    except ValueError:
        return False
    return True


__all__ = [
    "DEFAULT_ENTRYPOINT",
    "DEFAULT_EXECUTION_TIMEOUT_SECONDS",
    "FAILURE_CALLABLE_IMPORT",
    "FAILURE_CONTRACT",
    "FAILURE_MALFORMED_ARTIFACTS",
    "FAILURE_MISSING_ARTIFACTS",
    "FAILURE_PATH_GUARD",
    "FAILURE_STALE_ARTIFACTS",
    "FAILURE_SUBPROCESS_NONZERO",
    "FAILURE_TIMEOUT",
    "METHOD_NODE_GENERATED_ARTIFACTS",
    "METHOD_NODE_JSON_ARTIFACTS",
    "METHOD_NODE_REQUIRED_ARTIFACTS",
    "NOVELTY_SELECTION_OVERLAP_THRESHOLD",
    "PROPOSAL_REQUIRED_FIELDS",
    "FilesystemChange",
    "FilesystemEntry",
    "MethodNode",
    "MethodNodeContractError",
    "MethodNodeError",
    "MethodNodeExecutionResult",
    "MethodNodeValidationResult",
    "diff_filesystem_snapshots",
    "execute_method_node",
    "find_escaping_symlinks",
    "load_method_node_callables",
    "load_method_node",
    "snapshot_filesystem",
    "validate_method_node",
]
