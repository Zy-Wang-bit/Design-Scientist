"""Build durable project design state from allowlisted data."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from design_scientist.evidence import (
    current_anti_hbsag_evidence_cards,
    generic_source_inventory_evidence_cards,
    source_warning_card,
)
from design_scientist.io import read_yaml, write_json
from design_scientist.schemas import DesignState, EvidenceCard, to_plain_data
from design_scientist.table_io import IDENTIFIER_COLUMNS, read_identifier_safe_csv


KNOWN_MODULES = {
    "HD110H": {"n_primary_edges": 11, "primary_median_tau": -0.131, "status": "candidate_reusable_module"},
    "HG56H": {"n_primary_edges": 4, "primary_median_tau": -0.150, "status": "candidate_reusable_module"},
    "HN54H": {"n_primary_edges": 4, "primary_median_tau": -0.145, "status": "candidate_reusable_module"},
    "HV105H": {"n_primary_edges": 1, "primary_median_tau": -1.039, "status": "strong_but_narrow_signal"},
}


FASTA_SUFFIXES = {".fa", ".faa", ".fasta", ".fna"}


def _safe_evidence_id(prefix: str, source_key: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in source_key).strip("_").lower()
    return f"{prefix}_{safe or 'source'}"


def _identifier_columns(contract: dict[str, Any]) -> tuple[str, ...]:
    columns = list(IDENTIFIER_COLUMNS)
    for column in contract.get("identifier_columns") or []:
        text = str(column)
        if text and text not in columns:
            columns.append(text)
    return tuple(columns)


def _resolve_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def _source_key(path_value: str | Path, resolved_path: Path, local_root: Path) -> str:
    original = Path(path_value).expanduser()
    if not original.is_absolute():
        return Path(path_value).as_posix()
    try:
        return resolved_path.relative_to(local_root).as_posix()
    except ValueError:
        return str(resolved_path)


def _csv_header_errors(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
    if not header:
        return ["CSV file is empty or missing an explicit header row."]
    empty_positions = [str(index) for index, column in enumerate(header, start=1) if not column.strip()]
    if empty_positions:
        return [f"CSV header contains empty column name(s) at position(s): {', '.join(empty_positions)}."]
    return []


def _invalid_csv_header_card(source_key: str, errors: list[str]) -> EvidenceCard:
    return source_warning_card(
        evidence_id=_safe_evidence_id("invalid_csv_header", source_key),
        claim="Allowlisted CSV source was rejected because its header is not explicit and single-row.",
        source_table=source_key,
        limitations=errors,
        actionability="needs_data",
        tags=["invalid_csv_header", "source_rejected"],
    )


def _missing_source_card(source_key: str, path: Path) -> EvidenceCard:
    return source_warning_card(
        evidence_id=_safe_evidence_id("missing_source", source_key),
        claim="Allowlisted source file is missing.",
        source_table=source_key,
        limitations=[str(path)],
        actionability="needs_data",
        tags=["missing_source"],
    )


def _unsupported_source_card(source_key: str, path: Path) -> EvidenceCard:
    return source_warning_card(
        evidence_id=_safe_evidence_id("unsupported_source_type", source_key),
        claim="Allowlisted source file type is not counted by build-state.",
        source_table=source_key,
        limitations=[f"Unsupported suffix: {path.suffix or '<none>'}"],
        actionability="monitor",
        tags=["unsupported_source_type"],
    )


def _count_csv_rows(
    path: Path,
    *,
    source_key: str,
    identifier_columns: tuple[str, ...],
) -> tuple[int, list[EvidenceCard]]:
    header_errors = _csv_header_errors(path)
    if header_errors:
        return 0, [_invalid_csv_header_card(source_key, header_errors)]
    return len(read_identifier_safe_csv(path, identifier_columns=identifier_columns).records()), []


def _count_fasta_records(path: Path) -> int:
    record_count = 0
    has_nonempty_sequence = False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(">"):
                record_count += 1
            else:
                has_nonempty_sequence = True
    return record_count or int(has_nonempty_sequence)


def _count_pdb_coordinate_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.startswith(("ATOM", "HETATM")))


def _count_allowlisted_file(
    path: Path,
    *,
    source_key: str,
    identifier_columns: tuple[str, ...],
) -> tuple[int, list[EvidenceCard]]:
    if not path.exists():
        return 0, [_missing_source_card(source_key, path)]
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _count_csv_rows(path, source_key=source_key, identifier_columns=identifier_columns)
    if suffix in FASTA_SUFFIXES:
        return _count_fasta_records(path), []
    if suffix == ".pdb":
        return _count_pdb_coordinate_rows(path), []
    return 0, [_unsupported_source_card(source_key, path)]


def _read_module_status(module_summary_path: Path) -> list[dict[str, Any]]:
    modules: list[dict[str, Any]] = []
    if module_summary_path.is_file():
        for row in read_identifier_safe_csv(module_summary_path).records():
            module_id = row.get("module_id")
            if module_id in KNOWN_MODULES:
                modules.append(
                    {
                        "system": row.get("system") or "sdAb",
                        "module_id": module_id,
                        "classification": row.get("classification") or KNOWN_MODULES[module_id]["status"],
                        "n_primary_edges": int(float(row.get("n_primary_edges") or KNOWN_MODULES[module_id]["n_primary_edges"])),
                        "primary_median_tau": float(row.get("primary_median_tau") or KNOWN_MODULES[module_id]["primary_median_tau"]),
                    }
                )
    seen = {m["module_id"] for m in modules}
    for module_id, payload in KNOWN_MODULES.items():
        if module_id not in seen:
            modules.append({"system": "sdAb", "module_id": module_id, **payload})
    return modules


def _read_missing_edges(path: Path, limit: int = 50) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for record in read_identifier_safe_csv(path).records()[:limit]:
        rows.append({str(k): (None if v == "" else str(v)) for k, v in record.items()})
    return rows


def _project_context(root: Path, contract: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    project_path = root / "project.yaml"
    project = read_yaml(project_path) if project_path.exists() else {}
    project_id = str(
        project.get("project_id")
        or contract.get("project_id")
        or contract.get("contract_id")
        or root.name
    )
    context = {"project_id": project_id}
    if project.get("domain"):
        context["domain"] = str(project["domain"])
    objective = project.get("objective") or project.get("goal")
    if objective:
        context["objective"] = str(objective)
    return context, project


def _system_roles_from_project(project: dict[str, Any]) -> dict[str, str]:
    constraints = project.get("constraints")
    if isinstance(constraints, dict) and isinstance(constraints.get("system_roles"), dict):
        return {str(key): str(value) for key, value in constraints["system_roles"].items()}
    target_systems = project.get("target_systems")
    if isinstance(target_systems, list):
        return {str(system): "target_system" for system in target_systems}
    if project.get("system"):
        return {str(project["system"]): "target_system"}
    return {}


def _state_payload(state: DesignState, project_context: dict[str, str]) -> dict[str, Any]:
    payload = to_plain_data(state)
    for key in ("domain", "objective"):
        if key in project_context:
            payload[key] = project_context[key]
    return payload


def _apply_project_context_attributes(
    state: DesignState,
    project_context: dict[str, str],
) -> None:
    for key in ("domain", "objective"):
        if key in project_context:
            setattr(state, key, project_context[key])


def _build_allowed_sources_state(
    root: Path,
    contract: dict[str, Any],
    project_context: dict[str, str],
    project: dict[str, Any],
) -> tuple[DesignState, list[EvidenceCard]]:
    local_root = _resolve_path(contract["local_root"], root) if contract.get("local_root") else root
    identifier_columns = _identifier_columns(contract)
    row_counts: dict[str, int] = {}
    source_keys: list[str] = []
    warning_cards: list[EvidenceCard] = []

    allowed_sources = contract.get("allowed_sources") or {}
    if not isinstance(allowed_sources, dict):
        raise ValueError("data_contract.allowed_sources must be a mapping when present")

    for source_spec in allowed_sources.values():
        if not isinstance(source_spec, dict):
            continue
        for file_entry in source_spec.get("files", []):
            path_value = file_entry.get("path") if isinstance(file_entry, dict) else file_entry
            if not path_value:
                continue
            path = _resolve_path(str(path_value), local_root)
            source_key = _source_key(str(path_value), path, local_root)
            source_keys.append(source_key)
            row_count, cards = _count_allowlisted_file(
                path,
                source_key=source_key,
                identifier_columns=identifier_columns,
            )
            row_counts[source_key] = row_count
            warning_cards.extend(cards)

    state = DesignState(
        project_id=project_context["project_id"],
        data_version=str(contract.get("version") or contract.get("contract_id") or "unknown"),
        current_round=0,
        row_counts=row_counts,
        system_roles=_system_roles_from_project(project),
        module_status=[],
        unresolved_edges=[],
        unsupported_claims=[str(claim) for claim in contract.get("forbidden_claims", [])],
    )
    _apply_project_context_attributes(state, project_context)
    evidence = generic_source_inventory_evidence_cards(
        row_counts,
        project_id=project_context["project_id"],
        source_tables=source_keys,
    )
    return state, evidence + warning_cards


def _build_legacy_allowed_tables_state(
    root: Path,
    contract: dict[str, Any],
    project_context: dict[str, str],
    project: dict[str, Any],
) -> tuple[DesignState, list[EvidenceCard]]:
    identifier_columns = _identifier_columns(contract)
    row_counts: dict[str, int] = {}
    table_paths: dict[str, Path] = {}
    warning_cards: list[EvidenceCard] = []

    for table in contract.get("allowed_tables", []):
        name = table["name"]
        path = _resolve_path(table["path"], root)
        table_paths[name] = path
        if path.exists() and path.suffix.lower() == ".csv":
            row_counts[name], cards = _count_csv_rows(
                path,
                source_key=name,
                identifier_columns=identifier_columns,
            )
            warning_cards.extend(cards)
        else:
            row_counts[name] = 0
        if table.get("required", True) and not path.exists():
            warning_cards.append(
                EvidenceCard(
                    evidence_id=f"missing_table_{name}",
                    claim=f"Required allowlisted table is missing: {name}",
                    evidence_tier="descriptive",
                    source_tables=[name],
                    limitations=[str(path)],
                    actionability="monitor",
                    tags=["missing_table"],
                )
            )

    state = DesignState(
        project_id=project_context["project_id"],
        data_version=str(contract.get("version", "unknown")),
        current_round=0,
        row_counts=row_counts,
        system_roles=_system_roles_from_project(project) or {
            "sdAb": "module_learning",
            "1E62": "target_system_design_genotype_coverage",
        },
        module_status=_read_module_status(table_paths.get("module_summary", Path(""))),
        unresolved_edges=_read_missing_edges(table_paths.get("missing_primary_edge_candidates", Path(""))),
        unsupported_claims=[
            "Current 1E62 combo data cannot support module causal claims; exact primary matched edges are missing.",
            "sdAb module effects cannot be claimed to transfer to 1E62 without validation.",
            "Model-derived or synthetic records cannot be reported as wet-lab measurements.",
        ],
    )
    _apply_project_context_attributes(state, project_context)
    evidence = current_anti_hbsag_evidence_cards(row_counts) + warning_cards
    return state, evidence


def build_state(project_dir: str | Path) -> DesignState:
    root = Path(project_dir).expanduser().resolve()
    contract = read_yaml(root / "data_contract.yaml")
    project_context, project = _project_context(root, contract)
    if "allowed_sources" in contract:
        state, evidence = _build_allowed_sources_state(root, contract, project_context, project)
    else:
        state, evidence = _build_legacy_allowed_tables_state(root, contract, project_context, project)
    write_json(root / "state" / "design_state.json", _state_payload(state, project_context))
    write_json(root / "state" / "evidence_cards.json", evidence)
    print(f"Built design state at {root / 'state'}")
    return state
