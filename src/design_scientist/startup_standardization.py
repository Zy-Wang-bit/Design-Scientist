"""Schema audit and generic startup-data standardization.

This module handles raw startup projects described by
``data_contract.allowed_sources``.  It deliberately does not infer experiment
rounds or silently join unrelated variant-id namespaces.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from design_scientist.io import ensure_dir, read_yaml, write_json
from design_scientist.table_io import IDENTIFIER_COLUMNS, read_identifier_safe_csv


OBSERVATION_FIELDS = (
    "observation_id",
    "source_file",
    "source_row",
    "id_namespace",
    "variant_id",
    "antigen_genotype",
    "endpoint",
    "value",
    "value_status",
    "unit",
    "pH",
    "concentration_ng_ml",
    "replicate_id",
    "evidence_role",
)
SOURCE_INVENTORY_FIELDS = (
    "source_group",
    "path",
    "status",
    "file_type",
    "row_count",
    "header_valid",
    "columns",
    "evidence_tier",
    "allowed_as_experimental_evidence",
)
SEQUENCE_FIELDS = (
    "variant_id",
    "heavy_chain_seq",
    "light_chain_seq",
    "heavy_construct",
    "light_construct",
    "source_file",
    "source_row",
)


def audit_schema(project_dir: str | Path) -> dict[str, Any]:
    """Audit allowlisted startup files and write ``schema_audit_report.json``."""

    root = Path(project_dir).expanduser().resolve()
    standardized_dir = ensure_dir(root / "standardized")
    contract = read_yaml(root / "data_contract.yaml")
    project = _read_yaml_if_exists(root / "project.yaml")
    estimands = _read_yaml_if_exists(root / "estimands.yaml")
    local_root = _local_root(root, contract)

    sources = list(_iter_allowed_sources(contract, local_root))
    inventory: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    id_namespaces: dict[str, dict[str, Any]] = {}

    for source in sources:
        inventory_row, source_diagnostics, namespace_values = _audit_source(source, root)
        inventory.append(inventory_row)
        diagnostics.extend(source_diagnostics)
        if namespace_values:
            id_namespaces[inventory_row["path"]] = namespace_values

    namespace_summary = _namespace_summary(id_namespaces)
    diagnostics.extend(namespace_summary["diagnostics"])
    report = {
        "schema_version": 1,
        "project_id": _project_id(root, project, contract),
        "domain": project.get("domain"),
        "objective": project.get("objective") or project.get("goal"),
        "contract_path": "data_contract.yaml",
        "estimand_set_id": estimands.get("estimand_set_id"),
        "source_count": len(inventory),
        "sources": inventory,
        "identifier_namespaces": namespace_summary["namespaces"],
        "diagnostics": diagnostics,
        "status": "ok" if not any(item["severity"] == "error" for item in diagnostics) else "error",
    }
    write_json(standardized_dir / "schema_audit_report.json", report)
    _write_csv(standardized_dir / "source_inventory.csv", inventory, SOURCE_INVENTORY_FIELDS)
    return report


def standardize_startup_data(project_dir: str | Path) -> dict[str, Any]:
    """Audit and standardize allowlisted startup sources into common artifacts."""

    root = Path(project_dir).expanduser().resolve()
    standardized_dir = ensure_dir(root / "standardized")
    contract = read_yaml(root / "data_contract.yaml")
    project = _read_yaml_if_exists(root / "project.yaml")
    estimands = _read_yaml_if_exists(root / "estimands.yaml")
    local_root = _local_root(root, contract)

    sources = list(_iter_allowed_sources(contract, local_root))
    observations: list[dict[str, Any]] = []
    sequences: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    namespace_values_by_source: dict[str, dict[str, Any]] = {}

    for source in sources:
        inventory_row, source_diagnostics, namespace_values = _audit_source(source, root)
        inventory.append(inventory_row)
        diagnostics.extend(source_diagnostics)
        source_key = inventory_row["path"]
        if namespace_values:
            namespace_values_by_source[source_key] = namespace_values
        if inventory_row["status"] != "ok":
            continue
        rows = _read_csv_records(source["resolved_path"], contract)
        observations.extend(_standardize_observations(source_key, source, rows))
        sequences.extend(_standardize_sequences(source_key, source, rows))

    namespace_summary = _namespace_summary(namespace_values_by_source)
    diagnostics.extend(namespace_summary["diagnostics"])

    _write_csv(standardized_dir / "source_inventory.csv", inventory, SOURCE_INVENTORY_FIELDS)
    _write_csv(standardized_dir / "observations_long.csv", observations, OBSERVATION_FIELDS)
    _write_csv(standardized_dir / "variant_sequences.csv", sequences, SEQUENCE_FIELDS)

    project_context = {
        "schema_version": 1,
        "project_id": _project_id(root, project, contract),
        "domain": project.get("domain"),
        "objective": project.get("objective") or project.get("goal"),
        "endpoints": _estimand_endpoints(estimands),
        "constraints": _estimand_constraints(estimands),
        "observed_row_counts": {
            row["path"]: int(row["row_count"])
            for row in inventory
            if str(row.get("row_count", "")).isdigit()
        },
        "allowed_sources": _allowed_sources_summary(contract),
        "identifier_namespaces": namespace_summary["namespaces"],
        "diagnostics": diagnostics,
        "standardized_artifacts": {
            "schema_audit_report": "standardized/schema_audit_report.json",
            "source_inventory": "standardized/source_inventory.csv",
            "observations_long": "standardized/observations_long.csv",
            "variant_sequences": "standardized/variant_sequences.csv",
            "project_context": "standardized/project_context.json",
        },
        "evidence_boundary": (
            "Standardized startup artifacts preserve raw source provenance. "
            "They do not establish wet-lab validation for newly generated designs."
        ),
    }
    report = {
        "schema_version": 1,
        "project_id": project_context["project_id"],
        "source_count": len(inventory),
        "observation_count": len(observations),
        "sequence_count": len(sequences),
        "sources": inventory,
        "identifier_namespaces": namespace_summary["namespaces"],
        "diagnostics": diagnostics,
        "status": "ok" if not any(item["severity"] == "error" for item in diagnostics) else "error",
    }
    write_json(standardized_dir / "schema_audit_report.json", report)
    write_json(standardized_dir / "project_context.json", project_context)
    return {
        "status": report["status"],
        "artifacts": {
            "schema_audit_report": str(standardized_dir / "schema_audit_report.json"),
            "source_inventory": str(standardized_dir / "source_inventory.csv"),
            "observations_long": str(standardized_dir / "observations_long.csv"),
            "variant_sequences": str(standardized_dir / "variant_sequences.csv"),
            "project_context": str(standardized_dir / "project_context.json"),
        },
        "summary": {
            "sources": len(inventory),
            "observations": len(observations),
            "sequences": len(sequences),
            "diagnostics": len(diagnostics),
        },
    }


def _read_yaml_if_exists(path: Path) -> dict[str, Any]:
    return read_yaml(path) if path.exists() else {}


def _local_root(root: Path, contract: dict[str, Any]) -> Path:
    value = contract.get("local_root")
    if not value:
        return root
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


def _project_id(root: Path, project: dict[str, Any], contract: dict[str, Any]) -> str:
    return str(
        project.get("project_id")
        or contract.get("project_id")
        or contract.get("contract_id")
        or root.name
    )


def _iter_allowed_sources(contract: dict[str, Any], local_root: Path) -> Iterable[dict[str, Any]]:
    allowed_sources = contract.get("allowed_sources")
    if not isinstance(allowed_sources, dict):
        raise ValueError("data_contract.allowed_sources must be a mapping")
    for source_group, raw_spec in allowed_sources.items():
        if not isinstance(raw_spec, dict):
            continue
        files = raw_spec.get("files") or []
        for file_entry in files:
            path_value = file_entry.get("path") if isinstance(file_entry, dict) else file_entry
            if not path_value:
                continue
            rel_path = str(path_value)
            path = Path(rel_path).expanduser()
            resolved = path if path.is_absolute() else local_root / rel_path
            yield {
                "source_group": str(source_group),
                "path": rel_path if not path.is_absolute() else str(resolved),
                "resolved_path": resolved,
                "evidence_tier": str(raw_spec.get("evidence_tier") or ""),
                "allowed_as_experimental_evidence": bool(
                    raw_spec.get("allowed_as_experimental_evidence", False)
                ),
            }


def _audit_source(source: dict[str, Any], root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    path: Path = source["resolved_path"]
    rel_path = _rel(path, root)
    diagnostics: list[dict[str, Any]] = []
    namespace_values: dict[str, Any] = {}
    inventory = {
        "source_group": source["source_group"],
        "path": rel_path,
        "status": "ok",
        "file_type": path.suffix.lower().lstrip(".") or "unknown",
        "row_count": 0,
        "header_valid": "",
        "columns": "",
        "evidence_tier": source["evidence_tier"],
        "allowed_as_experimental_evidence": str(source["allowed_as_experimental_evidence"]).lower(),
    }
    if not path.exists():
        inventory["status"] = "missing"
        diagnostics.append(_diagnostic("error", "missing_source", rel_path, "Allowlisted source is missing."))
        return inventory, diagnostics, namespace_values
    if path.suffix.lower() != ".csv":
        inventory["row_count"] = _count_non_csv(path)
        return inventory, diagnostics, namespace_values

    try:
        header = _csv_header(path)
    except OSError as exc:
        inventory["status"] = "unreadable"
        diagnostics.append(_diagnostic("error", "unreadable_source", rel_path, str(exc)))
        return inventory, diagnostics, namespace_values
    inventory["columns"] = "|".join(header)
    empty = [str(index) for index, column in enumerate(header, start=1) if not column.strip()]
    duplicate = sorted({column for column in header if column and header.count(column) > 1})
    if empty:
        inventory["status"] = "invalid_header"
        inventory["header_valid"] = "false"
        diagnostics.append(
            _diagnostic(
                "error",
                "empty_column_name",
                rel_path,
                f"Empty CSV header column(s): {', '.join(empty)}",
            )
        )
        return inventory, diagnostics, namespace_values
    if duplicate:
        inventory["status"] = "invalid_header"
        inventory["header_valid"] = "false"
        diagnostics.append(
            _diagnostic(
                "error",
                "duplicate_column_name",
                rel_path,
                f"Duplicate CSV header column(s): {', '.join(duplicate)}",
            )
        )
        return inventory, diagnostics, namespace_values
    inventory["header_valid"] = "true"

    rows = _read_csv_records(path, {})
    inventory["row_count"] = len(rows)
    namespace_values = _source_identifier_namespaces(rows)
    diagnostics.extend(_value_diagnostics(rel_path, rows))
    return inventory, diagnostics, namespace_values


def _count_non_csv(path: Path) -> int:
    suffix = path.suffix.lower()
    try:
        if suffix in {".fa", ".faa", ".fasta", ".fna"}:
            return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.startswith(">"))
        if suffix == ".pdb":
            return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.startswith(("ATOM", "HETATM")))
    except OSError:
        return 0
    return 0


def _csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(next(csv.reader(handle), []))


def _read_csv_records(path: Path, contract: dict[str, Any]) -> list[dict[str, str]]:
    identifiers = list(IDENTIFIER_COLUMNS)
    for column in contract.get("identifier_columns") or []:
        if column not in identifiers:
            identifiers.append(str(column))
    return read_identifier_safe_csv(path, identifier_columns=identifiers).records()


def _source_identifier_namespaces(rows: list[dict[str, str]]) -> dict[str, Any]:
    namespaces: dict[str, Any] = {}
    for column in ("variant_id", "variant", "编号"):
        values = sorted({str(row.get(column) or "").strip() for row in rows if str(row.get(column) or "").strip()})
        if values:
            namespaces[column] = {
                "count": len(values),
                "prefixes": sorted({_id_namespace(value) for value in values}),
                "examples": values[:8],
            }
    return namespaces


def _namespace_summary(id_namespaces: dict[str, dict[str, Any]]) -> dict[str, Any]:
    namespaces = []
    source_prefixes: dict[str, set[str]] = {}
    diagnostics: list[dict[str, Any]] = []
    for source, columns in id_namespaces.items():
        prefixes: set[str] = set()
        for column, payload in columns.items():
            item = {"source_file": source, "column": column, **payload}
            namespaces.append(item)
            prefixes.update(payload.get("prefixes") or [])
        if prefixes:
            source_prefixes[source] = prefixes
    all_prefix_sets = [prefixes for prefixes in source_prefixes.values() if prefixes]
    if len(all_prefix_sets) > 1 and not set.intersection(*all_prefix_sets):
        diagnostics.append(
            {
                "severity": "warning",
                "code": "disjoint_identifier_namespaces",
                "source_file": "",
                "message": "Variant identifiers occupy disjoint namespaces across sources; no cross-source join was inferred.",
                "details": {
                    source: sorted(prefixes)
                    for source, prefixes in sorted(source_prefixes.items())
                },
            }
        )
    return {"namespaces": namespaces, "diagnostics": diagnostics}


def _value_diagnostics(source_file: str, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    missing_sentinel_count = 0
    for row in rows:
        for value in row.values():
            if str(value).strip() == "-1.0":
                missing_sentinel_count += 1
    if missing_sentinel_count:
        diagnostics.append(
            {
                "severity": "warning",
                "code": "missing_value_sentinel",
                "source_file": source_file,
                "message": "-1.0 sentinel values were treated as missing rather than numeric evidence.",
                "count": missing_sentinel_count,
            }
        )
    return diagnostics


def _standardize_observations(
    source_key: str,
    source: dict[str, Any],
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    if not rows:
        return []
    columns = set(rows[0])
    if {"variant_id", "antigen_genotype", "endpoint", "value"} <= columns:
        return _standardize_long_endpoint(source_key, source, rows)
    if {"variant_id", "antigen_genotype", "pH", "concentration_ng_ml", "replicate_id", "elisa_signal"} <= columns:
        return _standardize_elisa_dilution(source_key, source, rows)
    if "variant" in columns and any(column.endswith("_ratio") or "_pH" in column for column in columns):
        return _standardize_wide_elisa_summary(source_key, source, rows)
    if "编号" in columns or "浓度ug/ul" in columns:
        return _standardize_expression(source_key, source, rows)
    return []


def _standardize_long_endpoint(
    source_key: str,
    source: dict[str, Any],
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    observations = []
    for row_index, row in enumerate(rows, start=1):
        observations.append(
            _observation(
                source_key,
                row_index,
                variant_id=row.get("variant_id"),
                antigen_genotype=row.get("antigen_genotype"),
                endpoint=row.get("endpoint"),
                value=row.get("value"),
                unit=row.get("unit"),
                evidence_role=_evidence_role(source, default="primary_observation"),
            )
        )
    return observations


def _standardize_elisa_dilution(
    source_key: str,
    source: dict[str, Any],
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    observations = []
    for row_index, row in enumerate(rows, start=1):
        observations.append(
            _observation(
                source_key,
                row_index,
                variant_id=row.get("variant_id"),
                antigen_genotype=row.get("antigen_genotype"),
                endpoint="elisa_signal",
                value=row.get("elisa_signal"),
                unit="absorbance",
                pH=row.get("pH"),
                concentration_ng_ml=row.get("concentration_ng_ml"),
                replicate_id=row.get("replicate_id"),
                evidence_role=_evidence_role(source, default="primary_observation"),
            )
        )
    return observations


def _standardize_wide_elisa_summary(
    source_key: str,
    source: dict[str, Any],
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    observations = []
    value_columns = [
        column
        for column in rows[0]
        if re.match(r"^[A-Za-z0-9]+_pH[0-9.]+$", column) or column.endswith("_ratio")
    ]
    for row_index, row in enumerate(rows, start=1):
        variant_id = row.get("variant")
        for column in value_columns:
            parsed = _parse_summary_endpoint(column)
            observations.append(
                _observation(
                    source_key,
                    row_index,
                    variant_id=variant_id,
                    antigen_genotype=parsed["antigen_genotype"],
                    endpoint=parsed["endpoint"],
                    value=row.get(column),
                    unit="summary_value",
                    pH=parsed.get("pH"),
                    evidence_role="derived_summary",
                )
            )
    return observations


def _standardize_expression(
    source_key: str,
    source: dict[str, Any],
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    observations = []
    for row_index, row in enumerate(rows, start=1):
        variant_id = row.get("variant_id") or row.get("variant") or row.get("编号")
        for endpoint, raw_column, unit in (
            ("expression_concentration", "浓度ug/ul", "ug/ul"),
            ("first_well_volume", "首孔500ng所需体积ul", "ul"),
        ):
            if raw_column not in row:
                continue
            observations.append(
                _observation(
                    source_key,
                    row_index,
                    variant_id=variant_id,
                    endpoint=endpoint,
                    value=row.get(raw_column),
                    unit=unit,
                    evidence_role=_evidence_role(source, default="qc_observation"),
                )
            )
    return observations


def _standardize_sequences(
    source_key: str,
    source: dict[str, Any],
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    del source
    if not rows or "variant_id" not in rows[0]:
        return []
    if not ({"heavy_chain_seq", "light_chain_seq"} & set(rows[0])):
        return []
    sequences = []
    for row_index, row in enumerate(rows, start=1):
        sequences.append(
            {
                "variant_id": row.get("variant_id", ""),
                "heavy_chain_seq": row.get("heavy_chain_seq", ""),
                "light_chain_seq": row.get("light_chain_seq", ""),
                "heavy_construct": row.get("heavy_construct", ""),
                "light_construct": row.get("light_construct", ""),
                "source_file": source_key,
                "source_row": row_index,
            }
        )
    return sequences


def _parse_summary_endpoint(column: str) -> dict[str, str]:
    if column.endswith("_ratio"):
        antigen = column.removesuffix("_ratio")
        return {"antigen_genotype": antigen, "endpoint": "summary_pH74_over_pH60_ratio"}
    antigen, raw_ph = column.split("_pH", 1)
    return {"antigen_genotype": antigen, "endpoint": "summary_elisa_signal", "pH": raw_ph}


def _observation(
    source_file: str,
    source_row: int,
    *,
    variant_id: str | None = None,
    antigen_genotype: str | None = None,
    endpoint: str | None = None,
    value: str | None = None,
    unit: str | None = None,
    pH: str | None = None,
    concentration_ng_ml: str | None = None,
    replicate_id: str | None = None,
    evidence_role: str = "observation",
) -> dict[str, Any]:
    value_status = "missing_sentinel" if str(value).strip() == "-1.0" else "observed"
    if value is None or str(value).strip() == "":
        value_status = "missing"
    variant = str(variant_id or "").strip()
    row = {
        "observation_id": f"{_safe_id(source_file)}:{source_row}:{_safe_id(endpoint or 'endpoint')}:{len(str(value or ''))}",
        "source_file": source_file,
        "source_row": source_row,
        "id_namespace": _id_namespace(variant) if variant else "",
        "variant_id": variant,
        "antigen_genotype": str(antigen_genotype or "").strip(),
        "endpoint": str(endpoint or "").strip(),
        "value": "" if value_status != "observed" else str(value).strip(),
        "value_status": value_status,
        "unit": str(unit or "").strip(),
        "pH": str(pH or "").strip(),
        "concentration_ng_ml": str(concentration_ng_ml or "").strip(),
        "replicate_id": str(replicate_id or "").strip(),
        "evidence_role": evidence_role,
    }
    return row


def _evidence_role(source: dict[str, Any], *, default: str) -> str:
    if source.get("allowed_as_experimental_evidence"):
        return default
    return "reference_or_qc"


def _id_namespace(value: str) -> str:
    text = str(value).strip()
    if not text:
        return ""
    match = re.match(r"^[A-Za-z]+", text)
    if match:
        return match.group(0).lower()
    return "numeric_or_other"


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._-") or "item"


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _diagnostic(severity: str, code: str, source_file: str, message: str) -> dict[str, str]:
    return {"severity": severity, "code": code, "source_file": source_file, "message": message}


def _estimand_endpoints(estimands: dict[str, Any]) -> list[Any]:
    endpoints: list[Any] = []
    for key in ("primary_endpoints", "derived_endpoints", "secondary_endpoints"):
        raw = estimands.get(key)
        if isinstance(raw, list):
            endpoints.extend(raw)
    return endpoints


def _estimand_constraints(estimands: dict[str, Any]) -> dict[str, Any]:
    constraints = estimands.get("constraints")
    if isinstance(constraints, dict):
        return constraints
    if isinstance(constraints, list):
        return {"items": constraints}
    return {}


def _allowed_sources_summary(contract: dict[str, Any]) -> dict[str, Any]:
    allowed = contract.get("allowed_sources")
    if not isinstance(allowed, dict):
        return {}
    summary = {}
    for name, spec in allowed.items():
        if not isinstance(spec, dict):
            continue
        summary[str(name)] = {
            "evidence_tier": spec.get("evidence_tier"),
            "allowed_as_experimental_evidence": bool(spec.get("allowed_as_experimental_evidence", False)),
            "files": list(spec.get("files") or []),
        }
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


__all__ = ["audit_schema", "standardize_startup_data"]
