"""Build durable project design state from allowlisted data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from design_scientist.evidence import current_anti_hbsag_evidence_cards
from design_scientist.io import read_yaml, write_json
from design_scientist.schemas import DesignState, EvidenceCard
from design_scientist.table_io import read_identifier_safe_csv


KNOWN_MODULES = {
    "HD110H": {"n_primary_edges": 11, "primary_median_tau": -0.131, "status": "candidate_reusable_module"},
    "HG56H": {"n_primary_edges": 4, "primary_median_tau": -0.150, "status": "candidate_reusable_module"},
    "HN54H": {"n_primary_edges": 4, "primary_median_tau": -0.145, "status": "candidate_reusable_module"},
    "HV105H": {"n_primary_edges": 1, "primary_median_tau": -1.039, "status": "strong_but_narrow_signal"},
}


def _count_rows(path: Path) -> int:
    if not path.exists() or path.suffix.lower() != ".csv":
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


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


def build_state(project_dir: str | Path) -> DesignState:
    root = Path(project_dir).expanduser().resolve()
    contract = read_yaml(root / "data_contract.yaml")
    row_counts: dict[str, int] = {}
    table_paths: dict[str, Path] = {}
    warning_cards: list[EvidenceCard] = []

    for table in contract.get("allowed_tables", []):
        name = table["name"]
        path = Path(table["path"]).expanduser()
        table_paths[name] = path
        row_counts[name] = _count_rows(path)
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
        project_id=str(contract.get("project_id", "anti_hbsag")),
        data_version=str(contract.get("version", "unknown")),
        current_round=0,
        row_counts=row_counts,
        system_roles={
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
    evidence = current_anti_hbsag_evidence_cards(row_counts) + warning_cards
    write_json(root / "state" / "design_state.json", state)
    write_json(root / "state" / "evidence_cards.json", evidence)
    print(f"Built design state at {root / 'state'}")
    return state
