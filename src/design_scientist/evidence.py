"""Evidence-card construction helpers."""

from __future__ import annotations

from typing import Any

from design_scientist.schemas import EvidenceCard


def generic_source_inventory_evidence_cards(
    row_counts: dict[str, int],
    *,
    project_id: str,
    source_tables: list[str],
) -> list[EvidenceCard]:
    """Build generic inventory evidence for allowlisted project sources."""
    return [
        EvidenceCard(
            evidence_id="allowlisted_source_inventory",
            claim="Allowlisted project sources were inventoried for build-state input availability.",
            evidence_tier="descriptive",
            source_tables=source_tables,
            effect={"project_id": project_id, "row_counts": row_counts},
            limitations=["Inventory counts are provenance/QC evidence and do not imply causal support."],
            actionability="monitor",
            tags=["source_inventory", "allowlist"],
        )
    ]


def source_warning_card(
    *,
    evidence_id: str,
    claim: str,
    source_table: str,
    limitations: list[str],
    actionability: str = "needs_data",
    tags: list[str] | None = None,
    effect: dict[str, Any] | None = None,
) -> EvidenceCard:
    """Build a descriptive source-level warning card."""
    return EvidenceCard(
        evidence_id=evidence_id,
        claim=claim,
        evidence_tier="descriptive",
        source_tables=[source_table],
        effect=effect or {},
        limitations=limitations,
        actionability=actionability,
        tags=tags or [],
    )


def current_anti_hbsag_evidence_cards(row_counts: dict[str, int]) -> list[EvidenceCard]:
    return [
        EvidenceCard(
            evidence_id="data_inventory_current",
            claim=(
                "Current standardized anti-HBsAg data contain real binding, primary endpoint, "
                "and auxiliary QC records with schema-audit provenance."
            ),
            evidence_tier="descriptive",
            source_tables=[
                "standardized_binding_long_table",
                "standardized_binding_primary_table",
                "standardized_auxiliary_table",
            ],
            effect=row_counts,
            limitations=["Inventory evidence does not by itself justify module causality."],
            actionability="monitor",
            tags=["schema_audit", "real_data"],
        ),
        EvidenceCard(
            evidence_id="role_split_sdab_1e62",
            claim="sdAb and 1E62 must be treated as different evidence roles, not one homogeneous training table.",
            evidence_tier="descriptive",
            source_tables=["project.yaml", "data_contract.yaml"],
            limitations=[
                "sdAb module effects may inspire 1E62 experiments but do not automatically transfer."
            ],
            actionability="advance",
            tags=["role_split", "provenance_guard"],
        ),
        EvidenceCard(
            evidence_id="onee62_no_module_causality",
            claim="Current 1E62 combo data do not support module-level causal claims.",
            evidence_tier="descriptive",
            source_tables=["1e62_endpoint_table", "module_summary", "missing_primary_edge_candidates"],
            limitations=["Exact primary matched edges are missing for many 1E62 modules."],
            actionability="needs_data",
            tags=["unsupported_claim", "1E62", "lattice_repair"],
        ),
    ]
