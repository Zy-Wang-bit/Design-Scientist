"""Small V4 mechanism novelty audit helpers."""

from __future__ import annotations

from typing import Any, Mapping


REQUIRED_FIELDS = ("architecture_delta", "literature_refs")
WEAK_DELTA_OVERLAP = 0.75


def audit_mechanism_novelty(
    mechanism: Mapping[str, Any],
    baselines: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    """Audit whether a mechanism is more than a component-level baseline clone."""

    mechanism_components = _component_keys(mechanism)
    nearest = _nearest_baseline(mechanism_components, baselines)
    baseline_components = nearest["components"]
    new_components = [component for component in mechanism_components if component not in baseline_components]
    missing = [field for field in REQUIRED_FIELDS if not _has_value(mechanism.get(field))]

    baseline_clone = bool(mechanism_components) and mechanism_components == baseline_components
    weak_delta = (
        not baseline_clone
        and bool(nearest["name"])
        and nearest["overlap"] >= WEAK_DELTA_OVERLAP
        and len(new_components) <= 1
    )

    findings: list[str] = []
    if baseline_clone:
        findings.append(f"component clone of baseline {nearest['name']}")
    if weak_delta:
        findings.append(f"weak delta versus baseline {nearest['name']}")
    for field in missing:
        findings.append(f"missing required field: {field}")

    return {
        "schema_version": 1,
        "verdict": "reject" if findings else "pass",
        "baseline_clone": baseline_clone,
        "weak_delta": weak_delta,
        "nearest_baseline": nearest["name"],
        "component_overlap": round(nearest["overlap"], 3),
        "shared_components": nearest["shared"],
        "new_components": new_components,
        "missing_required_fields": missing,
        "findings": findings,
    }


def _nearest_baseline(
    components: list[str],
    baselines: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    best = {"name": None, "components": [], "overlap": 0.0, "shared": []}
    component_set = set(components)
    for baseline in baselines:
        baseline_components = _component_keys(baseline)
        baseline_set = set(baseline_components)
        union = component_set | baseline_set
        shared = [component for component in components if component in baseline_set]
        overlap = len(component_set & baseline_set) / len(union) if union else 0.0
        if overlap > best["overlap"]:
            best = {
                "name": baseline.get("name") or baseline.get("baseline_id") or baseline.get("id"),
                "components": baseline_components,
                "overlap": overlap,
                "shared": shared,
            }
    return best


def _component_keys(data: Mapping[str, Any]) -> list[str]:
    raw_components = data.get("components") or data.get("mechanism_components") or data.get("operator_refs") or []
    keys: list[str] = []
    for component in raw_components:
        key = _component_key(component)
        if key and key not in keys:
            keys.append(key)
    return keys


def _component_key(component: Any) -> str | None:
    if isinstance(component, str):
        return component.strip() or None
    if isinstance(component, Mapping):
        for field in ("component_id", "operator_id", "id", "name"):
            value = component.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _has_value(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set)):
        return any(_has_value(item) for item in value)
    return value is not None
