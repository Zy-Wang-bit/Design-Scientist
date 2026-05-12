"""V3 mechanism lifecycle replay benchmark over deterministic synthetic worlds."""

from __future__ import annotations

import csv
import os
import random
import re
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from design_scientist import policies as policy_api
from design_scientist import synthetic_replay
from design_scientist.method_nodes import (
    _is_relative_to,
    _snapshot_entry,
    diff_filesystem_snapshots,
    snapshot_filesystem,
)


DEFAULT_RUN_ID = "mechanism_replay_seed_1729"
BASELINE_MECHANISM_NAMES = ("random_feasible", "fixed_mix")
DEFAULT_MECHANISM_NAMES = (*BASELINE_MECHANISM_NAMES, "mechanism_aware")
CLONE_REFERENCE_MECHANISM_NAMES = DEFAULT_MECHANISM_NAMES
KEY_ABLATION_NAME = "key_component_removed"
LIFECYCLE_FUNCTION_NAMES = (
    "fit_state",
    "generate_candidates",
    "score_candidates",
    "select_panel",
    "plan_ablations",
)

LifecycleFunction = Callable[..., Any]


@dataclass(frozen=True)
class MechanismLifecycle:
    """Minimal V3 lifecycle used by the mechanism replay benchmark."""

    name: str
    fit_state: LifecycleFunction
    generate_candidates: LifecycleFunction
    score_candidates: LifecycleFunction
    select_panel: LifecycleFunction
    plan_ablations: LifecycleFunction
    claims: tuple[str, ...] = ()
    architecture_clone: bool = False
    claim_guardrail: bool = True
    confounding_correction: bool = True


class ReplayPathGuardError(RuntimeError):
    """Raised when replay lifecycle execution writes outside the project workspace."""


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
    "shutil.copyfile": (0, 1),
    "shutil.copymode": (0, 1),
    "shutil.copystat": (0, 1),
    "shutil.copytree": (0, 1),
    "shutil.move": (0, 1),
    "tempfile.mkstemp": (0,),
    "tempfile.mkdtemp": (0,),
}
_REPLAY_AUDIT_GUARD_INSTALLED = False
_REPLAY_AUDIT_GUARD_ACTIVE = False


def make_policy_mechanism(
    name: str,
    policy: policy_api.PolicyCallable,
    *,
    claims: Iterable[str] | None = None,
    architecture_clone: bool = False,
    key_policy_ablation: str | None = None,
    claim_guardrail: bool | None = None,
    confounding_correction: bool | None = None,
) -> MechanismLifecycle:
    """Wrap a policy API callable in the V3 lifecycle shape."""

    if claim_guardrail is None:
        claim_guardrail = name == "mechanism_aware"
    if confounding_correction is None:
        confounding_correction = name == "mechanism_aware"

    def fit_state(context: Mapping[str, Any]) -> dict[str, Any]:
        return dict(context)

    def generate_candidates(state: Mapping[str, Any]) -> list[dict[str, Any]]:
        return list(state["candidate_records"])

    def score_candidates(
        state: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        del state
        return [dict(candidate) for candidate in candidates]

    def select_panel(
        state: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        budget: int,
        rng: random.Random,
    ) -> list[str]:
        return policy(
            state["observed_records"],
            candidates,
            budget,
            state["round_index"],
            rng,
        )

    def plan_ablations(state: Mapping[str, Any]) -> list[dict[str, Any]]:
        del state
        return [
            {"name": "none"},
            {"name": KEY_ABLATION_NAME, "policy_ablation": key_policy_ablation},
        ]

    return MechanismLifecycle(
        name=_safe_name(name),
        fit_state=fit_state,
        generate_candidates=generate_candidates,
        score_candidates=score_candidates,
        select_panel=select_panel,
        plan_ablations=plan_ablations,
        claims=_normalize_claims(claims),
        architecture_clone=bool(architecture_clone),
        claim_guardrail=bool(claim_guardrail),
        confounding_correction=bool(confounding_correction),
    )


def run_mechanism_benchmark(
    project_dir: str | Path,
    run_id: str = DEFAULT_RUN_ID,
    mechanisms: (
        Iterable[str | policy_api.PolicyCallable | MechanismLifecycle | Mapping[str, Any] | object]
        | str
        | policy_api.PolicyCallable
        | MechanismLifecycle
        | Mapping[str, Any]
        | object
        | None
    ) = None,
    rounds: int = 3,
    budget: int = 12,
) -> dict[str, Any]:
    """Run a V3 mechanism lifecycle replay benchmark and write CSV artifacts."""

    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if budget <= 0:
        raise ValueError("budget must be positive")

    safe_run_id = _safe_run_id(run_id)
    root = Path(project_dir).expanduser().resolve()
    requested_mechanisms = _normalize_requested_mechanisms(mechanisms)
    benchmark_mechanisms = _with_required_baselines(requested_mechanisms)
    world_ids = world_ids_for_claims(_mechanism_claims(requested_mechanisms))
    world_specs = _select_worlds(world_ids)
    run_dir = root / "runs" / safe_run_id
    replay_snapshot_roots = _replay_snapshot_roots(root)

    benchmark_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []

    config = {
        "benchmark": "v3_mechanism_lifecycle_replay",
        "seed": synthetic_replay.SYNTHETIC_SEED,
        "rounds": rounds,
        "budget_per_round": budget,
        "mechanisms": [mechanism.name for mechanism in benchmark_mechanisms],
        "requested_mechanisms": [mechanism.name for mechanism in requested_mechanisms],
        "required_baselines": list(BASELINE_MECHANISM_NAMES),
        "worlds": [world.world_id for world in world_specs],
    }

    for world in world_specs:
        observations_by_id = synthetic_replay._build_observation_table(  # noqa: SLF001
            seed=synthetic_replay.SYNTHETIC_SEED,
            world=world,
        )
        variants = [record.variant for record in observations_by_id.values()]
        initial_ids = synthetic_replay._initial_observation_ids(  # noqa: SLF001
            observations_by_id,
            world=world,
        )
        truth = synthetic_replay._truth_summary(observations_by_id)  # noqa: SLF001

        full_results = {
            mechanism.name: _run_lifecycle_safely(
                mechanism,
                world=world,
                variants=variants,
                observations_by_id=observations_by_id,
                initial_ids=initial_ids,
                rounds=rounds,
                budget=budget,
                ablation={"name": "none"},
                project_root=root,
                snapshot_roots=replay_snapshot_roots,
            )
            for mechanism in benchmark_mechanisms
        }
        reference_selected_ids = {
            baseline: set(full_results[baseline]["selected_ids"])
            for baseline in BASELINE_MECHANISM_NAMES
        }

        for mechanism in benchmark_mechanisms:
            result = full_results[mechanism.name]
            row = _benchmark_row(
                result,
                mechanism=mechanism,
                truth=truth,
                rounds=rounds,
                budget=budget,
                initial_ids=initial_ids,
                observations_by_id=observations_by_id,
                reference_selected_ids=reference_selected_ids,
            )
            benchmark_rows.append(row)

            if result.get("status") != "completed":
                ablation_rows.append(
                    _ablation_row(
                        result,
                        mechanism=mechanism,
                        truth=truth,
                        rounds=rounds,
                        budget=budget,
                        initial_ids=initial_ids,
                        observations_by_id=observations_by_id,
                        reference_selected_ids=reference_selected_ids,
                        ablation_spec={"name": "none", "is_key": False},
                        full_best_feasible_utility=0.0,
                    )
                )
                continue

            full_best = float(result["metrics"]["best_feasible_utility"])
            plan_context = {
                "mechanism": mechanism.name,
                "world_id": world.world_id,
                "rounds": rounds,
                "budget": budget,
            }
            try:
                raw_ablation_specs = _guarded_replay_call(
                    lambda: _call_plan_ablations(mechanism, plan_context),
                    project_root=root,
                    snapshot_roots=replay_snapshot_roots,
                )
            except Exception as exc:  # pragma: no cover - exercised through failed CSV rows
                ablation_rows.append(
                    _ablation_row(
                        _failed_lifecycle_result(
                            mechanism,
                            world=world,
                            initial_ids=initial_ids,
                            ablation={"name": KEY_ABLATION_NAME},
                            error=exc,
                        ),
                        mechanism=mechanism,
                        truth=truth,
                        rounds=rounds,
                        budget=budget,
                        initial_ids=initial_ids,
                        observations_by_id=observations_by_id,
                        reference_selected_ids=reference_selected_ids,
                        ablation_spec={"name": KEY_ABLATION_NAME, "is_key": True},
                        full_best_feasible_utility=full_best,
                    )
                )
                continue

            ablation_specs = _normalize_ablation_specs(raw_ablation_specs)
            for ablation_spec in ablation_specs:
                ablation_name = str(ablation_spec["name"])
                if ablation_name == "none":
                    ablation_result = result
                else:
                    ablation_result = _run_lifecycle_safely(
                        mechanism,
                        world=world,
                        variants=variants,
                        observations_by_id=observations_by_id,
                        initial_ids=initial_ids,
                        rounds=rounds,
                        budget=budget,
                        ablation=ablation_spec,
                        project_root=root,
                        snapshot_roots=replay_snapshot_roots,
                    )
                ablation_row = _ablation_row(
                    ablation_result,
                    mechanism=mechanism,
                    truth=truth,
                    rounds=rounds,
                    budget=budget,
                    initial_ids=initial_ids,
                    observations_by_id=observations_by_id,
                    reference_selected_ids=reference_selected_ids,
                    ablation_spec=ablation_spec,
                    full_best_feasible_utility=full_best,
                )
                ablation_rows.append(ablation_row)

    summary_rows = _summary_rows(
        benchmark_rows,
        ablation_rows,
        mechanisms=benchmark_mechanisms,
    )

    benchmark_path = run_dir / "mechanism_benchmark_results.csv"
    summary_path = run_dir / "mechanism_benchmark_summary.csv"
    ablation_path = run_dir / "mechanism_ablation_results.csv"
    _write_csv(benchmark_path, benchmark_rows)
    _write_csv(summary_path, summary_rows)
    _write_csv(ablation_path, ablation_rows)

    return {
        "run_id": safe_run_id,
        "benchmark_results_path": str(benchmark_path),
        "summary_results_path": str(summary_path),
        "ablation_results_path": str(ablation_path),
        "mechanism_benchmark_results_path": str(benchmark_path),
        "mechanism_benchmark_summary_path": str(summary_path),
        "mechanism_ablation_results_path": str(ablation_path),
        "benchmark_results": benchmark_rows,
        "ablation_results": ablation_rows,
        "summary": {"mechanism_rankings": summary_rows},
        "config": config,
    }


def world_ids_for_claims(claims: Iterable[str]) -> list[str]:
    """Map mechanism claims to deterministic stress-test world IDs."""

    text = " ".join(str(claim) for claim in claims).lower()
    selected: list[str] = []
    if "transfer" in text or "confound" in text:
        selected.append("confounded_transfer")
    if "sparse" in text or "tiny" in text or "early" in text:
        selected.append("sparse_early_round")
    if "interaction" in text or "epistasis" in text or "epistatic" in text:
        selected.append("epistatic")
    if "noise" in text or "noisy" in text or "assay" in text:
        selected.append("noisy_endpoint")
    return _dedupe(selected) if selected else ["additive", "epistatic"]


def _normalize_requested_mechanisms(
    mechanisms: (
        Iterable[str | policy_api.PolicyCallable | MechanismLifecycle | Mapping[str, Any]]
        | str
        | policy_api.PolicyCallable
        | MechanismLifecycle
        | Mapping[str, Any]
        | None
    ),
) -> list[MechanismLifecycle]:
    if mechanisms is None:
        raw_mechanisms: list[Any] = [BASELINE_MECHANISMS[name] for name in DEFAULT_MECHANISM_NAMES]
    elif (
        isinstance(mechanisms, (str, MechanismLifecycle, Mapping))
        or _is_lifecycle_object(mechanisms)
        or callable(mechanisms)
    ):
        raw_mechanisms = [mechanisms]
    else:
        raw_mechanisms = list(mechanisms)

    normalized: list[MechanismLifecycle] = []
    seen: set[str] = set()
    for raw_mechanism in raw_mechanisms:
        mechanism = _normalize_mechanism(raw_mechanism)
        if mechanism.name not in seen:
            normalized.append(mechanism)
            seen.add(mechanism.name)
    if not normalized:
        raise ValueError("At least one mechanism is required")
    return normalized


def _normalize_mechanism(
    raw_mechanism: str | policy_api.PolicyCallable | MechanismLifecycle | Mapping[str, Any],
) -> MechanismLifecycle:
    if isinstance(raw_mechanism, MechanismLifecycle):
        return raw_mechanism
    if isinstance(raw_mechanism, str):
        if raw_mechanism not in BASELINE_MECHANISMS:
            raise ValueError(f"Unknown mechanism: {raw_mechanism}")
        return BASELINE_MECHANISMS[raw_mechanism]
    if isinstance(raw_mechanism, Mapping):
        return _dict_lifecycle(raw_mechanism)
    if _is_lifecycle_object(raw_mechanism):
        return _object_lifecycle(raw_mechanism)
    if callable(raw_mechanism):
        name = getattr(raw_mechanism, "policy_name", None) or getattr(raw_mechanism, "__name__", None)
        return make_policy_mechanism(_safe_name(str(name or "policy_callable")), raw_mechanism)
    raise TypeError("mechanisms must be names, policy callables, lifecycle objects, or lifecycle dicts")


def _dict_lifecycle(raw: Mapping[str, Any]) -> MechanismLifecycle:
    missing = [
        key
        for key in (
            *LIFECYCLE_FUNCTION_NAMES,
        )
        if not callable(raw.get(key))
    ]
    if missing:
        raise ValueError(f"Mechanism lifecycle is missing callable(s): {', '.join(missing)}")
    name = _safe_name(str(raw.get("name") or raw.get("mechanism") or "mechanism"))
    return MechanismLifecycle(
        name=name,
        fit_state=raw["fit_state"],
        generate_candidates=raw["generate_candidates"],
        score_candidates=raw["score_candidates"],
        select_panel=raw["select_panel"],
        plan_ablations=raw["plan_ablations"],
        claims=_normalize_claims(raw.get("claims")),
        architecture_clone=_as_bool(raw.get("architecture_clone", False)),
        claim_guardrail=_as_bool(raw.get("claim_guardrail", True)),
        confounding_correction=_as_bool(raw.get("confounding_correction", True)),
    )


def _object_lifecycle(raw: object) -> MechanismLifecycle:
    name = _safe_name(
        str(
            getattr(raw, "name", None)
            or getattr(raw, "mechanism", None)
            or raw.__class__.__name__
        )
    )
    return MechanismLifecycle(
        name=name,
        fit_state=getattr(raw, "fit_state"),
        generate_candidates=getattr(raw, "generate_candidates"),
        score_candidates=getattr(raw, "score_candidates"),
        select_panel=getattr(raw, "select_panel"),
        plan_ablations=getattr(raw, "plan_ablations"),
        claims=_normalize_claims(getattr(raw, "claims", None)),
        architecture_clone=_as_bool(getattr(raw, "architecture_clone", False)),
        claim_guardrail=_as_bool(getattr(raw, "claim_guardrail", True)),
        confounding_correction=_as_bool(getattr(raw, "confounding_correction", True)),
    )


def _is_lifecycle_object(value: object) -> bool:
    if isinstance(value, type):
        return False
    return all(callable(getattr(value, name, None)) for name in LIFECYCLE_FUNCTION_NAMES)


def _with_required_baselines(
    mechanisms: Sequence[MechanismLifecycle],
) -> list[MechanismLifecycle]:
    resolved: list[MechanismLifecycle] = []
    seen: set[str] = set()
    for baseline in CLONE_REFERENCE_MECHANISM_NAMES:
        mechanism = BASELINE_MECHANISMS[baseline]
        resolved.append(mechanism)
        seen.add(mechanism.name)
    for mechanism in mechanisms:
        if mechanism.name not in seen:
            resolved.append(mechanism)
            seen.add(mechanism.name)
    return resolved


def _mechanism_claims(mechanisms: Sequence[MechanismLifecycle]) -> list[str]:
    claims: list[str] = []
    for mechanism in mechanisms:
        claims.extend(mechanism.claims)
    return claims


def _select_worlds(world_ids: Sequence[str]) -> list[synthetic_replay.WorldSpec]:
    world_by_id = {world.world_id: world for world in synthetic_replay._world_suite()}  # noqa: SLF001
    worlds: list[synthetic_replay.WorldSpec] = []
    for world_id in world_ids:
        try:
            worlds.append(world_by_id[world_id])
        except KeyError as exc:
            raise ValueError(f"Unknown mechanism replay world: {world_id}") from exc
    return worlds


def _run_lifecycle_safely(
    mechanism: MechanismLifecycle,
    *,
    world: synthetic_replay.WorldSpec,
    variants: list[synthetic_replay.SyntheticVariant],
    observations_by_id: dict[str, synthetic_replay.SyntheticObservation],
    initial_ids: list[str],
    rounds: int,
    budget: int,
    ablation: Mapping[str, Any],
    project_root: Path,
    snapshot_roots: Sequence[Path],
) -> dict[str, Any]:
    try:
        result = _guarded_replay_call(
            lambda: _run_lifecycle_replay(
                mechanism,
                world=world,
                variants=variants,
                observations_by_id=observations_by_id,
                initial_ids=initial_ids,
                rounds=rounds,
                budget=budget,
                ablation=ablation,
            ),
            project_root=project_root,
            snapshot_roots=snapshot_roots,
        )
    except Exception as exc:  # pragma: no cover - exercised via CSV status in integration paths
        return _failed_lifecycle_result(
            mechanism,
            world=world,
            initial_ids=initial_ids,
            ablation=ablation,
            error=exc,
        )
    result["status"] = "completed"
    result["error"] = ""
    return result


def _failed_lifecycle_result(
    mechanism: MechanismLifecycle,
    *,
    world: synthetic_replay.WorldSpec,
    initial_ids: list[str],
    ablation: Mapping[str, Any],
    error: Exception,
) -> dict[str, Any]:
    return {
        "world_id": world.world_id,
        "method": mechanism.name,
        "ablation": str(ablation.get("name") or "none"),
        "selected_count": 0,
        "total_observations": len(initial_ids),
        "selected_ids": tuple(),
        "metrics": {
            "best_feasible_utility": 0.0,
            "hit_rate": 0.0,
            "regret_proxy": 1.0,
            "false_claim_rate": 1.0,
            "evidence_coverage": 0.0,
            "round_efficiency": 0.0,
        },
        "status": "failed",
        "error": f"{type(error).__name__}: {error}",
    }


def _run_lifecycle_replay(
    mechanism: MechanismLifecycle,
    *,
    world: synthetic_replay.WorldSpec,
    variants: list[synthetic_replay.SyntheticVariant],
    observations_by_id: dict[str, synthetic_replay.SyntheticObservation],
    initial_ids: list[str],
    rounds: int,
    budget: int,
    ablation: Mapping[str, Any],
) -> dict[str, Any]:
    observed_ids = set(initial_ids)
    observations = [observations_by_id[variant_id] for variant_id in initial_ids]
    selected_ids: list[str] = []
    selected_count = 0
    round_best: list[float] = []
    ablation_name = str(ablation.get("name") or "none")
    policy_ablation = _optional_str(ablation.get("policy_ablation"))

    for round_index in range(1, rounds + 1):
        available = [variant for variant in variants if variant.variant_id not in observed_ids]
        if not available:
            round_best.append(synthetic_replay._best_feasible_utility(observations))  # noqa: SLF001
            continue

        observed_records = [
            synthetic_replay._policy_observation_record(observation, world=world)  # noqa: SLF001
            for observation in observations
        ]
        candidate_records = [
            synthetic_replay._policy_candidate_record(  # noqa: SLF001
                variant,
                world=world,
                ablation=policy_ablation,
            )
            for variant in available
        ]
        state = _call_fit_state(
            mechanism,
            {
                "mechanism": mechanism.name,
                "world": world,
                "world_id": world.world_id,
                "round_index": round_index,
                "budget": budget,
                "observations": tuple(observations),
                "observed_ids": frozenset(observed_ids),
                "available_variants": tuple(available),
                "observed_records": observed_records,
                "candidate_records": candidate_records,
                "ablation": ablation_name,
                "policy_ablation": policy_ablation,
            },
        )
        candidates = _call_generate_candidates(mechanism, state)
        scored_candidates = _call_score_candidates(mechanism, state, candidates)
        rng = random.Random(
            synthetic_replay._stable_seed(  # noqa: SLF001
                synthetic_replay.SYNTHETIC_SEED,
                "mechanism",
                world.world_id,
                mechanism.name,
                round_index,
                policy_ablation or "full",
            )
        )
        selected_raw = _call_select_panel(mechanism, state, scored_candidates, budget, rng)
        selected = synthetic_replay._resolve_selected_variants(  # noqa: SLF001
            selected_raw,
            available,
            budget,
        )
        for variant in selected:
            observed_ids.add(variant.variant_id)
            observations.append(observations_by_id[variant.variant_id])
            selected_ids.append(variant.variant_id)
        selected_count += len(selected)
        round_best.append(synthetic_replay._best_feasible_utility(observations))  # noqa: SLF001

    claim_guardrail = _as_bool(ablation.get("claim_guardrail", mechanism.claim_guardrail))
    confounding_correction = _as_bool(
        ablation.get("confounding_correction", mechanism.confounding_correction)
    )
    if policy_ablation == "evidence_guardrail":
        claim_guardrail = False
    if policy_ablation == "confounding_correction":
        confounding_correction = False

    metrics = synthetic_replay._metrics(  # noqa: SLF001
        observations=observations,
        selected_count=selected_count,
        observations_by_id=observations_by_id,
        round_best=round_best,
        claim_guardrail=claim_guardrail,
        confounding_correction=confounding_correction,
    )
    return {
        "world_id": world.world_id,
        "method": mechanism.name,
        "ablation": ablation_name,
        "selected_count": selected_count,
        "total_observations": len(observations),
        "selected_ids": tuple(selected_ids),
        "metrics": metrics,
    }


def _benchmark_row(
    result: dict[str, Any],
    *,
    mechanism: MechanismLifecycle,
    truth: dict[str, Any],
    rounds: int,
    budget: int,
    initial_ids: list[str],
    observations_by_id: dict[str, synthetic_replay.SyntheticObservation],
    reference_selected_ids: Mapping[str, set[str]],
) -> dict[str, Any]:
    row = synthetic_replay._result_row(  # noqa: SLF001
        result,
        truth,
        rounds,
        budget,
        len(initial_ids),
    )
    row["mechanism"] = row.pop("method")
    row["architecture_clone_declared"] = _bool_text(mechanism.architecture_clone)
    row["novelty_score"] = synthetic_replay._selection_novelty(  # noqa: SLF001
        result["selected_ids"],
        initial_ids,
        observations_by_id,
    )
    row["selected_ids_digest"] = synthetic_replay._selection_digest(result["selected_ids"])  # noqa: SLF001
    for baseline, selected_ids in reference_selected_ids.items():
        row[f"selection_overlap_{baseline}"] = synthetic_replay._baseline_overlap(  # noqa: SLF001
            result["selected_ids"],
            selected_ids,
        )
    return row


def _ablation_row(
    result: dict[str, Any],
    *,
    mechanism: MechanismLifecycle,
    truth: dict[str, Any],
    rounds: int,
    budget: int,
    initial_ids: list[str],
    observations_by_id: dict[str, synthetic_replay.SyntheticObservation],
    reference_selected_ids: Mapping[str, set[str]],
    ablation_spec: Mapping[str, Any],
    full_best_feasible_utility: float,
) -> dict[str, Any]:
    row = _benchmark_row(
        result,
        mechanism=mechanism,
        truth=truth,
        rounds=rounds,
        budget=budget,
        initial_ids=initial_ids,
        observations_by_id=observations_by_id,
        reference_selected_ids=reference_selected_ids,
    )
    row["ablation"] = str(ablation_spec["name"])
    row["policy_ablation"] = _optional_str(ablation_spec.get("policy_ablation")) or ""
    row["is_key_ablation"] = _bool_text(_as_bool(ablation_spec.get("is_key", False)))
    row["delta_from_full_best_feasible_utility"] = _round_metric(
        full_best_feasible_utility - float(row["best_feasible_utility"])
    )
    return row


def _summary_rows(
    benchmark_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    *,
    mechanisms: Sequence[MechanismLifecycle],
) -> list[dict[str, Any]]:
    lifecycle_by_name = {mechanism.name: mechanism for mechanism in mechanisms}
    rows_by_mechanism: dict[str, list[dict[str, Any]]] = {}
    by_world_mechanism: dict[tuple[str, str], dict[str, Any]] = {}
    for row in benchmark_rows:
        mechanism = str(row["mechanism"])
        world_id = str(row["world_id"])
        rows_by_mechanism.setdefault(mechanism, []).append(row)
        by_world_mechanism[(world_id, mechanism)] = row

    random_false = _mean_false_claim(rows_by_mechanism.get("random_feasible", []))
    fixed_false = _mean_false_claim(rows_by_mechanism.get("fixed_mix", []))

    summary: list[dict[str, Any]] = []
    for mechanism_name, mechanism_rows in rows_by_mechanism.items():
        lifecycle = lifecycle_by_name[mechanism_name]
        majority_random = _majority_win(
            mechanism_rows,
            by_world_mechanism,
            baseline="random_feasible",
        )
        majority_fixed = _majority_win(
            mechanism_rows,
            by_world_mechanism,
            baseline="fixed_mix",
        )
        mean_best = _round_metric(
            mean(float(row["best_feasible_utility"]) for row in mechanism_rows)
        )
        mean_false = _round_metric(
            mean(float(row["false_claim_rate"]) for row in mechanism_rows)
        )
        key_delta = _key_ablation_delta(ablation_rows, mechanism_name)
        architecture_clone = lifecycle.architecture_clone or _selection_clone(
            mechanism_rows,
            by_world_mechanism,
        )
        false_claim_worse_than_both = mean_false > random_false and mean_false > fixed_false
        completed_required_rows = _required_rows_completed(
            mechanism_rows,
            ablation_rows,
            mechanism_name,
        )
        selected_eligible = not (
            architecture_clone
            or not majority_random
            or not majority_fixed
            or key_delta <= 0.0
            or false_claim_worse_than_both
            or not completed_required_rows
        )
        summary.append(
            {
                "mechanism": mechanism_name,
                "rank": 0,
                "worlds_tested": len({str(row["world_id"]) for row in mechanism_rows}),
                "majority_win_vs_random_feasible": _bool_text(majority_random),
                "majority_win_vs_fixed_mix": _bool_text(majority_fixed),
                "mean_best_feasible_utility": mean_best,
                "mean_false_claim_rate": mean_false,
                "key_ablation_delta": key_delta,
                "architecture_clone": _bool_text(architecture_clone),
                "selected_eligible": _bool_text(selected_eligible),
            }
        )

    summary.sort(
        key=lambda row: (
            row["selected_eligible"] == "true",
            float(row["mean_best_feasible_utility"]),
            float(row["key_ablation_delta"]),
            -float(row["mean_false_claim_rate"]),
            row["mechanism"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(summary, start=1):
        row["rank"] = rank
    return summary


def _majority_win(
    mechanism_rows: list[dict[str, Any]],
    by_world_mechanism: Mapping[tuple[str, str], dict[str, Any]],
    *,
    baseline: str,
) -> bool:
    wins = 0
    comparable = 0
    for row in mechanism_rows:
        baseline_row = by_world_mechanism.get((str(row["world_id"]), baseline))
        if baseline_row is None:
            continue
        comparable += 1
        if float(row["best_feasible_utility"]) > float(baseline_row["best_feasible_utility"]):
            wins += 1
    return comparable > 0 and wins > comparable / 2


def _selection_clone(
    mechanism_rows: list[dict[str, Any]],
    by_world_mechanism: Mapping[tuple[str, str], dict[str, Any]],
) -> bool:
    if not mechanism_rows:
        return False
    mechanism_name = str(mechanism_rows[0]["mechanism"])
    if mechanism_name in CLONE_REFERENCE_MECHANISM_NAMES:
        return False
    for baseline in CLONE_REFERENCE_MECHANISM_NAMES:
        comparable = 0
        matching = 0
        for row in mechanism_rows:
            baseline_row = by_world_mechanism.get((str(row["world_id"]), baseline))
            if baseline_row is None:
                continue
            comparable += 1
            if row.get("selected_ids_digest") == baseline_row.get("selected_ids_digest"):
                matching += 1
        if comparable > 0 and comparable == matching:
            return True
    return False


def _key_ablation_delta(ablation_rows: list[dict[str, Any]], mechanism_name: str) -> float:
    deltas = [
        float(row["delta_from_full_best_feasible_utility"])
        for row in ablation_rows
        if row["mechanism"] == mechanism_name
        and row.get("is_key_ablation") == "true"
        and row.get("status") == "completed"
    ]
    return _round_metric(mean(deltas)) if deltas else 0.0


def _required_rows_completed(
    benchmark_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    mechanism_name: str,
) -> bool:
    if any(row.get("status") != "completed" for row in benchmark_rows):
        return False
    key_rows = [
        row
        for row in ablation_rows
        if row["mechanism"] == mechanism_name and row.get("is_key_ablation") == "true"
    ]
    return not key_rows or all(row.get("status") == "completed" for row in key_rows)


def _mean_false_claim(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return _round_metric(mean(float(row["false_claim_rate"]) for row in rows))


def _normalize_ablation_specs(raw_specs: Any) -> list[dict[str, Any]]:
    if raw_specs is None:
        raw_items: list[Any] = []
    elif isinstance(raw_specs, (str, Mapping)):
        raw_items = [raw_specs]
    else:
        raw_items = list(raw_specs)

    specs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        if isinstance(raw_item, str):
            spec = {"name": raw_item}
        elif isinstance(raw_item, Mapping):
            raw_name = raw_item.get("name") or raw_item.get("ablation") or raw_item.get("removed_component")
            spec = dict(raw_item)
            spec["name"] = str(raw_name or "none")
        else:
            raise TypeError("Ablation specs must be strings or mappings")
        spec["name"] = _safe_name(str(spec["name"])) if spec["name"] != "none" else "none"
        if spec["name"] in seen:
            continue
        specs.append(spec)
        seen.add(spec["name"])

    if "none" not in seen:
        specs.insert(0, {"name": "none"})
    key_seen = False
    for spec in specs:
        if spec["name"] == "none":
            spec["is_key"] = False
        elif not key_seen:
            spec["is_key"] = True
            key_seen = True
        else:
            spec["is_key"] = _as_bool(spec.get("is_key", False))
    return specs


def _guarded_replay_call(
    call: Callable[[], Any],
    *,
    project_root: Path,
    snapshot_roots: Sequence[Path],
) -> Any:
    _ensure_replay_audit_guard_installed()
    before = _snapshot_replay_guard(snapshot_roots)
    result: Any = None
    call_error: Exception | None = None
    _set_replay_audit_guard_active(True)
    try:
        result = call()
    except Exception as exc:  # pragma: no cover - returned as failed rows by callers
        call_error = exc
    finally:
        _set_replay_audit_guard_active(False)
    after = _snapshot_replay_guard(snapshot_roots)
    filesystem_writes = _replay_filesystem_writes(diff_filesystem_snapshots(before, after))
    if filesystem_writes:
        raise ReplayPathGuardError(
            "filesystem writes outside replay workspace: "
            + ", ".join(filesystem_writes)
        )
    if call_error is not None:
        raise call_error
    return result


def _replay_filesystem_writes(filesystem_changes: Iterable[Any]) -> list[str]:
    return sorted(
        {
            str(change.path)
            for change in filesystem_changes
        }
    )


def _ensure_replay_audit_guard_installed() -> None:
    global _REPLAY_AUDIT_GUARD_INSTALLED
    if _REPLAY_AUDIT_GUARD_INSTALLED:
        return

    def guard(event: str, args: tuple[Any, ...]) -> None:
        if not _REPLAY_AUDIT_GUARD_ACTIVE:
            return
        if event in _AUDIT_BLOCKED_PROCESS_EVENTS:
            raise ReplayPathGuardError(
                f"filesystem writes outside replay workspace: blocked process event {event}"
            )
        for candidate in _audit_write_paths(event, args):
            path = _resolve_audit_path(candidate)
            if path is None:
                continue
            raise ReplayPathGuardError(
                f"filesystem writes outside replay workspace: {path}"
            )

    sys.addaudithook(guard)
    _REPLAY_AUDIT_GUARD_INSTALLED = True


def _set_replay_audit_guard_active(active: bool) -> None:
    global _REPLAY_AUDIT_GUARD_ACTIVE
    _REPLAY_AUDIT_GUARD_ACTIVE = active


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


def _replay_snapshot_roots(project_root: Path) -> list[Path]:
    return _unique_paths([project_root, *_temporary_guard_roots()])


def _snapshot_replay_guard(roots: Iterable[Path]) -> dict[Path, Any]:
    temp_roots = set(_temporary_guard_roots())
    recursive_roots = [root for root in roots if root not in temp_roots]
    snapshot = snapshot_filesystem(recursive_roots)
    snapshot.update(_snapshot_shallow_roots(temp_roots))
    return snapshot


def _snapshot_shallow_roots(roots: Iterable[Path]) -> dict[Path, Any]:
    snapshot: dict[Path, Any] = {}
    for root in _unique_paths(roots):
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
            root = Path(candidate).expanduser().resolve()
        except OSError:
            continue
        if root.exists():
            roots.append(root)
    return _unique_paths(roots)


def _absolute_no_symlink_resolve(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _call_fit_state(mechanism: MechanismLifecycle, context: Mapping[str, Any]) -> Any:
    return mechanism.fit_state(context)


def _call_generate_candidates(mechanism: MechanismLifecycle, state: Any) -> list[Any]:
    return list(mechanism.generate_candidates(state))


def _call_score_candidates(
    mechanism: MechanismLifecycle,
    state: Any,
    candidates: Sequence[Any],
) -> list[Any]:
    return list(mechanism.score_candidates(state, candidates))


def _call_select_panel(
    mechanism: MechanismLifecycle,
    state: Any,
    candidates: Sequence[Any],
    budget: int,
    rng: random.Random,
) -> list[Any]:
    return list(mechanism.select_panel(state, candidates, budget, rng))


def _call_plan_ablations(mechanism: MechanismLifecycle, state: Mapping[str, Any]) -> Any:
    return mechanism.plan_ablations(state)


def _safe_run_id(run_id: str | None) -> str:
    if run_id is None:
        return DEFAULT_RUN_ID
    if not run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a non-empty relative name")
    if "/" in run_id or "\\" in run_id or ".." in Path(run_id).parts:
        raise ValueError("run_id must not contain path separators or '..'")
    return run_id


def _safe_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._-")
    return safe or "mechanism"


def _dedupe(values: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        deduped.append(value)
        seen.add(value)
    return deduped


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _normalize_claims(claims: Any) -> tuple[str, ...]:
    if claims is None:
        return ()
    if isinstance(claims, str):
        return (claims,)
    return tuple(str(claim) for claim in claims)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _bool_text(value: Any) -> str:
    return "true" if _as_bool(value) else "false"


def _round_metric(value: float) -> float:
    return round(float(value), 6)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _ensure_dir(path.parent)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


random_feasible_mechanism = make_policy_mechanism(
    "random_feasible",
    policy_api.random_feasible,
    claim_guardrail=False,
    confounding_correction=False,
)
fixed_mix_mechanism = make_policy_mechanism(
    "fixed_mix",
    policy_api.fixed_mix,
    claim_guardrail=False,
    confounding_correction=False,
)
mechanism_aware_mechanism = make_policy_mechanism(
    "mechanism_aware",
    policy_api.mechanism_aware,
    key_policy_ablation="interaction_prior",
    claim_guardrail=True,
    confounding_correction=True,
)

BASELINE_MECHANISMS = {
    "random_feasible": random_feasible_mechanism,
    "fixed_mix": fixed_mix_mechanism,
    "mechanism_aware": mechanism_aware_mechanism,
}


__all__ = [
    "BASELINE_MECHANISMS",
    "DEFAULT_RUN_ID",
    "MechanismLifecycle",
    "fixed_mix_mechanism",
    "make_policy_mechanism",
    "mechanism_aware_mechanism",
    "random_feasible_mechanism",
    "run_mechanism_benchmark",
    "world_ids_for_claims",
]
