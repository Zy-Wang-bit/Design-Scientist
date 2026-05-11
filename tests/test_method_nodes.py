from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

from design_scientist.method_nodes import execute_method_node, load_method_node


def test_execute_method_node_accepts_valid_node(tmp_path: Path) -> None:
    workspace = tmp_path / "nodes" / "valid_node"
    _write_node(
        workspace,
        """
        def run(workspace):
            import json
            from pathlib import Path

            root = Path(workspace)
            (root / "proposal.json").write_text(
                json.dumps({
                    "method_hypothesis": "repair sparse transfer evidence with interaction-aware acquisition",
                    "literature_basis": ["active learning under transfer shift"],
                    "literature_gap_ids": ["gap_transfer_tiny_data"],
                    "reused_components": ["batch acquisition scoring"],
                    "architecture_delta": "adds interaction-aware repair quotas before final ranking",
                    "new_mechanism_claim": "gap-conditioned repair can improve sparse transfer panels",
                    "algorithm_mechanism": "score candidates with interaction priors and repair quotas",
                    "expected_advantage": "higher feasible utility under epistatic transfer worlds",
                    "failure_modes": ["prior misspecification", "over-repairing sparse regions"],
                    "planned_ablation": "disable_gap_conditioned_repair",
                    "planned_ablations": ["remove_interaction_prior", "remove_repair_quota"],
                }),
                encoding="utf-8",
            )
            (root / "candidate_policy.json").write_text(
                json.dumps({"policy_id": "balanced", "policy_entrypoint": "select_batch"}),
                encoding="utf-8",
            )
            (root / "novelty_report.json").write_text(
                json.dumps({
                    "baseline_clone": False,
                    "selection_overlap_vs_baselines": 0.42,
                    "novelty_score": 0.58,
                }),
                encoding="utf-8",
            )
            (root / "benchmark_metrics.json").write_text(
                json.dumps({"score": 1.25, "baseline": "fixed_mix"}),
                encoding="utf-8",
            )
            (root / "validation_report.json").write_text(
                json.dumps({"valid": True, "findings": []}),
                encoding="utf-8",
            )
        """,
    )

    loaded = load_method_node(workspace)
    result = execute_method_node(workspace, guard_roots=[tmp_path])

    assert loaded.entrypoint == "run"
    assert result.valid
    assert result.executed
    assert result.missing_artifacts == []
    assert result.malformed_json == {}
    assert result.out_of_bounds_writes == []
    assert set(result.artifacts) == {
        "manifest.json",
        "proposal.json",
        "candidate_policy.json",
        "novelty_report.json",
        "benchmark_metrics.json",
        "validation_report.json",
    }


def test_execute_method_node_accepts_structured_architecture_delta_and_ablation(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "nodes" / "structured_proposal"
    _write_node(
        workspace,
        """
        def run(workspace):
            import json
            from pathlib import Path

            root = Path(workspace)
            (root / "proposal.json").write_text(
                json.dumps({
                    "method_hypothesis": "structured architecture delta is still a concrete method claim",
                    "literature_basis": ["active learning benchmarks"],
                    "literature_gap_ids": ["gap_structured_delta"],
                    "reused_components": ["helper scoring code"],
                    "architecture_delta": {
                        "delta_type": "new_acquisition_architecture",
                        "changes": ["adds evidence-tiered allocation before utility ranking"],
                    },
                    "new_mechanism_claim": "evidence-tiered allocation changes the acquisition architecture",
                    "algorithm_mechanism": ["score", "allocate", "repair"],
                    "expected_advantage": "less baseline cloning under sparse evidence",
                    "failure_modes": ["misweighted evidence tier"],
                    "planned_ablation": [
                        "disable evidence-tier allocation",
                        "disable repair quota",
                    ],
                }),
                encoding="utf-8",
            )
            (root / "candidate_policy.json").write_text(
                json.dumps({"policy_id": "structured", "policy_entrypoint": "select_batch"}),
                encoding="utf-8",
            )
            (root / "novelty_report.json").write_text(
                json.dumps({
                    "baseline_clone": False,
                    "selection_overlap_vs_baselines": 0.31,
                    "novelty_score": 0.69,
                }),
                encoding="utf-8",
            )
            (root / "benchmark_metrics.json").write_text(
                json.dumps({"score": 1.1}),
                encoding="utf-8",
            )
            (root / "validation_report.json").write_text(
                json.dumps({"valid": True, "findings": []}),
                encoding="utf-8",
            )
        """,
    )

    result = execute_method_node(workspace, guard_roots=[tmp_path])

    assert result.valid
    assert not result.errors


def test_execute_method_node_does_not_pass_workspace_to_optional_observed_run(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "nodes" / "optional_observed_run"
    _write_node(
        workspace,
        """
        def run(observed=None, candidates=None, budget=2, round_index=0, rng=None):
            import json
            from pathlib import Path

            if observed is not None:
                raise TypeError("observed should default to None when no workspace is requested")
            root = Path(".").resolve()
            (root / "proposal.json").write_text(
                json.dumps({
                    "method_hypothesis": "optional observed run writes artifacts without workspace injection",
                    "literature_basis": ["active learning benchmarks"],
                    "literature_gap_ids": ["gap_optional_observed_run"],
                    "reused_components": ["helper code"],
                    "architecture_delta": {"delta": "new optional-run architecture"},
                    "new_mechanism_claim": "optional run signatures are policy simulators",
                    "algorithm_mechanism": "write artifacts from cwd",
                    "expected_advantage": "compatible with Codex generated method nodes",
                    "failure_modes": ["cwd unavailable"],
                    "planned_ablation": ["disable optional-run defaults"],
                }),
                encoding="utf-8",
            )
            (root / "candidate_policy.json").write_text(
                json.dumps({"policy_id": "optional", "policy_entrypoint": "select_batch"}),
                encoding="utf-8",
            )
            (root / "novelty_report.json").write_text(
                json.dumps({"baseline_clone": False, "selection_overlap_vs_baselines": 0.25}),
                encoding="utf-8",
            )
            (root / "benchmark_metrics.json").write_text(
                json.dumps({"score": 1.0}),
                encoding="utf-8",
            )
            (root / "validation_report.json").write_text(
                json.dumps({"valid": True, "findings": []}),
                encoding="utf-8",
            )

        def select_batch(observed, candidates, budget, round_index, rng):
            return []
        """,
    )

    result = execute_method_node(workspace, guard_roots=[tmp_path])

    assert result.valid
    assert result.executed


def test_execute_method_node_accepts_module_colon_entrypoint(tmp_path: Path) -> None:
    workspace = tmp_path / "nodes" / "module_colon"
    _write_node(
        workspace,
        """
        def run(workspace):
            import json
            from pathlib import Path

            root = Path(workspace)
            (root / "proposal.json").write_text(
                json.dumps({
                    "method_hypothesis": "test hypothesis",
                    "literature_basis": [],
                    "literature_gap_ids": ["gap_test"],
                    "reused_components": [],
                    "architecture_delta": "test architecture delta",
                    "new_mechanism_claim": "test mechanism claim",
                    "algorithm_mechanism": "test mechanism",
                    "expected_advantage": "test advantage",
                    "failure_modes": [],
                    "planned_ablation": "test ablation",
                    "planned_ablations": [],
                }),
                encoding="utf-8",
            )
            (root / "candidate_policy.json").write_text(
                json.dumps({"policy_id": "mechanism_aware"}),
                encoding="utf-8",
            )
            (root / "novelty_report.json").write_text(
                json.dumps({"baseline_clone": False, "selection_overlap_vs_baselines": 0.25}),
                encoding="utf-8",
            )
            (root / "benchmark_metrics.json").write_text(
                json.dumps({"score": 1.0}),
                encoding="utf-8",
            )
            (root / "validation_report.json").write_text(
                json.dumps({"valid": True, "findings": []}),
                encoding="utf-8",
            )
        """,
        entrypoint="method:run",
    )

    result = execute_method_node(workspace, guard_roots=[tmp_path])

    assert result.valid
    assert result.executed


def test_execute_method_node_marks_missing_artifact_invalid(tmp_path: Path) -> None:
    workspace = tmp_path / "nodes" / "missing_artifact"
    _write_node(
        workspace,
        """
        def run(workspace):
            import json
            from pathlib import Path

            root = Path(workspace)
            (root / "proposal.json").write_text(
                json.dumps({
                    "method_hypothesis": "test hypothesis",
                    "literature_basis": [],
                    "literature_gap_ids": ["gap_test"],
                    "reused_components": [],
                    "architecture_delta": "test architecture delta",
                    "new_mechanism_claim": "test mechanism claim",
                    "algorithm_mechanism": "test mechanism",
                    "expected_advantage": "test advantage",
                    "failure_modes": [],
                    "planned_ablation": "test ablation",
                    "planned_ablations": [],
                }),
                encoding="utf-8",
            )
            (root / "candidate_policy.json").write_text(
                json.dumps({"policy_id": "missing_metrics"}),
                encoding="utf-8",
            )
            (root / "novelty_report.json").write_text(
                json.dumps({"baseline_clone": False, "selection_overlap_vs_baselines": 0.25}),
                encoding="utf-8",
            )
            (root / "validation_report.json").write_text(
                json.dumps({"valid": True, "findings": []}),
                encoding="utf-8",
            )
        """,
    )

    result = execute_method_node(workspace, guard_roots=[tmp_path])

    assert not result.valid
    assert result.executed
    assert result.missing_artifacts == ["benchmark_metrics.json"]
    assert result.exception is None


def test_execute_method_node_marks_malformed_json_invalid(tmp_path: Path) -> None:
    workspace = tmp_path / "nodes" / "malformed_json"
    _write_node(
        workspace,
        """
        def run(workspace):
            import json
            from pathlib import Path

            root = Path(workspace)
            (root / "proposal.json").write_text(
                json.dumps({
                    "method_hypothesis": "test hypothesis",
                    "literature_basis": [],
                    "literature_gap_ids": ["gap_test"],
                    "reused_components": [],
                    "architecture_delta": "test architecture delta",
                    "new_mechanism_claim": "test mechanism claim",
                    "algorithm_mechanism": "test mechanism",
                    "expected_advantage": "test advantage",
                    "failure_modes": [],
                    "planned_ablation": "test ablation",
                    "planned_ablations": [],
                }),
                encoding="utf-8",
            )
            (root / "candidate_policy.json").write_text("{not-json", encoding="utf-8")
            (root / "novelty_report.json").write_text(
                json.dumps({"baseline_clone": False, "selection_overlap_vs_baselines": 0.25}),
                encoding="utf-8",
            )
            (root / "benchmark_metrics.json").write_text(
                json.dumps({"score": 0.0}),
                encoding="utf-8",
            )
            (root / "validation_report.json").write_text(
                json.dumps({"valid": True, "findings": []}),
                encoding="utf-8",
            )
        """,
    )

    result = execute_method_node(workspace, guard_roots=[tmp_path])

    assert not result.valid
    assert result.executed
    assert "candidate_policy.json" in result.malformed_json
    assert result.exception is None


def test_execute_method_node_requires_proposal_and_novelty_report(tmp_path: Path) -> None:
    workspace = tmp_path / "nodes" / "missing_v2_artifacts"
    _write_node(
        workspace,
        """
        def run(workspace):
            import json
            from pathlib import Path

            root = Path(workspace)
            (root / "candidate_policy.json").write_text(
                json.dumps({"policy_id": "missing_v2"}),
                encoding="utf-8",
            )
            (root / "benchmark_metrics.json").write_text(
                json.dumps({"score": 0.0}),
                encoding="utf-8",
            )
            (root / "validation_report.json").write_text(
                json.dumps({"valid": True, "findings": []}),
                encoding="utf-8",
            )
        """,
    )

    result = execute_method_node(workspace, guard_roots=[tmp_path])

    assert not result.valid
    assert result.executed
    assert result.missing_artifacts == ["proposal.json", "novelty_report.json"]


def test_execute_method_node_rejects_incomplete_proposal(tmp_path: Path) -> None:
    workspace = tmp_path / "nodes" / "incomplete_proposal"
    _write_node(
        workspace,
        """
        def run(workspace):
            import json
            from pathlib import Path

            root = Path(workspace)
            (root / "proposal.json").write_text(
                json.dumps({"method_hypothesis": "too sparse"}),
                encoding="utf-8",
            )
            (root / "candidate_policy.json").write_text(
                json.dumps({"policy_id": "candidate", "policy_entrypoint": "select_batch"}),
                encoding="utf-8",
            )
            (root / "novelty_report.json").write_text(
                json.dumps({"baseline_clone": False, "selection_overlap_vs_baselines": 0.30}),
                encoding="utf-8",
            )
            (root / "benchmark_metrics.json").write_text(
                json.dumps({"score": 0.0}),
                encoding="utf-8",
            )
            (root / "validation_report.json").write_text(
                json.dumps({"valid": True, "findings": []}),
                encoding="utf-8",
            )
        """,
    )

    result = execute_method_node(workspace, guard_roots=[tmp_path])

    assert not result.valid
    assert "proposal.json missing required fields" in "; ".join(result.errors)


def test_execute_method_node_rejects_missing_literature_gap_provenance(tmp_path: Path) -> None:
    workspace = tmp_path / "nodes" / "missing_gap_provenance"
    _write_node(
        workspace,
        """
        def run(workspace):
            import json
            from pathlib import Path

            root = Path(workspace)
            (root / "proposal.json").write_text(
                json.dumps({
                    "method_hypothesis": "test hypothesis",
                    "literature_basis": ["active learning"],
                    "reused_components": ["shared scoring helper"],
                    "architecture_delta": "adds a gap-conditioned allocation layer",
                    "new_mechanism_claim": "allocation by explicit gap improves search",
                    "algorithm_mechanism": "test mechanism",
                    "expected_advantage": "test advantage",
                    "failure_modes": [],
                    "planned_ablation": "remove_gap_conditioning",
                    "planned_ablations": ["remove_gap_conditioning"],
                }),
                encoding="utf-8",
            )
            (root / "candidate_policy.json").write_text(
                json.dumps({"policy_id": "missing_gap", "policy_entrypoint": "select_batch"}),
                encoding="utf-8",
            )
            (root / "novelty_report.json").write_text(
                json.dumps({"baseline_clone": False, "selection_overlap_vs_baselines": 0.30}),
                encoding="utf-8",
            )
            (root / "benchmark_metrics.json").write_text(
                json.dumps({"score": 0.0}),
                encoding="utf-8",
            )
            (root / "validation_report.json").write_text(
                json.dumps({"valid": True, "findings": []}),
                encoding="utf-8",
            )
        """,
    )

    result = execute_method_node(workspace, guard_roots=[tmp_path])

    assert not result.valid
    assert "literature_gap_ids" in "; ".join(result.errors)


def test_execute_method_node_allows_helper_reuse_with_architecture_delta(tmp_path: Path) -> None:
    workspace = tmp_path / "nodes" / "helper_reuse_delta"
    _write_node(
        workspace,
        """
        def run(workspace):
            import json
            from pathlib import Path

            root = Path(workspace)
            (root / "proposal.json").write_text(
                json.dumps({
                    "method_hypothesis": "gap allocation can reuse policy helpers without cloning ranking",
                    "literature_basis": ["active learning under wet-lab constraints"],
                    "literature_gap_ids": ["gap_constraints_tiny_data"],
                    "reused_components": ["design_scientist.policies._candidate_pool"],
                    "architecture_delta": "adds a new gap-prior allocation stage before helper scoring",
                    "new_mechanism_claim": "the gap-prior stage changes which candidates enter scoring",
                    "algorithm_mechanism": "preallocate budget to literature gaps, then score candidates",
                    "expected_advantage": "keeps reusable parsing while changing selection architecture",
                    "failure_modes": ["gap prior can over-allocate weak regions"],
                    "planned_ablation": "disable_gap_prior_allocation",
                    "planned_ablations": ["disable_gap_prior_allocation"],
                }),
                encoding="utf-8",
            )
            (root / "candidate_policy.json").write_text(
                json.dumps({"policy_id": "helper_reuse", "policy_entrypoint": "select_batch"}),
                encoding="utf-8",
            )
            (root / "novelty_report.json").write_text(
                json.dumps({
                    "baseline_clone": False,
                    "selection_overlap_vs_baselines": 0.42,
                    "reused_helper_code": True,
                }),
                encoding="utf-8",
            )
            (root / "benchmark_metrics.json").write_text(
                json.dumps({"score": 0.0}),
                encoding="utf-8",
            )
            (root / "validation_report.json").write_text(
                json.dumps({"valid": True, "findings": []}),
                encoding="utf-8",
            )
        """,
    )

    result = execute_method_node(workspace, guard_roots=[tmp_path])

    assert result.valid


def test_execute_method_node_rejects_tail_swap_baseline_wrapper(tmp_path: Path) -> None:
    workspace = tmp_path / "nodes" / "tail_swap_wrapper"
    _write_node(
        workspace,
        """
        def run(workspace):
            import json
            from pathlib import Path

            root = Path(workspace)
            (root / "proposal.json").write_text(
                json.dumps({
                    "method_hypothesis": "test hypothesis",
                    "literature_basis": ["active learning"],
                    "literature_gap_ids": ["gap_clone_guard"],
                    "reused_components": ["mechanism_aware"],
                    "architecture_delta": "only swaps the final baseline-selected item",
                    "new_mechanism_claim": "tail swaps are enough",
                    "algorithm_mechanism": "wrap a baseline and replace the tail item",
                    "expected_advantage": "test advantage",
                    "failure_modes": [],
                    "planned_ablation": "disable_tail_swap",
                    "planned_ablations": ["disable_tail_swap"],
                }),
                encoding="utf-8",
            )
            (root / "candidate_policy.json").write_text(
                json.dumps({"policy_id": "tail_swap", "policy_entrypoint": "select_batch"}),
                encoding="utf-8",
            )
            (root / "novelty_report.json").write_text(
                json.dumps({
                    "baseline_clone": False,
                    "selection_overlap_vs_baselines": 0.40,
                    "tail_swap_baseline_wrapper": True,
                }),
                encoding="utf-8",
            )
            (root / "benchmark_metrics.json").write_text(
                json.dumps({"score": 0.0}),
                encoding="utf-8",
            )
            (root / "validation_report.json").write_text(
                json.dumps({"valid": True, "findings": []}),
                encoding="utf-8",
            )
        """,
    )

    result = execute_method_node(workspace, guard_roots=[tmp_path])

    assert not result.valid
    assert "tail-swap baseline wrapper" in "; ".join(result.errors)


def test_execute_method_node_rejects_baseline_clone_novelty_report(tmp_path: Path) -> None:
    workspace = tmp_path / "nodes" / "baseline_clone"
    _write_node(
        workspace,
        """
        def run(workspace):
            import json
            from pathlib import Path

            root = Path(workspace)
            (root / "proposal.json").write_text(
                json.dumps({
                    "method_hypothesis": "test hypothesis",
                    "literature_basis": [],
                    "literature_gap_ids": ["gap_test"],
                    "reused_components": [],
                    "architecture_delta": "test architecture delta",
                    "new_mechanism_claim": "test mechanism claim",
                    "algorithm_mechanism": "test mechanism",
                    "expected_advantage": "test advantage",
                    "failure_modes": [],
                    "planned_ablation": "test ablation",
                    "planned_ablations": [],
                }),
                encoding="utf-8",
            )
            (root / "candidate_policy.json").write_text(
                json.dumps({"policy_id": "clone", "policy_entrypoint": "select_batch"}),
                encoding="utf-8",
            )
            (root / "novelty_report.json").write_text(
                json.dumps({"baseline_clone": False, "selection_overlap_vs_baselines": 0.91}),
                encoding="utf-8",
            )
            (root / "benchmark_metrics.json").write_text(
                json.dumps({"score": 0.0}),
                encoding="utf-8",
            )
            (root / "validation_report.json").write_text(
                json.dumps({"valid": True, "findings": []}),
                encoding="utf-8",
            )
        """,
    )

    result = execute_method_node(workspace, guard_roots=[tmp_path])

    assert not result.valid
    assert "baseline clone guard" in "; ".join(result.errors)


def test_execute_method_node_marks_runtime_exception_invalid(tmp_path: Path) -> None:
    workspace = tmp_path / "nodes" / "raises"
    _write_node(
        workspace,
        """
        def run(workspace):
            raise RuntimeError("node boom")
        """,
    )

    result = execute_method_node(workspace, guard_roots=[tmp_path])

    assert not result.valid
    assert not result.executed
    assert result.exception is not None
    assert "node boom" in result.exception


def test_execute_method_node_marks_out_of_bounds_write_invalid(tmp_path: Path) -> None:
    workspace = tmp_path / "nodes" / "writes_outside"
    _write_node(
        workspace,
        """
        def run(workspace):
            import json
            from pathlib import Path

            root = Path(workspace)
            (root / "proposal.json").write_text(
                json.dumps({
                    "method_hypothesis": "test hypothesis",
                    "literature_basis": [],
                    "literature_gap_ids": ["gap_test"],
                    "reused_components": [],
                    "architecture_delta": "test architecture delta",
                    "new_mechanism_claim": "test mechanism claim",
                    "algorithm_mechanism": "test mechanism",
                    "expected_advantage": "test advantage",
                    "failure_modes": [],
                    "planned_ablation": "test ablation",
                    "planned_ablations": [],
                }),
                encoding="utf-8",
            )
            (root / "candidate_policy.json").write_text(
                json.dumps({"policy_id": "leaky"}),
                encoding="utf-8",
            )
            (root / "novelty_report.json").write_text(
                json.dumps({"baseline_clone": False, "selection_overlap_vs_baselines": 0.25}),
                encoding="utf-8",
            )
            (root / "benchmark_metrics.json").write_text(
                json.dumps({"score": 1.0}),
                encoding="utf-8",
            )
            (root / "validation_report.json").write_text(
                json.dumps({"valid": True, "findings": []}),
                encoding="utf-8",
            )
            (root.parent / "outside.txt").write_text("leak", encoding="utf-8")
        """,
    )

    result = execute_method_node(workspace, guard_roots=[tmp_path])

    assert not result.valid
    assert result.executed
    assert result.exception is None
    assert any(path.endswith("outside.txt") for path in result.out_of_bounds_writes)


def _write_node(workspace: Path, method_source: str, entrypoint: str = "run") -> None:
    workspace.mkdir(parents=True)
    (workspace / "manifest.json").write_text(
        json.dumps({"name": workspace.name, "entrypoint": entrypoint}),
        encoding="utf-8",
    )
    (workspace / "method.py").write_text(dedent(method_source).lstrip(), encoding="utf-8")
