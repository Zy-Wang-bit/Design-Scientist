from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_renderer():
    script = Path(__file__).resolve().parents[1] / "scripts" / "render_bioinformatics_paper.py"
    spec = importlib.util.spec_from_file_location("render_bioinformatics_paper", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_bioinformatics_structured_abstract_uses_external_ph_switch_table(
    tmp_path: Path,
    monkeypatch,
) -> None:
    renderer = _load_renderer()
    paper_dir = tmp_path / "paper"
    tables_dir = paper_dir / "tables"
    tables_dir.mkdir(parents=True)
    (tables_dir / "external_ph_switch_benchmark_summary.csv").write_text(
        "\n".join(
            [
                "method,dataset_id,dataset_count,mean_best_selected_normalized_log_ratio",
                "leave_one_study_transition_calibration,overall,5,0.845875",
                "histidine_count,overall,5,0.709451",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (paper_dir / "algorithm_results_summary.json").write_text(
        json.dumps(
            {
                "external_ph_switch_benchmark": {
                    "variant_replay_summary": {
                        "variant_count": 35,
                        "predicted_vs_observed_normalized_log_ratio_pearson": 0.377779,
                        "method_summaries": [
                            {
                                "method": "transition_context_prior",
                                "top_tertile_enrichment_at_top_third_by_score": 1.296296,
                            }
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DS_ALLOW_SUBMISSION_PLACEHOLDERS", "1")
    monkeypatch.setenv("DS_PAPER_ARCHIVE_URL", "https://archive.softwareheritage.org/swh:1:snp:test")

    abstract = renderer._structured_bioinformatics_abstract(paper_dir)

    assert "5-table public pH-switch replay" in abstract
    assert "0.846 versus 0.709" in abstract
    assert "leave-one-variant replay over 35 variants gave r=0.378 and context-prior enrichment 1.296" in abstract
    assert "Public pH-switch replays, project-data masking, and computational stress tests provide bounded, auditable checks" in abstract
    renderer._validate_structured_bioinformatics_abstract(abstract)


def test_bioinformatics_source_package_excludes_literature_caches(tmp_path: Path) -> None:
    renderer = _load_renderer()
    project_dir = tmp_path / "project"
    cache_raw = project_dir / "framework" / "cache" / "literature_raw" / "query" / "raw.json"
    cache_fulltext = project_dir / "framework" / "cache" / "literature_fulltext" / "paper.txt"
    regular_framework = project_dir / "framework" / "paper_cards.json"
    cache_raw.parent.mkdir(parents=True)
    cache_fulltext.parent.mkdir(parents=True)
    regular_framework.parent.mkdir(parents=True, exist_ok=True)
    cache_raw.write_text("{}", encoding="utf-8")
    cache_fulltext.write_text("full text", encoding="utf-8")
    regular_framework.write_text("[]", encoding="utf-8")

    assert renderer._is_submission_cache_file(cache_raw, project_dir)
    assert renderer._is_submission_cache_file(cache_fulltext, project_dir)
    assert not renderer._is_submission_cache_file(regular_framework, project_dir)
