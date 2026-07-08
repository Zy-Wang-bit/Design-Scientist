from __future__ import annotations

import importlib.util
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
    monkeypatch.setenv("DS_ALLOW_SUBMISSION_PLACEHOLDERS", "1")
    monkeypatch.setenv("DS_PAPER_ARCHIVE_URL", "https://archive.softwareheritage.org/swh:1:snp:test")

    abstract = renderer._structured_bioinformatics_abstract(paper_dir)

    assert "5-table public pH-switch replay" in abstract
    assert "0.846 versus 0.709" in abstract
    renderer._validate_structured_bioinformatics_abstract(abstract)
