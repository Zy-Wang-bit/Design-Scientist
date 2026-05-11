from __future__ import annotations

from pathlib import Path

from design_scientist.method_report import write_method_report
from test_framework_validation_v3 import _build_v3_framework_run


def test_v3_method_report_contains_mechanism_spec_kernel_sections(tmp_path: Path) -> None:
    project = _build_v3_framework_run(tmp_path)

    report_path = write_method_report(project, run_id="v3_unit")

    text = report_path.read_text(encoding="utf-8")
    assert "MechanismSpec" in text
    assert "MechanismSpec Kernel" in text
    assert "components" in text
    assert "Stress Tests" in text
    assert "Ablation" in text
    assert "Mechanism Library" in text
    assert "literature_kernel" in text
