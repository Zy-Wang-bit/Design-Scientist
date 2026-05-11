from __future__ import annotations

from pathlib import Path

from design_scientist.table_io import (
    find_unsafe_identifier_values,
    read_identifier_safe_csv,
)


def test_identifier_safe_csv_preserves_canonical_onee62(tmp_path: Path) -> None:
    path = tmp_path / "systems.csv"
    path.write_text('system,variant_id\n"1E62","00123"\n', encoding="utf-8")

    result = read_identifier_safe_csv(path)

    assert result.frame.loc[0, "system"] == "1E62"
    assert result.frame.loc[0, "variant_id"] == "00123"
    assert result.warnings == []


def test_identifier_safe_csv_canonicalizes_unquoted_onee62_alias(tmp_path: Path) -> None:
    path = tmp_path / "systems.csv"
    path.write_text("system,variant_id\n1e+62,abc\n", encoding="utf-8")

    result = read_identifier_safe_csv(path)

    assert result.frame.loc[0, "system"] == "1E62"
    assert result.warnings
    assert result.warnings[0].column == "system"
    assert result.warnings[0].original_value == "1e+62"
    assert result.warnings[0].canonical_value == "1E62"


def test_validation_helper_detects_unknown_scientific_notation_identifier() -> None:
    issues = find_unsafe_identifier_values(
        [{"system": "2e+05", "module_id": "HD110H"}],
        artifact="candidate_pool.csv",
    )

    assert len(issues) == 1
    assert issues[0].code == "scientific_notation_identifier"
    assert issues[0].artifact == "candidate_pool.csv"
    assert issues[0].column == "system"
