"""Identifier-safe CSV IO helpers."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


IDENTIFIER_COLUMNS = (
    "system",
    "target_system",
    "variant_id",
    "base_variant",
    "module_id",
    "paper_id",
    "external_id",
)

ONEE62_CANONICAL = "1E62"
ONEE62_ALIAS_NORMALIZED = {"1E+62", "1.0E+62", "1.00E+62"}

_SCIENTIFIC_NOTATION_RE = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)[eE][+-]?\d+$"
)
_STALE_ONEE62_RE = re.compile(
    r"(?<![A-Za-z0-9_.+-])(?:1e\+62|1E\+62|1\.0[eE]\+62|1\.00[eE]\+62)(?![A-Za-z0-9_.+-])"
)


@dataclass(frozen=True)
class IdentifierCanonicalizationWarning:
    artifact: str
    row_number: int
    column: str
    original_value: str
    canonical_value: str


@dataclass(frozen=True)
class UnsafeIdentifierIssue:
    code: str
    artifact: str
    path: str
    column: str | None = None
    row_number: int | None = None


@dataclass(frozen=True)
class IdentifierSafeCSVResult:
    frame: pd.DataFrame
    warnings: list[IdentifierCanonicalizationWarning]

    def records(self) -> list[dict[str, str]]:
        return self.frame.to_dict(orient="records")


def read_identifier_safe_csv(
    path: str | Path,
    *,
    identifier_columns: Iterable[str] = IDENTIFIER_COLUMNS,
) -> IdentifierSafeCSVResult:
    """Read a CSV while preserving identifier columns as strings."""
    source = Path(path)
    identifiers = set(identifier_columns)
    warnings: list[IdentifierCanonicalizationWarning] = []
    rows: list[dict[str, str]] = []

    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for row_number, row in enumerate(reader, start=1):
            cleaned: dict[str, str] = {}
            for column in fieldnames:
                value = row.get(column)
                text = "" if value is None else str(value)
                if column in identifiers:
                    canonical = canonicalize_identifier_value(text)
                    if canonical != text:
                        warnings.append(
                            IdentifierCanonicalizationWarning(
                                artifact=str(source),
                                row_number=row_number,
                                column=column,
                                original_value=text,
                                canonical_value=canonical,
                            )
                        )
                    text = canonical
                cleaned[column] = text
            rows.append(cleaned)

    return IdentifierSafeCSVResult(
        frame=pd.DataFrame(rows, columns=fieldnames, dtype=object),
        warnings=warnings,
    )


def canonicalize_identifier_value(value: str) -> str:
    """Canonicalize known identifier aliases without guessing unknown IDs."""
    stripped = value.strip()
    if stripped.upper() in ONEE62_ALIAS_NORMALIZED:
        return ONEE62_CANONICAL
    return value


def find_unsafe_identifier_values(
    data: Any,
    *,
    artifact: str,
    identifier_columns: Iterable[str] = IDENTIFIER_COLUMNS,
) -> list[UnsafeIdentifierIssue]:
    """Find stale 1E62 aliases and unknown scientific notation in identifiers."""
    identifiers = set(identifier_columns)
    issues: list[UnsafeIdentifierIssue] = []

    def add_stale(path: str, column: str | None, row_number: int | None) -> None:
        issues.append(
            UnsafeIdentifierIssue(
                code="stale_onee62_identifier",
                artifact=artifact,
                path=path,
                column=column,
                row_number=row_number,
            )
        )

    def add_scientific(path: str, column: str | None, row_number: int | None) -> None:
        issues.append(
            UnsafeIdentifierIssue(
                code="scientific_notation_identifier",
                artifact=artifact,
                path=path,
                column=column,
                row_number=row_number,
            )
        )

    def check_text(text: str, path: str, column: str | None, row_number: int | None) -> None:
        stripped = text.strip()
        if _STALE_ONEE62_RE.search(stripped):
            add_stale(path, column, row_number)
            return
        if column in identifiers and _is_unknown_scientific_identifier(stripped):
            add_scientific(path, column, row_number)

    def visit(value: Any, path: str, column: str | None = None, row_number: int | None = None) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                key_text = str(key)
                key_path = f"{path}.{key_text}" if path else key_text
                check_text(key_text, key_path, None, row_number)
                child_column = key_text if key_text in identifiers else None
                visit(child, key_path, child_column, row_number)
            return

        if isinstance(value, list):
            for index, item in enumerate(value):
                item_path = f"{path}[{index}]"
                item_row = index + 1 if isinstance(item, dict) else row_number
                visit(item, item_path, None, item_row)
            return

        if isinstance(value, str):
            check_text(value, path, column, row_number)

    visit(data, "")
    return issues


def _is_unknown_scientific_identifier(value: str) -> bool:
    if value == ONEE62_CANONICAL:
        return False
    return bool(_SCIENTIFIC_NOTATION_RE.fullmatch(value))


__all__ = [
    "IDENTIFIER_COLUMNS",
    "IdentifierCanonicalizationWarning",
    "IdentifierSafeCSVResult",
    "ONEE62_CANONICAL",
    "UnsafeIdentifierIssue",
    "canonicalize_identifier_value",
    "find_unsafe_identifier_values",
    "read_identifier_safe_csv",
]
