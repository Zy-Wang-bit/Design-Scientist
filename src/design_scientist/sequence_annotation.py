"""Conservative sequence-index annotation for antibody candidate mutations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Iterable, Mapping


EDIT_TOKEN_PATTERN = re.compile(r"^(VH|VL):([A-Z])([1-9][0-9]*)([A-Z])$")
STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
REGION_NOTE = "region is a sequence-index heuristic from raw chain positions; not IMGT/Kabat numbering"
STANDARD_REGION_NOTE = "region and numbering from ANARCI/abnumber IMGT numbering"
__all__ = [
    "CandidateAnnotation",
    "MutationAnnotation",
    "MutationEdit",
    "annotate_candidate",
    "parse_edit_notation",
]

CDR_LIKE_SEQUENCE_WINDOWS: Mapping[str, tuple[tuple[int, int], ...]] = {
    "VH": ((24, 35), (50, 65), (89, 110)),
    "VL": ((24, 34), (50, 56), (89, 97)),
}

FLAG_NAMES = (
    "histidine_switch",
    "glycosylation_motif_created",
    "cysteine_created",
    "proline_created",
    "charge_change",
)
RISK_FLAGS = (
    "glycosylation_motif_created",
    "cysteine_created",
    "proline_created",
    "charge_change",
)
NEW_SITE_FLAGS = (
    "glycosylation_motif_created",
    "cysteine_created",
    "proline_created",
)


@dataclass(frozen=True)
class MutationEdit:
    chain: str
    one_based_position: int
    wildtype: str
    mutant: str
    raw_token: str


@dataclass(frozen=True)
class MutationAnnotation:
    chain: str
    one_based_position: int
    wildtype: str
    mutant: str
    region: str
    region_is_heuristic: bool
    region_note: str
    numbering_scheme: str
    numbering_position: str
    numbering_region: str
    numbering_source: str
    numbering_note: str
    is_new_site: bool
    flags: dict[str, bool]


@dataclass(frozen=True)
class CandidateAnnotation:
    candidate_id: str | None
    edit_notation: str
    mutations: list[MutationAnnotation]
    warnings: list[str]
    mutation_count: int
    flag_counts: dict[str, int]
    risk_count: int
    new_site_count: int


def parse_edit_notation(edit_notation: str, *, strict: bool = False) -> tuple[list[MutationEdit], list[str]]:
    """Parse semicolon-delimited VH/VL edit tokens like ``VH:S30H``."""

    edits: list[MutationEdit] = []
    warnings: list[str] = []
    for raw_token in edit_notation.split(";"):
        token = raw_token.strip()
        if not token:
            continue
        match = EDIT_TOKEN_PATTERN.fullmatch(token.upper())
        if match is None:
            _warn_or_raise(warnings, strict, f"invalid edit token {token!r}; expected VH:S30H or VL:K24H")
            continue

        chain, wildtype, position_text, mutant = match.groups()
        if wildtype not in STANDARD_AMINO_ACIDS or mutant not in STANDARD_AMINO_ACIDS:
            _warn_or_raise(warnings, strict, f"invalid amino acid in edit token {token!r}")
            continue

        edits.append(
            MutationEdit(
                chain=chain,
                one_based_position=int(position_text),
                wildtype=wildtype,
                mutant=mutant,
                raw_token=token,
            )
        )
    return edits, warnings


def annotate_candidate(
    heavy_sequence: str,
    light_sequence: str,
    edit_notation: str,
    candidate_id: str | None = None,
    new_site_notation: str | Iterable[str] | None = None,
    strict: bool = False,
) -> CandidateAnnotation:
    """Annotate candidate mutations from base heavy/light sequences and edit notation.

    When ANARCI/abnumber and HMMER are available, mutations are mapped to IMGT
    positions. Otherwise the function falls back to conservative sequence-index
    regions and records a warning rather than pretending to have standard
    antibody numbering.
    """

    sequences = {
        "VH": _normalise_sequence(heavy_sequence),
        "VL": _normalise_sequence(light_sequence),
    }
    edits, warnings = parse_edit_notation(edit_notation, strict=strict)
    new_sites = _parse_new_sites(new_site_notation, warnings, strict)
    numbering_maps, numbering_warnings = _standard_numbering_maps(
        sequences["VH"],
        sequences["VL"],
        "imgt",
    )
    warnings.extend(numbering_warnings)
    annotations: list[MutationAnnotation] = []

    for edit in edits:
        sequence = sequences[edit.chain]
        if edit.one_based_position > len(sequence):
            _warn_or_raise(
                warnings,
                strict,
                f"{edit.raw_token!r} position {edit.one_based_position} is outside {edit.chain} length {len(sequence)}",
            )
            continue

        zero_based_position = edit.one_based_position - 1
        observed = sequence[zero_based_position]
        if observed != edit.wildtype:
            _warn_or_raise(
                warnings,
                strict,
                f"{edit.raw_token!r} wildtype mismatch: sequence has {observed!r} at {edit.chain}:{edit.one_based_position}",
            )
            continue

        numbering = numbering_maps.get(edit.chain, {}).get(edit.one_based_position, {})
        numbering_position = str(numbering.get("position") or "")
        numbering_region = str(numbering.get("region") or "")
        numbering_source = str(numbering.get("source") or "")
        has_standard_numbering = bool(numbering_position and numbering_region)
        if has_standard_numbering:
            region = numbering_region
            region_is_heuristic = False
            region_note = STANDARD_REGION_NOTE
            numbering_note = "standard IMGT numbering available"
        else:
            region = _region_for(edit.chain, edit.one_based_position)
            region_is_heuristic = True
            region_note = REGION_NOTE
            numbering_source = "sequence_index_fallback"
            numbering_note = "standard IMGT numbering unavailable; using raw sequence index"

        annotations.append(
            MutationAnnotation(
                chain=edit.chain,
                one_based_position=edit.one_based_position,
                wildtype=edit.wildtype,
                mutant=edit.mutant,
                region=region,
                region_is_heuristic=region_is_heuristic,
                region_note=region_note,
                numbering_scheme="imgt" if has_standard_numbering else "",
                numbering_position=numbering_position,
                numbering_region=numbering_region,
                numbering_source=numbering_source,
                numbering_note=numbering_note,
                is_new_site=(edit.chain, edit.one_based_position) in new_sites,
                flags=_developability_flags(sequence, zero_based_position, edit.wildtype, edit.mutant),
            )
        )

    flag_counts = {
        flag: sum(1 for annotation in annotations if annotation.flags[flag])
        for flag in FLAG_NAMES
    }
    risk_count = sum(flag_counts[flag] for flag in RISK_FLAGS)
    new_site_count = sum(1 for annotation in annotations if annotation.is_new_site)
    return CandidateAnnotation(
        candidate_id=candidate_id,
        edit_notation=edit_notation,
        mutations=annotations,
        warnings=warnings,
        mutation_count=len(annotations),
        flag_counts=flag_counts,
        risk_count=risk_count,
        new_site_count=new_site_count,
    )


def _normalise_sequence(sequence: str) -> str:
    return "".join(sequence.split()).upper()


@lru_cache(maxsize=64)
def _standard_numbering_maps(
    heavy_sequence: str,
    light_sequence: str,
    scheme: str,
) -> tuple[dict[str, dict[int, dict[str, str]]], tuple[str, ...]]:
    warnings: list[str] = []
    try:
        from abnumber import Chain  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional package
        return (
            {"VH": {}, "VL": {}},
            (f"standard antibody numbering unavailable: abnumber import failed ({type(exc).__name__})",),
        )

    maps: dict[str, dict[int, dict[str, str]]] = {"VH": {}, "VL": {}}
    for chain, sequence in (("VH", heavy_sequence), ("VL", light_sequence)):
        if not sequence:
            continue
        try:
            numbered = Chain(sequence, scheme=scheme)
        except Exception as exc:  # pragma: no cover - depends on local hmmer/ANARCI state
            warnings.append(
                f"standard antibody numbering unavailable for {chain}: {type(exc).__name__}: {exc}"
            )
            continue
        for raw_index, (position, residue) in enumerate(numbered.positions.items(), start=1):
            maps[chain][raw_index] = {
                "position": str(position.format()),
                "region": str(position.get_region()),
                "source": "anarci_abnumber",
                "residue": str(residue),
                "scheme": scheme,
            }
    return maps, tuple(warnings)


def _warn_or_raise(warnings: list[str], strict: bool, message: str) -> None:
    if strict:
        raise ValueError(message)
    warnings.append(message)


def _parse_new_sites(
    new_site_notation: str | Iterable[str] | None,
    warnings: list[str],
    strict: bool,
) -> set[tuple[str, int]]:
    if new_site_notation is None:
        return set()
    if isinstance(new_site_notation, str):
        raw_tokens = [item.strip() for item in new_site_notation.replace(",", ";").split(";")]
    else:
        raw_tokens = [str(item).strip() for item in new_site_notation]
    sites: set[tuple[str, int]] = set()
    for raw_token in raw_tokens:
        if not raw_token:
            continue
        edits, token_warnings = parse_edit_notation(raw_token, strict=False)
        if token_warnings or not edits:
            _warn_or_raise(
                warnings,
                strict,
                f"invalid new-site token {raw_token!r}; expected VH:S30H or VL:K24H",
            )
            continue
        for edit in edits:
            sites.add((edit.chain, edit.one_based_position))
    return sites


def _region_for(chain: str, one_based_position: int) -> str:
    for start, end in CDR_LIKE_SEQUENCE_WINDOWS[chain]:
        if start <= one_based_position <= end:
            return "cdr_like"
    return "framework"


def _developability_flags(sequence: str, zero_based_position: int, wildtype: str, mutant: str) -> dict[str, bool]:
    mutated_sequence = sequence[:zero_based_position] + mutant + sequence[zero_based_position + 1 :]
    glycosylation_created = _glycosylation_motif_created(
        before=sequence,
        after=mutated_sequence,
        zero_based_position=zero_based_position,
    )
    return {
        "histidine_switch": wildtype != "H" and mutant == "H",
        "glycosylation_motif_created": glycosylation_created,
        "cysteine_created": wildtype != "C" and mutant == "C",
        "proline_created": wildtype != "P" and mutant == "P",
        "charge_change": _charge_class(wildtype) != _charge_class(mutant),
    }


def _glycosylation_motif_created(*, before: str, after: str, zero_based_position: int) -> bool:
    before_sites = _glycosylation_motif_starts(before)
    after_sites = _glycosylation_motif_starts(after)
    for start in after_sites - before_sites:
        if start <= zero_based_position <= start + 2:
            return True
    return False


def _glycosylation_motif_starts(sequence: str) -> set[int]:
    starts: set[int] = set()
    for index in range(0, max(0, len(sequence) - 2)):
        if sequence[index] == "N" and sequence[index + 1] != "P" and sequence[index + 2] in {"S", "T"}:
            starts.add(index)
    return starts


def _charge_class(residue: str) -> str:
    if residue in {"K", "R", "H"}:
        return "positive"
    if residue in {"D", "E"}:
        return "negative"
    return "neutral"
