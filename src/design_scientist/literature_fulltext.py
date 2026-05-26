"""Build an offline-friendly fulltext corpus from literature paper cards.

This module intentionally avoids paid fulltext scraping and does not query
Google Scholar. In offline fixture mode it reads local XML, HTML, PDF, or text
files and degrades to metadata-only records when no readable open fulltext is
available.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup, FeatureNotFound, XMLParsedAsHTMLWarning

from design_scientist.io import ensure_dir, read_json, write_json


VERSION = "literature_fulltext_v3"
FIXTURE_ENV_VAR = "DESIGN_SCIENTIST_FULLTEXT_FIXTURES"
FULLTEXT_EXTENSIONS = (".xml", ".nxml", ".html", ".htm", ".pdf", ".txt", ".text")
HTML_EXTENSIONS = (".html", ".htm")
XML_EXTENSIONS = (".xml", ".nxml")
PDF_EXTENSIONS = (".pdf",)
TEXT_EXTENSIONS = (".txt", ".text")
DEFAULT_CHUNK_SIZE = 4000


@dataclass(frozen=True)
class FulltextResult:
    status: str
    text: str
    text_source: str
    reason: str | None = None
    error_type: str | None = None
    fixture_path: Path | None = None
    cache_path: Path | None = None


def build_literature_corpus(
    project_dir: str | Path,
    offline_fixtures: bool = False,
    max_chars_per_paper: int = 20000,
    client: Any | None = None,
) -> Path:
    """Build ``framework/literature_corpus.jsonl`` from paper cards.

    The input is ``framework/paper_cards.json``. Each JSONL record includes
    paper metadata text even when fulltext is unavailable. In offline fixture
    mode, local fixture files are matched by paper id, DOI, source id, and
    external ids using filesystem-safe filenames.
    """

    if max_chars_per_paper <= 0:
        raise ValueError("max_chars_per_paper must be positive")

    root = Path(project_dir).expanduser().resolve()
    framework_dir = ensure_dir(root / "framework")
    paper_cards_path = framework_dir / "paper_cards.json"
    cards = read_json(paper_cards_path)
    if not isinstance(cards, list):
        raise ValueError(f"Expected list in {paper_cards_path}")

    cache_dir = ensure_dir(framework_dir / "cache" / "literature_fulltext")
    fixture_dir = _fixture_dir() if offline_fixtures else None
    corpus_path = framework_dir / "literature_corpus.jsonl"
    trace_path = framework_dir / "literature_reading_trace.json"

    records: list[dict[str, Any]] = []
    trace_papers: list[dict[str, Any]] = []
    for index, raw_card in enumerate(cards, start=1):
        card = raw_card if isinstance(raw_card, dict) else {}
        paper_id = _clean_str(card.get("paper_id")) or f"paper:{index}"
        abstract_or_summary = _abstract_or_summary(card)
        metadata_text = _metadata_text(card, paper_id=paper_id, abstract_or_summary=abstract_or_summary)
        result = _read_fulltext(card, fixture_dir=fixture_dir, cache_dir=cache_dir, client=client)
        text = _merge_text(metadata_text, result.text, max_chars=max_chars_per_paper)
        chunks = _chunk_text(text, paper_id=paper_id, text_source=result.text_source)

        provenance = _drop_empty(
            {
                "paper_card_path": _rel(paper_cards_path, root),
                "source": _clean_str(card.get("source")),
                "source_id": _clean_str(card.get("source_id")),
                "doi": _clean_str(card.get("doi")),
                "url": _clean_str(card.get("url")),
                "offline_fixtures": bool(offline_fixtures),
                "fixture_path": _rel(result.fixture_path, root) if result.fixture_path else None,
                "raw_cache_path": _rel(result.cache_path, root) if result.cache_path else None,
                "fulltext_reason": result.reason,
            }
        )
        record = {
            "paper_id": paper_id,
            "source": _clean_str(card.get("source")) or "unknown",
            "title": _clean_str(card.get("title")) or _clean_str(card.get("citation")) or paper_id,
            "abstract_or_summary": abstract_or_summary,
            "fulltext_status": result.status,
            "text": text,
            "text_source": result.text_source,
            "chunks": chunks,
            "provenance": provenance,
        }
        records.append(record)
        trace_papers.append(
            _drop_empty(
                {
                    "paper_id": paper_id,
                    "source": record["source"],
                    "status": result.status,
                    "text_source": result.text_source,
                    "reason": result.reason,
                    "error_type": result.error_type,
                    "fixture_path": _rel(result.fixture_path, root) if result.fixture_path else None,
                    "raw_cache_path": _rel(result.cache_path, root) if result.cache_path else None,
                    "char_count": len(text),
                    "chunk_count": len(chunks),
                }
            )
        )

    _write_jsonl(corpus_path, records)
    write_json(
        trace_path,
        {
            "version": VERSION,
            "offline_fixtures": bool(offline_fixtures),
            "max_chars_per_paper": max_chars_per_paper,
            "corpus_path": _rel(corpus_path, root),
            "cache_dir": _rel(cache_dir, root),
            "papers": trace_papers,
            "summary": _trace_summary(trace_papers),
        },
    )
    return corpus_path


def _read_fulltext(
    card: dict[str, Any],
    *,
    fixture_dir: Path | None,
    cache_dir: Path,
    client: Any | None = None,
) -> FulltextResult:
    if fixture_dir is None:
        return _read_live_open_fulltext(card, cache_dir=cache_dir, client=client)

    fixture_path = _find_fixture(card, fixture_dir)
    if fixture_path is None:
        return FulltextResult(
            status="metadata_only",
            text="",
            text_source="metadata",
            reason="fixture_not_found",
        )

    cache_path = cache_dir / fixture_path.name
    shutil.copyfile(fixture_path, cache_path)
    suffix = fixture_path.suffix.lower()
    try:
        if suffix in XML_EXTENSIONS:
            text = _extract_readable_html_or_xml(fixture_path, prefer_xml=True)
            text_source = "fixture_xml"
        elif suffix in HTML_EXTENSIONS:
            text = _extract_readable_html_or_xml(fixture_path, prefer_xml=False)
            text_source = "fixture_html"
        elif suffix in PDF_EXTENSIONS:
            text = _extract_pdf_text(fixture_path)
            text_source = "fixture_pdf"
        elif suffix in TEXT_EXTENSIONS:
            text = _normalize_text(fixture_path.read_text(encoding="utf-8", errors="replace"))
            text_source = "fixture_text"
        else:
            text = ""
            text_source = "fixture_unknown"
    except Exception as exc:
        reason = "pdf_parse_failed" if suffix in PDF_EXTENSIONS else "fulltext_parse_failed"
        return FulltextResult(
            status="metadata_only",
            text="",
            text_source="metadata",
            reason=reason,
            error_type=type(exc).__name__,
            fixture_path=fixture_path,
            cache_path=cache_path,
        )

    if not text:
        return FulltextResult(
            status="metadata_only",
            text="",
            text_source="metadata",
            reason="fulltext_empty",
            fixture_path=fixture_path,
            cache_path=cache_path,
        )
    return FulltextResult(
        status="open_fulltext",
        text=text,
        text_source=text_source,
        fixture_path=fixture_path,
        cache_path=cache_path,
    )


def _read_live_open_fulltext(
    card: dict[str, Any],
    *,
    cache_dir: Path,
    client: Any | None,
) -> FulltextResult:
    """Resolve allowed open fulltext sources without paid scraping."""

    source = _clean_str(card.get("source")).lower()
    if source == "biorxiv":
        resolvers = (_resolve_biorxiv_pdf, _resolve_pmc_oa, _resolve_arxiv_pdf)
    elif source == "arxiv":
        resolvers = (_resolve_arxiv_pdf, _resolve_pmc_oa, _resolve_biorxiv_pdf)
    else:
        resolvers = (_resolve_pmc_oa, _resolve_arxiv_pdf, _resolve_biorxiv_pdf)
    first_error_type: str | None = None
    for resolver in resolvers:
        try:
            result = resolver(card, cache_dir=cache_dir, client=client)
        except Exception as exc:  # pragma: no cover - defensive degradation path.
            first_error_type = first_error_type or type(exc).__name__
            continue
        if result is not None:
            return result
    return FulltextResult(
        status="metadata_only",
        text="",
        text_source="metadata",
        reason="open_fulltext_not_found",
        error_type=first_error_type,
    )


def _resolve_pmc_oa(card: dict[str, Any], *, cache_dir: Path, client: Any | None) -> FulltextResult | None:
    pmcid = _pmcid_from_card(card)
    doi = _clean_str(card.get("doi"))
    pmid = _clean_str(card.get("pmid"))
    if not pmcid and (doi or pmid):
        response = _http_get(
            client,
            "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/",
            params={"ids": doi or pmid, "format": "json"},
        )
        response.raise_for_status()
        records = response.json().get("records", []) if hasattr(response, "json") else []
        if records and isinstance(records[0], dict):
            pmcid = _clean_str(records[0].get("pmcid"))
    if not pmcid:
        return None

    pmc_digits = re.sub(r"^PMC", "", pmcid, flags=re.I)
    cache_path = cache_dir / f"pmc_PMC{pmc_digits}_oa.xml"
    if cache_path.exists():
        text = _extract_readable_html_or_xml(cache_path, prefer_xml=True)
        if text:
            return FulltextResult(
                status="open_fulltext",
                text=text,
                text_source="pmc_oa_xml",
                cache_path=cache_path,
            )
    response = _http_get(
        client,
        "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi",
        params={
            "verb": "GetRecord",
            "metadataPrefix": "pmc",
            "identifier": f"oai:pubmedcentral.nih.gov:{pmc_digits}",
        },
    )
    response.raise_for_status()
    _write_bytes_or_text(cache_path, response)
    text = _extract_readable_html_or_xml(cache_path, prefer_xml=True)
    if not text:
        return None
    return FulltextResult(
        status="open_fulltext",
        text=text,
        text_source="pmc_oa_xml",
        cache_path=cache_path,
    )


def _resolve_arxiv_pdf(card: dict[str, Any], *, cache_dir: Path, client: Any | None) -> FulltextResult | None:
    arxiv_id = _arxiv_id_from_card(card)
    if not arxiv_id:
        return None
    cache_path = cache_dir / f"arxiv_{_safe_filename(arxiv_id)}_pdf.pdf"
    if cache_path.exists():
        text = _extract_pdf_text(cache_path)
        if text:
            return FulltextResult(
                status="open_fulltext",
                text=text,
                text_source="arxiv_pdf",
                cache_path=cache_path,
            )
    response = _http_get(client, f"https://arxiv.org/pdf/{arxiv_id}.pdf")
    response.raise_for_status()
    _write_bytes_or_text(cache_path, response)
    text = _extract_pdf_text(cache_path)
    if not text:
        return None
    return FulltextResult(
        status="open_fulltext",
        text=text,
        text_source="arxiv_pdf",
        cache_path=cache_path,
    )


def _resolve_biorxiv_pdf(card: dict[str, Any], *, cache_dir: Path, client: Any | None) -> FulltextResult | None:
    source = _clean_str(card.get("source")).lower()
    doi = _clean_str(card.get("doi")) or _clean_str(card.get("source_id"))
    if source != "biorxiv" or not doi:
        return None
    cache_path = cache_dir / f"biorxiv_{_safe_filename(_strip_doi_prefix(doi))}_pdf.pdf"
    if cache_path.exists():
        text = _extract_pdf_text(cache_path)
        if text:
            return FulltextResult(
                status="open_fulltext",
                text=text,
                text_source="biorxiv_pdf",
                cache_path=cache_path,
            )
    landing_url = _clean_str(card.get("url")) or f"https://doi.org/{doi}"
    landing_response = _http_get(client, landing_url)
    landing_response.raise_for_status()
    pdf_url = _find_pdf_url(landing_response.text, base_url=landing_url)
    if not pdf_url:
        return None
    pdf_response = _http_get(client, pdf_url)
    pdf_response.raise_for_status()
    _write_bytes_or_text(cache_path, pdf_response)
    text = _extract_pdf_text(cache_path)
    if not text:
        return None
    return FulltextResult(
        status="open_fulltext",
        text=text,
        text_source="biorxiv_pdf",
        cache_path=cache_path,
    )


def _http_get(client: Any | None, url: str, *, params: dict[str, str] | None = None) -> Any:
    headers = {"User-Agent": os.getenv("DESIGN_SCIENTIST_USER_AGENT", "design-scientist/0.1")}
    if client is not None:
        return client.get(url, params=params, headers=headers)
    with httpx.Client(timeout=30.0, follow_redirects=True, headers=headers) as http_client:
        return http_client.get(url, params=params)


def _write_bytes_or_text(path: Path, response: Any) -> None:
    ensure_dir(path.parent)
    content = getattr(response, "content", None)
    if content:
        path.write_bytes(content)
        return
    path.write_text(getattr(response, "text", ""), encoding="utf-8")


def _pmcid_from_card(card: dict[str, Any]) -> str:
    for key in ("pmcid", "pmc_id"):
        value = _clean_str(card.get(key))
        if value:
            return value
    external_ids = card.get("external_ids")
    if isinstance(external_ids, dict):
        for key in ("PMCID", "pmcid", "PubMedCentral"):
            value = _clean_str(external_ids.get(key))
            if value:
                return value
    return ""


def _arxiv_id_from_card(card: dict[str, Any]) -> str:
    source = _clean_str(card.get("source")).lower()
    candidates = []
    if source == "arxiv":
        candidates.extend([card.get("source_id"), card.get("paper_id"), card.get("url")])
    external_ids = card.get("external_ids")
    if isinstance(external_ids, dict):
        candidates.extend([external_ids.get("arxiv"), external_ids.get("ArXiv")])
    for candidate in candidates:
        text = _clean_str(candidate)
        if not text:
            continue
        text = text.rstrip("/").split("/")[-1]
        text = re.sub(r"^arxiv:", "", text, flags=re.I)
        text = re.sub(r"\.pdf$", "", text, flags=re.I)
        if re.search(r"\d{4}\.\d{4,5}", text) or re.search(r"[a-z-]+/\d{7}", text, flags=re.I):
            return text
    return ""


def _find_pdf_url(html: str, *, base_url: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for link in soup.find_all("a"):
        href = _clean_str(link.get("href"))
        if not href:
            continue
        lower = href.lower()
        if ".pdf" in lower or lower.endswith("/pdf"):
            if href.startswith("http://") or href.startswith("https://"):
                return href
            if href.startswith("/"):
                match = re.match(r"^(https?://[^/]+)", base_url)
                return f"{match.group(1)}{href}" if match else href
            return href
    return ""


def _fixture_dir() -> Path:
    configured = _clean_str(os.environ.get(FIXTURE_ENV_VAR))
    if configured:
        return Path(configured).expanduser().resolve()
    cwd_fixture_dir = Path.cwd() / "tests" / "fixtures" / "literature_fulltext"
    if cwd_fixture_dir.exists():
        return cwd_fixture_dir.resolve()
    return Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "literature_fulltext"


def _find_fixture(card: dict[str, Any], fixture_dir: Path) -> Path | None:
    direct_path = _direct_fixture_path(card)
    if direct_path is not None and direct_path.exists():
        return direct_path.resolve()

    for stem in _fixture_stems(card):
        for extension in FULLTEXT_EXTENSIONS:
            for candidate in (
                fixture_dir / f"{stem}{extension}",
                fixture_dir / f"{stem.lower()}{extension}",
            ):
                if candidate.exists() and candidate.is_file():
                    return candidate.resolve()
    return None


def _direct_fixture_path(card: dict[str, Any]) -> Path | None:
    for key in ("fulltext_fixture", "fixture_path", "fulltext_path", "pdf_path", "xml_path", "html_path", "text_path"):
        value = _clean_str(card.get(key))
        if value:
            return Path(value).expanduser()
    return None


def _fixture_stems(card: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("paper_id", "doi", "source_id", "pmcid", "pmid"):
        value = _clean_str(card.get(key))
        if value:
            values.append(_strip_doi_prefix(value))
    external_ids = card.get("external_ids")
    if isinstance(external_ids, dict):
        for value in external_ids.values():
            clean_value = _clean_str(value)
            if clean_value:
                values.append(_strip_doi_prefix(clean_value))

    stems: list[str] = []
    for value in values:
        safe = _safe_filename(value)
        if safe and safe not in stems:
            stems.append(safe)
        lower_safe = safe.lower()
        if lower_safe and lower_safe not in stems:
            stems.append(lower_safe)
    return stems


def _extract_readable_html_or_xml(path: Path, *, prefer_xml: bool) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    if prefer_xml:
        try:
            soup = BeautifulSoup(raw, "xml")
        except FeatureNotFound:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
                soup = BeautifulSoup(raw, "html.parser")
    else:
        soup = BeautifulSoup(raw, "html.parser")
    for tag in soup.find_all(
        [
            "script",
            "style",
            "nav",
            "header",
            "footer",
            "aside",
            "noscript",
            "form",
            "iframe",
            "svg",
            "ref-list",
            "references",
        ]
    ):
        tag.decompose()

    root = None
    if prefer_xml:
        root = soup.find("article") or soup.find("body")
    else:
        root = soup.find("article") or soup.find("main") or soup.find("body")
    return _normalize_text((root or soup).get_text(" ", strip=True))


def _extract_pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    page_texts: list[str] = []
    for page in reader.pages:
        page_texts.append(page.extract_text() or "")
    return _normalize_text("\n".join(page_texts))


def _metadata_text(card: dict[str, Any], *, paper_id: str, abstract_or_summary: str) -> str:
    lines: list[str] = []
    title = _clean_str(card.get("title")) or _clean_str(card.get("citation")) or paper_id
    if title:
        lines.append(f"Title: {title}")
    if abstract_or_summary:
        lines.append(f"Abstract: {abstract_or_summary}")
    authors = _authors_text(card.get("authors"))
    if authors:
        lines.append(f"Authors: {authors}")
    for label, key in (("Year", "year"), ("Venue", "venue"), ("DOI", "doi"), ("URL", "url"), ("Source", "source")):
        value = _clean_str(card.get(key))
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def _abstract_or_summary(card: dict[str, Any]) -> str:
    for key in ("abstract", "summary", "method_summary", "problem", "citation"):
        value = _clean_str(card.get(key))
        if value:
            return value
    return ""


def _authors_text(value: Any) -> str:
    if isinstance(value, list):
        authors = [_clean_str(item) for item in value]
        return ", ".join(author for author in authors if author)
    return _clean_str(value)


def _merge_text(metadata_text: str, fulltext: str, *, max_chars: int) -> str:
    if fulltext:
        text = f"{metadata_text}\n\nFull text:\n{fulltext}" if metadata_text else fulltext
    else:
        text = metadata_text
    if len(text) <= max_chars:
        return text
    if metadata_text and len(metadata_text) < max_chars:
        remaining = max_chars - len(metadata_text) - len("\n\nFull text:\n")
        truncated_fulltext = fulltext[: max(0, remaining)].rstrip()
        return f"{metadata_text}\n\nFull text:\n{truncated_fulltext}".rstrip()
    return text[:max_chars].rstrip()


def _chunk_text(text: str, *, paper_id: str, text_source: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for index, start in enumerate(range(0, len(text), DEFAULT_CHUNK_SIZE), start=1):
        end = min(len(text), start + DEFAULT_CHUNK_SIZE)
        chunks.append(
            {
                "chunk_id": f"{paper_id}::chunk_{index}",
                "index": index,
                "start_char": start,
                "end_char": end,
                "text": text[start:end],
                "text_source": text_source,
            }
        )
    return chunks


def _trace_summary(trace_papers: list[dict[str, Any]]) -> dict[str, int]:
    open_fulltext = sum(1 for paper in trace_papers if paper.get("status") == "open_fulltext")
    metadata_only = sum(1 for paper in trace_papers if paper.get("status") == "metadata_only")
    degraded = sum(1 for paper in trace_papers if paper.get("reason") in {"pdf_parse_failed", "fulltext_parse_failed"})
    return {
        "total_papers": len(trace_papers),
        "open_fulltext": open_fulltext,
        "metadata_only": metadata_only,
        "degraded": degraded,
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=False))
            handle.write("\n")


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9-]+", "_", value)
    return safe.strip("_")


def _strip_doi_prefix(value: str) -> str:
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", value.strip(), flags=re.I)
    text = re.sub(r"^doi:", "", text, flags=re.I)
    return text


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _drop_empty(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value not in (None, "")}


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
