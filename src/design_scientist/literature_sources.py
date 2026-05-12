"""Adapters for external literature search sources.

The adapters normalize source-specific payloads into paper-card dictionaries
and can run entirely from local fixtures for deterministic tests.
"""

from __future__ import annotations

import json
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import httpx

from design_scientist.io import ensure_dir


DEFAULT_TIMEOUT = 30.0
ARXIV_TIMEOUT = 10.0
ARXIV_RATE_LIMIT_SECONDS = 5.0
ARXIV_MAX_RETRIES = 1
ARXIV_RETRY_STATUS_CODES = {429, 503}
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_LAST_ARXIV_REQUEST_AT = 0.0
_CACHE_MISS = object()


@dataclass(frozen=True)
class LiteratureContext:
    """Project-derived context used to search and annotate paper cards."""

    query: str
    relevance: str


class LiteratureSourceTemporarilyUnavailable(RuntimeError):
    """Raised when a live source is rate-limited or temporarily unavailable."""


class LiteratureCacheError(ValueError):
    """Raised when an existing raw cache file is unreadable or malformed."""


class LiteratureSourceFormatError(ValueError):
    """Raised when a live source response is parseable enough to fetch but malformed."""


def search_pubmed(
    context: LiteratureContext,
    *,
    max_papers: int,
    cache_dir: str | Path,
    offline_fixtures: bool = False,
    fixture_dir: str | Path | None = None,
    client: httpx.Client | None = None,
) -> list[dict[str, Any]]:
    cache = ensure_dir(cache_dir)
    esearch_path = cache / "pubmed_esearch.json"
    efetch_path = cache / "pubmed_efetch.xml"
    if offline_fixtures:
        raw = _read_fixture_json(fixture_dir, "pubmed_esearch.json")
        xml_text = _read_fixture_text(fixture_dir, "pubmed_efetch.xml")
        _write_cache_json(esearch_path, raw)
        _write_cache_text(efetch_path, xml_text)
    else:
        cached_xml = _read_cached_text(efetch_path)
        if cached_xml is not None:
            return _parse_pubmed_xml(
                cached_xml,
                context,
                strict=True,
                source_label=f"PubMed XML cache {efetch_path}",
                error_cls=LiteratureCacheError,
            )[:max_papers]
        raw = _read_cached_json(esearch_path)
        if raw is _CACHE_MISS:
            params = _pubmed_common_params()
            params.update(
                {
                    "db": "pubmed",
                    "term": context.query,
                    "retmode": "json",
                    "retmax": str(max_papers),
                    "sort": "relevance",
                }
            )
            raw = _get_json(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params=params,
                client=client,
            )
            _write_cache_json(esearch_path, raw)
        ids = _extract_pubmed_ids(raw)[:max_papers]
        if not ids:
            return []
        fetch_params = _pubmed_common_params()
        fetch_params.update({"db": "pubmed", "id": ",".join(ids), "retmode": "xml"})
        xml_text = _get_text(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params=fetch_params,
            client=client,
        )
        _write_cache_text(efetch_path, xml_text)
    return _parse_pubmed_xml(
        xml_text,
        context,
        strict=not offline_fixtures,
        source_label="PubMed XML response",
        error_cls=LiteratureSourceFormatError,
    )[:max_papers]


def search_biorxiv(
    context: LiteratureContext,
    *,
    max_papers: int,
    cache_dir: str | Path,
    offline_fixtures: bool = False,
    fixture_dir: str | Path | None = None,
    client: httpx.Client | None = None,
) -> list[dict[str, Any]]:
    cache = ensure_dir(cache_dir)
    cache_path = cache / "biorxiv.json"
    if offline_fixtures:
        raw = _read_fixture_json(fixture_dir, "biorxiv.json")
        _write_cache_json(cache_path, raw)
    else:
        raw = _read_cached_json(cache_path)
        if raw is _CACHE_MISS:
            url = f"https://api.biorxiv.org/details/biorxiv/2020-01-01/{date.today().isoformat()}/0"
            raw = _get_json(url, client=client)
            _write_cache_json(cache_path, raw)
    records = raw.get("collection", []) if isinstance(raw, dict) else []
    query_terms = _query_terms(context.query)
    cards = []
    for item in records:
        if not isinstance(item, dict):
            continue
        haystack = f"{item.get('title', '')} {item.get('abstract', '')}".lower()
        if query_terms and not any(term in haystack for term in query_terms):
            continue
        doi = _clean_str(item.get("doi"))
        cards.append(
            _paper_card(
                source="biorxiv",
                source_id=doi or _clean_str(item.get("server")) or _clean_str(item.get("title")),
                title=item.get("title"),
                abstract=item.get("abstract"),
                authors=_split_authors(item.get("authors")),
                published=item.get("date"),
                year=_year_from_text(item.get("date")),
                doi=doi,
                venue="bioRxiv",
                url=f"https://doi.org/{doi}" if doi else None,
                context=context,
                external_ids={"doi": doi} if doi else {},
            )
        )
    return [card for card in cards if card][:max_papers]


def search_arxiv(
    context: LiteratureContext,
    *,
    max_papers: int,
    cache_dir: str | Path,
    offline_fixtures: bool = False,
    fixture_dir: str | Path | None = None,
    client: httpx.Client | None = None,
) -> list[dict[str, Any]]:
    cache = ensure_dir(cache_dir)
    cache_path = cache / "arxiv.xml"
    if offline_fixtures:
        xml_text = _read_fixture_text(fixture_dir, "arxiv.xml")
        _write_cache_text(cache_path, xml_text)
    else:
        cached_xml = _read_cached_text(cache_path)
        if cached_xml is not None:
            return _parse_arxiv_xml(
                cached_xml,
                context,
                strict=True,
                source_label=f"arXiv XML cache {cache_path}",
                error_cls=LiteratureCacheError,
            )[:max_papers]
        else:
            xml_text = _get_arxiv_text(
                "https://export.arxiv.org/api/query",
                params={
                    "search_query": _arxiv_search_query(context.query),
                    "start": "0",
                    "max_results": str(max_papers),
                    "sortBy": "relevance",
                    "sortOrder": "descending",
                },
                headers=_arxiv_headers(),
                client=client,
            )
            _write_cache_text(cache_path, xml_text)
    return _parse_arxiv_xml(
        xml_text,
        context,
        strict=not offline_fixtures,
        source_label="arXiv XML response",
        error_cls=LiteratureSourceFormatError,
    )[:max_papers]


def search_semantic_scholar(
    context: LiteratureContext,
    *,
    max_papers: int,
    cache_dir: str | Path,
    offline_fixtures: bool = False,
    fixture_dir: str | Path | None = None,
    client: httpx.Client | None = None,
) -> list[dict[str, Any]]:
    cache = ensure_dir(cache_dir)
    api_key = os.getenv("S2_API_KEY")
    cache_path = cache / "semantic_scholar.json"
    if offline_fixtures:
        raw = _read_fixture_json(fixture_dir, "semantic_scholar.json")
        _write_cache_json(cache_path, raw)
    else:
        raw = _read_cached_json(cache_path)
        if raw is _CACHE_MISS:
            if not api_key:
                return []
            raw = _get_json(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={
                    "query": context.query,
                    "limit": str(max_papers),
                    "fields": (
                        "paperId,title,abstract,authors,year,venue,url,externalIds,"
                        "publicationDate,citationCount,referenceCount,influentialCitationCount,"
                        "citations.paperId,citations.title,citations.year,citations.url,citations.externalIds,"
                        "references.paperId,references.title,references.year,references.url,references.externalIds"
                    ),
                },
                headers={"x-api-key": api_key},
                client=client,
            )
            _write_cache_json(cache_path, raw)
    records = raw.get("data", []) if isinstance(raw, dict) else []
    cards = []
    for item in records:
        if not isinstance(item, dict):
            continue
        external_ids = item.get("externalIds") if isinstance(item.get("externalIds"), dict) else {}
        doi = _clean_str(external_ids.get("DOI") or external_ids.get("doi"))
        paper_id = _clean_str(item.get("paperId"))
        authors = [
            author.get("name")
            for author in item.get("authors", [])
            if isinstance(author, dict) and author.get("name")
        ]
        card = _paper_card(
            source="semantic_scholar",
            source_id=paper_id,
            title=item.get("title"),
            abstract=item.get("abstract"),
            authors=authors,
            published=item.get("publicationDate"),
            year=item.get("year") or _year_from_text(item.get("publicationDate")),
            doi=doi,
            venue=item.get("venue"),
            url=item.get("url"),
            context=context,
            external_ids={**external_ids, "paperId": paper_id} if paper_id else external_ids,
            citation_count=item.get("citationCount"),
            reference_count=item.get("referenceCount"),
            influential_citation_count=item.get("influentialCitationCount"),
        )
        if card:
            citations = _semantic_scholar_related_papers(item.get("citations"))
            references = _semantic_scholar_related_papers(item.get("references"))
            if citations:
                card["citations"] = citations
            if references:
                card["references"] = references
        cards.append(card)
    return [card for card in cards if card][:max_papers]


def dedupe_paper_cards(cards: Iterable[dict[str, Any]], *, max_papers: int | None = None) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for card in cards:
        key = _dedupe_key(card)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(card)
        if max_papers is not None and len(unique) >= max_papers:
            break
    return unique


def paper_card_dedupe_key(card: dict[str, Any]) -> str | None:
    """Return the stable key used for cross-source paper-card deduplication."""
    return _dedupe_key(card)


def default_fixture_dir() -> Path:
    env_path = os.getenv("DESIGN_SCIENTIST_LITERATURE_FIXTURES")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "literature"


def _pubmed_common_params() -> dict[str, str]:
    params = {"tool": os.getenv("NCBI_TOOL", "design_scientist")}
    if os.getenv("NCBI_EMAIL"):
        params["email"] = os.environ["NCBI_EMAIL"]
    if os.getenv("NCBI_API_KEY"):
        params["api_key"] = os.environ["NCBI_API_KEY"]
    return params


def _extract_pubmed_ids(raw: Any) -> list[str]:
    try:
        ids = raw["esearchresult"]["idlist"]
    except (KeyError, TypeError):
        return []
    return [str(item) for item in ids if item]


def _parse_pubmed_xml(
    xml_text: str,
    context: LiteratureContext,
    *,
    strict: bool = False,
    source_label: str = "PubMed XML",
    error_cls: type[Exception] = LiteratureSourceFormatError,
) -> list[dict[str, Any]]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        if strict:
            raise error_cls(f"Malformed {source_label}: {exc}") from exc
        return []
    if _local_xml_name(root.tag) != "PubmedArticleSet":
        if strict:
            raise error_cls(
                f"Malformed {source_label}: expected PubmedArticleSet root, got {_local_xml_name(root.tag)}"
            )
        return []
    cards = []
    for article in root.findall(".//PubmedArticle"):
        pmid = _text(article.find(".//PMID"))
        title = _xml_text(article.find(".//ArticleTitle"))
        abstract = " ".join(
            part.strip()
            for part in (_xml_text(node) for node in article.findall(".//Abstract/AbstractText"))
            if part.strip()
        )
        authors = []
        for author in article.findall(".//Author"):
            collective = _text(author.find("CollectiveName"))
            if collective:
                authors.append(collective)
                continue
            last = _text(author.find("LastName"))
            fore = _text(author.find("ForeName")) or _text(author.find("Initials"))
            name = " ".join(part for part in [fore, last] if part)
            if name:
                authors.append(name)
        doi = None
        for node in article.findall(".//ArticleId"):
            if node.attrib.get("IdType", "").lower() == "doi":
                doi = _clean_str(node.text)
                break
        year = _text(article.find(".//PubDate/Year")) or _year_from_text(_text(article.find(".//PubDate/MedlineDate")))
        journal = _xml_text(article.find(".//Journal/Title"))
        cards.append(
            _paper_card(
                source="pubmed",
                source_id=pmid,
                title=title,
                abstract=abstract,
                authors=authors,
                published=str(year) if year else None,
                year=int(year) if str(year).isdigit() else None,
                doi=doi,
                venue=journal,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                context=context,
                external_ids={"pmid": pmid, "doi": doi} if doi else {"pmid": pmid},
            )
        )
    return [card for card in cards if card]


def _parse_arxiv_xml(
    xml_text: str,
    context: LiteratureContext,
    *,
    strict: bool = False,
    source_label: str = "arXiv XML",
    error_cls: type[Exception] = LiteratureSourceFormatError,
) -> list[dict[str, Any]]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        if strict:
            raise error_cls(f"Malformed {source_label}: {exc}") from exc
        return []
    if root.tag != "{http://www.w3.org/2005/Atom}feed":
        if strict:
            raise error_cls(f"Malformed {source_label}: expected Atom feed root, got {_local_xml_name(root.tag)}")
        return []
    cards = []
    for entry in root.findall("atom:entry", ARXIV_NS):
        entry_id = _text(entry.find("atom:id", ARXIV_NS))
        arxiv_id = entry_id.rstrip("/").split("/")[-1] if entry_id else None
        doi = _clean_str(_text(entry.find("arxiv:doi", ARXIV_NS)))
        authors = [_text(node.find("atom:name", ARXIV_NS)) for node in entry.findall("atom:author", ARXIV_NS)]
        authors = [author for author in authors if author]
        published = _text(entry.find("atom:published", ARXIV_NS))
        cards.append(
            _paper_card(
                source="arxiv",
                source_id=arxiv_id,
                title=_xml_text(entry.find("atom:title", ARXIV_NS)),
                abstract=_xml_text(entry.find("atom:summary", ARXIV_NS)),
                authors=authors,
                published=published,
                year=_year_from_text(published),
                doi=doi,
                venue="arXiv",
                url=entry_id,
                context=context,
                external_ids={"arxiv": arxiv_id, "doi": doi} if doi else {"arxiv": arxiv_id},
            )
        )
    return [card for card in cards if card]


def _semantic_scholar_related_papers(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    papers: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        paper_id = _clean_str(item.get("paperId"))
        title = _clean_str(item.get("title"))
        if not paper_id and not title:
            continue
        external_ids = item.get("externalIds") if isinstance(item.get("externalIds"), dict) else {}
        related = {
            "paper_id": paper_id,
            "title": title,
            "year": int(item["year"]) if str(item.get("year")).isdigit() else None,
            "url": _clean_str(item.get("url")),
            "external_ids": {
                str(key): val for key, val in external_ids.items() if val not in (None, "")
            },
        }
        papers.append({key: val for key, val in related.items() if val not in (None, "", {})})
    return papers


def _paper_card(
    *,
    source: str,
    source_id: str | None,
    title: Any,
    abstract: Any,
    authors: Any,
    published: Any,
    year: Any,
    doi: str | None,
    venue: Any,
    url: Any,
    context: LiteratureContext,
    external_ids: dict[str, Any],
    citation_count: Any = None,
    reference_count: Any = None,
    influential_citation_count: Any = None,
) -> dict[str, Any] | None:
    clean_title = _clean_str(title)
    clean_source_id = _clean_str(source_id)
    if not clean_title or not clean_source_id:
        return None
    clean_doi = _normal_doi(doi)
    clean_external_ids = {
        str(key): value for key, value in external_ids.items() if value not in (None, "")
    }
    if clean_doi:
        clean_external_ids["doi"] = clean_doi
    card = {
        "paper_id": f"{source}:{clean_source_id}",
        "source": source,
        "source_id": clean_source_id,
        "title": clean_title,
        "authors": authors if isinstance(authors, list) else _split_authors(authors),
        "year": int(year) if str(year).isdigit() else None,
        "published": _clean_str(published),
        "venue": _clean_str(venue),
        "doi": clean_doi,
        "url": _clean_str(url),
        "abstract": _clean_str(abstract),
        "relevance_to_current_project": context.relevance,
        "review_status": "source_imported_unreviewed",
        "external_ids": clean_external_ids,
    }
    for key, value in (
        ("citation_count", citation_count),
        ("reference_count", reference_count),
        ("influential_citation_count", influential_citation_count),
    ):
        parsed = _int_or_none(value)
        if parsed is not None:
            card[key] = parsed
    return card


def _dedupe_key(card: dict[str, Any]) -> str | None:
    doi = _normal_doi(card.get("doi"))
    if doi:
        return f"doi:{doi}"
    external_ids = card.get("external_ids") if isinstance(card.get("external_ids"), dict) else {}
    for key in ("pmid", "arxiv", "paperId"):
        value = _clean_str(external_ids.get(key))
        if value:
            return f"{key.lower()}:{value.lower()}"
    title = _clean_str(card.get("title"))
    year = card.get("year")
    if title:
        return f"title:{re.sub(r'[^a-z0-9]+', ' ', title.lower()).strip()}:{year or ''}"
    return None


def _query_terms(query: str) -> list[str]:
    return [term.lower() for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]{2,}", query)]


def _arxiv_search_query(query: str) -> str:
    """Convert a free-text project query into arXiv's fielded query syntax."""
    if re.search(r"\b(?:ti|au|abs|co|jr|cat|rn|all):", query):
        return query
    stop_words = {
        "method",
        "methods",
        "benchmark",
        "benchmarks",
        "experimental",
        "experiments",
        "retrospective",
        "replay",
        "sequential",
    }
    terms: list[str] = []
    seen: set[str] = set()
    for term in _query_terms(query.replace("_", " ")):
        if term in stop_words or term in seen:
            continue
        seen.add(term)
        terms.append(term)
        if len(terms) >= 8:
            break
    if not terms:
        fallback = _clean_str(query) or "active learning"
        return f'all:"{fallback}"'
    return " OR ".join(f"all:{term}" for term in terms)


def _arxiv_headers() -> dict[str, str]:
    return {"User-Agent": os.getenv("DESIGN_SCIENTIST_ARXIV_USER_AGENT", "Design-Scientist/0.1")}


def _get_arxiv_text(
    url: str,
    *,
    params: dict[str, str],
    headers: dict[str, str],
    client: httpx.Client | None = None,
) -> str:
    last_error: Exception | None = None
    for attempt in range(ARXIV_MAX_RETRIES + 1):
        _respect_arxiv_rate_limit()
        try:
            return _get_text(
                url,
                params=params,
                headers=headers,
                client=client,
                timeout=ARXIV_TIMEOUT,
            )
        except httpx.HTTPStatusError as exc:
            last_error = exc
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code not in ARXIV_RETRY_STATUS_CODES or attempt >= ARXIV_MAX_RETRIES:
                if status_code in ARXIV_RETRY_STATUS_CODES:
                    raise LiteratureSourceTemporarilyUnavailable(str(exc)) from exc
                raise
            _sleep_before_arxiv_retry(exc, attempt)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_error = exc
            if attempt >= ARXIV_MAX_RETRIES:
                raise LiteratureSourceTemporarilyUnavailable(str(exc)) from exc
            _sleep_before_arxiv_retry(exc, attempt)
    if last_error is not None:  # pragma: no cover - defensive guard
        raise last_error
    raise RuntimeError("arXiv request failed before any attempt was made")  # pragma: no cover


def _respect_arxiv_rate_limit() -> None:
    global _LAST_ARXIV_REQUEST_AT
    now = time.monotonic()
    if _LAST_ARXIV_REQUEST_AT > 0:
        elapsed = now - _LAST_ARXIV_REQUEST_AT
        if elapsed < ARXIV_RATE_LIMIT_SECONDS:
            time.sleep(ARXIV_RATE_LIMIT_SECONDS - elapsed)
    _LAST_ARXIV_REQUEST_AT = time.monotonic()


def _sleep_before_arxiv_retry(exc: Exception, attempt: int) -> None:
    retry_after = _retry_after_seconds(exc)
    delay = retry_after if retry_after is not None else ARXIV_RATE_LIMIT_SECONDS * (attempt + 1)
    time.sleep(max(ARXIV_RATE_LIMIT_SECONDS, delay))


def _retry_after_seconds(exc: Exception) -> float | None:
    if not isinstance(exc, httpx.HTTPStatusError) or exc.response is None:
        return None
    retry_after = exc.response.headers.get("Retry-After")
    if not retry_after:
        return None
    try:
        return max(float(retry_after), 0.0)
    except ValueError:
        return None


def _split_authors(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if not value:
        return []
    return [part.strip() for part in re.split(r";|\band\b", str(value)) if part.strip()]


def _year_from_text(value: Any) -> int | None:
    if value is None:
        return None
    match = re.search(r"\b(19|20)\d{2}\b", str(value))
    return int(match.group(0)) if match else None


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normal_doi(value: Any) -> str | None:
    text = _clean_str(value)
    if not text:
        return None
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text, flags=re.I)
    return text.lower()


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def _text(node: ET.Element | None) -> str | None:
    if node is None or node.text is None:
        return None
    return _clean_str(node.text)


def _xml_text(node: ET.Element | None) -> str | None:
    if node is None:
        return None
    return _clean_str("".join(node.itertext()))


def _local_xml_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _read_fixture_json(fixture_dir: str | Path | None, name: str) -> Any:
    path = _fixture_path(fixture_dir, name)
    if not path.exists():
        raise FileNotFoundError(f"Missing literature fixture: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_fixture_text(fixture_dir: str | Path | None, name: str) -> str:
    path = _fixture_path(fixture_dir, name)
    if not path.exists():
        raise FileNotFoundError(f"Missing literature fixture: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _fixture_path(fixture_dir: str | Path | None, name: str) -> Path:
    root = Path(fixture_dir).expanduser().resolve() if fixture_dir else default_fixture_dir()
    return root / name


def _read_cached_json(path: Path) -> Any:
    if not path.exists():
        return _CACHE_MISS
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise LiteratureCacheError(f"Unreadable JSON cache {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise LiteratureCacheError(f"Malformed JSON cache {path}: {exc}") from exc


def _read_cached_text(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LiteratureCacheError(f"Unreadable text cache {path}: {exc}") from exc


def _write_cache_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _write_cache_text(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8")


def _get_json(
    url: str,
    *,
    params: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    client: httpx.Client | None = None,
) -> Any:
    if client is not None:
        response = client.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    with httpx.Client(timeout=DEFAULT_TIMEOUT, follow_redirects=True) as managed:
        response = managed.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()


def _get_text(
    url: str,
    *,
    params: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    client: httpx.Client | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    if client is not None:
        response = client.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.text
    with httpx.Client(timeout=timeout, follow_redirects=True) as managed:
        response = managed.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.text
