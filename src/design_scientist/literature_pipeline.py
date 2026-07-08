"""Literature search pipeline writing staged framework literature artifacts."""

from __future__ import annotations

import csv
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from design_scientist.io import ensure_dir, read_yaml, write_json
from design_scientist.literature_sources import (
    LiteratureContext,
    LiteratureCacheError,
    LiteratureSourceTemporarilyUnavailable,
    default_fixture_dir,
    paper_card_dedupe_key,
    search_arxiv,
    search_biorxiv,
    search_pubmed,
    search_semantic_scholar,
)


SOURCE_ADAPTERS = {
    "pubmed": search_pubmed,
    "biorxiv": search_biorxiv,
    "arxiv": search_arxiv,
    "semantic_scholar": search_semantic_scholar,
    "s2": search_semantic_scholar,
}

DEFAULT_SOURCES = ("pubmed", "biorxiv", "arxiv", "semantic_scholar")
LITERATURE_SOURCES_ENV = "DESIGN_SCIENTIST_LITERATURE_SOURCES"
SOURCE_PRIORITY = {
    "semantic_scholar": 4.0,
    "s2": 4.0,
    "pubmed": 3.0,
    "biorxiv": 2.0,
    "arxiv": 2.0,
}
METHOD_KEYWORDS = (
    "active learning",
    "adaptive design",
    "acquisition",
    "bayesian optimization",
    "benchmark",
    "constraint",
    "contrast",
    "experimental design",
    "matched contrast",
    "mechanism aware",
    "mechanism-aware",
    "neutral binding",
    "ph dependent",
    "ph-dependent",
    "ph sensitive",
    "ph-sensitive",
    "acidic ph",
    "histidine",
    "antibody",
    "antibodies",
    "antigen binding",
    "dissociation",
    "hbsag",
    "protein engineering",
    "protein design",
    "retrospective",
    "sequential design",
    "variant",
    "wet lab",
)
SCORE_FIELDS = (
    "rank",
    "selected",
    "exclusion_reason",
    "dedupe_key",
    "paper_id",
    "query_id",
    "source",
    "title",
    "year",
    "recency",
    "citation_count",
    "method_keyword_score",
    "query_match_score",
    "source_priority",
    "total_score",
)


def run_literature_search(
    project_dir: str | Path,
    max_papers: int = 60,
    offline_fixtures: bool = False,
    sources: list[str] | tuple[str, ...] | None = None,
) -> Path:
    """Search literature sources and write normalized paper cards.

    Offline fixture mode reads tests/fixtures/literature by default, or the path
    in DESIGN_SCIENTIST_LITERATURE_FIXTURES.
    """

    if max_papers <= 0:
        raise ValueError("max_papers must be positive")

    root = Path(project_dir).expanduser().resolve()
    framework_dir = ensure_dir(root / "framework")
    cache_dir = ensure_dir(framework_dir / "cache" / "literature_raw")
    context = _build_context(root)
    selected_sources = _sources_from_argument_or_env(sources)
    requested = _validate_sources(list(selected_sources or DEFAULT_SOURCES))
    plan_queries = _build_query_plan(root, context, requested, explicit_sources=selected_sources is not None)
    plan = {
        "version": "literature_search_v2",
        "offline_fixtures": bool(offline_fixtures),
        "max_papers": max_papers,
        "queries": plan_queries,
    }
    write_json(framework_dir / "literature_search_plan.json", plan)

    per_source_limit = max(max_papers, 1)

    cards: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    events: list[dict[str, Any]] = []
    fixture_dir = default_fixture_dir() if offline_fixtures else None
    cooldown_sources: set[str] = set()
    for query in plan_queries:
        query_context = LiteratureContext(query=query["query"], relevance=context.relevance)
        for source in query["sources"]:
            adapter = SOURCE_ADAPTERS.get(source)
            if adapter is None:
                raise ValueError(f"Unknown literature source: {source}")
            source_cache_dir = _source_cache_dir(cache_dir, query_id=query["query_id"], source=source)
            pre_cache_files = _cache_files(source_cache_dir)
            cache_status = _source_cache_status(source, source_cache_dir, pre_cache_files)
            cache_complete = cache_status == "cache_hit"
            if cache_status == "cache_error" and not offline_fixtures:
                exc = _source_cache_error(source, source_cache_dir) or LiteratureCacheError(
                    f"Malformed literature cache at {source_cache_dir}"
                )
                raw_count = _raw_count_for_source(source, source_cache_dir, fallback=0)
                reason = _safe_error_text(exc)
                error = {
                    "query_id": query["query_id"],
                    "source": source,
                    "cache_path": _relative_artifact_path(source_cache_dir, root),
                    "error_type": type(exc).__name__,
                    "error": reason,
                    "reason": reason,
                }
                errors.append(error)
                event = {
                    "event": "source_error",
                    "status": "cache_error",
                    "query_id": query["query_id"],
                    "source": source,
                    "raw_count": raw_count,
                    "normalized_count": 0,
                    "reason": reason,
                    "error_type": type(exc).__name__,
                    "error": reason,
                }
                events.append(
                    _with_source_event_metadata(
                        event,
                        source=source,
                        source_cache_dir=source_cache_dir,
                        project_root=root,
                        offline_fixtures=offline_fixtures,
                        cache_status="cache_error",
                        network_fetch=False,
                    )
                )
                continue
            if _should_skip_semantic_scholar(
                source,
                offline_fixtures=offline_fixtures,
                cache_complete=cache_complete,
            ):
                event = {
                    "event": "source_skip",
                    "status": "skipped",
                    "query_id": query["query_id"],
                    "source": source,
                    "raw_count": 0,
                    "normalized_count": 0,
                    "reason": "missing_s2_api_key",
                }
                events.append(
                    _with_source_event_metadata(
                        event,
                        source=source,
                        source_cache_dir=source_cache_dir,
                        project_root=root,
                        offline_fixtures=offline_fixtures,
                        cache_status=cache_status,
                        network_fetch=False,
                    )
                )
                continue
            if source in cooldown_sources:
                event = {
                    "event": "source_skip",
                    "status": "skipped",
                    "query_id": query["query_id"],
                    "source": source,
                    "raw_count": 0,
                    "normalized_count": 0,
                    "reason": f"previous_{source}_error",
                }
                events.append(
                    _with_source_event_metadata(
                        event,
                        source=source,
                        source_cache_dir=source_cache_dir,
                        project_root=root,
                        offline_fixtures=offline_fixtures,
                        cache_status=cache_status,
                        network_fetch=False,
                    )
                )
                continue
            try:
                source_cards = adapter(
                    query_context,
                    max_papers=per_source_limit,
                    cache_dir=source_cache_dir,
                    offline_fixtures=offline_fixtures,
                    fixture_dir=fixture_dir,
                )
                annotated_cards = _annotate_cards(source_cards, query=query, requested_source=source)
                cards.extend(annotated_cards)
                event = {
                    "event": "source_result",
                    "status": "ok",
                    "query_id": query["query_id"],
                    "source": source,
                    "raw_count": _raw_count_for_source(source, source_cache_dir, fallback=len(source_cards)),
                    "normalized_count": len(annotated_cards),
                    "result_count": len(annotated_cards),
                    "paper_ids": [card["paper_id"] for card in annotated_cards if card.get("paper_id")],
                }
                events.append(
                    _with_source_event_metadata(
                        event,
                        source=source,
                        source_cache_dir=source_cache_dir,
                        project_root=root,
                        offline_fixtures=offline_fixtures,
                        cache_status=cache_status,
                        network_fetch=cache_status == "cache_miss" and not offline_fixtures,
                    )
                )
            except LiteratureSourceTemporarilyUnavailable as exc:
                reason = _safe_error_text(exc)
                event = {
                    "event": "source_skip",
                    "status": "degraded",
                    "query_id": query["query_id"],
                    "source": source,
                    "raw_count": _raw_count_for_source(source, source_cache_dir, fallback=0),
                    "normalized_count": 0,
                    "reason": reason,
                    "error_type": type(exc).__name__,
                }
                events.append(
                    _with_source_event_metadata(
                        event,
                        source=source,
                        source_cache_dir=source_cache_dir,
                        project_root=root,
                        offline_fixtures=offline_fixtures,
                        cache_status=cache_status,
                        network_fetch=cache_status == "cache_miss" and not offline_fixtures,
                    )
                )
                cooldown_sources.add(source)
            except Exception as exc:
                is_cache_error = cache_status == "cache_error" or isinstance(exc, LiteratureCacheError)
                raw_count = _raw_count_for_source(source, source_cache_dir, fallback=0)
                reason = _safe_error_text(exc)
                error = {
                    "query_id": query["query_id"],
                    "source": source,
                    "cache_path": _relative_artifact_path(source_cache_dir, root),
                    "error_type": type(exc).__name__,
                    "error": reason,
                    "reason": reason,
                }
                errors.append(error)
                event = {
                    "event": "source_error",
                    "status": "cache_error" if is_cache_error else "error",
                    "query_id": query["query_id"],
                    "source": source,
                    "raw_count": raw_count,
                    "normalized_count": 0,
                    "reason": reason,
                    "error_type": type(exc).__name__,
                    "error": reason,
                }
                events.append(
                    _with_source_event_metadata(
                        event,
                        source=source,
                        source_cache_dir=source_cache_dir,
                        project_root=root,
                        offline_fixtures=offline_fixtures,
                        cache_status="cache_error" if is_cache_error else cache_status,
                        network_fetch=(
                            not is_cache_error and cache_status == "cache_miss" and not offline_fixtures
                        ),
                    )
                )
                if source == "arxiv" and not offline_fixtures:
                    cooldown_sources.add(source)

    final_cards, score_rows = _rank_and_dedupe(cards, max_papers=max_papers)
    _write_scores_csv(framework_dir / "paper_scores.csv", score_rows)

    trace = {
        "version": "literature_search_v2",
        "offline_fixtures": bool(offline_fixtures),
        "events": events,
        "summary": {
            "planned_queries": len(plan_queries),
            "source_results": sum(1 for event in events if event["event"] == "source_result"),
            "source_errors": len(errors),
            "source_skips": sum(1 for event in events if event["event"] == "source_skip"),
            "raw_paper_count": len(cards),
            "final_paper_count": len(final_cards),
        },
    }
    write_json(framework_dir / "literature_search_trace.json", trace)
    write_json(framework_dir / "citation_graph.json", _build_citation_graph(final_cards))

    out = framework_dir / "paper_cards.json"
    write_json(out, final_cards)
    source_errors_path = cache_dir / "source_errors.json"
    if errors:
        write_json(source_errors_path, {"errors": errors})
    elif source_errors_path.exists():
        source_errors_path.unlink()
    return out


def _validate_sources(sources: list[str]) -> list[str]:
    requested = []
    for source in sources:
        clean = _clean_str(source)
        if not clean:
            continue
        if clean not in SOURCE_ADAPTERS:
            raise ValueError(f"Unknown literature source: {clean}")
        if clean not in requested:
            requested.append(clean)
    return requested or list(DEFAULT_SOURCES)


def _sources_from_argument_or_env(sources: list[str] | tuple[str, ...] | None) -> list[str] | None:
    if sources is not None:
        return list(sources)
    raw = os.getenv(LITERATURE_SOURCES_ENV)
    if raw is None:
        return None
    return [source for source in re.split(r"[,\s]+", raw) if source]


def _build_query_plan(
    project_dir: Path,
    context: LiteratureContext,
    requested_sources: list[str],
    *,
    explicit_sources: bool,
) -> list[dict[str, Any]]:
    queries_path = project_dir / "framework" / "literature_queries.yaml"
    if not queries_path.exists():
        return [_default_query_plan(context, requested_sources)]

    data = read_yaml(queries_path)
    raw_queries = data.get("queries")
    if not isinstance(raw_queries, list) or not raw_queries:
        return [_default_query_plan(context, requested_sources)]

    plan: list[dict[str, Any]] = []
    seen_ids: dict[str, int] = {}
    for index, raw_query in enumerate(raw_queries, start=1):
        if not isinstance(raw_query, dict):
            continue
        query_id = _query_id(raw_query.get("query_id"), index=index)
        seen_ids[query_id] = seen_ids.get(query_id, 0) + 1
        if seen_ids[query_id] > 1:
            query_id = f"{query_id}_{seen_ids[query_id]}"
        query_text = _clean_str(raw_query.get("query")) or context.query
        plan.append(
            {
                "query_id": query_id,
                "query": query_text,
                "purpose": _clean_str(raw_query.get("purpose")) or "method_discovery",
                "sources": _sources_for_query(
                    raw_query.get("sources"),
                    requested_sources,
                    explicit_sources=explicit_sources,
                ),
            }
        )
    return plan or [_default_query_plan(context, requested_sources)]


def _default_query_plan(context: LiteratureContext, requested_sources: list[str]) -> dict[str, Any]:
    return {
        "query_id": "project_query",
        "query": context.query,
        "purpose": "project_literature_search",
        "sources": requested_sources,
    }


def _sources_for_query(raw_sources: Any, requested_sources: list[str], *, explicit_sources: bool) -> list[str]:
    if explicit_sources:
        return requested_sources
    if isinstance(raw_sources, str):
        candidates = [raw_sources]
    elif isinstance(raw_sources, list):
        candidates = raw_sources
    else:
        candidates = []

    planned = []
    for candidate in candidates:
        source = _clean_str(candidate)
        if not source or source == "adapter_pending":
            continue
        if source not in SOURCE_ADAPTERS:
            raise ValueError(f"Unknown literature source: {source}")
        if source not in planned:
            planned.append(source)
    return planned or requested_sources


def _query_id(value: Any, *, index: int) -> str:
    text = _clean_str(value) or f"query_{index}"
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_").lower()
    return text or f"query_{index}"


def _should_skip_semantic_scholar(source: str, *, offline_fixtures: bool, cache_complete: bool) -> bool:
    return (
        source in {"semantic_scholar", "s2"}
        and not offline_fixtures
        and not cache_complete
        and not os.getenv("S2_API_KEY")
    )


def _source_cache_dir(cache_dir: Path, *, query_id: str, source: str) -> Path:
    return ensure_dir(cache_dir / _path_component(query_id) / _path_component(source))


def _path_component(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._-")
    return text or "unknown"


def _with_cache_metadata(event: dict[str, Any], *, source_cache_dir: Path, project_root: Path) -> dict[str, Any]:
    event["cache_path"] = _relative_artifact_path(source_cache_dir, project_root)
    cache_files = _cache_files(source_cache_dir)
    if cache_files:
        event["cache_files"] = [_relative_artifact_path(path, project_root) for path in cache_files]
    return event


def _with_source_event_metadata(
    event: dict[str, Any],
    *,
    source: str,
    source_cache_dir: Path,
    project_root: Path,
    offline_fixtures: bool,
    cache_status: str,
    network_fetch: bool,
) -> dict[str, Any]:
    event = _with_cache_metadata(event, source_cache_dir=source_cache_dir, project_root=project_root)
    if not offline_fixtures:
        event["cache_status"] = cache_status
        event["cache_hit"] = cache_status == "cache_hit"
        event["cache_miss"] = cache_status == "cache_miss"
        event["cache_error"] = cache_status == "cache_error"
        event["network_fetch"] = bool(network_fetch)
    if source == "biorxiv" and not offline_fixtures:
        event["behavior"] = "recent_feed_scan"
    return event


def _cache_files(source_cache_dir: Path) -> list[Path]:
    if not source_cache_dir.exists():
        return []
    return sorted(path for path in source_cache_dir.iterdir() if path.is_file())


def _source_cache_complete(source: str, source_cache_dir: Path, cache_files: list[Path]) -> bool:
    return _source_cache_status(source, source_cache_dir, cache_files) == "cache_hit"


def _source_cache_status(source: str, source_cache_dir: Path, cache_files: list[Path]) -> str:
    source_name = "semantic_scholar" if source == "s2" else source
    if _source_cache_error(source, source_cache_dir) is not None:
        return "cache_error"
    if source_name == "pubmed":
        path = source_cache_dir / "pubmed_efetch.xml"
        if path.exists():
            return "cache_hit"
        return "cache_miss"
    if source_name == "biorxiv":
        return "cache_hit" if _read_cache_json(source_cache_dir / "biorxiv.json") is not None else "cache_miss"
    if source_name == "arxiv":
        if (source_cache_dir / "arxiv.xml").exists() or (source_cache_dir / "arxiv_search.html").exists():
            return "cache_hit"
        return "cache_miss"
    if source_name == "semantic_scholar":
        return "cache_hit" if _read_cache_json(source_cache_dir / "semantic_scholar.json") is not None else "cache_miss"
    return "cache_hit" if cache_files else "cache_miss"


def _source_cache_error(source: str, source_cache_dir: Path) -> LiteratureCacheError | None:
    source_name = "semantic_scholar" if source == "s2" else source
    if source_name == "pubmed":
        efetch_path = source_cache_dir / "pubmed_efetch.xml"
        if efetch_path.exists():
            reason = _pubmed_xml_cache_error(efetch_path)
            if reason:
                return LiteratureCacheError(f"Malformed PubMed XML cache {efetch_path}: {reason}")
            return None
        esearch_path = source_cache_dir / "pubmed_esearch.json"
        reason = _json_cache_error(esearch_path)
        if reason:
            return LiteratureCacheError(reason)
        return None
    if source_name == "biorxiv":
        reason = _json_cache_error(source_cache_dir / "biorxiv.json")
        return LiteratureCacheError(reason) if reason else None
    if source_name == "arxiv":
        path = source_cache_dir / "arxiv.xml"
        web_path = source_cache_dir / "arxiv_search.html"
        if not path.exists() and not web_path.exists():
            return None
        if path.exists():
            reason = _arxiv_xml_cache_error(path)
            return LiteratureCacheError(f"Malformed arXiv XML cache {path}: {reason}") if reason else None
        reason = _arxiv_web_cache_error(web_path)
        return LiteratureCacheError(f"Malformed arXiv search cache {web_path}: {reason}") if reason else None
    if source_name == "semantic_scholar":
        reason = _json_cache_error(source_cache_dir / "semantic_scholar.json")
        return LiteratureCacheError(reason) if reason else None
    return None


def _json_cache_error(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return f"Unreadable JSON cache {path}: {exc}"
    except json.JSONDecodeError as exc:
        return f"Malformed JSON cache {path}: {exc}"
    return None


def _relative_artifact_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _raw_count_for_source(source: str, source_cache_dir: Path, *, fallback: int) -> int:
    source_name = "semantic_scholar" if source == "s2" else source
    if source_name == "pubmed":
        efetch_count = _count_pubmed_articles(source_cache_dir / "pubmed_efetch.xml")
        if efetch_count is not None:
            return efetch_count
        raw = _read_cache_json(source_cache_dir / "pubmed_esearch.json")
        ids = _pubmed_ids_from_cache(raw)
        if ids is not None:
            return len(ids)
    if source_name == "biorxiv":
        raw = _read_cache_json(source_cache_dir / "biorxiv.json")
        records = raw.get("collection") if isinstance(raw, dict) else None
        if isinstance(records, list):
            return len(records)
    if source_name == "arxiv":
        arxiv_count = _count_arxiv_entries(source_cache_dir / "arxiv.xml")
        if arxiv_count is not None:
            return arxiv_count
        arxiv_search_count = _count_arxiv_search_results(source_cache_dir / "arxiv_search.html")
        if arxiv_search_count is not None:
            return arxiv_search_count
    if source_name == "semantic_scholar":
        raw = _read_cache_json(source_cache_dir / "semantic_scholar.json")
        records = raw.get("data") if isinstance(raw, dict) else None
        if isinstance(records, list):
            return len(records)
    return max(_int_or_zero(fallback), 0)


def _read_cache_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _pubmed_ids_from_cache(raw: Any) -> list[str] | None:
    try:
        ids = raw["esearchresult"]["idlist"]
    except (KeyError, TypeError):
        return None
    return [str(item) for item in ids if item]


def _count_pubmed_articles(path: Path) -> int | None:
    try:
        root = ET.fromstring(path.read_text(encoding="utf-8"))
    except (OSError, ET.ParseError):
        return None
    return len(root.findall(".//PubmedArticle"))


def _pubmed_xml_cache_error(path: Path) -> str | None:
    try:
        root = ET.fromstring(path.read_text(encoding="utf-8"))
    except (OSError, ET.ParseError) as exc:
        return str(exc)
    if _local_xml_name(root.tag) != "PubmedArticleSet":
        return f"expected PubmedArticleSet root, got {_local_xml_name(root.tag)}"
    return None


def _count_arxiv_entries(path: Path) -> int | None:
    try:
        root = ET.fromstring(path.read_text(encoding="utf-8"))
    except (OSError, ET.ParseError):
        return None
    return len(root.findall("{http://www.w3.org/2005/Atom}entry"))


def _count_arxiv_search_results(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    return len(re.findall(r'class=["\'][^"\']*\barxiv-result\b', text))


def _arxiv_xml_cache_error(path: Path) -> str | None:
    try:
        root = ET.fromstring(path.read_text(encoding="utf-8"))
    except (OSError, ET.ParseError) as exc:
        return str(exc)
    if root.tag != "{http://www.w3.org/2005/Atom}feed":
        return f"expected Atom feed root, got {_local_xml_name(root.tag)}"
    return None


def _arxiv_web_cache_error(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return str(exc)
    if (
        not re.search(r'class=["\'][^"\']*\barxiv-result\b', text)
        and not re.search(r"produced\s+no\s+results", text, flags=re.I)
    ):
        return "no arXiv result rows found"
    return None


def _local_xml_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _annotate_cards(
    source_cards: list[dict[str, Any]],
    *,
    query: dict[str, Any],
    requested_source: str,
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for card in source_cards:
        if not isinstance(card, dict) or not card.get("paper_id"):
            continue
        clean_card = dict(card)
        trace_item = {
            "query_id": query["query_id"],
            "source": requested_source,
            "paper_id": clean_card.get("paper_id"),
            "source_id": clean_card.get("source_id"),
        }
        clean_card["query_id"] = query["query_id"]
        clean_card["search_query"] = query["query"]
        clean_card["search_purpose"] = query["purpose"]
        clean_card["source_trace"] = _merge_source_trace(clean_card.get("source_trace"), [trace_item])
        annotated.append(clean_card)
    return annotated


def _rank_and_dedupe(cards: list[dict[str, Any]], *, max_papers: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not cards:
        return [], []

    scored = sorted(_score_cards(cards), key=_score_sort_key)
    selected: dict[str, dict[str, Any]] = {}
    ordered_keys: list[str] = []
    ranked_items: list[dict[str, Any]] = []
    for rank, item in enumerate(scored, start=1):
        card = item["card"]
        key = _dedupe_key(card)
        ranked_item = {**item, "rank": rank, "original_rank": rank, "dedupe_key": key}
        ranked_items.append(ranked_item)
        if key is None:
            continue
        if key not in selected:
            selected[key] = {"card": dict(card), "components": item["components"], "rank": rank}
            selected[key]["card"]["source_trace"] = _merge_source_trace(card.get("source_trace"), [])
            ordered_keys.append(key)
            continue
        _merge_duplicate_card(selected[key]["card"], card)

    selected_limit = max(max_papers, 0)
    selected_keys = set(ordered_keys[:selected_limit])
    first_rank_by_key = {key: selected[key]["rank"] for key in ordered_keys}
    final_items = [selected[key] for key in ordered_keys[:selected_limit]]
    final_cards = [item["card"] for item in final_items]
    display_items = []
    for item in ranked_items:
        selected_flag = _score_item_selected(
            item,
            selected_keys=selected_keys,
            first_rank_by_key=first_rank_by_key,
        )
        exclusion_reason = _score_item_exclusion_reason(
            item,
            selected_keys=selected_keys,
            first_rank_by_key=first_rank_by_key,
        )
        display_items.append(
            {
                **item,
                "selected_flag": selected_flag,
                "exclusion_reason": exclusion_reason,
            }
        )
    display_items = sorted(display_items, key=_score_row_display_sort_key)
    score_rows = [
        _score_row(
            rank=display_rank,
            card=item["card"],
            components=item["components"],
            selected=item["selected_flag"],
            exclusion_reason=item["exclusion_reason"],
            dedupe_key=item["dedupe_key"],
        )
        for display_rank, item in enumerate(display_items, start=1)
    ]
    return final_cards, score_rows


def _score_row_display_sort_key(item: dict[str, Any]) -> tuple[int, int, float, str]:
    reason = item.get("exclusion_reason") or ""
    if item.get("selected_flag"):
        group = 0
    elif reason == "below_max_papers_cutoff":
        group = 1
    elif reason == "duplicate_dedupe_key":
        group = 2
    else:
        group = 3
    components = item.get("components", {})
    return (
        group,
        int(item.get("original_rank") or item.get("rank") or 0),
        -float(components.get("total_score") or 0.0),
        str(item.get("card", {}).get("paper_id") or ""),
    )


def _score_item_selected(
    item: dict[str, Any],
    *,
    selected_keys: set[str],
    first_rank_by_key: dict[str, int],
) -> bool:
    key = item.get("dedupe_key")
    return bool(key and key in selected_keys and first_rank_by_key.get(key) == item.get("rank"))


def _score_item_exclusion_reason(
    item: dict[str, Any],
    *,
    selected_keys: set[str],
    first_rank_by_key: dict[str, int],
) -> str:
    key = item.get("dedupe_key")
    if not key:
        return "missing_dedupe_key"
    if first_rank_by_key.get(key) != item.get("rank"):
        return "duplicate_dedupe_key"
    if key not in selected_keys:
        return "below_max_papers_cutoff"
    return ""


def _score_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    years = [_int_or_zero(card.get("year")) for card in cards if _int_or_zero(card.get("year")) > 0]
    min_year = min(years) if years else 0
    scored = []
    for card in cards:
        year = _int_or_zero(card.get("year"))
        recency = min(float(year - min_year), 10.0) / 2.0 if year and min_year else 0.0
        citation_count = _int_or_zero(card.get("citation_count"))
        method_keyword_score = float(_method_keyword_score(card))
        query_match_score = float(_query_match_score(card))
        source_priority = SOURCE_PRIORITY.get(str(card.get("source") or ""), 1.0)
        total_score = (
            recency
            + min(citation_count, 1000) / 100.0
            + method_keyword_score
            + query_match_score
            + source_priority
        )
        scored.append(
            {
                "card": card,
                "components": {
                    "recency": recency,
                    "citation_count": citation_count,
                    "method_keyword_score": method_keyword_score,
                    "query_match_score": query_match_score,
                    "source_priority": source_priority,
                    "total_score": round(total_score, 6),
                },
            }
        )
    return scored


def _score_sort_key(item: dict[str, Any]) -> tuple[float, int, str, str]:
    card = item["card"]
    components = item["components"]
    return (
        -float(components["total_score"]),
        -_int_or_zero(card.get("year")),
        str(card.get("title") or "").lower(),
        str(card.get("paper_id") or ""),
    )


def _method_keyword_score(card: dict[str, Any]) -> int:
    haystack = f"{card.get('title', '')} {card.get('abstract', '')}".lower()
    return sum(1 for keyword in METHOD_KEYWORDS if keyword in haystack)


def _query_match_score(card: dict[str, Any]) -> int:
    haystack = f"{card.get('title', '')} {card.get('abstract', '')}".lower()
    terms = _query_terms(str(card.get("search_query") or ""))
    return sum(1 for term in terms if term in haystack)


def _query_terms(query: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9-]{2,}", query.replace("_", " ").lower())
    stop = {
        "and",
        "are",
        "benchmark",
        "design",
        "experimental",
        "framework",
        "method",
        "methods",
        "paper",
        "papers",
        "project",
        "query",
        "retained",
        "search",
        "the",
        "with",
    }
    terms: list[str] = []
    for token in tokens:
        if token in stop or token in terms:
            continue
        terms.append(token)
    return terms[:20]


def _dedupe_key(card: dict[str, Any]) -> str | None:
    return paper_card_dedupe_key(card) or (
        f"paper_id:{card['paper_id']}" if _clean_str(card.get("paper_id")) else None
    )


def _merge_duplicate_card(target: dict[str, Any], duplicate: dict[str, Any]) -> None:
    target["source_trace"] = _merge_source_trace(target.get("source_trace"), duplicate.get("source_trace"))
    target_external = target.get("external_ids") if isinstance(target.get("external_ids"), dict) else {}
    duplicate_external = duplicate.get("external_ids") if isinstance(duplicate.get("external_ids"), dict) else {}
    if duplicate_external:
        target["external_ids"] = {**duplicate_external, **target_external}
    for key in ("citation_count", "reference_count", "influential_citation_count"):
        target_value = _int_or_zero(target.get(key))
        duplicate_value = _int_or_zero(duplicate.get(key))
        if duplicate_value > target_value:
            target[key] = duplicate_value


def _merge_source_trace(existing: Any, additions: Any) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for trace in _trace_items(existing) + _trace_items(additions):
        query_id = _clean_str(trace.get("query_id"))
        source = _clean_str(trace.get("source"))
        paper_id = _clean_str(trace.get("paper_id"))
        source_id = _clean_str(trace.get("source_id"))
        if not query_id or not source:
            continue
        key = (query_id, source, paper_id or "", source_id or "")
        if key in seen:
            continue
        seen.add(key)
        merged.append(
            {
                "query_id": query_id,
                "source": source,
                "paper_id": paper_id,
                "source_id": source_id,
            }
        )
    return sorted(merged, key=lambda item: (item["query_id"], item["source"], item.get("paper_id") or ""))


def _trace_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _score_row(
    *,
    rank: int,
    card: dict[str, Any],
    components: dict[str, Any],
    selected: bool,
    exclusion_reason: str,
    dedupe_key: str | None,
) -> dict[str, Any]:
    return {
        "rank": rank,
        "selected": str(bool(selected)).lower(),
        "exclusion_reason": exclusion_reason,
        "dedupe_key": dedupe_key,
        "paper_id": card.get("paper_id"),
        "query_id": card.get("query_id"),
        "source": card.get("source"),
        "title": card.get("title"),
        "year": card.get("year"),
        "recency": _format_score(components["recency"]),
        "citation_count": components["citation_count"],
        "method_keyword_score": _format_score(components["method_keyword_score"]),
        "query_match_score": _format_score(components["query_match_score"]),
        "source_priority": _format_score(components["source_priority"]),
        "total_score": _format_score(components["total_score"]),
    }


def _write_scores_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SCORE_FIELDS})


def _build_citation_graph(cards: list[dict[str, Any]]) -> dict[str, Any]:
    nodes = []
    edges = []
    for card in cards:
        node = {
            "paper_id": card.get("paper_id"),
            "title": card.get("title"),
            "source": card.get("source"),
            "year": card.get("year"),
            "doi": card.get("doi"),
            "url": card.get("url"),
            "citation_count": card.get("citation_count"),
            "external_ids": card.get("external_ids") if isinstance(card.get("external_ids"), dict) else {},
        }
        nodes.append(node)
        edges.extend(_related_edges(card, field="references", relationship="references"))
        edges.extend(_related_edges(card, field="related_papers", relationship="related"))
        edges.extend(_related_edges(card, field="citations", relationship="cited_by"))
    return {"version": "literature_search_v2", "nodes": nodes, "edges": edges}


def _related_edges(card: dict[str, Any], *, field: str, relationship: str) -> list[dict[str, Any]]:
    related = card.get(field)
    if not isinstance(related, list):
        return []
    edges = []
    for item in related:
        target_id = _related_paper_id(item)
        if target_id:
            edges.append(
                {
                    "source_paper_id": card.get("paper_id"),
                    "target_paper_id": target_id,
                    "relationship": relationship,
                }
            )
    return edges


def _related_paper_id(value: Any) -> str | None:
    if isinstance(value, str):
        return _clean_str(value)
    if isinstance(value, dict):
        for key in ("paper_id", "paperId", "id", "doi", "source_id"):
            clean = _clean_str(value.get(key))
            if clean:
                return clean
    return None


def _format_score(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "0"
    if number.is_integer():
        return str(int(number))
    return f"{number:.6f}".rstrip("0").rstrip(".")


def _int_or_zero(value: Any) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def build_project_literature_context(
    project_dir: str | Path,
    *,
    fallback_domain: str | None = None,
) -> LiteratureContext:
    """Build search context from project, estimand, and data-contract metadata."""
    root = Path(project_dir).expanduser().resolve()
    project = _read_yaml_if_exists(root / "project.yaml")
    estimands = _read_yaml_if_exists(root / "estimands.yaml")
    data_contract = _read_yaml_if_exists(root / "data_contract.yaml")

    project_phrases = _project_context_phrases(project, fallback_domain=fallback_domain)
    estimand_phrases = _estimand_context_phrases(estimands)
    source_hints = _data_contract_source_hints(data_contract)
    project_terms = project_phrases + estimand_phrases + source_hints
    query_terms = (
        project_phrases
        + estimand_phrases
        + _conditional_ph_sensitive_terms(project_terms)
        + source_hints
        + [
            "active learning",
            "experimental design",
            "sequential design",
        ]
    )
    query = _join_query_phrases(query_terms) or _join_query_phrases(
        [fallback_domain, "active learning", "experimental design"]
    )
    relevance = _join_relevance_phrases(project_terms) or "general iterative design-scientist project"
    return LiteratureContext(query=query, relevance=relevance)


def _build_context(project_dir: Path) -> LiteratureContext:
    return build_project_literature_context(project_dir)


def _read_yaml_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return read_yaml(path)


def _project_context_phrases(project: dict[str, Any], *, fallback_domain: str | None) -> list[str]:
    phrases: list[str] = []
    for key in (
        "objective",
        "primary_goal",
        "domain",
        "system",
        "name",
        "goal",
        "project_id",
    ):
        phrases.extend(_text_phrases(project.get(key)))
    target_systems = project.get("target_systems")
    if isinstance(target_systems, list):
        for target_system in target_systems:
            phrases.extend(_text_phrases(target_system))
    constraints = project.get("constraints")
    if isinstance(constraints, dict):
        system_roles = constraints.get("system_roles")
        if isinstance(system_roles, dict):
            for system, role in system_roles.items():
                phrases.extend(_text_phrases(system))
                phrases.extend(_text_phrases(role))
    if fallback_domain:
        phrases.extend(_text_phrases(fallback_domain))
    return _dedupe_phrases(phrases)


def _estimand_context_phrases(estimands: dict[str, Any]) -> list[str]:
    phrases: list[str] = []
    for section in ("primary", "secondary", "guardrails", "estimands", "objectives"):
        phrases.extend(_estimand_item_phrases(estimands.get(section)))
    return _dedupe_phrases(phrases)


def _estimand_item_phrases(value: Any) -> list[str]:
    phrases: list[str] = []
    if isinstance(value, dict):
        for key in ("name", "endpoint", "interpretation", "description", "success_direction"):
            phrases.extend(_text_phrases(value.get(key)))
        for nested in value.values():
            if isinstance(nested, (dict, list)):
                phrases.extend(_estimand_item_phrases(nested))
    elif isinstance(value, list):
        for item in value:
            phrases.extend(_estimand_item_phrases(item))
    return phrases


def _data_contract_source_hints(data_contract: dict[str, Any]) -> list[str]:
    phrases: list[str] = []
    allowed_sources = data_contract.get("allowed_sources")
    if isinstance(allowed_sources, dict):
        for source_name, source in allowed_sources.items():
            phrases.extend(_text_phrases(source_name))
            if isinstance(source, dict):
                phrases.extend(_text_phrases(source.get("evidence_tier")))
    allowed_tables = data_contract.get("allowed_tables")
    if isinstance(allowed_tables, list):
        for table in allowed_tables:
            if not isinstance(table, dict):
                continue
            for key in ("name", "role"):
                phrases.extend(_text_phrases(table.get(key)))
            measurement_types = table.get("measurement_types")
            if isinstance(measurement_types, list):
                for measurement_type in measurement_types:
                    phrases.extend(_text_phrases(measurement_type))
    return _dedupe_phrases(phrases)


def _conditional_ph_sensitive_terms(phrases: list[str]) -> list[str]:
    text = " ".join(phrases).lower()
    has_ph_context = bool(
        re.search(r"\bph\b|ph[-\s]?\d|ph-sensitive|ph-dependent|neutral-ph|acidic-ph", text)
        or "acidic ph" in text
        or "neutral ph" in text
        or "acid binding" in text
        or "neutral binding" in text
    )
    has_antibody_or_protein_context = any(
        term in text
        for term in (
            "antibody",
            "anti-hbsag",
            "hbsag",
            "protein",
            "binding",
        )
    )
    if not has_ph_context or not has_antibody_or_protein_context:
        return []
    return [
        "pH-sensitive antibody protein engineering",
        "pH-dependent binding",
        "neutral pH binding retention",
        "acidic pH release",
    ]


def _text_phrases(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple, set)):
        phrases: list[str] = []
        for item in value:
            phrases.extend(_text_phrases(item))
        return phrases
    if isinstance(value, dict):
        return []
    text = str(value).replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip(" .;,")
    return [text] if text else []


def _dedupe_phrases(phrases: list[str]) -> list[str]:
    kept: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        clean = _clean_str(phrase)
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append(clean)
    return kept


def _join_query_phrases(phrases: list[str | None], *, max_phrases: int = 32) -> str:
    clean_phrases = _dedupe_phrases([phrase for phrase in phrases if phrase])
    return " ".join(clean_phrases[:max_phrases])


def _join_relevance_phrases(phrases: list[str]) -> str:
    clean_phrases = _dedupe_phrases(phrases)
    return "; ".join(clean_phrases[:12])


def _safe_error_text(exc: BaseException) -> str:
    return _redact_sensitive_text(str(exc))


def _redact_sensitive_text(text: str) -> str:
    redacted = text
    for key, value in os.environ.items():
        if not value or len(value) < 4:
            continue
        key_upper = key.upper()
        if not any(marker in key_upper for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            continue
        redacted = redacted.replace(value, "[redacted]")
    redacted = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[redacted]", redacted)
    return redacted


def _compact_query(text: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9-]{2,}", text)
    stop = {
        "and",
        "for",
        "over",
        "the",
        "that",
        "where",
        "with",
        "avoid",
        "obvious",
        "project",
    }
    kept = []
    for token in tokens:
        lower = token.lower()
        if lower in stop or lower in kept:
            continue
        kept.append(lower)
    return " ".join(kept[:12])
