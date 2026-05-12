from __future__ import annotations

import csv
import json
from pathlib import Path

import httpx
import pytest

from design_scientist.io import read_json
from design_scientist.literature_pipeline import run_literature_search
import design_scientist.literature_pipeline as literature_pipeline
import design_scientist.literature_sources as literature_sources
from design_scientist.literature_sources import (
    LiteratureCacheError,
    LiteratureSourceTemporarilyUnavailable,
    LiteratureContext,
    default_fixture_dir,
    dedupe_paper_cards,
    search_arxiv,
    search_biorxiv,
    search_pubmed,
    search_semantic_scholar,
)


FIXTURES = Path(__file__).parent / "fixtures" / "literature"


def test_offline_default_fixture_dir_is_independent_of_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DESIGN_SCIENTIST_LITERATURE_FIXTURES", raising=False)
    monkeypatch.chdir(tmp_path)

    fixture_dir = default_fixture_dir()
    cards = search_pubmed(
        LiteratureContext(query="protein engineering antibody", relevance="test"),
        max_papers=5,
        cache_dir=tmp_path / "cache",
        offline_fixtures=True,
    )

    assert fixture_dir == FIXTURES.resolve()
    assert len(cards) == 2


def test_offline_adapters_standardize_and_dedupe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("S2_API_KEY", raising=False)
    context = LiteratureContext(query="protein engineering antibody", relevance="test project")
    cache = tmp_path / "cache"

    cards = []
    cards.extend(search_pubmed(context, max_papers=10, cache_dir=cache, offline_fixtures=True, fixture_dir=FIXTURES))
    cards.extend(search_biorxiv(context, max_papers=10, cache_dir=cache, offline_fixtures=True, fixture_dir=FIXTURES))
    cards.extend(search_arxiv(context, max_papers=10, cache_dir=cache, offline_fixtures=True, fixture_dir=FIXTURES))
    cards.extend(
        search_semantic_scholar(
            context,
            max_papers=10,
            cache_dir=cache,
            offline_fixtures=True,
            fixture_dir=FIXTURES,
        )
    )

    unique = dedupe_paper_cards(cards)
    assert len(cards) > len(unique)
    assert {card["source"] for card in unique} >= {"pubmed", "biorxiv", "arxiv"}
    assert all(card["paper_id"] and card["title"] for card in unique)
    assert sum(card["doi"] == "10.1101/2024.01.01.123456" for card in unique) == 1


def test_empty_and_malformed_payloads_return_empty(tmp_path: Path) -> None:
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "pubmed_esearch.json").write_text("{bad json", encoding="utf-8")
    (bad / "pubmed_efetch.xml").write_text("<not xml", encoding="utf-8")
    (bad / "biorxiv.json").write_text(json.dumps({"collection": [None, {"doi": "", "title": ""}]}), encoding="utf-8")
    (bad / "arxiv.xml").write_text("<feed><entry>", encoding="utf-8")
    (bad / "semantic_scholar.json").write_text(json.dumps({"data": [{"paperId": None}]}), encoding="utf-8")
    context = LiteratureContext(query="anything", relevance="test")

    assert search_pubmed(context, max_papers=5, cache_dir=tmp_path / "cache", offline_fixtures=True, fixture_dir=bad) == []
    assert search_biorxiv(context, max_papers=5, cache_dir=tmp_path / "cache", offline_fixtures=True, fixture_dir=bad) == []
    assert search_arxiv(context, max_papers=5, cache_dir=tmp_path / "cache", offline_fixtures=True, fixture_dir=bad) == []
    assert (
        search_semantic_scholar(context, max_papers=5, cache_dir=tmp_path / "cache", offline_fixtures=True, fixture_dir=bad)
        == []
    )


def test_arxiv_live_adapter_uses_fielded_query_headers_and_retries_rate_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import design_scientist.literature_sources as literature_sources

    class RateLimitedResponse:
        text = ""

        def raise_for_status(self) -> None:
            request = httpx.Request("GET", "https://export.arxiv.org/api/query")
            response = httpx.Response(429, request=request, headers={"Retry-After": "0"})
            raise httpx.HTTPStatusError("rate limited", request=request, response=response)

    class GoodResponse:
        text = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <title>Active learning for protein design</title>
    <summary>Sequential design for protein variants.</summary>
    <published>2024-01-01T00:00:00Z</published>
    <author><name>Ada Lovelace</name></author>
  </entry>
</feed>
"""

        def raise_for_status(self) -> None:
            return None

    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def get(self, url: str, *, params: dict | None = None, headers: dict | None = None):
            self.calls.append({"url": url, "params": params, "headers": headers})
            return RateLimitedResponse() if len(self.calls) == 1 else GoodResponse()

    sleep_calls: list[float] = []
    monkeypatch.setattr(literature_sources.time, "sleep", lambda seconds: sleep_calls.append(seconds))
    monkeypatch.setattr(literature_sources, "_LAST_ARXIV_REQUEST_AT", 0.0)

    client = FakeClient()
    cards = search_arxiv(
        LiteratureContext(
            query="protein_variant_design active learning experimental design method benchmark",
            relevance="test",
        ),
        max_papers=3,
        cache_dir=tmp_path / "cache",
        client=client,
    )

    assert len(cards) == 1
    assert len(client.calls) == 2
    params = client.calls[0]["params"]
    headers = client.calls[0]["headers"]
    assert "all:protein" in params["search_query"]
    assert " OR " in params["search_query"]
    assert headers["User-Agent"].startswith("Design-Scientist/")
    assert sleep_calls


def test_semantic_scholar_skips_without_key_and_reads_fixture_with_key_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = LiteratureContext(query="protein engineering antibody", relevance="test")
    monkeypatch.delenv("S2_API_KEY", raising=False)
    assert search_semantic_scholar(context, max_papers=5, cache_dir=tmp_path / "cache") == []

    cards = search_semantic_scholar(
        context,
        max_papers=5,
        cache_dir=tmp_path / "cache",
        offline_fixtures=True,
        fixture_dir=FIXTURES,
    )
    assert len(cards) == 2
    assert cards[0]["external_ids"]["paperId"] == "S2DUPLICATE"


def test_semantic_scholar_with_key_sends_x_api_key_and_enriches_citation_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "data": [
                    {
                        "paperId": "S2KEYED",
                        "title": "Active learning for protein design",
                        "abstract": "Bayesian optimization for protein engineering.",
                        "authors": [{"name": "Ada Lovelace"}],
                        "year": 2025,
                        "venue": "Test Venue",
                        "url": "https://example.test/s2keyed",
                        "externalIds": {"DOI": "10.1000/keyed"},
                        "publicationDate": "2025-01-02",
                        "citationCount": 17,
                        "referenceCount": 3,
                        "influentialCitationCount": 4,
                        "citations": [{"paperId": "CITEDBY1", "title": "A citing paper"}],
                        "references": [{"paperId": "REF1", "title": "A reference paper"}],
                    }
                ]
            }

    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def get(self, url: str, *, params: dict | None = None, headers: dict | None = None) -> FakeResponse:
            self.calls.append({"url": url, "params": params, "headers": headers})
            return FakeResponse()

    monkeypatch.setenv("S2_API_KEY", "test-secret")
    client = FakeClient()
    cards = search_semantic_scholar(
        LiteratureContext(query="protein design", relevance="test"),
        max_papers=5,
        cache_dir=tmp_path / "cache",
        client=client,
    )

    assert client.calls
    assert client.calls[0]["headers"] == {"x-api-key": "test-secret"}
    assert "citations.paperId" in client.calls[0]["params"]["fields"]
    assert "references.paperId" in client.calls[0]["params"]["fields"]
    assert cards[0]["citation_count"] == 17
    assert cards[0]["reference_count"] == 3
    assert cards[0]["influential_citation_count"] == 4
    assert cards[0]["citations"][0]["paper_id"] == "CITEDBY1"
    assert cards[0]["references"][0]["paper_id"] == "REF1"


def test_run_literature_search_writes_paper_cards_and_raw_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DESIGN_SCIENTIST_LITERATURE_FIXTURES", str(FIXTURES))
    project = tmp_path / "project"
    project.mkdir()
    (project / "project.yaml").write_text(
        "name: anti-HBsAg pH-dependent antibody design\n"
        "goal: Improve antibody protein engineering over rounds\n"
        "target_systems:\n"
        "- sdAb\n",
        encoding="utf-8",
    )

    out = run_literature_search(project, max_papers=4, offline_fixtures=True)
    cards = read_json(out)

    assert out == project / "framework" / "paper_cards.json"
    assert len(cards) == 4
    assert all(card["review_status"] == "source_imported_unreviewed" for card in cards)
    raw_cache = project / "framework" / "cache" / "literature_raw"
    assert list(raw_cache.glob("project_query/pubmed/pubmed_esearch.json"))
    assert list(raw_cache.glob("project_query/biorxiv/biorxiv.json"))
    assert list(raw_cache.glob("project_query/arxiv/arxiv.xml"))
    assert list(raw_cache.glob("project_query/semantic_scholar/semantic_scholar.json"))


def test_run_literature_search_rejects_non_positive_max_papers(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()

    for max_papers in (0, -1):
        with pytest.raises(ValueError, match="max_papers must be positive"):
            run_literature_search(project, max_papers=max_papers, offline_fixtures=True, sources=["pubmed"])


def test_missing_offline_fixture_records_source_error_not_ok_empty_cards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_fixtures = tmp_path / "missing-fixtures"
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("DESIGN_SCIENTIST_LITERATURE_FIXTURES", str(missing_fixtures))

    out = run_literature_search(project, max_papers=5, offline_fixtures=True, sources=["pubmed"])
    cards = read_json(out)
    trace = read_json(project / "framework" / "literature_search_trace.json")
    errors = read_json(project / "framework" / "cache" / "literature_raw" / "source_errors.json")

    assert cards == []
    assert trace["events"][0]["event"] == "source_error"
    assert trace["events"][0]["status"] == "error"
    assert trace["events"][0]["source"] == "pubmed"
    assert trace["events"][0]["cache_path"] == "framework/cache/literature_raw/project_query/pubmed"
    assert "missing literature fixture" in trace["events"][0]["reason"].lower()
    assert errors["errors"][0]["cache_path"] == trace["events"][0]["cache_path"]
    assert errors["errors"][0]["error_type"] == "FileNotFoundError"


@pytest.mark.parametrize(
    ("source", "cache_name", "payload"),
    [
        ("pubmed", "pubmed_efetch.xml", "<PubmedArticleSet><PubmedArticle>"),
        ("arxiv", "arxiv.xml", "<feed><entry>"),
    ],
)
def test_corrupt_pubmed_arxiv_xml_cache_is_traced_as_cache_error_not_ok_empty_cards(
    tmp_path: Path,
    source: str,
    cache_name: str,
    payload: str,
) -> None:
    project = tmp_path / "project"
    source_cache = project / "framework" / "cache" / "literature_raw" / "project_query" / source
    source_cache.mkdir(parents=True)
    (source_cache / cache_name).write_text(payload, encoding="utf-8")

    out = run_literature_search(project, max_papers=5, sources=[source])
    cards = read_json(out)
    trace = read_json(project / "framework" / "literature_search_trace.json")
    errors = read_json(project / "framework" / "cache" / "literature_raw" / "source_errors.json")

    assert cards == []
    assert trace["events"][0]["event"] == "source_error"
    assert trace["events"][0]["status"] == "cache_error"
    assert trace["events"][0]["source"] == source
    assert trace["events"][0]["cache_hit"] is False
    assert trace["events"][0]["cache_error"] is True
    assert trace["events"][0]["network_fetch"] is False
    assert "xml cache" in trace["events"][0]["reason"].lower()
    assert errors["errors"][0]["source"] == source
    assert errors["errors"][0]["error_type"] == "LiteratureCacheError"


def test_run_literature_search_isolates_raw_cache_by_query_and_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DESIGN_SCIENTIST_LITERATURE_FIXTURES", str(FIXTURES))
    project = tmp_path / "project"
    framework = project / "framework"
    framework.mkdir(parents=True)
    (framework / "literature_queries.yaml").write_text(
        "queries:\n"
        "- query_id: foundations\n"
        "  query: protein engineering antibody\n"
        "  sources: [pubmed]\n"
        "- query_id: applications\n"
        "  query: antibody active design\n"
        "  sources: [pubmed]\n",
        encoding="utf-8",
    )

    run_literature_search(project, max_papers=5, offline_fixtures=True)
    trace = read_json(framework / "literature_search_trace.json")
    pubmed_events = [event for event in trace["events"] if event["source"] == "pubmed"]

    assert len(pubmed_events) == 2
    assert {event["status"] for event in pubmed_events} == {"ok"}
    assert all(event["raw_count"] == event["normalized_count"] == 2 for event in pubmed_events)
    cache_paths = [project / event["cache_path"] for event in pubmed_events]
    assert len({path.as_posix() for path in cache_paths}) == 2
    for cache_path in cache_paths:
        assert (cache_path / "pubmed_esearch.json").exists()
        assert (cache_path / "pubmed_efetch.xml").exists()
    assert all("cache_files" in event for event in pubmed_events)


def test_run_literature_search_writes_v2_artifacts_from_query_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("S2_API_KEY", raising=False)
    monkeypatch.setenv("DESIGN_SCIENTIST_LITERATURE_FIXTURES", str(FIXTURES))
    project = tmp_path / "project"
    framework = project / "framework"
    framework.mkdir(parents=True)
    (project / "project.yaml").write_text(
        "name: anti-HBsAg pH-dependent antibody design\n"
        "goal: benchmark active antibody design policies\n",
        encoding="utf-8",
    )
    (framework / "literature_queries.yaml").write_text(
        "queries:\n"
        "- query_id: method_foundations\n"
        "  query: protein engineering antibody\n"
        "  purpose: find reusable method families\n"
        "  sources: [pubmed, semantic_scholar]\n"
        "- query_id: preprint_methods\n"
        "  query: antibody active design\n"
        "  purpose: find recent preprints\n"
        "  sources: [arxiv]\n",
        encoding="utf-8",
    )

    out = run_literature_search(project, max_papers=5, offline_fixtures=True)
    cards = read_json(out)
    plan = read_json(framework / "literature_search_plan.json")
    trace = read_json(framework / "literature_search_trace.json")
    graph = read_json(framework / "citation_graph.json")
    score_rows = _read_score_rows(framework / "paper_scores.csv")

    assert [query["query_id"] for query in plan["queries"]] == ["method_foundations", "preprint_methods"]
    assert plan["queries"][0]["sources"] == ["pubmed", "semantic_scholar"]
    assert plan["queries"][1]["sources"] == ["arxiv"]

    result_events = [event for event in trace["events"] if event["event"] == "source_result"]
    assert {(event["query_id"], event["source"]) for event in result_events} == {
        ("method_foundations", "pubmed"),
        ("method_foundations", "semantic_scholar"),
        ("preprint_methods", "arxiv"),
    }
    assert all(event["paper_ids"] for event in result_events)
    assert cards
    for card in cards:
        assert card["query_id"]
        assert card["source_trace"]
        assert all(trace_item["query_id"] and trace_item["source"] for trace_item in card["source_trace"])
        assert any(
            card["paper_id"] in event["paper_ids"]
            for trace_item in card["source_trace"]
            for event in result_events
            if event["query_id"] == trace_item["query_id"] and event["source"] == trace_item["source"]
        )

    selected_score_rows = [row for row in score_rows if row["selected"] == "true"]
    assert {row["paper_id"] for row in selected_score_rows} == {card["paper_id"] for card in cards}
    assert {card["paper_id"] for card in cards}.issubset({row["paper_id"] for row in score_rows})
    for field in ("recency", "citation_count", "method_keyword_score", "source_priority", "total_score"):
        assert field in score_rows[0]
    assert graph["nodes"]
    assert {node["paper_id"] for node in graph["nodes"]} == {card["paper_id"] for card in cards}
    assert isinstance(graph["edges"], list)


def test_semantic_scholar_pipeline_skips_without_key_but_reads_offline_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("S2_API_KEY", raising=False)
    project = tmp_path / "project"
    project.mkdir()

    run_literature_search(project, max_papers=5, sources=["semantic_scholar"])
    online_trace = read_json(project / "framework" / "literature_search_trace.json")
    assert len(online_trace["events"]) == 1
    skip_event = online_trace["events"][0]
    assert skip_event["event"] == "source_skip"
    assert skip_event["status"] == "skipped"
    assert skip_event["query_id"] == "project_query"
    assert skip_event["source"] == "semantic_scholar"
    assert skip_event["raw_count"] == 0
    assert skip_event["normalized_count"] == 0
    assert skip_event["reason"] == "missing_s2_api_key"
    assert skip_event["cache_path"] == "framework/cache/literature_raw/project_query/semantic_scholar"
    assert set(skip_event) >= {
        "event",
        "status",
        "query_id",
        "source",
        "raw_count",
        "normalized_count",
        "reason",
        "cache_path",
    }

    monkeypatch.setenv("DESIGN_SCIENTIST_LITERATURE_FIXTURES", str(FIXTURES))
    run_literature_search(project, max_papers=5, offline_fixtures=True, sources=["semantic_scholar"])
    offline_trace = read_json(project / "framework" / "literature_search_trace.json")
    cards = read_json(project / "framework" / "paper_cards.json")

    assert len(cards) == 2
    assert [event["event"] for event in offline_trace["events"]] == ["source_result"]
    assert offline_trace["events"][0]["source"] == "semantic_scholar"
    assert offline_trace["events"][0]["status"] == "ok"
    assert offline_trace["events"][0]["raw_count"] == 2
    assert offline_trace["events"][0]["normalized_count"] == 2
    assert "cache_path" in offline_trace["events"][0]


def test_pubmed_live_adapter_reads_existing_raw_cache_before_network_calls(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "pubmed_esearch.json").write_text(
        (FIXTURES / "pubmed_esearch.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (cache / "pubmed_efetch.xml").write_text(
        (FIXTURES / "pubmed_efetch.xml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    class FailingClient:
        def get(self, *args, **kwargs):
            raise AssertionError("network should not be called when pubmed raw cache exists")

    cards = search_pubmed(
        LiteratureContext(query="protein engineering antibody", relevance="test"),
        max_papers=5,
        cache_dir=cache,
        client=FailingClient(),
    )

    assert len(cards) == 2


@pytest.mark.parametrize("failure_mode", ["corrupt", "unreadable"])
def test_biorxiv_live_adapter_treats_existing_bad_json_cache_as_cache_error_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_mode: str
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    cache_path = cache / "biorxiv.json"
    cache_path.write_text("{bad json" if failure_mode == "corrupt" else "{}", encoding="utf-8")
    original_read_text = Path.read_text
    network_calls: list[str] = []

    if failure_mode == "unreadable":
        def unreadable_cache(self: Path, *args, **kwargs):
            if self == cache_path:
                raise OSError("permission denied")
            return original_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", unreadable_cache)

    def fake_get_json(*args, **kwargs):
        network_calls.append("biorxiv")
        return {"collection": []}

    monkeypatch.setattr(literature_sources, "_get_json", fake_get_json)

    with pytest.raises(LiteratureCacheError):
        search_biorxiv(
            LiteratureContext(query="protein engineering antibody", relevance="test"),
            max_papers=5,
            cache_dir=cache,
        )

    assert network_calls == []


def test_run_literature_search_does_not_call_live_fetch_when_existing_xml_cache_is_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    source_cache = project / "framework" / "cache" / "literature_raw" / "project_query" / "pubmed"
    source_cache.mkdir(parents=True)
    cache_path = source_cache / "pubmed_efetch.xml"
    cache_path.write_text((FIXTURES / "pubmed_efetch.xml").read_text(encoding="utf-8"), encoding="utf-8")
    original_read_text = Path.read_text
    network_calls: list[str] = []

    def unreadable_cache(self: Path, *args, **kwargs):
        if self == cache_path:
            raise OSError("permission denied")
        return original_read_text(self, *args, **kwargs)

    def fake_get_json(*args, **kwargs):
        network_calls.append("json")
        return {"esearchresult": {"idlist": []}}

    def fake_get_text(*args, **kwargs):
        network_calls.append("text")
        return ""

    monkeypatch.setattr(Path, "read_text", unreadable_cache)
    monkeypatch.setattr(literature_sources, "_get_json", fake_get_json)
    monkeypatch.setattr(literature_sources, "_get_text", fake_get_text)

    out = run_literature_search(project, max_papers=5, sources=["pubmed"])
    cards = read_json(out)
    trace = read_json(project / "framework" / "literature_search_trace.json")
    errors = read_json(project / "framework" / "cache" / "literature_raw" / "source_errors.json")
    event = trace["events"][0]

    assert cards == []
    assert network_calls == []
    assert event["event"] == "source_error"
    assert event["status"] == "cache_error"
    assert event["cache_status"] == "cache_error"
    assert event["cache_hit"] is False
    assert event["cache_error"] is True
    assert event["network_fetch"] is False
    assert "permission denied" in event["reason"].lower()
    assert errors["errors"][0]["error_type"] == "LiteratureCacheError"


def test_run_literature_search_records_corrupt_json_cache_error_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    source_cache = project / "framework" / "cache" / "literature_raw" / "project_query" / "biorxiv"
    source_cache.mkdir(parents=True)
    (source_cache / "biorxiv.json").write_text("{bad json", encoding="utf-8")
    network_calls: list[str] = []

    def fake_get_json(*args, **kwargs):
        network_calls.append("biorxiv")
        return {"collection": []}

    monkeypatch.setattr(literature_sources, "_get_json", fake_get_json)

    out = run_literature_search(project, max_papers=5, sources=["biorxiv"])
    cards = read_json(out)
    trace = read_json(project / "framework" / "literature_search_trace.json")
    event = trace["events"][0]

    assert cards == []
    assert network_calls == []
    assert event["event"] == "source_error"
    assert event["status"] == "cache_error"
    assert event["cache_status"] == "cache_error"
    assert event["network_fetch"] is False
    assert event["cache_error"] is True


def test_biorxiv_live_pipeline_reuses_raw_cache_and_records_cache_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    source_cache = project / "framework" / "cache" / "literature_raw" / "project_query" / "biorxiv"
    source_cache.mkdir(parents=True)
    (source_cache / "biorxiv.json").write_text(
        json.dumps(
            {
                "collection": [
                    {
                        "doi": "10.1101/live-cache",
                        "title": "Active learning protein engineering from cached preprints",
                        "abstract": "A cached bioRxiv record for protein engineering.",
                        "authors": "Ada Lovelace",
                        "date": "2025-01-01",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    def fail_network(*args, **kwargs):
        raise AssertionError("network should not be called when biorxiv raw cache exists")

    monkeypatch.setattr(literature_sources, "_get_json", fail_network)

    run_literature_search(project, max_papers=5, sources=["biorxiv"])
    cards = read_json(project / "framework" / "paper_cards.json")
    trace = read_json(project / "framework" / "literature_search_trace.json")
    event = trace["events"][0]

    assert [card["doi"] for card in cards] == ["10.1101/live-cache"]
    assert event["status"] == "ok"
    assert event["cache_hit"] is True
    assert event["cache_miss"] is False
    assert event["network_fetch"] is False


def test_biorxiv_live_pipeline_records_cache_miss_network_fetch_and_recent_feed_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"

    def fake_get_json(*args, **kwargs):
        return {
            "collection": [
                {
                    "doi": "10.1101/live-fetch",
                    "title": "Active learning protein engineering from recent feed",
                    "abstract": "A live bioRxiv record for protein engineering.",
                    "authors": "Ada Lovelace",
                    "date": "2025-01-02",
                }
            ]
        }

    monkeypatch.setattr(literature_sources, "_get_json", fake_get_json)

    run_literature_search(project, max_papers=5, sources=["biorxiv"])
    trace = read_json(project / "framework" / "literature_search_trace.json")
    event = trace["events"][0]

    assert event["status"] == "ok"
    assert event["cache_hit"] is False
    assert event["cache_miss"] is True
    assert event["network_fetch"] is True
    assert event["behavior"] == "recent_feed_scan"


def test_run_literature_search_records_source_errors_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()

    def good_adapter(*args, **kwargs):
        return [
            {
                "paper_id": "good:1",
                "source": "good",
                "source_id": "1",
                "title": "Good paper",
                "abstract": "active learning",
                "review_status": "source_imported_unreviewed",
            }
        ]

    def bad_adapter(*args, **kwargs):
        raise RuntimeError("rate limited")

    monkeypatch.setitem(literature_pipeline.SOURCE_ADAPTERS, "good", good_adapter)
    monkeypatch.setitem(literature_pipeline.SOURCE_ADAPTERS, "bad", bad_adapter)

    out = run_literature_search(project, max_papers=5, sources=["bad", "good"])
    cards = read_json(out)
    errors = read_json(project / "framework" / "cache" / "literature_raw" / "source_errors.json")
    trace = read_json(project / "framework" / "literature_search_trace.json")

    assert [card["paper_id"] for card in cards] == ["good:1"]
    assert errors["errors"][0]["source"] == "bad"
    assert errors["errors"][0]["error_type"] == "RuntimeError"
    error_events = [event for event in trace["events"] if event["event"] == "source_error"]
    assert error_events[0]["query_id"] == "project_query"
    assert error_events[0]["source"] == "bad"
    assert error_events[0]["status"] == "error"
    assert error_events[0]["reason"] == "rate limited"
    assert error_events[0]["error_type"] == "RuntimeError"


def test_run_literature_search_circuit_breaks_arxiv_after_live_source_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    framework = project / "framework"
    framework.mkdir(parents=True)
    (framework / "literature_queries.yaml").write_text(
        "queries:\n"
        "- query_id: first\n"
        "  query: protein design\n"
        "  sources: [arxiv]\n"
        "- query_id: second\n"
        "  query: active learning\n"
        "  sources: [arxiv]\n",
        encoding="utf-8",
    )

    def bad_arxiv(*args, **kwargs):
        raise TimeoutError("arxiv timeout")

    monkeypatch.setitem(literature_pipeline.SOURCE_ADAPTERS, "arxiv", bad_arxiv)

    run_literature_search(project, max_papers=5, sources=["arxiv"])
    trace = read_json(framework / "literature_search_trace.json")

    assert [event["event"] for event in trace["events"]] == ["source_error", "source_skip"]
    assert trace["events"][0]["query_id"] == "first"
    assert trace["events"][0]["source"] == "arxiv"
    assert trace["events"][1]["query_id"] == "second"
    assert trace["events"][1]["source"] == "arxiv"
    assert trace["events"][1]["reason"] == "previous_arxiv_error"


def test_run_literature_search_records_retryable_arxiv_failure_as_degraded_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    framework = project / "framework"
    framework.mkdir(parents=True)
    (framework / "literature_queries.yaml").write_text(
        "queries:\n"
        "- query_id: first\n"
        "  query: protein design\n"
        "  sources: [arxiv]\n",
        encoding="utf-8",
    )

    def degraded_arxiv(*args, **kwargs):
        raise LiteratureSourceTemporarilyUnavailable("arxiv temporarily unavailable: 429")

    monkeypatch.setitem(literature_pipeline.SOURCE_ADAPTERS, "arxiv", degraded_arxiv)

    run_literature_search(project, max_papers=5, sources=["arxiv"])
    trace = read_json(framework / "literature_search_trace.json")

    assert trace["summary"]["source_errors"] == 0
    assert trace["summary"]["source_skips"] == 1
    assert trace["events"][0]["event"] == "source_skip"
    assert trace["events"][0]["status"] == "degraded"
    assert trace["events"][0]["reason"] == "arxiv temporarily unavailable: 429"


def test_run_literature_search_allows_multiple_arxiv_live_requests_until_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    framework = project / "framework"
    framework.mkdir(parents=True)
    (framework / "literature_queries.yaml").write_text(
        "queries:\n"
        "- query_id: first\n"
        "  query: protein design\n"
        "  sources: [arxiv]\n"
        "- query_id: second\n"
        "  query: active learning\n"
        "  sources: [arxiv]\n",
        encoding="utf-8",
    )

    calls = 0

    def good_arxiv(*args, **kwargs):
        nonlocal calls
        calls += 1
        return [
            {
                "paper_id": "arxiv:2401.00001v1",
                "source": "arxiv",
                "source_id": "2401.00001v1",
                "title": "Active learning for protein design",
                "abstract": "Sequential protein design.",
                "review_status": "source_imported_unreviewed",
            }
        ]

    monkeypatch.setitem(literature_pipeline.SOURCE_ADAPTERS, "arxiv", good_arxiv)

    run_literature_search(project, max_papers=5, sources=["arxiv"])
    trace = read_json(framework / "literature_search_trace.json")

    assert calls == 2
    assert [event["event"] for event in trace["events"]] == ["source_result", "source_result"]


def test_run_literature_search_deduplicates_before_final_limit_and_merges_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()

    def first_adapter(*args, **kwargs):
        return [
            {
                "paper_id": "first:1",
                "source": "first",
                "source_id": "1",
                "title": "Active learning for protein panels",
                "abstract": "active learning acquisition for protein engineering",
                "year": 2020,
                "doi": "10.1234/duplicate",
                "citation_count": 2,
                "review_status": "source_imported_unreviewed",
            }
        ]

    def second_adapter(*args, **kwargs):
        return [
            {
                "paper_id": "second:2",
                "source": "second",
                "source_id": "2",
                "title": "Active learning for protein panels",
                "abstract": "active learning acquisition for protein engineering",
                "year": 2024,
                "doi": "10.1234/duplicate",
                "citation_count": 20,
                "review_status": "source_imported_unreviewed",
            },
            {
                "paper_id": "second:3",
                "source": "second",
                "source_id": "3",
                "title": "Matched contrast design for variants",
                "abstract": "matched contrast design for variant panels",
                "year": 2023,
                "doi": "10.1234/unique",
                "citation_count": 5,
                "review_status": "source_imported_unreviewed",
            },
        ]

    monkeypatch.setitem(literature_pipeline.SOURCE_ADAPTERS, "first", first_adapter)
    monkeypatch.setitem(literature_pipeline.SOURCE_ADAPTERS, "second", second_adapter)

    run_literature_search(project, max_papers=2, sources=["first", "second"])
    cards = read_json(project / "framework" / "paper_cards.json")
    score_rows = _read_score_rows(project / "framework" / "paper_scores.csv")

    assert [card["doi"] for card in cards] == ["10.1234/duplicate", "10.1234/unique"]
    assert len(cards[0]["source_trace"]) == 2
    assert {trace["source"] for trace in cards[0]["source_trace"]} == {"first", "second"}
    assert {row["paper_id"] for row in score_rows} == {"second:2", "second:3", "first:1"}
    selected_rows = [row for row in score_rows if row["selected"] == "true"]
    assert [row["paper_id"] for row in selected_rows] == [card["paper_id"] for card in cards]


def test_paper_scores_include_all_ranked_candidates_with_dedupe_and_selection_reasons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()

    def first_adapter(*args, **kwargs):
        return [
            {
                "paper_id": "first:1",
                "source": "first",
                "source_id": "1",
                "title": "Active learning for protein panels",
                "abstract": "active learning acquisition for protein engineering",
                "year": 2020,
                "doi": "10.1234/duplicate",
                "citation_count": 0,
                "review_status": "source_imported_unreviewed",
            }
        ]

    def second_adapter(*args, **kwargs):
        return [
            {
                "paper_id": "second:2",
                "source": "second",
                "source_id": "2",
                "title": "Active learning for protein panels",
                "abstract": "active learning acquisition for protein engineering",
                "year": 2024,
                "doi": "10.1234/duplicate",
                "citation_count": 20,
                "review_status": "source_imported_unreviewed",
            },
            {
                "paper_id": "second:3",
                "source": "second",
                "source_id": "3",
                "title": "Matched contrast design for variants",
                "abstract": "matched contrast design for variant panels",
                "year": 2023,
                "doi": "10.1234/unique",
                "citation_count": 5,
                "review_status": "source_imported_unreviewed",
            },
        ]

    monkeypatch.setitem(literature_pipeline.SOURCE_ADAPTERS, "first", first_adapter)
    monkeypatch.setitem(literature_pipeline.SOURCE_ADAPTERS, "second", second_adapter)

    run_literature_search(project, max_papers=1, sources=["first", "second"])
    cards = read_json(project / "framework" / "paper_cards.json")
    score_rows = _read_score_rows(project / "framework" / "paper_scores.csv")

    assert [card["paper_id"] for card in cards] == ["second:2"]
    assert [row["paper_id"] for row in score_rows] == ["second:2", "second:3", "first:1"]
    assert [row["rank"] for row in score_rows] == ["1", "2", "3"]
    assert score_rows[0]["selected"] == "true"
    assert score_rows[0]["exclusion_reason"] == ""
    assert score_rows[0]["dedupe_key"] == "doi:10.1234/duplicate"
    assert score_rows[1]["selected"] == "false"
    assert score_rows[1]["exclusion_reason"] == "below_max_papers_cutoff"
    assert score_rows[2]["selected"] == "false"
    assert score_rows[2]["exclusion_reason"] == "duplicate_dedupe_key"
    for field in ("recency", "citation_count", "method_keyword_score", "source_priority", "total_score"):
        assert score_rows[0][field] != ""


def _read_score_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
