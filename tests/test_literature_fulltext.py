from __future__ import annotations

import json
from pathlib import Path

import pytest

from design_scientist.literature_fulltext import build_literature_corpus


FIXTURES = Path(__file__).parent / "fixtures" / "literature_fulltext"
REQUIRED_RECORD_FIELDS = {
    "paper_id",
    "source",
    "title",
    "abstract_or_summary",
    "fulltext_status",
    "text",
    "text_source",
    "chunks",
    "provenance",
}


def test_fixture_xml_and_html_are_extracted_into_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DESIGN_SCIENTIST_FULLTEXT_FIXTURES", str(FIXTURES))
    project = _project_with_cards(
        tmp_path,
        [
            {
                "paper_id": "pmc:PMC12345",
                "source": "pubmed",
                "source_id": "PMC12345",
                "title": "Active allocation for protein design",
                "abstract": "Metadata abstract about Bayesian optimization.",
                "doi": "10.1000/xml-fixture",
                "url": "https://pmc.example/articles/PMC12345/",
                "year": 2025,
            },
            {
                "paper_id": "s2:HTML123",
                "source": "semantic_scholar",
                "source_id": "HTML123",
                "title": "Matched contrast panels",
                "abstract": "Abstract for contrast lattice design.",
                "doi": "10.1000/html-paper",
                "url": "https://example.test/html-paper",
                "year": 2024,
            },
        ],
    )

    corpus_path = build_literature_corpus(project, offline_fixtures=True)

    assert corpus_path == project / "framework" / "literature_corpus.jsonl"
    records = _read_jsonl(corpus_path)
    assert len(records) == 2
    assert all(REQUIRED_RECORD_FIELDS <= record.keys() for record in records)

    by_id = {record["paper_id"]: record for record in records}
    xml_record = by_id["pmc:PMC12345"]
    assert xml_record["fulltext_status"] == "open_fulltext"
    assert xml_record["text_source"] == "fixture_xml"
    assert "Title: Active allocation for protein design" in xml_record["text"]
    assert "Abstract: Metadata abstract about Bayesian optimization." in xml_record["text"]
    assert "PMC body text describes batch active learning" in xml_record["text"]
    assert xml_record["chunks"]
    assert xml_record["chunks"][0]["text"] in xml_record["text"]

    html_record = by_id["s2:HTML123"]
    assert html_record["fulltext_status"] == "open_fulltext"
    assert html_record["text_source"] == "fixture_html"
    assert "HTML article body covers matched contrast repair" in html_record["text"]
    assert "navigation that should not appear" not in html_record["text"]

    cache_dir = project / "framework" / "cache" / "literature_fulltext"
    assert (cache_dir / "pmc_PMC12345.xml").exists()
    assert (cache_dir / "10_1000_html-paper.html").exists()


def test_missing_fixture_degrades_to_metadata_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DESIGN_SCIENTIST_FULLTEXT_FIXTURES", str(FIXTURES))
    project = _project_with_cards(
        tmp_path,
        [
            {
                "paper_id": "missing:paper",
                "source": "pubmed",
                "title": "Metadata-only paper",
                "abstract": "This abstract remains available without open fulltext.",
                "doi": "10.1000/no-fixture",
            }
        ],
    )

    corpus_path = build_literature_corpus(project, offline_fixtures=True)

    records = _read_jsonl(corpus_path)
    assert len(records) == 1
    record = records[0]
    assert record["fulltext_status"] == "metadata_only"
    assert record["text_source"] == "metadata"
    assert "Title: Metadata-only paper" in record["text"]
    assert "Abstract: This abstract remains available without open fulltext." in record["text"]

    trace = json.loads((project / "framework" / "literature_reading_trace.json").read_text(encoding="utf-8"))
    assert trace["papers"][0]["paper_id"] == "missing:paper"
    assert trace["papers"][0]["status"] == "metadata_only"
    assert trace["papers"][0]["reason"] == "fixture_not_found"


def test_pdf_parser_exception_degrades_cleanly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DESIGN_SCIENTIST_FULLTEXT_FIXTURES", str(FIXTURES))
    project = _project_with_cards(
        tmp_path,
        [
            {
                "paper_id": "pdf_exception",
                "source": "pubmed",
                "title": "PDF exception paper",
                "abstract": "PDF parsing failure keeps this abstract.",
                "doi": "10.1000/pdf-exception",
            }
        ],
    )

    corpus_path = build_literature_corpus(project, offline_fixtures=True)

    record = _read_jsonl(corpus_path)[0]
    assert record["fulltext_status"] == "metadata_only"
    assert record["text_source"] == "metadata"
    assert "PDF parsing failure keeps this abstract." in record["text"]

    trace = json.loads((project / "framework" / "literature_reading_trace.json").read_text(encoding="utf-8"))
    assert trace["papers"][0]["status"] == "metadata_only"
    assert trace["papers"][0]["reason"] == "pdf_parse_failed"
    assert trace["papers"][0]["error_type"]


def test_trace_records_per_paper_status_without_api_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DESIGN_SCIENTIST_FULLTEXT_FIXTURES", str(FIXTURES))
    monkeypatch.setenv("S2_API_KEY", "super-secret-api-key")
    project = _project_with_cards(
        tmp_path,
        [
            {
                "paper_id": "pmc:PMC12345",
                "source": "pubmed",
                "title": "Active allocation for protein design",
                "abstract": "Metadata abstract about Bayesian optimization.",
                "doi": "10.1000/xml-fixture",
            },
            {
                "paper_id": "missing:paper",
                "source": "semantic_scholar",
                "title": "Missing fulltext",
                "abstract": "Metadata is still readable.",
                "doi": "10.1000/no-fixture",
            },
        ],
    )

    build_literature_corpus(project, offline_fixtures=True)

    trace_path = project / "framework" / "literature_reading_trace.json"
    trace_text = trace_path.read_text(encoding="utf-8")
    trace = json.loads(trace_text)
    statuses = {paper["paper_id"]: paper["status"] for paper in trace["papers"]}
    assert statuses == {"pmc:PMC12345": "open_fulltext", "missing:paper": "metadata_only"}
    assert trace["summary"]["total_papers"] == 2
    assert trace["summary"]["open_fulltext"] == 1
    assert trace["summary"]["metadata_only"] == 1
    assert "super-secret-api-key" not in trace_text
    assert "api_key" not in trace_text.lower()


def _project_with_cards(tmp_path: Path, cards: list[dict]) -> Path:
    project = tmp_path / "project"
    framework = project / "framework"
    framework.mkdir(parents=True)
    (framework / "paper_cards.json").write_text(json.dumps(cards, indent=2), encoding="utf-8")
    return project


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
