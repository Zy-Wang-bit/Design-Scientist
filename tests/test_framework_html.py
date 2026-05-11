from html.parser import HTMLParser
from pathlib import Path


HTML_PATH = Path("output/html/design-scientist-framework.html")


class HeadingAndLinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids: set[str] = set()
        self.links: list[str] = []
        self.classes: list[str] = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if "id" in attributes:
            self.ids.add(attributes["id"])
        if tag == "a" and "href" in attributes:
            self.links.append(attributes["href"])
        if "class" in attributes:
            self.classes.extend(attributes["class"].split())


def parse_html() -> tuple[str, HeadingAndLinkParser]:
    html = HTML_PATH.read_text(encoding="utf-8")
    parser = HeadingAndLinkParser()
    parser.feed(html)
    return html, parser


def test_framework_html_artifact_exists():
    assert HTML_PATH.exists()


def test_framework_html_contains_core_story_sections():
    html, parser = parse_html()

    expected_ids = {
        "overview",
        "architecture",
        "workflow",
        "evidence",
        "benchmark",
        "handoff",
    }
    assert expected_ids <= parser.ids

    expected_copy = [
        "Framework R&D",
        "Project execution",
        "Research OS",
        "synthetic multi-world replay",
        "human review",
        "1E62",
    ]
    for phrase in expected_copy:
        assert phrase in html


def test_framework_html_supports_navigation_presentation_and_print():
    html, parser = parse_html()

    for anchor in ["#overview", "#architecture", "#workflow", "#benchmark", "#handoff"]:
        assert anchor in parser.links

    assert "presentation-mode" in html
    assert "@media print" in html
    assert "window.print()" in html
    assert "TODO" not in html
