"""Report metadata is text, chart labels are JSON, and history links are local."""

import json
import re
from html import escape
from html.parser import HTMLParser
from urllib.parse import quote

from f1sim.analysis.montecarlo import DriverStatistics, SimulationResults
from f1sim.output.export import Exporter


class Tags(HTMLParser):
    def __init__(self, markup):
        super().__init__()
        self.tags = []
        self.feed(markup)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


def test_report_metadata_cannot_inject_html_and_chart_json_roundtrips(tmp_path):
    payload = '</ScRiPt><img src=x onerror="alert(1)"> & Montréal 東京'
    stats = DriverStatistics("A", payload, payload, wins=1, positions=[1], total_points=25)
    results = SimulationResults(1, payload, {"A": stats}, [], [], seed=payload)
    html = Exporter(tmp_path).export_report_html(results).read_text(encoding="utf-8")
    tags = Tags(html).tags
    assert not any(tag == "img" for tag, _ in tags)
    assert sum(tag == "script" for tag, _ in tags) == 2
    assert escape(payload) in html
    arrays = re.findall(r"^\s*x: (\[.*\]),$", html, re.MULTILINE)
    assert len(arrays) == 2
    for array in arrays:
        assert "<" not in array and ">" not in array and "&" not in array
        assert json.loads(array) == [payload]
    assert ("script", {"src": Exporter._PLOTLY_CDN}) in tags


def test_history_escapes_all_metadata_and_encodes_filename_component(tmp_path):
    exporter = Exporter(tmp_path)
    payload = '<svg onload="alert(1)"> & Montréal 東京'
    filename = 'Montréal 東京 " & report.html'
    exporter._write_history([{
        "timestamp": payload, "track": payload, "num_simulations": payload, "seed": payload,
        "race_engine": payload,
        "files": {"report_html": filename, "statistics_json": "javascript:alert(1)"},
    }])
    html = exporter.export_run_index_html().read_text(encoding="utf-8")
    tags = Tags(html).tags
    assert not any(tag in ("svg", "script") for tag, _ in tags)
    assert html.count(escape(payload)) == 5
    links = [attrs["href"] for tag, attrs in tags if tag == "a"]
    assert links == ["./" + quote(filename, safe=""), "./javascript%3Aalert%281%29"]
    assert exporter._read_history()[0]["track"] == payload


def test_forged_history_urls_and_unknown_values_are_text_not_links(tmp_path):
    exporter = Exporter(tmp_path)
    values = ["https://example.com/x", "//example.com/x", "../x", "..\\x",
              {"markup": "<img>"}, ["<svg>"]]
    exporter._write_history([{"files": {"report_html": value}} for value in values])
    html = exporter.export_run_index_html().read_text(encoding="utf-8")
    assert not any(tag in ("a", "img", "svg") for tag, _ in Tags(html).tags)
    for value in values:
        assert escape(str(value)) in html
