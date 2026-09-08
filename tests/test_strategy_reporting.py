"""Tyre sequence summaries survive exports with coverage and escaped labels."""

import json

from f1sim.analysis.montecarlo import SimulationResults
from f1sim.output.export import Exporter
from f1sim.simulation.race import DriverStatus, RaceResult


def test_recorded_sequences_in_json_comparison_and_html(tmp_path):
    name = "<img src=x onerror=alert(1)>"
    strategy = ["soft", "<script>bad()</script>", "soft"]
    rows = [RaceResult(name, name, "T", 1, 1000, 0, 1, 90, status, strategy=sequence)
            for status, sequence in [(DriverStatus.FINISHED, strategy),
                                     (DriverStatus.DNF, strategy), (DriverStatus.DNF, [])]]
    result = SimulationResults(100, "Test", {}, [[row] for row in rows], [])
    exporter = Exporter(tmp_path)
    expected = result.get_strategy_statistics()
    stats = json.loads(exporter.export_statistics_json(result).read_text(encoding="utf-8"))
    comparison = json.loads(exporter.export_scenario_comparison_json(
        {"dry": result},
    ).read_text(encoding="utf-8"))
    assert stats["strategy_statistics"] == expected
    assert comparison["scenarios"]["dry"]["strategy_statistics"] == expected
    html = exporter.export_report_html(result).read_text(encoding="utf-8")
    assert "2 recorded of 3 observed races" in html
    assert "Missing tyre sequences: 1" in html
    assert "2 (100.0%)" in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
    assert "soft → &lt;script&gt;bad()&lt;/script&gt; → soft" in html
    assert name not in html and "<script>bad()" not in html


def test_empty_sequence_report_is_explicit(tmp_path):
    result = SimulationResults(100, "Test", {}, [], [])
    html = Exporter(tmp_path).export_report_html(result).read_text(encoding="utf-8")
    assert "No tyre sequences were recorded." in html
