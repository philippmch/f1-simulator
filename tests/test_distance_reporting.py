"""Saved reports retain the denominators behind lap-aware distance summaries."""

import json

from f1sim.analysis.montecarlo import SimulationResults
from f1sim.output.export import Exporter
from f1sim.simulation.race import DriverStatus, RaceResult


def test_distance_summary_is_consistent_in_json_comparison_and_html(tmp_path):
    race = [
        RaceResult("A", "A", "T", 1, 1000, 0, 0, 90, DriverStatus.FINISHED,
                   laps_completed=10, race_time_limited=True),
        RaceResult("B", "B", "T", 2, 1010, 10, 0, 100, DriverStatus.FINISHED,
                   laps_completed=9, race_time_limited=True),
        RaceResult("C", "C", "T", 3, 900, 0, 0, 80, DriverStatus.DNF,
                   laps_completed=12, classified=True, race_time_limited=True),
    ]
    results = SimulationResults(1, "Test", {}, [race], [], race_engine="chronological")
    exporter = Exporter(tmp_path)
    stats = json.loads(exporter.export_statistics_json(results).read_text(encoding="utf-8"))
    comparison = json.loads(exporter.export_scenario_comparison_json(
        {"rain": results},
    ).read_text(encoding="utf-8"))
    expected = results.get_race_distance_statistics()
    assert stats["race_distance_statistics"] == expected
    assert comparison["scenarios"]["rain"]["race_distance_statistics"] == expected
    html = exporter.export_report_html(results).read_text(encoding="utf-8")
    assert "Mean winning distance: 10.0 laps (1 race)" in html
    assert "Lapped finishers: 1 of 2 finishers with known distance (50.0%)" in html
    assert "Time-limited races: 1 of 1 race (100.0%)" in html
