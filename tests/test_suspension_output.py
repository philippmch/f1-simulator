"""Suspension observations stay consistent across output surfaces."""

import csv
import json
from types import SimpleNamespace

from f1sim.analysis.montecarlo import DriverStatistics, SimulationResults
from f1sim.output import ConsoleOutput, Exporter
from f1sim.output.timing import suspension_statistics
from f1sim.simulation.race import DriverStatus, RaceResult


def _result(driver_id: str, suspension: float | None, position: int = 1) -> RaceResult:
    return RaceResult(
        driver_id, driver_id, "Team", position, 100.0, 0.0, 0, 90.0,
        DriverStatus.FINISHED, race_suspension_seconds=suspension,
    )


def _results() -> SimulationResults:
    return SimulationResults(
        3,
        "Suspension Test",
        {"A": DriverStatistics("A", "A", "Team", positions=[1, 1], wins=2)},
        [[_result("A", 0.0)], [_result("A", 12.0), _result("B", 12.0)],
         [_result("A", None)]],
        [],
        seed=9,
    )


def test_csv_and_json_preserve_zero_positive_and_unknown(tmp_path):
    results = _results()
    exporter = Exporter(tmp_path)

    csv_path = exporter.export_race_results_csv(results)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["race_suspension_seconds"] for row in rows] == ["0.000", "12.000", "12.000", ""]

    expected = {
        "recorded_races": 2,
        "races_with_recorded_suspension": 1,
        "mean_completed_suspension_seconds": 6.0,
    }
    statistics = json.loads(exporter.export_statistics_json(results).read_text(encoding="utf-8"))
    comparison_path = exporter.export_scenario_comparison_json({"test": results})
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert statistics["suspension_statistics"] == expected
    assert comparison["scenarios"]["test"]["suspension_statistics"] == expected


def test_reports_and_console_name_race_wide_pause_and_denominator(tmp_path, capsys):
    results = _results()
    exporter = Exporter(tmp_path)
    report = exporter.export_report_html(results).read_text(encoding="utf-8")
    comparison = exporter.export_scenario_comparison_html({"test": results}).read_text(
        encoding="utf-8"
    )

    for content in (report, comparison):
        assert "Completed race suspension" in content
        assert "6.000 s" in content
        assert "2 recorded races" in content
        assert "1 with suspension" in content
        assert "Race-wide collection + restart pause" in content
        assert "individual driver" in content and "stopped or driving" in content

    ConsoleOutput.print_race_results([_result("A", 0.0)])
    ConsoleOutput.print_monte_carlo_summary(results)
    ConsoleOutput.print_scenario_comparison({"test": results})
    output = capsys.readouterr().out
    assert "Completed race suspension: 0.000 s" in output
    assert "Mean completed race suspension: 6.000 s; 2 recorded races; 1 with suspension" in output
    assert "RACE SUSPENSION CONTEXT" in output
    assert "Race-wide collection + restart pause" in output


def test_old_duck_typed_aggregate_stays_unknown():
    legacy = SimpleNamespace(race_results=[[_result("A", 20.0)]])
    assert suspension_statistics(legacy) == {
        "recorded_races": 0,
        "races_with_recorded_suspension": 0,
        "mean_completed_suspension_seconds": None,
    }
