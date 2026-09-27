"""Overtaking-call statistics stay distinct across text and export formats."""

import csv
import json

from f1sim.analysis.montecarlo import SimulationResults
from f1sim.output import ConsoleOutput, Exporter
from f1sim.output.comparison import _overtaking_statistics_html, render_comparison_report
from f1sim.simulation.race import DriverStatus, RaceResult


def _result(*, legacy=False):
    hostile = '<script>alert("x")</script>'

    def row(driver_id, attempts, successes, contacts):
        return RaceResult(
            driver_id, driver_id, "T", 1, 90, 0, 0, 90, DriverStatus.FINISHED,
            overtake_attempts=attempts, overtake_successes=successes,
            overtake_contacts=contacts,
        )

    first = [
        row("A", None if legacy else 2, None if legacy else 1, None if legacy else 1),
        row("B", 0 if not legacy else None, 0 if not legacy else None, 0 if not legacy else None),
        row(hostile, *(None, None, None) if legacy else (2, 2, 0)),
        row("Z", 0 if not legacy else None, 0 if not legacy else None, 0 if not legacy else None),
    ]
    second = [
        row("A", None, None, None),
        row("B", 1 if not legacy else None, 1 if not legacy else None, 0 if not legacy else None),
        row(hostile, *(None, None, None) if legacy else (0, 0, 0)),
        row("Z", 0 if not legacy else None, 0 if not legacy else None, 0 if not legacy else None),
    ]
    return SimulationResults(2, "Test", {}, [first, second], []), hostile


def test_console_csv_json_and_html_keep_missing_zero_and_pooled_rates_distinct(
    tmp_path, capsys,
):
    results, hostile = _result()
    expected = results.get_overtake_statistics()
    assert expected["overall"] == {
        "status": "partial",
        "recorded_driver_races": 7,
        "missing_driver_races": 1,
        "attempts": 5,
        "successes": 4,
        "contacts": 1,
        "success_rate": 0.8,
        "contact_rate": 0.2,
    }
    assert expected["drivers"]["Z"]["status"] == "recorded"
    assert expected["drivers"]["Z"]["attempts"] == 0
    assert expected["drivers"]["Z"]["success_rate"] is None

    ConsoleOutput.print_monte_carlo_summary(results)
    printed = capsys.readouterr().out
    assert "5 / 4 / 1" in printed
    assert "success per attempt 80.0%" in printed
    assert "contact per attempt 20.0%" in printed
    assert "7 recorded / 1 missing available driver-race rows" in printed

    exporter = Exporter(tmp_path)
    csv_rows = list(csv.DictReader(
        exporter.export_race_results_csv(results).open(encoding="utf-8"),
    ))
    assert csv_rows[0]["overtake_attempts"] == "2"
    assert csv_rows[0]["overtake_successes"] == "1"
    assert csv_rows[0]["overtake_contacts"] == "1"
    assert csv_rows[4]["overtake_attempts"] == ""
    assert csv_rows[4]["overtake_successes"] == ""
    assert csv_rows[4]["overtake_contacts"] == ""
    saved = json.loads(exporter.export_statistics_json(results).read_text(encoding="utf-8"))
    scenario = json.loads(exporter.export_scenario_comparison_json(
        {"Test": results},
    ).read_text(encoding="utf-8"))["scenarios"]["Test"]
    assert saved["overtaking_statistics"] == expected
    assert scenario["overtaking_statistics"] == expected

    html = exporter.export_report_html(results).read_text(encoding="utf-8")
    assert "5 / 4 / 1" in html
    assert "7 recorded / 1 missing available driver-race rows" in html
    focused = _overtaking_statistics_html(results, focus_driver=hostile)
    assert "Full field" in focused and "5 / 4 / 1" in focused
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in focused
    assert "<script>alert" not in focused
    assert "<th scope=\"row\">B</th>" not in focused
    comparison = render_comparison_report(
        {"Test": results}, focus_driver=hostile,
    )
    assert "5 / 4 / 1" in comparison
    assert "Overtaking attempts and outcomes" in comparison


def test_legacy_rows_are_not_recorded_and_do_not_turn_into_zero(tmp_path, capsys):
    results, _ = _result(legacy=True)
    statistics = results.get_overtake_statistics()
    assert statistics["overall"]["status"] == "not_recorded"
    assert statistics["overall"]["attempts"] is None

    ConsoleOutput.print_monte_carlo_summary(results)
    assert "Full field: Not recorded" in capsys.readouterr().out
    payload = json.loads(Exporter(tmp_path).export_statistics_json(results).read_text())
    assert payload["overtaking_statistics"] == statistics
    html = _overtaking_statistics_html(results)
    assert "Not recorded" in html
    assert "0 / 0 / 0" not in html

    zero = _result()[0]
    zero.race_results = [[
        RaceResult("Z", "Z", "T", 1, 90, 0, 0, 90, DriverStatus.FINISHED,
                   overtake_attempts=0, overtake_successes=0, overtake_contacts=0),
    ]]
    zero_html = _overtaking_statistics_html(zero)
    assert "0 / 0 / 0" in zero_html
    assert "Not defined (0 attempts)" in zero_html
