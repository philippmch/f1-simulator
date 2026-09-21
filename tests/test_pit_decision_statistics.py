"""Paid-stop decision reason summaries retain their observation coverage."""

import json
from types import SimpleNamespace

from f1sim.analysis.montecarlo import DriverStatistics, SimulationResults
from f1sim.output.export import Exporter
from f1sim.simulation.race import DriverStatus

REASONS = (
    "forced_repair", "critical_weather", "weather_reaction", "compound_requirement",
    "dry_forecast", "rain_forecast", "inventory_forecast", "neutralization_window",
    "planned_window",
)


def row(driver="A", stops=0, details=None, *, retired=False):
    return SimpleNamespace(
        driver_id=driver,
        driver_name=driver,
        team="Team",
        position=1,
        total_time=90.0,
        gap_to_leader=0.0,
        pit_stops=stops,
        pit_stop_details=details,
        status=DriverStatus.DNF if retired else DriverStatus.FINISHED,
    )


def result(*rows):
    return SimulationResults(5, "Test", {}, [[item] for item in rows], [])


def test_mixed_reasons_report_paid_stop_and_race_coverage():
    sample = result(
        row(details=[{"decision_reason": "dry_forecast"},
                     {"decision_reason": "forced_repair"}], stops=2),
        row(details=[{"decision_reason": "<img src=x>"}], stops=1, retired=True),
        row(details=[], stops=0),
        row(details=None, stops=0),
        row(details=[{"decision_reason": "dry_forecast"}], stops=2),
    )

    assert sample.get_pit_decision_statistics() == {
        "A": {
            "races": 5,
            "races_with_recorded_details": 3,
            "missing_details_races": 2,
            "recorded_stops": 3,
            "stops_with_recorded_reasons": 2,
            "missing_reason_stops": 3,
            "reasons": {
                "dry_forecast": {"stops": 1, "share": 0.5},
                "forced_repair": {"stops": 1, "share": 0.5},
            },
        },
    }


def test_zero_stops_and_legacy_missing_details_stay_distinct():
    sample = result(row(details=[], stops=0), row(details=None, stops=0, retired=True))

    assert sample.get_pit_decision_statistics()["A"] == {
        "races": 2,
        "races_with_recorded_details": 1,
        "missing_details_races": 1,
        "recorded_stops": 0,
        "stops_with_recorded_reasons": 0,
        "missing_reason_stops": 0,
        "reasons": {},
    }


def test_malformed_detail_rows_are_missing_without_inventing_counts():
    sample = result(
        row(details=[None], stops=1),
        row(details=[{"decision_reason": "dry_forecast"}], stops=True),
        row(details=[{"decision_reason": "dry_forecast"}], stops=1.0),
        row(details=[{"decision_reason": "dry_forecast"}], stops=1),
    )

    summary = sample.get_pit_decision_statistics()["A"]
    assert summary["races_with_recorded_details"] == 1
    assert summary["missing_details_races"] == 3
    assert summary["recorded_stops"] == 1
    assert summary["missing_reason_stops"] == 1
    assert summary["reasons"] == {"dry_forecast": {"stops": 1, "share": 1.0}}


def test_unhashable_unknown_reason_is_missing_not_an_output_label():
    summary = (
        result(row(details=[{"decision_reason": ["hostile"]}], stops=1))
        .get_pit_decision_statistics()["A"]
    )

    assert summary["recorded_stops"] == 1
    assert summary["missing_reason_stops"] == 1
    assert summary["reasons"] == {}


def test_each_native_reason_is_allowlisted_and_observed():
    sample = result(*[row(details=[{"decision_reason": reason}], stops=1)
                      for reason in REASONS])

    summary = sample.get_pit_decision_statistics()["A"]
    assert summary["stops_with_recorded_reasons"] == len(REASONS)
    assert list(summary["reasons"]) == sorted(REASONS)
    assert all(item["stops"] == 1 and item["share"] == 1 / len(REASONS)
               for item in summary["reasons"].values())


def test_json_exports_include_decision_statistics_for_single_and_scenario(tmp_path):
    sample = result(row(details=[{"decision_reason": "dry_forecast"}], stops=1))
    exporter = Exporter(tmp_path)
    expected = sample.get_pit_decision_statistics()

    single = json.loads(exporter.export_statistics_json(sample).read_text(encoding="utf-8"))
    combined = json.loads(exporter.export_scenario_comparison_json({"dry": sample})
                          .read_text(encoding="utf-8"))
    assert single["pit_decision_statistics"] == expected
    assert combined["scenarios"]["dry"]["pit_decision_statistics"] == expected


def test_comparison_html_shows_reason_share_coverage_and_escapes_unknown(tmp_path):
    driver = DriverStatistics(
        driver_id="A", driver_name="A", team="Team", positions=[1, 2, 3], wins=1,
    )
    sample = SimulationResults(
        3, "Test", {"A": driver},
        [[row(details=[{"decision_reason": "dry_forecast"}], stops=1)],
         [row(details=[{"decision_reason": "<script>bad</script>"}], stops=1)],
         [row(details=[], stops=0)]],
        [],
    )
    no_stops = SimulationResults(
        1, "Test", {"A": driver}, [[row(details=[], stops=0)]], [],
    )

    report = Exporter(tmp_path).export_scenario_comparison_html(
        {"dry": sample, "no_stops": no_stops},
    )
    html = report.read_text(encoding="utf-8")
    assert "Paid-stop decisions for A" in html
    assert 'aria-label="A (A) · Team paid-stop decisions"' in html
    assert "Dry forecast" in html
    assert "50.0%" not in html
    assert "&lt;script&gt;bad&lt;/script&gt;" not in html
    assert "No paid stops recorded" in html
    assert "missing reasons" in html
