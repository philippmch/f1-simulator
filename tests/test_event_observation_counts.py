"""Event summaries share denominators and do not tune from an empty failure sample."""

import json

import pytest

from f1sim.analysis.montecarlo import RaceEventStatistics, SimulationResults
from f1sim.output import ConsoleOutput, Exporter
from f1sim.web.server import _summarize_scenario_results


def result():
    return SimulationResults(1000, "Partial", {}, [], [], event_stats=RaceEventStatistics(
        safety_car_count=2, vsc_count=3, red_flag_count=1, total_incidents=10,
        races_with_safety_car=1, races_with_red_flag=1, num_simulations=4,
    ))


def test_event_rates_and_calibration_use_ledger_count():
    observed = result()
    assert observed.get_event_rate_trials() == 4
    assert observed.get_event_rates() == {
        "safety_car_race_rate": .25, "red_flag_race_rate": .25,
        "avg_safety_cars": .5, "avg_vsc": .75, "avg_incidents": 2.5,
    }
    assert observed.get_safety_car_calibration_delta(.5) == .25
    assert observed.event_stats.safety_car_rate == 25
    assert observed.event_stats.red_flag_rate == 25


def test_legacy_event_counts_retain_nominal_denominator():
    observed = result()
    observed.event_stats.num_simulations = 0
    assert observed.get_event_rate_trials() == 1000
    assert observed.get_event_rates()["safety_car_race_rate"] == .001


@pytest.mark.parametrize("breakdown", [{}, {"engine": 0, "gearbox": 0}])
def test_no_failure_sample_has_no_tuning_advice(breakdown):
    observed = result()
    observed.event_stats.mechanical_failure_breakdown = breakdown
    expected = {"engine": .7, "gearbox": .3}
    assert observed.get_mechanical_failure_component_rates() == {}
    assert observed.get_mechanical_calibration_delta(expected) is None
    assert observed.get_mechanical_tuning_suggestions(expected) == {}
    assert observed.get_reliability_adjustment_recommendations(expected) == {}


def test_console_api_and_exports_share_event_denominator_and_empty_advice(tmp_path, capsys):
    observed = result()
    observed.event_stats.mechanical_failure_breakdown = {"engine": 0}
    ConsoleOutput.print_monte_carlo_summary(observed)
    ConsoleOutput.print_event_calibration(observed, .5)
    console = capsys.readouterr().out
    assert "Event-rate denominator: 4 trials" in console
    assert "25.0% of races" in console
    assert "No reference component shares are configured" in console
    assert "share unknown" in console
    assert "Calibration delta: Not recorded" not in console
    assert "vs target" not in console
    assert "Suggested tuning" not in console
    assert "decrease reliability" not in console
    exporter = Exporter(tmp_path)
    saved = json.loads(exporter.export_statistics_json(observed).read_text(encoding="utf-8"))
    combined = json.loads(exporter.export_scenario_comparison_json(
        {"partial": observed},
    ).read_text(encoding="utf-8"))["scenarios"]["partial"]
    shown = _summarize_scenario_results({"partial": observed})["scenarios"]["partial"]
    for payload in (saved, combined, shown):
        assert payload["event_rate_trials"] == 4
        assert payload["event_rates"] == observed.get_event_rates()
    for payload in (saved, combined, shown):
        assert payload["mechanical_failure_breakdown"] == {"engine": 0}
        assert payload["mechanical_failure_component_rates"] == {}
        assert payload["mechanical_tuning_suggestions"] == {}
        assert payload["reliability_adjustment_recommendations"] == {}


@pytest.mark.parametrize(
    ("breakdown", "expected_rates"),
    [
        (
            {"engine": 2, "gearbox": 2, "electrical": 2, "cooling": 2, "brakes": 2},
            {"engine": .2, "gearbox": .2, "electrical": .2, "cooling": .2, "brakes": .2},
        ),
        (
            {"engine": 5, "gearbox": 3, "electrical": 1},
            {"engine": 5 / 9, "gearbox": 3 / 9, "electrical": 1 / 9},
        ),
    ],
)
def test_automatic_reports_preserve_observed_shares_without_reference_advice(
    tmp_path, capsys, monkeypatch, breakdown, expected_rates,
):
    observed = result()
    observed.event_stats.mechanical_failure_breakdown = breakdown

    def unexpected_advice(*args, **kwargs):
        pytest.fail("automatic reporting must not request reference-based advice")

    monkeypatch.setattr(observed, "get_mechanical_tuning_suggestions", unexpected_advice)
    monkeypatch.setattr(observed, "get_reliability_adjustment_recommendations", unexpected_advice)

    ConsoleOutput.print_monte_carlo_summary(observed)
    console = capsys.readouterr().out
    assert "No reference component shares are configured" in console
    assert "observed share" in console
    assert "vs target" not in console
    assert "Calibration delta" not in console
    assert "Suggested tuning" not in console

    exporter = Exporter(tmp_path)
    saved = json.loads(exporter.export_statistics_json(observed).read_text(encoding="utf-8"))
    combined = json.loads(exporter.export_scenario_comparison_json(
        {"partial": observed},
    ).read_text(encoding="utf-8"))["scenarios"]["partial"]
    shown = _summarize_scenario_results({"partial": observed})["scenarios"]["partial"]

    for payload in (saved, combined, shown):
        assert payload["mechanical_failure_breakdown"] == breakdown
        assert payload["mechanical_failure_component_rates"] == expected_rates
        assert payload["mechanical_tuning_suggestions"] == {}
        assert payload["reliability_adjustment_recommendations"] == {}
