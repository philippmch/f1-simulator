"""Saved rain-scenario reports retain their actual experiment and escape context."""

import json
from copy import deepcopy

import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.models import Car, Driver, Track, Weather
from f1sim.output import ConsoleOutput, Exporter
from f1sim.output.comparison import render_comparison_report, render_rival_strategy_selection_report
from f1sim.output.qualifying_context import qualifying_weather_context
from f1sim.output.weather_schedule_context import weather_schedule_context


def snapshot():
    return {
        "schema_version": 8,
        "track": {"total_laps": 6},
        "weather": Weather(change_probability=.6).model_dump(mode="json"),
        "weather_schedule": [
            {"lap": 3, "rain_intensity": .8, "condition": "heavy_rain"},
            {"lap": 6, "rain_intensity": 0},
        ],
    }


def test_context_uses_saved_scenario_without_mutating_it():
    saved = snapshot()
    before = deepcopy(saved)
    text = weather_schedule_context(saved)
    assert "known to strategy; shared leading laps" in text
    assert "lap 3: rain 80%, heavy rain" in text
    assert "lap 6: rain 0%, condition unchanged" in text
    assert "Surface water continues to evolve" in text
    assert "random atmosphere changes are disabled" in text
    assert saved == before
    saved["weather_schedule"][0]["rain_intensity"] = .1
    assert "rain 80%" in text


@pytest.mark.parametrize("saved", [None, {}, {"schema_version": 7},
                                  {"weather_schedule": []}])
def test_legacy_or_empty_schedules_add_no_context(saved):
    assert weather_schedule_context(saved) == ""


@pytest.mark.parametrize("schedule", [
    "<script>", [{"lap": 3, "rain_intensity": True}],
    [{"lap": 3, "rain_intensity": "0.8"}],
    [{"lap": 3, "rain_intensity": .8, "condition": "<img src=x>"}],
    [{"lap": 7, "rain_intensity": .8}],
])
def test_invalid_saved_schedule_is_identified_without_echoing_values(schedule):
    saved = snapshot()
    saved["weather_schedule"] = schedule
    assert weather_schedule_context(saved) == "Prescribed race rainfall: invalid saved schedule."


def test_wrong_schema_and_combined_qualifying_context():
    saved = snapshot()
    saved["schema_version"] = 7
    assert weather_schedule_context(saved) == "Prescribed race rainfall: unrecognized saved schema."
    saved["schema_version"] = 8
    saved["qualifying_weather"] = {"Q1": {"rain_intensity": .8, "track_wetness": .8}}
    text = qualifying_weather_context(saved)
    assert "Q1: dry, rain 80%, surface water 80%" in text
    assert "Q2: dry, rain 0%, surface water 0% (race weather)" in text


def test_console_html_json_and_comparison_use_completed_run_inputs(tmp_path, capsys):
    schedule = [{"lap": 2, "rain_intensity": .8, "condition": "heavy_rain"},
                {"lap": 3, "rain_intensity": 0, "condition": "cloudy"}]
    runner = MonteCarloRunner(
        [Driver(id="A", name="A", team_id="T")],
        {"T": Car(team_id="T", team_name="T")},
        Track(id="t", name="Report", country="Test", total_laps=3, base_lap_time=90),
        Weather(change_probability=.6), seed=29, weather_schedule=schedule,
    )
    result = runner.run(1, parallel=False)
    expected = weather_schedule_context(result.input_snapshot)
    assert result.input_snapshot["schema_version"] == 8
    # Current controls and caller objects are not observations about the completed run.
    schedule[0]["rain_intensity"] = .1
    runner.weather_schedule[0]["rain_intensity"] = .2
    runner.weather.track_wetness = .5
    ConsoleOutput.print_monte_carlo_summary(result)
    assert expected in capsys.readouterr().out
    ConsoleOutput.print_scenario_comparison({"rain then dry": result})
    assert expected in capsys.readouterr().out
    comparison = render_comparison_report({"rain then dry": result})
    assert expected in comparison
    assert "weather change 60%/lap" not in comparison
    exporter = Exporter(tmp_path)
    assert expected in exporter.export_report_html(result).read_text(encoding="utf-8")
    saved = json.loads(exporter.export_statistics_json(result).read_text(encoding="utf-8"))
    assert saved["simulation_inputs"]["weather_schedule"][0]["rain_intensity"] == .8
    assert saved["simulation_inputs"]["weather"]["track_wetness"] == 0


def test_weighted_rival_summary_retains_and_escapes_scenario_context():
    expected = weather_schedule_context(snapshot())
    manifest = {"report_context": {"weather_schedule_context": expected}}
    assert expected in render_rival_strategy_selection_report(manifest)
    manifest["report_context"]["weather_schedule_context"] = "Schedule: <img src=x>"
    html = render_rival_strategy_selection_report(manifest)
    assert "Schedule: &lt;img src=x&gt;" in html and "<img src=x>" not in html
