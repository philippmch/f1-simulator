"""Reports use the completed run's session conditions and preserve raw inputs."""

import json
from copy import deepcopy

import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.models import Car, Driver, Track, Weather
from f1sim.output import ConsoleOutput, Exporter
from f1sim.output.comparison import render_comparison_report, render_rival_strategy_selection_report
from f1sim.output.qualifying_context import qualifying_weather_context


def snapshot():
    return {
        "schema_version": 7,
        "weather": Weather(change_probability=0).model_dump(mode="json"),
        "qualifying_weather": {
            "Q1": {"condition": "heavy_rain", "rain_intensity": .8, "track_wetness": .8},
        },
    }


def test_context_resolves_saved_session_fallback_without_mutation():
    saved = snapshot()
    before = deepcopy(saved)
    text = qualifying_weather_context(saved)
    assert "fixed within each session" in text
    assert "Q1: heavy rain, rain 80%, surface water 80%" in text
    assert "Q2: dry, rain 0%, surface water 0% (race weather)" in text
    assert "Q3: dry, rain 0%, surface water 0% (race weather)" in text
    assert saved == before
    saved["qualifying_weather"]["Q1"]["rain_intensity"] = .1
    assert "Q1: heavy rain, rain 80%" in text


@pytest.mark.parametrize("saved", [None, {}, {"schema_version": 2},
                                  {"qualifying_weather": {}}])
def test_absent_settings_add_no_context(saved):
    assert qualifying_weather_context(saved) == ""


@pytest.mark.parametrize("fields", [None, [], {"condition": "<script>"},
                                   {"rain_intensity": "0.8"}, {"rain_intensity": True},
                                   {"unknown": "<img src=x>"}])
def test_malformed_saved_session_is_explicit_without_echoing_input(fields):
    saved = snapshot()
    saved["qualifying_weather"]["Q1"] = fields
    assert qualifying_weather_context(saved) == "Qualifying weather: invalid saved conditions."


def test_context_rejects_wrong_schema_and_bad_race_fallback():
    saved = snapshot()
    saved["schema_version"] = 6
    assert qualifying_weather_context(saved) == "Qualifying weather: unrecognized saved schema."
    saved["schema_version"] = 7
    saved["weather"]["track_wetness"] = True
    assert qualifying_weather_context(saved) == "Qualifying weather: invalid saved conditions."


def test_console_html_and_json_show_completed_settings(tmp_path, capsys):
    runner = MonteCarloRunner(
        [Driver(id="A", name="A", team_id="T")],
        {"T": Car(team_id="T", team_name="T")},
        Track(id="t", name="Report", country="Test", total_laps=3, base_lap_time=90),
        Weather(change_probability=0), seed=29,
        qualifying_weather=snapshot()["qualifying_weather"],
    )
    result = runner.run(1, parallel=False)
    expected = qualifying_weather_context(result.input_snapshot)
    # Later form/runner changes must not become observations in completed reports.
    runner.qualifying_weather["Q1"]["rain_intensity"] = .1
    runner.weather.track_wetness = .5
    ConsoleOutput.print_monte_carlo_summary(result)
    assert expected in capsys.readouterr().out
    ConsoleOutput.print_scenario_comparison({"dry race": result})
    assert expected in capsys.readouterr().out
    assert expected in render_comparison_report({"dry race": result})
    exporter = Exporter(tmp_path)
    assert expected in exporter.export_report_html(result).read_text(encoding="utf-8")
    saved = json.loads(exporter.export_statistics_json(result).read_text(encoding="utf-8"))
    assert saved["simulation_inputs"]["qualifying_weather"] == {
        "Q1": Weather(condition="heavy_rain", rain_intensity=.8,
                      track_wetness=.8).model_dump(mode="json"),
    }
    assert saved["simulation_inputs"]["weather"]["track_wetness"] == 0


def test_weighted_summary_retains_and_escapes_session_context():
    expected = qualifying_weather_context(snapshot())
    manifest = {"report_context": {"qualifying_weather_context": expected}}
    assert expected in render_rival_strategy_selection_report(manifest)
    manifest["report_context"]["qualifying_weather_context"] = "Q1: <img src=x>"
    html = render_rival_strategy_selection_report(manifest)
    assert "Q1: &lt;img src=x&gt;" in html and "<img src=x>" not in html
