"""Weather observations stay aligned with exported and displayed trials."""

import csv
import json

import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner, SimulationResults
from f1sim.models import Car, Driver, Track, Weather, WeatherCondition
from f1sim.output import Exporter
from f1sim.web.server import _summarize_scenario_results


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_weather_exports_and_dashboard_select_the_same_trial(tmp_path, engine):
    results = MonteCarloRunner(
        [Driver(id="A", name="A", team_id="T")], {"T": Car(team_id="T", team_name="T")},
        Track(id="t", name="Track", country="Test", total_laps=4, base_lap_time=90),
        Weather(condition=WeatherCondition.LIGHT_RAIN, rain_intensity=.35,
                track_wetness=.5, change_probability=0),
        seed=41, race_engine=engine,
    ).run(3, parallel=False)
    exporter = Exporter(tmp_path)
    files = exporter.export_all(results)
    with files["weather_csv"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = [(index, row) for index, history in enumerate(results.weather_histories, 1)
                for row in history]
    assert len(rows) == len(expected) > 0
    for exported, (index, observed) in zip(rows, expected):
        assert int(exported["simulation"]) == index
        assert int(exported["weather_interval"]) == observed["lap"]
        assert exported["race_engine"] == engine
        assert exported["condition"] == observed["condition"]
        assert float(exported["rain_intensity"]) == observed["rain_intensity"]
        assert float(exported["track_wetness"]) == observed["track_wetness"]
    stats = json.loads(files["statistics_json"].read_text(encoding="utf-8"))
    assert stats["weather_histories"] == results.weather_histories
    assert "weather CSV" in files["runs_index_html"].read_text(encoding="utf-8")
    comparison = exporter.export_scenario_comparison_json({"rain": results})
    scenario = json.loads(comparison.read_text(encoding="utf-8"))["scenarios"]["rain"]
    assert scenario["weather_histories"] == results.weather_histories
    displayed = _summarize_scenario_results({"rain": results})["scenarios"]["rain"]
    assert displayed["sample_weather_history"] == (
        results.weather_histories[displayed["sample_index"]]
    )


def test_legacy_without_weather_history_has_no_fabricated_observations(tmp_path):
    results = SimulationResults(5, "Legacy", {}, [], [])
    exported = Exporter(tmp_path).export_weather_history_csv(results)
    with exported.open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == []
    scenario = _summarize_scenario_results({"legacy": results})["scenarios"]["legacy"]
    assert scenario["sample_weather_history"] == []
