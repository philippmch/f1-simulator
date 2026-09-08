"""Normal entry points forward model choice into real simulations and exports."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from f1sim.analysis.replay import replay_saved_simulation
from f1sim.models import Car, Driver, Track
from f1sim.web import server


class SyntheticLoader:
    def __init__(self, **kwargs):
        pass

    def resolve_race_identifier(self, *args):
        return 1

    def list_available_events(self, *args):
        return [{"round": 1, "race": "Synthetic"}]

    def get_weighted_driver_stats(self, **kwargs):
        return {}

    def get_track_stats(self, *args):
        return SimpleNamespace(track_name="Synthetic", country="Test", total_laps=3,
                               avg_lap_time=90)

    def create_drivers_from_stats(self, *args):
        return [Driver(id="A", name="A", team_id="A")]

    def create_cars_from_stats(self, *args):
        return {"A": Car(team_id="A", team_name="A")}

    def create_track_from_stats(self, *args):
        return Track(id="synthetic", name="Synthetic", country="Test", total_laps=3,
                     base_lap_time=90)

    def get_provenance(self):
        return {"source": "Synthetic test"}


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("starting_tires", [None, {"A": "hard"}])
@pytest.mark.parametrize("weather_mode", ["evolving", "fixed_rainfall"])
def test_dashboard_real_runner_propagates_engine_to_each_scenario(
    monkeypatch, tmp_path, engine, starting_tires, weather_mode,
):
    monkeypatch.setattr(server, "_get_loader", SyntheticLoader)
    payload = server.run_dashboard_simulation(server.DashboardRunRequest(
        simulations=10, scenarios="dry,light_rain", seed=7, parallel=False,
        race_engine=engine, starting_tires=starting_tires, weather_mode=weather_mode,
    ))
    assert payload["request"]["race_engine"] == engine
    assert payload["request"]["starting_tires"] == (starting_tires or {})
    assert payload["request"]["weather_mode"] == weather_mode
    report = payload["comparison_report_html"]
    assert "Simulation comparison" in report and "<script" not in report
    assert "dry; rain 0%" in report and "light_rain; rain 35%" in report
    assert engine in report
    saved = tmp_path / "dashboard.json"
    saved.write_text(json.dumps(payload), encoding="utf-8")
    for index, (name, scenario) in enumerate(payload["scenarios"].items()):
        assert scenario["race_engine"] == engine
        assert scenario["seed"] == 7 + index * 1000
        assert scenario["sample_race"]
        assert scenario["simulation_inputs"]["starting_tires"] == (starting_tires or {})
        expected_change = 0 if weather_mode == "fixed_rainfall" else .2
        assert scenario["simulation_inputs"]["weather"]["change_probability"] == expected_change
        if starting_tires:
            assert all(row["strategy"][0] == "hard" for row in scenario["sample_race"])
        strategies = scenario["strategy_statistics"]
        assert strategies
        for stats in strategies.values():
            assert stats["races"] == stats["races_with_recorded_strategy"] == 10
            assert sum(item["races"] for item in stats["strategies"]) == 10
        distance = scenario["race_distance_statistics"]
        assert distance["recorded_races"] == 10
        assert distance["mean_winner_laps"] == 3
        assert scenario["simulation_inputs"]["schema_version"] == 1
        assert scenario["simulation_inputs"]["track"]["total_laps"] == 3
        replay = replay_saved_simulation(saved, scenario["sample_index"] + 1, name)
        assert server._serialize_sample_race(replay) == scenario["sample_race"]
        assert server._serialize_sample_qualifying(replay) == scenario["sample_qualifying"]


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_cli_real_runner_records_selected_engine_in_exports(monkeypatch, tmp_path, engine):
    path = Path(__file__).resolve().parents[1] / "examples" / "simulate_race.py"
    spec = importlib.util.spec_from_file_location("simulate_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "CurrentSeasonDataLoader", SyntheticLoader)
    monkeypatch.setattr(sys, "argv", [str(path), "--race-engine", engine,
        "--simulations", "1", "--no-parallel", "--scenarios", "dry,light_rain",
        "--export", "--output-dir", str(tmp_path), "--starting-tyres", "A=hard",
        "--weather-mode", "fixed_rainfall"])
    assert module.main() == 0
    comparison = next(tmp_path.glob("*scenario_comparison_*.json"))
    report = comparison.with_suffix(".html")
    assert "Simulation comparison" in report.read_text(encoding="utf-8")
    scenarios = json.loads(comparison.read_text(encoding="utf-8"))["scenarios"]
    assert [entry["race_engine"] for entry in scenarios.values()] == [engine, engine]
    assert [entry["seed"] for entry in scenarios.values()] == [42, 1042]
    assert all(entry["race_distance_statistics"]["recorded_races"] == 1
               for entry in scenarios.values())
    assert all(entry["simulation_inputs"]["schema_version"] == 1
               for entry in scenarios.values())
    assert all(entry["simulation_inputs"]["starting_tires"] == {"A": "hard"}
               for entry in scenarios.values())
    assert all(entry["simulation_inputs"]["weather"]["change_probability"] == 0
               for entry in scenarios.values())
    assert all(entry["probability_intervals"]["A"]["trials"] == 1
               for entry in scenarios.values())


def test_dashboard_rejects_unknown_starting_driver(monkeypatch):
    monkeypatch.setattr(server, "_get_loader", SyntheticLoader)
    with pytest.raises(ValueError, match="UNKNOWN"):
        server.run_dashboard_simulation(server.DashboardRunRequest(
            simulations=10, scenarios="dry", starting_tires={"UNKNOWN": "soft"},
        ))


@pytest.mark.parametrize("overrides", [{"A": "bad"}, ["soft"], {"A": None}])
def test_dashboard_rejects_malformed_starting_tyres_before_loading(monkeypatch, overrides):
    monkeypatch.setattr(server, "_get_loader", lambda **kwargs: pytest.fail("live loading"))
    with pytest.raises(ValueError):
        server.run_dashboard_simulation(server.DashboardRunRequest(starting_tires=overrides))
