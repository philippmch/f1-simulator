"""Normal entry points forward model choice into real simulations and exports."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

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
def test_dashboard_real_runner_propagates_engine_to_each_scenario(monkeypatch, engine):
    monkeypatch.setattr(server, "_get_loader", SyntheticLoader)
    payload = server.run_dashboard_simulation(server.DashboardRunRequest(
        simulations=10, scenarios="dry,light_rain", seed=7, parallel=False,
        race_engine=engine,
    ))
    assert payload["request"]["race_engine"] == engine
    for index, scenario in enumerate(payload["scenarios"].values()):
        assert scenario["race_engine"] == engine
        assert scenario["seed"] == 7 + index * 1000
        assert scenario["sample_race"]
        distance = scenario["race_distance_statistics"]
        assert distance["recorded_races"] == 10
        assert distance["mean_winner_laps"] == 3


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_cli_real_runner_records_selected_engine_in_exports(monkeypatch, tmp_path, engine):
    path = Path(__file__).resolve().parents[1] / "examples" / "simulate_race.py"
    spec = importlib.util.spec_from_file_location("simulate_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "CurrentSeasonDataLoader", SyntheticLoader)
    monkeypatch.setattr(sys, "argv", [str(path), "--race-engine", engine,
        "--simulations", "1", "--no-parallel", "--scenarios", "dry,light_rain",
        "--export", "--output-dir", str(tmp_path)])
    assert module.main() == 0
    comparison = next(tmp_path.glob("*scenario_comparison.json"))
    scenarios = json.loads(comparison.read_text(encoding="utf-8"))["scenarios"]
    assert [entry["race_engine"] for entry in scenarios.values()] == [engine, engine]
    assert [entry["seed"] for entry in scenarios.values()] == [42, 1042]
    assert all(entry["race_distance_statistics"]["recorded_races"] == 1
               for entry in scenarios.values())
