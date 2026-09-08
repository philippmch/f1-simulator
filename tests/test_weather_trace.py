"""Weather observations follow shared race intervals without consuming randomness."""

import json

import numpy as np
import pytest

from f1sim.analysis import montecarlo
from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.analysis.replay import replay_saved_simulation
from f1sim.models import Car, Driver, Track, Weather
from f1sim.models.weather import WeatherCondition
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.race import RaceSimulator


def inputs():
    return ([Driver(id="A", name="A", team_id="A")],
            {"A": Car(team_id="A", team_name="A")},
            Track(id="test", name="Test", country="Test", total_laps=4, base_lap_time=90),
            Weather(condition=WeatherCondition.LIGHT_RAIN, rain_intensity=0.4,
                    track_wetness=0.2, change_probability=0))


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_fixed_rain_records_surface_and_resets(engine):
    drivers, cars, track, weather = inputs()
    simulator = RaceSimulator(rng=np.random.default_rng(7))
    run = (simulator.simulate_race if engine == "standard"
           else ChronologicalRace(simulator).run)
    run(drivers, cars, track, weather, ["A"])
    history = simulator.weather_history
    assert [row["lap"] for row in history] == [1, 2, 3, 4]
    assert [row["condition"] for row in history] == ["light_rain"] * 4
    assert [row["rain_intensity"] for row in history] == [0.4] * 4
    assert [row["track_wetness"] for row in history] == pytest.approx(
        [0.2, 0.24, 0.272, 0.2976])
    weather.track_wetness = 0.8
    assert history[0]["track_wetness"] == 0.2
    run(drivers, cars, track.model_copy(update={"total_laps": 1}), weather, ["A"])
    assert len(simulator.weather_history) == 1
    assert simulator.weather_history[0]["track_wetness"] == 0.8
    assert len(history) == 4
    run([], {}, track, weather, [])
    assert simulator.weather_history == []


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_evolving_trace_matches_activated_weather_without_extra_evolution(monkeypatch, engine):
    drivers, cars, track, weather = inputs()
    calls = []

    def evolve(current, rng):
        calls.append(current.model_dump())
        return current.model_copy(update={"condition": WeatherCondition.HEAVY_RAIN,
                                          "rain_intensity": 0.7,
                                          "track_wetness": current.track_wetness + 0.1})

    monkeypatch.setattr(Weather, "evolve", evolve)
    simulator = RaceSimulator(rng=np.random.default_rng(7))
    monkeypatch.setattr(simulator.event_manager, "process_lap", lambda *args, **kwargs: [])
    run = (simulator.simulate_race if engine == "standard"
           else ChronologicalRace(simulator).run)
    run(drivers, cars, track, weather, ["A"])
    assert len(calls) == 3
    assert [row["condition"] for row in simulator.weather_history] == [
        "light_rain", "heavy_rain", "heavy_rain", "heavy_rain"]
    assert [row["track_wetness"] for row in simulator.weather_history] == pytest.approx(
        [0.2, 0.3, 0.4, 0.5])


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_trial_order_and_replay_history(tmp_path, engine):
    drivers, cars, track, weather = inputs()
    weather.change_probability = 1
    runner = MonteCarloRunner(drivers, cars, track, weather, seed=17, race_engine=engine)
    result = runner.run(3, parallel=False)
    saved = tmp_path / "saved.json"
    saved.write_text(json.dumps({"simulation_inputs": result.input_snapshot,
                                 "metadata": {"num_simulations": 3, "seed": 17,
                                              "race_engine": engine}}), encoding="utf-8")
    for trial in range(1, 4):
        replay = replay_saved_simulation(saved, simulation=trial)
        assert replay.weather_histories == [result.weather_histories[trial - 1]]
    parallel = runner.run(3, parallel=True, max_workers=2)
    assert parallel.weather_histories == result.weather_histories
    assert parallel.race_results == result.race_results


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_recording_does_not_change_race_or_rng(monkeypatch, engine):
    drivers, cars, track, weather = inputs()
    weather.change_probability = 1

    def run():
        simulator = RaceSimulator(rng=np.random.default_rng(31))
        execute = (simulator.simulate_race if engine == "standard"
                   else ChronologicalRace(simulator).run)
        results = execute(drivers, cars, track, weather, ["A"])
        return results, simulator.rng.bit_generator.state

    observed = run()
    monkeypatch.setattr(RaceSimulator, "_record_weather", lambda *args: None)
    assert run() == observed


def test_legacy_worker_without_weather_history(monkeypatch):
    drivers, cars, track, weather = inputs()
    monkeypatch.setattr(montecarlo, "_run_single_simulation", lambda args: ([], [], {}))
    result = MonteCarloRunner(drivers, cars, track, weather, seed=1).run(2, parallel=False)
    assert result.weather_histories == [[], []]
