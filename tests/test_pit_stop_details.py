"""Paid-stop observations preserve modeled execution and exclude free fits."""

from dataclasses import asdict

import numpy as np
import pytest

from f1sim.models import Car, Driver, Track, Weather, WeatherCondition
from f1sim.models.tire import TireCompound
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.race import DriverRaceState, RaceSimulator


def inputs():
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="test", name="Test", country="Test", total_laps=4,
                  base_lap_time=90, pit_lane_delta=22)
    return driver, car, track


@pytest.mark.parametrize("control,factor", [("green", 1), ("safety_car", .55), ("vsc", .75)])
def test_stop_snapshots_actual_cost_age_weather_and_queue(monkeypatch, control, factor):
    driver, car, track = inputs()
    state = DriverRaceState(driver, car, 1, tire_laps=12)
    weather = Weather(condition=WeatherCondition.HEAVY_RAIN, rain_intensity=.8,
                      track_wetness=.9)
    simulator = RaceSimulator(np.random.default_rng(2))
    simulator.event_manager.safety_car_active = control == "safety_car"
    simulator.event_manager.vsc_active = control == "vsc"
    monkeypatch.setattr(simulator.lap_simulator, "calculate_pit_stop_time", lambda car: 3.5)
    releases = {"A": 103.0}
    loss = simulator._execute_pit_stop(state, track, weather, 13,
                                     pit_box_releases=releases, arrival_time=100)
    stop = state.pit_stop_details[0]
    assert stop == {"lap": 13, "from_compound": "medium", "to_compound": "wet",
                    "tire_age": 12, "condition": "heavy_rain", "rain_intensity": .8,
                    "track_wetness": .9, "control": control, "lane_loss": 22 * factor,
                    "service_time": 3.5, "queue_time": 3.0, "total_loss": loss}
    assert loss == stop["lane_loss"] + stop["service_time"] + stop["queue_time"]
    assert releases == {"A": 106.5}
    weather.track_wetness = 0
    state.tire_laps = 2
    simulator._handle_red_flag_stop([state], Weather(), track, 3)
    assert len(state.pit_stop_details) == 1
    assert stop["tire_age"] == 12 and stop["track_wetness"] == .9


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("paid", [False, True])
def test_actual_races_record_paid_opening_correction_and_reset(monkeypatch, engine, paid):
    driver, car, track = inputs()
    simulator = RaceSimulator(np.random.default_rng(3))
    monkeypatch.setattr(simulator.event_manager, "process_lap", lambda *a, **kw: [])
    run = simulator.simulate_race if engine == "standard" else ChronologicalRace(simulator).run
    weather = Weather(condition=WeatherCondition.HEAVY_RAIN, rain_intensity=.8,
                      track_wetness=.9, change_probability=0)
    result = run([driver], {"A": car}, track, weather, ["A"],
                 starting_tires={"A": TireCompound.SOFT if paid else TireCompound.WET})[0]
    assert result.pit_stops == len(result.pit_stop_details) == int(paid)
    assert result.pit_laps == [row["lap"] for row in result.pit_stop_details]
    if paid:
        assert result.pit_stop_details[0]["lap"] == 1
        assert result.pit_stop_details[0]["tire_age"] == 0
    assert run([driver], {"A": car}, track, weather, ["A"],
               starting_tires={"A": TireCompound.WET})[0].pit_stop_details == []
    assert len(result.pit_stop_details) == int(paid)


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_completed_stints_use_own_laps_and_snapshot_results(monkeypatch, engine):
    driver, car, track = inputs()
    track.total_laps = 60
    simulator = RaceSimulator(np.random.default_rng(3))
    monkeypatch.setattr(simulator.event_manager, "process_lap", lambda *a, **kw: [])
    states = []
    execute_stop = simulator._execute_pit_stop

    def capture(state, *args, **kwargs):
        states.append(state)
        return execute_stop(state, *args, **kwargs)

    monkeypatch.setattr(simulator, "_execute_pit_stop", capture)
    run = simulator.simulate_race if engine == "standard" else ChronologicalRace(simulator).run
    result = run([driver], {"A": car}, track, Weather(change_probability=0), ["A"])[0]
    assert result.pit_stops > 0
    assert result.pit_stops == len(result.pit_stop_details)
    assert result.pit_laps == [stop["lap"] for stop in result.pit_stop_details]
    previous_lap = 1
    for stop in result.pit_stop_details:
        assert stop["tire_age"] == stop["lap"] - previous_lap
        previous_lap = stop["lap"]
    states[0].pit_stop_details[0]["lap"] = -1
    assert result.pit_stop_details[0]["lap"] == result.pit_laps[0]


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_recording_does_not_change_physics_or_random_draws(monkeypatch, engine):
    driver, car, track = inputs()
    weather = Weather(condition=WeatherCondition.HEAVY_RAIN, rain_intensity=.8,
                      track_wetness=.9, change_probability=0)

    def run():
        simulator = RaceSimulator(np.random.default_rng(31))
        execute = (simulator.simulate_race if engine == "standard"
                   else ChronologicalRace(simulator).run)
        result = asdict(execute([driver], {"A": car}, track, weather, ["A"],
                               starting_tires={"A": TireCompound.SOFT})[0])
        result.pop("pit_stop_details")
        return result, simulator.rng.bit_generator.state

    observed = run()
    original = DriverRaceState.__post_init__

    class DiscardObservations(list):
        def append(self, value):
            pass

    def initialize(state):
        original(state)
        state.pit_stop_details = DiscardObservations()

    monkeypatch.setattr(DriverRaceState, "__post_init__", initialize)
    assert run() == observed
