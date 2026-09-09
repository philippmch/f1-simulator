"""Weather costs follow shared leading updates at projected own-lap starts."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather, WeatherCondition
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import DriverRaceState, RaceSimulator
from f1sim.simulation.surface_projection import projected_surfaces


def setup(monkeypatch, *, pace=180, wetness=0, rain=.8, service=None, laps=12,
          pit_lane_delta=22, condition=WeatherCondition.DRY):
    simulator = RaceSimulator(np.random.default_rng(8))
    engine = ChronologicalRace(simulator)
    drivers = [Driver(id=key, name=key, team_id=key) for key in "AB"]
    cars = {key: Car(team_id=key, team_name=key) for key in "AB"}
    track = Track(id="T", name="T", country="T", total_laps=laps, base_lap_time=90,
                  pit_lane_delta=pit_lane_delta)
    observations, forecasts = {}, {}
    control = simulator.event_manager
    monkeypatch.setattr(control, "_check_mechanical_failure", lambda *a: None)
    monkeypatch.setattr(control, "_check_random_incident", lambda *a, **kw: None)

    def events(lap, *args, **kwargs):
        if service is not None and lap == 2:
            engine.states["A"].force_pit_next_lap = True
        return []

    def decide(state, states, planning, lap, window, weather, **kwargs):
        key = state.driver.id, lap
        forecasts[key] = (weather.model_copy(deep=True), planning.total_laps - lap + 1,
                          kwargs["weather_intervals"])
        return False

    def physics(driver, car, track, tire, weather, lap, total_laps, **kwargs):
        observations[driver.id, lap] = weather.model_copy(deep=True)
        return 90 if driver.id == "A" else pace

    monkeypatch.setattr(control, "process_lap", events)
    monkeypatch.setattr(simulator, "_should_pit", decide)
    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake",
                        lambda *a, **kw: (True, False))
    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", physics)
    if service is not None:
        monkeypatch.setattr(simulator.lap_simulator, "calculate_pit_stop_time", lambda car: service)
    monkeypatch.setattr(Weather, "evolve", lambda self, rng: self.project_surface())

    def run():
        weather = Weather(condition=condition, track_wetness=wetness, rain_intensity=rain)
        return engine.run(drivers, cars, track, weather, list("AB"),
                          starting_tires={key: TireCompound.INTERMEDIATE for key in "AB"})

    return engine, run, observations, forecasts


@pytest.mark.parametrize("pace", [67.5, 90, 110, 180])
@pytest.mark.parametrize("wetness,rain", [(0, .8), (.74, 0)])
@pytest.mark.parametrize("timed", [False, True])
def test_projected_surfaces_match_actual_multi_pace_lap_starts(
    monkeypatch, pace, wetness, rain, timed,
):
    if timed:
        monkeypatch.setattr("f1sim.simulation.race_timing.RACING_TIME_LIMIT_SECONDS", 360)
    engine, run, actual, forecasts = setup(monkeypatch, pace=pace, wetness=wetness, rain=rain)
    results = run()
    assert forecasts["A", 1][2] is None and forecasts["B", 1][2] is None
    checked = 0
    for (driver_id, lap), (surface, horizon, intervals) in forecasts.items():
        if intervals is None:
            continue
        path = projected_surfaces(surface, horizon, intervals)
        for offset, expected in enumerate(path):
            assert expected == actual[driver_id, lap + offset]
            checked += 1
    assert checked > 1
    assert len(engine.simulator.weather_history) == engine.control_intervals
    assert all(result.laps_completed > 0 for result in results)


def test_two_leading_updates_replace_one_own_lap_projection(monkeypatch):
    _, run, actual, forecasts = setup(monkeypatch)
    run()
    surface, horizon, intervals = forecasts["B", 2]
    assert surface.track_wetness == pytest.approx(.288)
    assert intervals[:3] == (0, 2, 4)
    assert surface.project_surface().track_wetness == pytest.approx(.3904)
    assert projected_surfaces(surface, horizon, intervals)[1].track_wetness == pytest.approx(.47232)
    assert actual["B", 3].track_wetness == pytest.approx(.47232)


def test_unfinished_leader_service_uses_expected_timing_without_future_sample(monkeypatch):
    paths = []
    for service in (3, 8, 4000):
        with monkeypatch.context() as patch:
            _, run, _, forecasts = setup(patch, pace=181, service=service)
            run()
            # A is a lap ahead but still in service when B starts lap two.
            paths.append(forecasts["B", 2])
    assert paths[0] == paths[1] == paths[2]
    assert paths[0][2] is not None


def test_forecast_does_not_mutate_state_rng_or_evolve_weather(monkeypatch):
    engine, run, _, _ = setup(monkeypatch)
    decide = engine.simulator._should_pit
    checked = []

    def pure(state, states, planning, *args, **kwargs):
        before = copy.deepcopy((engine.states, engine.pending, engine.weather,
                                engine.simulator.rng.bit_generator.state))
        with monkeypatch.context() as patch:
            patch.setattr(Weather, "evolve", lambda *a: pytest.fail("Forecast drew future weather"))
            intervals = engine._weather_intervals(state, state.total_time, planning)
        assert intervals == kwargs["weather_intervals"]
        if intervals is not None:
            checked.append(intervals)
        assert (engine.states, engine.pending, engine.weather,
                engine.simulator.rng.bit_generator.state) == before
        return decide(state, states, planning, *args, **kwargs)

    monkeypatch.setattr(engine.simulator, "_should_pit", pure)
    run()
    assert len(checked) > 1


def test_shared_drying_projection_changes_executed_transition_decision(monkeypatch):
    stops = {}
    for legacy in (False, True):
        with monkeypatch.context() as patch:
            engine, run, _, _ = setup(
                patch, wetness=.21, rain=0, laps=6, pit_lane_delta=20,
                condition=WeatherCondition.CLOUDY,
            )
            simulator = engine.simulator
            patch.setattr(simulator, "_should_pit", RaceSimulator._should_pit.__get__(simulator))
            patch.setattr(simulator.lap_simulator, "calculate_pit_stop_time",
                          expected_stationary_time)
            if legacy:
                patch.setattr(engine, "_weather_intervals", lambda *a: None)
            result = next(row for row in run() if row.driver_id == "B")
            stops[legacy] = result.pit_laps
            assert result.pit_stops == 1
            assert result.strategy == ["intermediate", "soft"]
    # The full policy anticipates two leading updates per own lap and fits
    # its chosen transition set before the retained rain set becomes critical.
    assert stops == {False: [2], True: [4]}


@pytest.mark.parametrize("intervals,expected", [(None, "same"), ((0, 5), "transition")])
def test_shared_cadence_selects_and_reaches_the_appropriate_rain_planner(
    monkeypatch, intervals, expected,
):
    simulator = RaceSimulator(np.random.default_rng(8))
    state = DriverRaceState(
        Driver(id="A", name="A", team_id="T"), Car(team_id="T", team_name="T"), 1,
        current_tire=TIRE_COMPOUNDS[TireCompound.INTERMEDIATE], tire_laps=1,
        tire_compound_history=["intermediate"],
    )
    track = Track(id="T", name="T", country="T", total_laps=3, base_lap_time=90)
    weather = Weather(track_wetness=.6, rain_intensity=.8)
    calls = []

    def planner(name, **kwargs):
        calls.append((name, kwargs["weather_intervals"]))
        return SimpleNamespace(should_pit=lambda: False)

    monkeypatch.setattr("f1sim.simulation.race.plan_rain_stop",
                        lambda *a, **kw: planner("same", **kw))
    monkeypatch.setattr("f1sim.simulation.race.plan_rain_transition",
                        lambda *a, **kw: planner("transition", **kw))
    assert not simulator._should_pit(state, [state], track, 2, False, weather,
                                    weather_intervals=intervals)
    assert calls == [(expected, intervals)]
