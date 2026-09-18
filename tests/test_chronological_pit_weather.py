"""Paid-stop weather clocks in the chronological execution engine."""

import copy
from dataclasses import replace

import numpy as np
import pytest
from test_chronological_weather_intervals import setup

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock


def _forced_stop_run(monkeypatch, *, wetness, rain, pace=170, pit_lane_delta=22):
    """Build the two-car deterministic clock fixture and stop B on lap two."""
    engine, run, observations, forecasts = setup(
        monkeypatch,
        pace=pace,
        wetness=wetness,
        rain=rain,
        laps=8,
        pit_lane_delta=pit_lane_delta,
    )
    simulator = engine.simulator
    monkeypatch.setattr(
        simulator.lap_simulator,
        "calculate_pit_stop_time",
        lambda car: expected_stationary_time(car),
    )
    clocks = {}
    entries = {}

    def decide(state, states, planning, lap, window, weather, **kwargs):
        clocks[state.driver.id, lap] = kwargs.get("weather_clock")
        entries[state.driver.id, lap] = weather.model_copy(deep=True)
        return state.driver.id == "B" and lap == 2

    monkeypatch.setattr(simulator, "_should_pit", decide)
    return engine, run, observations, forecasts, clocks, entries


def test_external_leader_clock_prices_stop_exit_and_surface_updates(monkeypatch):
    engine, run, observations, _forecasts, clocks, entries = _forced_stop_run(
        monkeypatch, wetness=.4, rain=.6,
    )

    results = run()

    clock = clocks["B", 2]
    assert isinstance(clock, StrategyWeatherClock)
    assert clock.lap_start_offsets[0] == 0
    assert clock.first_update_after == pytest.approx(10)
    assert clock.updates(0, 1, True) == 1
    assert entries["B", 2].track_wetness == pytest.approx(.44)
    assert observations["B", 2].track_wetness == pytest.approx(.472)

    result = next(row for row in results if row.driver_id == "B")
    assert result.pit_laps == [2]
    assert result.pit_stops == 1
    assert result.pit_stop_details[0]["service_time"] == pytest.approx(
        expected_stationary_time(engine.states["B"].car)
    )
    assert engine.pit_exits == [("B", 2, pytest.approx(194.75094604485366))]


def test_clock_keeps_physical_delay_separate_from_traffic_pricing(monkeypatch):
    engine, run, _, _, clocks, _ = _forced_stop_run(
        monkeypatch, wetness=.4, rain=.6,
    )
    # A negative rejoin estimate is strategy relief, not a shorter physical
    # stop.  The chronological engine passes only the physical queue delay.
    original_traffic = engine._strategy_traffic

    def relieved_traffic(*args, **kwargs):
        snapshot = original_traffic(*args, **kwargs)
        return replace(snapshot, rejoin_traffic_cost=-100.0)

    monkeypatch.setattr(engine, "_strategy_traffic", relieved_traffic)
    before = copy.deepcopy(engine.simulator.rng.bit_generator.state)
    run()
    clock = clocks["B", 2]
    assert clock.current_stop_delay == pytest.approx(
        expected_stationary_time(engine.states["B"].car) + 22
    )
    assert clock.future_stop_delay == pytest.approx(
        expected_stationary_time(engine.states["B"].car) + 22
    )
    assert engine.simulator.rng.bit_generator.state == before


def test_drying_surface_uses_stop_exit_snapshot_for_the_outlap(monkeypatch):
    _, run, observations, _, clocks, entries = _forced_stop_run(
        monkeypatch, wetness=.25, rain=0,
    )
    run()
    assert clocks["B", 2] is not None
    assert entries["B", 2].track_wetness == pytest.approx(.22)
    assert observations["B", 2].track_wetness == pytest.approx(.19)


def test_steady_surface_keeps_legacy_clock_path(monkeypatch):
    engine, run, _, _, clocks, _ = _forced_stop_run(
        monkeypatch, wetness=0, rain=0,
    )
    run()
    assert clocks["B", 2] is None


def test_external_clock_is_absent_for_leader(monkeypatch):
    _, run, _, _, clocks, _ = _forced_stop_run(
        monkeypatch, wetness=.4, rain=.6,
    )
    run()
    assert all(clocks["A", lap] is None for lap in range(1, 9))


def test_fallback_stint_cost_uses_rebased_clock_surface_timeline():
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="T", name="T", country="T", total_laps=8, base_lap_time=90)
    weather = Weather(track_wetness=.4, rain_intensity=.6)
    tire = TIRE_COMPOUNDS[TireCompound.SOFT]
    simulator = LapSimulator(np.random.default_rng(9))
    intervals = (0, 2, 4)

    projected = simulator.projected_stint_lap_cost(
        driver, car, track, tire, 3, 2, weather,
        current_lap_time_modifier=1.1,
        weather_intervals=intervals,
    )
    manual_driver = driver.model_copy(deep=True)
    manual_surface = weather
    manual = 0.0
    for age, updates in enumerate(intervals):
        manual_surface = weather
        for _ in range(updates):
            manual_surface = manual_surface.project_surface()
        manual_driver.current_tire_laps = age
        manual += simulator.calculate_lap_time(
            manual_driver, car, track, tire, manual_surface, 2 + age, track.total_laps,
            active_aero_enabled=True, sample_variation=False,
        ) * (1.1 if age == 0 else 1.0)
    assert projected == pytest.approx(manual)
