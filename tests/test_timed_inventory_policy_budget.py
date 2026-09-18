"""Caller stop envelopes retain schedules across delayed weather transitions."""

from math import inf

import numpy as np
import pytest
from test_timed_inventory_weather_oracle import enumerate_sets

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.inventory_strategy import plan_inventory_strategy
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import (
    DriverRaceState,
    RaceSimulator,
    TeamStrategyArchetype,
)
from f1sim.simulation.rain_strategy import plan_rain_transition
from f1sim.simulation.strategy_traffic import StrategyTrafficSnapshot
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.tire_inventory import TireInventory


def _inventory_fixture():
    driver = Driver(id="d", name="D", team_id="t", tire_management=0)
    car = Car(team_id="t", team_name="T", tire_degradation_factor=1.5)
    track = Track(
        id="t", name="T", country="T", total_laps=8,
        base_lap_time=900, pit_lane_delta=.1, tire_stress=1,
    )
    weather = Weather(track_wetness=.12, rain_intensity=0, change_probability=0)
    records = [
        {"id": "S", "compound": "soft"},
        {"id": "S1", "compound": "soft"},
        {"id": "S2", "compound": "soft"},
        {"id": "M", "compound": "medium"},
        {"id": "H", "compound": "hard", "age": 1},
        {"id": "I", "compound": "intermediate"},
    ]
    future_delay = expected_stationary_time(car) + track.pit_lane_delta
    clock = StrategyWeatherClock(
        (0., 900., 1800.), 910., 900., 2, 20., future_delay,
    )
    intervals = tuple(clock.updates(offset, 0) for offset in range(3))
    return (driver, car, track, weather), records, clock, intervals


def _inventory_state(simulator, models, records):
    driver, car, _track, _weather = models
    inventory = TireInventory.from_sets(records)
    inventory.fit("S")
    state = DriverRaceState(driver, car, 1)
    simulator._initialize_inventory(state, inventory, inventory.sets["S"])
    state.laps_completed = 5
    state.pit_stops = 1
    state.strategy_archetype = TeamStrategyArchetype.AGGRESSIVE
    state.tire_compound_history = ["hard", "soft"]
    state.tire_laps = state.driver.current_tire_laps = 4
    return state, inventory


def test_timed_inventory_caller_matches_physical_schedule_oracle():
    models, records, clock, intervals = _inventory_fixture()
    driver, car, track, weather = models
    inventory = TireInventory.from_sets(records)
    inventory.fit("S")
    options = dict(
        tire_age=4,
        current_lap=6,
        free_fit=False,
        remaining_stops=2,
        remaining_dry_stops=2,
        remaining_damp_stops=1,
        used_compounds=(TireCompound.SOFT, TireCompound.HARD),
        physical_total_laps=8,
        current_traffic_gaps=(None, None),
        current_lap_time_modifier=1.,
        active_aero_enabled=True,
        pit_lane_factor=1.,
        additional_current_stop_cost=clock.current_stop_delay - clock.future_stop_delay,
    )
    expected = enumerate_sets(models, inventory, clock, options)
    assert intervals == (0, 0, 1)
    # Retaining stays damp. A paid stop shifts the last lap past the second
    # update, making a later dry stop possible before the projected finish.
    assert weather.track_wetness - .03 * intervals[-1] >= .08
    assert clock.updates(2, 1, True) == 2
    assert weather.track_wetness - .03 * clock.updates(2, 1, True) < .08

    simulator = RaceSimulator(np.random.default_rng(0))
    state, _ = _inventory_state(simulator, models, records)
    actual = simulator._plan_inventory(
        state, track, weather, 6,
        physical_total_laps=8, weather_intervals=intervals, weather_clock=clock,
        additional_current_stop_cost=clock.current_stop_delay - clock.future_stop_delay,
    )
    assert actual.wait_cost == pytest.approx(expected[0])
    assert actual.pit_now_cost == pytest.approx(expected[1])
    assert actual.set_id == expected[2]
    assert actual.should_pit()


def test_timed_inventory_keeps_damp_gate_after_global_envelope_expands():
    models, records, clock, intervals = _inventory_fixture()
    _driver, _car, track, weather = models
    simulator = RaceSimulator(np.random.default_rng(0))
    state, _ = _inventory_state(simulator, models, records)
    state.pit_stops = 2  # Exhaust the aggressive two-stop damp allowance.
    state.tire_compound_history = ["soft", "hard", "soft"]
    actual = simulator._plan_inventory(
        state, track, weather, 6, physical_total_laps=8,
        weather_intervals=intervals, weather_clock=clock,
        additional_current_stop_cost=clock.current_stop_delay - clock.future_stop_delay,
    )
    assert actual.pit_now_cost == inf
    assert actual.wait_cost < inf


def test_inventory_without_clock_keeps_original_global_allowance():
    models, records, clock, intervals = _inventory_fixture()
    driver, car, track, weather = models
    simulator = RaceSimulator(np.random.default_rng(0))
    state, inventory = _inventory_state(simulator, models, records)
    actual = simulator._plan_inventory(
        state, track, weather, 6,
        physical_total_laps=8, weather_intervals=intervals,
        additional_current_stop_cost=clock.current_stop_delay - clock.future_stop_delay,
    )
    expected = plan_inventory_strategy(
        driver, car, track, weather, inventory, 6,
        tire_age=4, remaining_stops=1, remaining_dry_stops=2,
        remaining_damp_stops=1, used_compounds=(TireCompound.SOFT, TireCompound.HARD),
        physical_total_laps=8, weather_intervals=intervals,
        additional_current_stop_cost=clock.current_stop_delay - clock.future_stop_delay,
    )
    assert actual == expected


def test_timed_transition_caller_matches_delayed_dry_schedule():
    driver = Driver(id="d", name="D", team_id="t", tire_management=0)
    car = Car(team_id="t", team_name="T", tire_degradation_factor=1.5)
    # Deliberately long laps and high wear expose the small benefit of one
    # additional fresh set; these are model checks, not calibrated race gains.
    track = Track(id="t", name="T", country="T", total_laps=8,
                  base_lap_time=900, pit_lane_delta=.1, tire_stress=1)
    weather = Weather(track_wetness=.12, rain_intensity=0, change_probability=0)
    tire = TIRE_COMPOUNDS[TireCompound.SOFT].model_copy()
    delay = expected_stationary_time(car) + track.pit_lane_delta
    clock = StrategyWeatherClock((0., 900., 1800.), 910., 900., 2, 20., delay)
    intervals = tuple(clock.updates(offset, 0) for offset in range(3))
    options = dict(
        additional_current_stop_cost=20. - delay, physical_total_laps=8,
        weather_intervals=intervals, remaining_dry_stops=2,
        remaining_damp_stops=1, used_compounds=(TireCompound.SOFT, TireCompound.HARD),
        current_traffic_gaps=(None, None), weather_clock=clock,
    )
    expected = plan_rain_transition(driver, car, track, weather, tire, 4, 6, 2, **options)
    truncated = plan_rain_transition(driver, car, track, weather, tire, 4, 6, 1, **options)
    assert intervals == (0, 0, 1)
    assert expected.should_pit()
    assert not truncated.should_pit()

    simulator = RaceSimulator(np.random.default_rng(0))
    state = DriverRaceState(
        driver, car, 1, current_tire=tire, tire_laps=4,
        prior_tire_laps=0, tire_compound_history=["hard", "soft"],
        strategy_archetype=TeamStrategyArchetype.AGGRESSIVE,
        pit_stops=1, laps_completed=5,
    )
    snapshot = StrategyTrafficSnapshot(None, None, 0., (None, None))
    assert simulator._should_pit(
        state, [state], track, 6, False, weather,
        additional_current_stop_cost=20. - delay,
        traffic_snapshot=snapshot, physical_total_laps=8,
        weather_intervals=intervals, weather_clock=clock,
    )
    assert state.weather_pit_proposal == (6, expected.compound)


def test_timed_envelope_does_not_add_elective_dry_stop_at_budget():
    driver = Driver(id="d", name="D", team_id="t")
    car = Car(team_id="t", team_name="T")
    track = Track(
        id="t", name="T", country="T", total_laps=10,
        base_lap_time=90, pit_lane_delta=.1,
    )
    future_delay = expected_stationary_time(car) + track.pit_lane_delta
    clock = StrategyWeatherClock(
        tuple(90. * offset for offset in range(6)), 100., 90., 5,
        future_delay + 3., future_delay,
    )
    intervals = tuple(clock.updates(offset, 0) for offset in range(6))
    simulator = RaceSimulator(np.random.default_rng(0))
    state = DriverRaceState(
        driver, car, 1,
        current_tire=TIRE_COMPOUNDS[TireCompound.MEDIUM].model_copy(),
        tire_laps=100, prior_tire_laps=0,
        tire_compound_history=["soft", "medium"],
        strategy_archetype=TeamStrategyArchetype.BALANCED,
        pit_stops=3,
    )
    snapshot = StrategyTrafficSnapshot(None, None, 0., (None, None))
    assert not simulator._should_pit(
        state, [state], track, 5, False, Weather(track_wetness=.02),
        additional_current_stop_cost=clock.current_stop_delay - clock.future_stop_delay,
        traffic_snapshot=snapshot, physical_total_laps=10,
        weather_intervals=intervals, weather_clock=clock,
    )
