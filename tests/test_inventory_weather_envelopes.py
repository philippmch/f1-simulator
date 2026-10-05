"""Weather-envelope reuse preserves full snapshots, physical inputs and cancellation."""

from copy import deepcopy

import pytest
from test_inventory_partial_strategies import assert_continuation, independent_schedules

from f1sim.cancellation import SimulationCancelled, cancellation_scope
from f1sim.models import Car, Driver, Track, Weather
from f1sim.models._native import forecast_json
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.models.weather import WeatherCondition
from f1sim.simulation import strategy_lap
from f1sim.simulation.inventory_strategy import (
    _inventory_surface_envelopes,
    plan_inventory_strategy,
)
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.strategy_lap import (
    control_lap_memo,
    control_lap_scope,
    memoized_control_envelope,
)
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.tire_inventory import TireInventory
from f1sim.simulation.weather_schedule import WeatherForecastContext


def models(laps=8):
    return (
        Driver(id="A", name="A", team_id="T"),
        Car(team_id="T", team_name="T"),
        Track(id="T", name="T", country="Synthetic", total_laps=laps, base_lap_time=90.),
    )


def test_envelopes_preserve_complete_snapshots_and_exact_native_minima():
    driver, car, track = models()
    first = Weather(track_wetness=.2, rain_intensity=.2)
    surfaces = [
        first, first.model_copy(deep=True),
        first.model_copy(update={"track_temperature": first.track_temperature + 1.}),
        first.model_copy(update={"condition": WeatherCondition.LIGHT_RAIN}),
        first.model_copy(update={"air_temperature": first.air_temperature + 1.}),
    ]
    original = {0: (0, 1, 2, 3), 1: (4, 3, 2, 1)}
    before = deepcopy(surfaces)
    snapshots = [forecast_json(surface) for surface in surfaces]
    reduced, keys = _inventory_surface_envelopes(original, snapshots)
    assert reduced[0] == (0, 2, 3)
    assert reduced[1] == (4, 3, 2, 0)
    assert all(len(set(key)) == len(key) for key in keys.values())
    assert keys[0] != keys[1]
    mean = LapSimulator().prepare_deterministic_lap_time(driver, car, track, 12)
    assert mean is not None
    for offset, observations in original.items():
        for tire in TIRE_COMPOUNDS.values():
            for age in (0, 11, 90):
                expected = min(mean(tire, surfaces[index], offset + 1, age)
                               for index in observations)
                actual = min(mean(tire, surfaces[index], offset + 1, age)
                             for index in reduced[offset])
                assert actual == expected
    assert surfaces == before


def test_running_envelopes_reuse_physics_across_planning_horizons():
    driver, car, track = models()
    weather = Weather(track_wetness=.2, rain_intensity=.2)
    tire = next(iter(TIRE_COMPOUNDS.values()))
    row = (forecast_json(weather),)
    calls = []

    @control_lap_scope
    def evaluate():
        values = []
        for horizon, selected_car, physical in (
            (8, car, 12), (10, car, 12),
            (10, car.model_copy(update={"base_pace": .95}), 12),
            (10, car, 18), (8, car, 12),
        ):
            selected_track = track.model_copy(update={"total_laps": horizon})
            memo = control_lap_memo(driver, selected_car, selected_track, physical)
            assert memo is not None
            mean = LapSimulator().prepare_deterministic_lap_time(
                driver, selected_car, selected_track, physical)
            assert mean is not None

            def calculate():
                calls.append((horizon, physical))
                return mean(tire, weather, 3, 11)

            value = memoized_control_envelope(memo, row, 3, tire.compound.value, 11, calculate)
            assert value == mean(tire, weather, 3, 11)
            values.append(value)
        return values

    values = evaluate()
    assert calls == [(8, 12), (10, 12), (10, 18)]
    assert values[0] == values[1]
    assert values[1] != values[2]
    assert values[0] != values[3]
    assert values[4] == values[0]
    assert strategy_lap._CONTROL_LAPS.get() is None


def test_envelope_cache_is_bounded_and_restores_after_cancellation(monkeypatch):
    driver, car, track = models()
    monkeypatch.setattr(strategy_lap, "_CONTROL_ENVELOPE_LIMIT", 3)
    calls = []

    @control_lap_scope
    def evaluate():
        memo = control_lap_memo(driver, car, track, 12)
        assert memo is not None
        for age in range(9):
            def calculate():
                calls.append(age)
                return float(age)

            assert memoized_control_envelope(memo, ("weather",), 1, "soft", age,
                                             calculate) == float(age)
            assert len(strategy_lap._CONTROL_LAPS.get()[4]) <= 3
        with cancellation_scope(lambda: True):
            memoized_control_envelope(memo, ("weather",), 1, "soft", 8, lambda: -1.)

    with pytest.raises(SimulationCancelled):
        evaluate()
    assert calls == list(range(9))
    assert strategy_lap._CONTROL_LAPS.get() is None


@pytest.mark.parametrize("warmup", [{}, {"soft": .5, "intermediate": .25}])
def test_shared_bounds_keep_actual_traffic_costs_separate(warmup):
    driver, car, track = models(6)
    weather = Weather(track_wetness=.2, rain_intensity=.2)
    stock = TireInventory.from_sets([
        {"id": "M", "compound": "medium"}, {"compound": "hard"},
        {"compound": "soft"}, {"compound": "intermediate"}, {"compound": "wet"},
    ])
    stock.fit("M")
    context = WeatherForecastContext.from_schedule([
        {"lap": 3, "rain_intensity": .5}, {"lap": 5, "rain_intensity": 0.},
    ])
    clock = StrategyWeatherClock(tuple(100. * offset for offset in range(6)),
                                 50., 90., 6, 23., 23.)
    options = dict(tire_age=0, remaining_stops=3, used_compounds=("medium",),
                   physical_total_laps=8, weather_clock=clock, forecast_context=context,
                   tire_warmup=warmup)
    gaps = [(0., None), (1.5, 1.5)]
    before = deepcopy((driver, car, track, weather, stock.__dict__))
    expected = [plan_inventory_strategy(driver, car, track, weather, stock, 1,
                                       current_traffic_gaps=gap, **options) for gap in gaps]

    @control_lap_scope
    def evaluate():
        return [plan_inventory_strategy(driver, car, track, weather, stock, 1,
                                        current_traffic_gaps=gap, **options) for gap in gaps]

    assert evaluate() == expected
    assert expected[0].wait_cost != expected[1].wait_cost
    assert (driver, car, track, weather, stock.__dict__) == before


@pytest.mark.parametrize("timed", [False, True])
@pytest.mark.parametrize("warmup", [{}, {"intermediate": 60., "wet": .5, "hard": 2.}])
@pytest.mark.parametrize("limited", [False, True])
def test_shared_weather_bounds_match_independent_physical_schedules(timed, warmup, limited):
    driver, car, track = models(4)
    weather = Weather(track_wetness=.3, rain_intensity=.2)
    records = [{"id": "I", "compound": "intermediate", "age": 1},
               {"id": "W", "compound": "wet"}, {"id": "H", "compound": "hard"}]
    if limited:
        for record, allowance in zip(records, (2, 2, 1), strict=True):
            record["remaining_laps"] = allowance
    stock = TireInventory.from_sets(records)
    stock.fit("I")
    context = WeatherForecastContext.from_schedule([
        {"lap": 2, "rain_intensity": .85}, {"lap": 4, "rain_intensity": 0.},
    ])
    options = dict(tire_age=1, remaining_stops=2, remaining_dry_stops=1,
                   remaining_damp_stops=2, used_compounds=("intermediate",),
                   physical_total_laps=9, tire_warmup=warmup, forecast_context=context,
                   current_traffic_gaps=(.2, 1.2))
    if timed:
        options["weather_clock"] = StrategyWeatherClock((0., 95., 185., 285.),
                                                        15., 65., 9, 23., 23.)
    inputs = driver, car, track, weather, stock
    before = deepcopy((inputs[:-1], stock.__dict__))
    expected, costs = independent_schedules(inputs, options)

    @control_lap_scope
    def evaluate():
        # A changed initial gap populates the same relaxed service tables.
        plan_inventory_strategy(*inputs, 1, **{**options, "current_traffic_gaps": (0., None)})
        return plan_inventory_strategy(*inputs, 1, **options)

    actual = evaluate()
    for stopped in (False, True):
        assert_continuation(actual.continuation(stopped), expected[stopped])
    if expected[True][1] < 0:
        assert actual.set_id is None
    else:
        assert costs[actual.set_id] == pytest.approx(expected[True], abs=1.e-8)
    assert actual.should_pit() == (expected[True] > expected[False])
    assert (inputs[:-1], stock.__dict__) == before


def test_failed_envelope_calculation_and_nested_decisions_do_not_leak():
    driver, car, track = models()
    calls = []

    def calculate():
        calls.append(None)
        if len(calls) == 1:
            raise RuntimeError("incomplete calculation")
        return 90.

    @control_lap_scope
    def inner():
        memo = control_lap_memo(driver, car, track, 12)
        assert memoized_control_envelope(memo, ("weather",), 1, "soft", 0,
                                         lambda: 95.) == 95.

    @control_lap_scope
    def outer():
        memo = control_lap_memo(driver, car, track, 12)
        with pytest.raises(RuntimeError, match="incomplete calculation"):
            memoized_control_envelope(memo, ("weather",), 1, "soft", 0, calculate)
        assert memoized_control_envelope(memo, ("weather",), 1, "soft", 0, calculate) == 90.
        inner()
        assert memoized_control_envelope(memo, ("weather",), 1, "soft", 0, calculate) == 90.

    outer()
    assert len(calls) == 2
    assert strategy_lap._CONTROL_LAPS.get() is None
