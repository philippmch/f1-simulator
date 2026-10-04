"""Shared fresh-service costs preserve physical schedules and decision isolation."""

import cProfile
import runpy
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from test_controlled_weather_strategy import context, inputs
from test_inventory_partial_strategies import assert_continuation, independent_schedules

from f1sim.cancellation import SimulationCancelled
from f1sim.models import TireCompound, _native
from f1sim.simulation import inventory_strategy
from f1sim.simulation.chronological_finish import ChronologicalFinishContext
from f1sim.simulation.inventory_strategy import plan_inventory_strategy
from f1sim.simulation.race_timing import RaceFinishTimeline
from f1sim.simulation.strategy_control_clock import StrategyControlContext
from f1sim.simulation.strategy_lap import (
    control_lap_memo,
    control_lap_scope,
    control_relaxation_memo,
)
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.weather_schedule import WeatherForecastContext


def preserve_native_helper(monkeypatch, replacement):
    monkeypatch.setattr(inventory_strategy, "control_relaxation_memo", replacement)
    # Change only cache reuse in the reference, preserving native physics and
    # its original exact finite-pool search on either side of the comparison.
    monkeypatch.setattr(_native, "_HELPERS", [
        (namespace, name, replacement if namespace is inventory_strategy.__dict__
         and name == "control_relaxation_memo" else value)
        for namespace, name, value in _native._HELPERS])


def profiled_decision(models, options):
    profiler = cProfile.Profile()
    result = profiler.runcall(plan_inventory_strategy, *models, 1, **options)
    profiler.create_stats()
    visits = sum(value[1] for key, value in profiler.stats.items() if key[2] == "service_frame")
    return result, visits


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("control", ["sc", "vsc"])
@pytest.mark.parametrize("horizon,limited,warmup", [
    (5, False, None), (14, False, None), (5, True, None),
    (14, False, {"intermediate": .3, "wet": .5, "hard": .1}),
])
def test_pool_independent_costs_preserve_exact_controlled_decisions(
    monkeypatch, engine, control, horizon, limited, warmup,
):
    driver, car, track, weather, stock = models = inputs(.2, .2, horizon)
    if limited:
        stock.sets = {key: replace(item, remaining_laps=3) for key, item in stock.sets.items()}
    observed = context(track, car, "standard", control, 3)
    if engine == "chronological":
        observed = StrategyControlContext(ChronologicalFinishContext(
            "A", RaceFinishTimeline(18, ["A"]), ("A",), (), 100.,
            1.4 if control == "sc" else 1.2, control == "sc", control_intervals=3,
        ), 50., observed.current_stop_delay)
    options = dict(
        control_context=observed, physical_total_laps=18, tire_age=11,
        remaining_stops=3, remaining_dry_stops=3, remaining_damp_stops=3,
        used_compounds=(TireCompound.INTERMEDIATE,), forecast_context=
        WeatherForecastContext.from_schedule([
            dict(lap=2, rain_intensity=.6), dict(lap=max(4, horizon * 2 // 3), rain_intensity=0.),
        ]),
        tire_warmup=warmup,
    )
    before = deepcopy((models, observed))
    shared, shared_visits = profiled_decision(models, options)
    preserve_native_helper(monkeypatch, lambda cache, key: None)
    independent, independent_visits = profiled_decision(models, options)
    assert shared == independent
    assert shared_visits <= independent_visits
    if horizon == 14:
        assert 0 < shared_visits < independent_visits
    assert models[:-1] == before[0][:-1]
    assert stock.__dict__ == before[0][-1].__dict__
    assert observed == before[1]


def test_shared_tables_keep_changed_native_inputs_and_reused_sets_independent():
    models = inputs(.2, .2, 4)
    driver, car, track, weather, stock = models
    clock = StrategyWeatherClock((0., 90., 180., 270.), 35., 90., 5, 20., 20.,
                                 update_offsets=(35., 125., 215., 305., 395.))
    options = dict(tire_age=11, remaining_stops=2, remaining_dry_stops=2,
                   remaining_damp_stops=2, used_compounds=(TireCompound.INTERMEDIATE,),
                   physical_total_laps=12, weather_clock=clock,
                   forecast_context=WeatherForecastContext.from_schedule([
                       dict(lap=2, rain_intensity=.6), dict(lap=4, rain_intensity=0.),
                   ]))

    @control_lap_scope
    def check_variants():
        for index in range(40):
            # More distinct forecasts than the LRU capacity, followed by a
            # repeated first case. Input changes must never reuse stale costs.
            step = index % 39
            alternate = deepcopy(models)
            alternate[3].track_temperature = 20. + step
            alternate[3].track_wetness = .13 + step * .01
            active = dict(options, physical_total_laps=12 + step % 2,
                          forecast_context=options["forecast_context"].advanced(step % 3))
            if step % 2:
                alternate[4].sets["I2"] = replace(alternate[4].sets["I2"], age=step)
            expected, _ = independent_schedules(alternate, active)
            actual = plan_inventory_strategy(*alternate, 1, **active)
            for stopped in (False, True):
                assert_continuation(actual.continuation(stopped), expected[stopped])

    check_variants()


def test_external_controlled_clocks_reuse_complete_fresh_costs_across_pool_histories(monkeypatch):
    benchmark = runpy.run_path(str(
        Path(__file__).parents[1] / "examples/benchmark_controlled_strategy.py"))["benchmark"]

    def execute():
        result = benchmark(laps=30, drivers=4, intervals=3, profile=True)
        assert result["native"]
        assert result["benchmark_version"] == 4
        return result, result["fresh_service_frame_visits"]

    shared, shared_visits = execute()
    preserve_native_helper(monkeypatch, lambda cache, key: None)
    independent, independent_visits = execute()
    assert shared["decision"] == independent["decision"]
    assert shared["outcome_sha256"] == independent["outcome_sha256"]
    assert shared["green_suffix_evaluations"] == independent["green_suffix_evaluations"]
    assert 0 < shared_visits < independent_visits


def test_relaxation_tables_are_bounded_and_reset_with_the_field_decision():
    driver, car, track, *_ = inputs(.2, .2)

    @control_lap_scope
    def populate():
        cache = control_lap_memo(driver, car, track, 12)
        first = control_relaxation_memo(cache, 0)
        first[0] = 123.
        for index in range(1, 33):
            control_relaxation_memo(cache, index)[0] = float(index)
        assert control_relaxation_memo(cache, 0) == {}
        control_relaxation_memo(cache, 32)[0] = 456.
        return cache

    cache = populate()
    assert control_relaxation_memo(cache, 32) is None
    populate()


def test_cancelled_nested_decision_discards_its_costs_and_restores_the_outer_scope():
    driver, car, track, *_ = inputs(.2, .2)

    @control_lap_scope
    def cancel_nested():
        cache = control_lap_memo(driver, car, track, 12)
        assert control_relaxation_memo(cache, "forecast") == {}
        control_relaxation_memo(cache, "forecast")[0] = 999.
        raise SimulationCancelled()

    @control_lap_scope
    def outer():
        cache = control_lap_memo(driver, car, track, 12)
        costs = control_relaxation_memo(cache, "forecast")
        costs[0] = 123.
        with pytest.raises(SimulationCancelled):
            cancel_nested()
        assert control_relaxation_memo(cache, "forecast") is costs
        assert costs == {0: 123.}
        return cache

    cache = outer()
    assert control_relaxation_memo(cache, "forecast") is None
    with pytest.raises(SimulationCancelled):
        cancel_nested()
