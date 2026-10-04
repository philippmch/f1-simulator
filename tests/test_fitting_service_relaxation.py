"""Fee-delay envelopes stay optimistic against complete physical schedules."""

import cProfile
import runpy
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from test_controlled_weather_strategy import inputs
from test_inventory_partial_strategies import assert_continuation, independent_schedules

from f1sim.models import Car, Driver, TireCompound, Track, Weather, _native
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.inventory_strategy import (
    _fitting_inventory_completion_bound,
    plan_inventory_strategy,
)
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.tire_inventory import TireInventory, tire_set_slot
from f1sim.simulation.weather_schedule import WeatherForecastContext


@pytest.mark.parametrize("profile", [
    {compound.value: .5 for compound in TireCompound},
    {"soft": .1, "medium": .2, "hard": .3, "intermediate": .5, "wet": .7},
    {"intermediate": 60.}, {"wet": 60., "hard": .1},
])
@pytest.mark.parametrize("limited", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_fee_bound_covers_all_finite_stock_weather_and_fitting_histories(
    profile, limited, explicit,
):
    driver, car, track, weather, stock = models = inputs(.25, .2, 4)
    if limited:
        stock.sets = {key: replace(item, remaining_laps=3)
                      for key, item in stock.sets.items()}
    green = track.pit_lane_delta + expected_stationary_time(car)
    clock = StrategyWeatherClock(
        (0., 90., 180., 270.), 20., 80., 8, green, green,
        update_offsets=tuple(20. + i * 80. for i in range(8)) if explicit else None)
    forecast = WeatherForecastContext.from_schedule([
        dict(lap=2, rain_intensity=.8), dict(lap=4, rain_intensity=0.),
    ])
    options = dict(tire_age=11, physical_total_laps=12, remaining_stops=2,
                   remaining_dry_stops=2, remaining_damp_stops=2,
                   used_compounds=("intermediate",), weather_clock=clock,
                   forecast_context=forecast, tire_warmup=profile)
    expected, _ = independent_schedules(models, options)
    finishes = [-time for finished, _, time in expected.values() if finished]
    assert finishes
    projected = [weather]
    for update in range(clock.max_updates):
        projected.append(forecast.advanced(update).project_next(projected[-1]))
    mean_driver = deepcopy(driver)
    physics = LapSimulator()

    def running(offset, compound, age, update):
        mean_driver.current_tire_laps = age
        return physics.calculate_lap_time(
            mean_driver, car, track, TIRE_COMPOUNDS[TireCompound(compound)], projected[update],
            offset + 1, options["physical_total_laps"], sample_variation=False)

    bound = _fitting_inventory_completion_bound(
        track.total_laps, tuple(compound.value for compound in TireCompound), running,
        lambda offset, paid, stopped, delay:
            clock.updates(offset, paid, stopped, fit_delay=delay),
        lambda update, compound: projected[update].tire_mismatch(TireCompound(compound))
            == "critical",
        lambda offset, paid, stopped, delay: (paid, stopped, delay), profile, green)
    expiry = tire_set_slot(stock.sets[stock.current_set_id])[2]
    actual = bound(0, "intermediate", options["tire_age"], (0, False, 0.), expiry)
    assert actual <= min(finishes) + 1.e-8


def test_deferred_service_bound_activates_and_keeps_the_exact_physical_optimum():
    driver = Driver(id="A", name="A", team_id="T")
    car = Car(team_id="T", team_name="T", pit_stop_avg=1.5, pit_stop_std=.1)
    track = Track(id="T", name="T", country="T", total_laps=12,
                  base_lap_time=300., pit_lane_delta=.1)
    stock = TireInventory.from_sets([dict(id=key, compound=compound) for key, compound in
        (("M", "medium"), ("S1", "soft"), ("S2", "soft"), ("H", "hard"))])
    stock.fit("M")
    models = driver, car, track, Weather(change_probability=0.), stock
    green = track.pit_lane_delta + expected_stationary_time(car)
    options = dict(
        tire_age=50, remaining_stops=5, remaining_dry_stops=5, remaining_damp_stops=5,
        physical_total_laps=20, used_compounds=("soft", "medium"),
        forecast_context=WeatherForecastContext.from_schedule([]),
        weather_clock=StrategyWeatherClock(tuple(300. * i for i in range(12)), 100., 300., 12,
                                           green, green),
        tire_warmup={"soft": .1, "medium": .2, "hard": .3})
    before = deepcopy((models, options))
    assert _native.native_physics(*models[:4])
    expected, _ = independent_schedules(models, options)
    profiler = cProfile.Profile()
    actual = profiler.runcall(plan_inventory_strategy, *models, 1, **options)
    profiler.create_stats()
    assert sum(value[1] for key, value in profiler.stats.items()
               if key[2] == "build_fitting_bound") == 1
    for stopped in (False, True):
        assert_continuation(actual.continuation(stopped), expected[stopped])
    assert models[:-1] == before[0][:-1]
    assert stock.__dict__ == before[0][-1].__dict__
    assert options == before[1] and _native.native_physics(*models[:4])


def test_opening_benchmark_records_a_valid_full_horizon_observed_field():
    benchmark = runpy.run_path(str(
        Path(__file__).parents[1] / "examples/benchmark_controlled_strategy.py"))["benchmark"]
    result = benchmark(laps=8, drivers=2, intervals=1, opening=True,
                       tire_warmup={"intermediate": .5})
    assert result["native"] and result["benchmark_version"] == 5
    assert result["opening"] and result["current_lap"] == 1 and result["now"] == 0.
    assert all(row["completed_laps"] == 0 and row["running_start"] == 0.
               and row["ready"] > 0. for row in result["rivals"])
    assert result["tire_warmup"] == {"intermediate": .5}
    assert result["inventory_state_expansions"] is None
    with pytest.raises(ValueError, match="opening must be a boolean"):
        benchmark(opening=1)
