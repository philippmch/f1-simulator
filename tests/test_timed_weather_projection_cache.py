"""Reuse deterministic weather and stint physics without changing the forecast."""

from copy import deepcopy
from functools import lru_cache
from sys import _getframe

import pytest
from test_strategy_pit_weather import clock_for, exhaustive_same, models

from f1sim.cancellation import SimulationCancelled, cancellation_scope
from f1sim.models import TireCompound, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation import rain_strategy
from f1sim.simulation.inventory_strategy import plan_inventory_strategy
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.rain_strategy import _running_row, plan_rain_stop, plan_rain_transition
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.tire_inventory import TireInventory
from f1sim.simulation.weather_strategy import weather_stop_costs


@pytest.mark.parametrize("planner", ["transition", "weather", "inventory"])
def test_timed_surface_cache_advances_each_update_once(monkeypatch, planner):
    driver, car, track, _ = models(laps=5)
    weather = Weather(track_wetness=.19, rain_intensity=.39)
    tire = TIRE_COMPOUNDS[TireCompound.MEDIUM]
    clock = clock_for(track)
    calls = 0
    original = Weather.project_surface

    def counted(self):
        nonlocal calls
        calls += 1
        return original(self)

    monkeypatch.setattr(Weather, "project_surface", counted)
    if planner == "inventory":
        inventory = TireInventory.from_sets([
            {"compound": "medium", "age": 3}, {"compound": "intermediate"},
            {"compound": "hard"},
        ])
        inventory.fit("set-1")
        plan_inventory_strategy(driver, car, track, weather, inventory, 1,
                                remaining_stops=2, weather_clock=clock)
    elif planner == "transition":
        plan_rain_transition(driver, car, track, weather, tire, 3, 1, 2,
                             used_compounds=("medium",), weather_clock=clock)
    else:
        weather_stop_costs(driver, car, track, weather, tire, 3, 1, weather_clock=clock)
    assert 0 < calls <= clock.max_updates


def test_clocked_stints_reuse_physics_but_include_changed_car_performance(monkeypatch, request):
    _running_row.cache_clear()
    request.addfinalizer(_running_row.cache_clear)
    driver, car, track, weather = models()
    tire = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE]
    clock = clock_for(track)
    calls = 0
    original = LapSimulator.calculate_lap_time

    def counted(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(LapSimulator, "calculate_lap_time", counted)
    first = plan_rain_stop(driver, car, track, weather, tire, 39, 1, 2,
                           weather_clock=clock)
    initial_calls = calls
    renamed_driver = driver.model_copy(update={"id": "other", "team_id": "other"})
    renamed_car = car.model_copy(update={"team_id": "other", "team_name": "Other"})
    same = plan_rain_stop(renamed_driver, renamed_car, track, weather, tire, 39, 1, 2,
                          weather_clock=clock)
    assert same == first
    assert calls - initial_calls == 2  # Current traffic/control laps stay outside the cache.

    changed_car = renamed_car.model_copy(update={"tire_degradation_factor": 1.5})
    before = calls
    changed = plan_rain_stop(renamed_driver, changed_car, track, weather, tire, 39, 1, 2,
                             weather_clock=clock)
    assert calls > before + 2
    expected = exhaustive_same(deepcopy(driver), changed_car, track, weather,
                               tire, 39, 1, 2, clock)
    assert changed.pit_now_cost == pytest.approx(expected[0])
    assert changed.wait_cost == pytest.approx(expected[1])


@pytest.mark.parametrize("budget", [0, 2])
def test_clocked_stints_match_control_traffic_and_physical_fuel(budget):
    driver, car, track, weather = models()
    tire = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE].model_copy(
        update={"degradation_rate": .08},
    )
    clock = clock_for(track, current=27, future=13)
    expected = exhaustive_same(
        deepcopy(driver), car, track, weather, tire, 12, 1, budget, clock,
        queue=-2, modifier=1.2, aero=False, lane=.75,
        current_gaps=(.2, 1.1), physical_total_laps=10,
    )
    actual = plan_rain_stop(
        driver, car, track, weather, tire, 12, 1, budget, weather_clock=clock,
        additional_current_stop_cost=-2, current_lap_time_modifier=1.2,
        active_aero_enabled=False, pit_lane_factor=.75,
        current_traffic_gaps=(.2, 1.1), physical_total_laps=10,
    )
    assert actual.pit_now_cost == pytest.approx(expected[0])
    assert actual.wait_cost == pytest.approx(expected[1])


@pytest.mark.parametrize("budget", [0, 1, 3])
@pytest.mark.parametrize("wetness", [.07999999999999999, .08, .08000000000000002,
                                   .29999999999999993, .3, .30000000000000004, .44731])
@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("cadence", [(23.25, 83.125, 9, 103.5, 17.125),
                                    (81.25, 47.5, 4, 0., 128.75),
                                    (0., 90., 0, 25., 25.)])
def test_native_clock_costs_are_exactly_generic_with_fitting_and_cadence(
    monkeypatch, request, budget, wetness, pending, cadence,
):
    _running_row.cache_clear()
    request.addfinalizer(_running_row.cache_clear)
    driver, car, track, _ = models(laps=7)
    weather = Weather(track_wetness=wetness, rain_intensity=.39127)
    tire = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE].model_copy(
        update={"degradation_rate": .08123, "initial_grip": .9317,
                "cliff_threshold": 14, "cliff_multiplier": 4.17},
    )
    clock = StrategyWeatherClock((0., 81.25, 176.125, 233., 310.5, 475., 521.),
                                 *cadence)
    inputs = tuple(item.model_dump_json() for item in (driver, car, track, weather, tire))
    options = dict(weather_clock=clock, physical_total_laps=17,
                   current_lap_time_modifier=1.4, active_aero_enabled=False,
                   current_traffic_gaps=(.3125, 1.375), pit_lane_factor=.6,
                   additional_current_stop_cost=-3.125,
                   tire_warmup={"intermediate": 11.75}, current_fit_pending=pending)
    native = plan_rain_stop(driver, car, track, weather, tire, 13, 1, budget, **options)
    monkeypatch.setattr(LapSimulator, "prepare_deterministic_lap_time",
                        lambda *args: None)
    generic = plan_rain_stop(driver, car, track, weather, tire, 13, 1, budget, **options)
    assert native == generic
    assert tuple(item.model_dump_json() for item in (driver, car, track, weather, tire)) == inputs


def test_native_clock_shares_surface_path_and_never_reconstructs_rows(monkeypatch, request):
    _running_row.cache_clear()
    request.addfinalizer(_running_row.cache_clear)
    driver, car, track, weather = models(laps=8)
    tire = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE]
    clock = StrategyWeatherClock(tuple(index * 83. for index in range(8)),
                                 20., 90., 10, 95., 13.)
    row_calls = surface_calls = 0
    original_row = rain_strategy._running_row
    original_surface = Weather.project_surface

    def row(*args):
        nonlocal row_calls
        row_calls += 1
        return original_row(*args)

    def surface(self):
        nonlocal surface_calls
        surface_calls += 1
        return original_surface(self)

    monkeypatch.setattr(rain_strategy, "_running_row", row)
    monkeypatch.setattr(Weather, "project_surface", surface)
    native = plan_rain_stop(driver, car, track, weather, tire, 18, 1, 3,
                           weather_clock=clock)
    assert row_calls == 0
    native_surfaces = surface_calls
    assert 0 < native_surfaces <= clock.max_updates
    monkeypatch.setattr(LapSimulator, "prepare_deterministic_lap_time",
                        lambda *args: None)
    generic = plan_rain_stop(driver, car, track, weather, tire, 18, 1, 3,
                            weather_clock=clock)
    assert generic == native
    assert row_calls > 1
    assert surface_calls - native_surfaces > native_surfaces


@pytest.mark.parametrize("custom", ["physics", "clock"])
def test_native_shortcut_keeps_custom_dispatch(monkeypatch, request, custom):
    _running_row.cache_clear()
    request.addfinalizer(_running_row.cache_clear)
    driver, car, track, weather = models()
    tire = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE]
    clock = clock_for(track)
    calls = 0
    original = rain_strategy._running_row

    def row(*args):
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(rain_strategy, "_running_row", row)
    if custom == "physics":
        physics = LapSimulator.calculate_lap_time
        monkeypatch.setattr(LapSimulator, "calculate_lap_time",
                            lambda self, *args, **kwargs: physics(self, *args, **kwargs) + .125)
    else:
        class CustomClock(StrategyWeatherClock):
            def updates(self, offset, paid_stops=0, stopped_first=False, *, fit_delay=0.):
                return min(self.max_updates, super().updates(offset, paid_stops, stopped_first) + 1)
        clock = CustomClock(**{name: getattr(clock, name) for name in clock.__slots__})
    actual = plan_rain_stop(driver, car, track, weather, tire, 12, 1, 2,
                           weather_clock=clock)
    assert calls > 1
    monkeypatch.setattr(LapSimulator, "prepare_deterministic_lap_time", lambda *args: None)
    assert actual == plan_rain_stop(driver, car, track, weather, tire, 12, 1, 2,
                                   weather_clock=clock)


def test_native_clock_planning_does_not_sample_randomness(monkeypatch):
    class NoRandomDraws:
        def normal(self, *args, **kwargs):
            pytest.fail("Deterministic planning sampled driver variation")

    monkeypatch.setattr(rain_strategy.np.random, "default_rng", lambda *args: NoRandomDraws())
    driver, car, track, weather = models()
    plan_rain_stop(driver, car, track, weather, TIRE_COMPOUNDS[TireCompound.INTERMEDIATE],
                   12, 1, 2, weather_clock=clock_for(track))


@pytest.mark.parametrize("compound", list(TireCompound))
@pytest.mark.parametrize("age", [0, 19, 43])
@pytest.mark.parametrize("intervals", [None, (0, 0, 1, 1, 3), (0, 2, 4, 6, 8)])
def test_cached_stint_preparation_is_bit_exact(monkeypatch, request, compound, age, intervals):
    _running_row.cache_clear()
    request.addfinalizer(_running_row.cache_clear)
    driver, car, track, _ = models(laps=9)
    driver.skill_rating = .87123
    car.tire_degradation_factor = 1.2317
    weather = Weather(track_wetness=.30000000000000004, rain_intensity=.32719)
    tire = TIRE_COMPOUNDS[compound].model_copy(
        update={"initial_grip": .95171, "degradation_rate": .07923,
                "cliff_threshold": 21, "cliff_multiplier": 4.173},
    )
    args = (tuple(item.model_dump_json() for item in (driver, car, track)),
            weather.model_dump_json(), tire.model_dump_json(), age, 5, 17, intervals)
    native = _running_row(*args)
    _running_row.cache_clear()
    monkeypatch.setattr(LapSimulator, "prepare_deterministic_lap_time", lambda *args: None)
    assert native == _running_row(*args)


def test_cached_stint_preparation_preserves_custom_physics(monkeypatch, request):
    _running_row.cache_clear()
    request.addfinalizer(_running_row.cache_clear)
    driver, car, track, weather = models()
    tire = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE]
    args = (tuple(item.model_dump_json() for item in (driver, car, track)),
            weather.model_dump_json(), tire.model_dump_json(), 9, 1, 10)
    native = _running_row(*args)
    _running_row.cache_clear()
    original = LapSimulator.calculate_lap_time
    calls = 0

    def physics(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs) + .125

    monkeypatch.setattr(LapSimulator, "calculate_lap_time", physics)
    assert _running_row(*args) == tuple(value + .125 for value in native)
    assert calls == track.total_laps


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("cancel_after", [1, 2])
def test_zero_stop_clock_row_cancels_immediately_or_mid_row(
    monkeypatch, request, native, cancel_after,
):
    _running_row.cache_clear()
    request.addfinalizer(_running_row.cache_clear)
    driver, car, track, weather = models(laps=40)
    clock = clock_for(track)
    if not native:
        monkeypatch.setattr(LapSimulator, "prepare_deterministic_lap_time", lambda *args: None)
    polls = 0

    def cancel():
        nonlocal polls
        polls += 1
        return polls >= cancel_after

    with cancellation_scope(cancel), pytest.raises(SimulationCancelled):
        plan_rain_stop(driver, car, track, weather, TIRE_COMPOUNDS[TireCompound.INTERMEDIATE],
                       12, 1, 0, weather_clock=clock)
    assert polls == cancel_after


def test_native_row_cost_cache_hits_still_poll_cancellation(monkeypatch):
    driver, car, track, weather = models(laps=40)
    clock = StrategyWeatherClock(tuple(index * 90. for index in range(40)),
                                 0., 90., 0, 25., 25.)
    hits = 0

    def observed_cache(*args, **kwargs):
        decorate = lru_cache(*args, **kwargs)

        def apply(function):
            cached = decorate(function)
            if function.__name__ != "running":
                return cached

            def running(*key):
                nonlocal hits
                before = cached.cache_info().hits
                value = cached(*key)
                hits += cached.cache_info().hits - before
                return value

            return running

        return apply

    original_checkpoint = rain_strategy.cancellation_checkpoint

    def row_checkpoint():
        # Isolate row cancellation: the outer search also polls, and would
        # otherwise mask missing checks when all scalar costs are cache hits.
        if _getframe(1).f_code.co_name == "row":
            original_checkpoint()

    monkeypatch.setattr(rain_strategy, "lru_cache", observed_cache)
    monkeypatch.setattr(rain_strategy, "cancellation_checkpoint", row_checkpoint)
    with cancellation_scope(lambda: hits > 0), pytest.raises(SimulationCancelled):
        plan_rain_stop(driver, car, track, weather, TIRE_COMPOUNDS[TireCompound.INTERMEDIATE],
                       12, 1, 3, weather_clock=clock)
    assert hits > 0
