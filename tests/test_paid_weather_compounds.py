"""Complete small action oracle based on critical safety, never recommendation."""

import sys
from collections import OrderedDict
from copy import deepcopy
from cProfile import Profile
from functools import lru_cache
from itertools import product
from math import inf

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation import rain_strategy
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.rain_strategy import plan_rain_stop, plan_rain_transition
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.surface_projection import normalize_weather_intervals
from f1sim.simulation.weather_schedule import WeatherForecastContext
from f1sim.simulation.weather_strategy import weather_stop_costs


@pytest.mark.parametrize("used", [(), (TireCompound.SOFT,), (TireCompound.INTERMEDIATE,),
                                  (TireCompound.SOFT, TireCompound.INTERMEDIATE),
                                  (TireCompound.SOFT, TireCompound.HARD, TireCompound.WET)])
@pytest.mark.parametrize("warmup", [{}, {"intermediate": 35., "wet": 7., "soft": 3.}])
def test_timed_native_safety_and_wet_mask_quotient_match_uncached_hooks(monkeypatch, used, warmup):
    driver, car, track = models(laps=6)
    weather = Weather(track_wetness=.5, rain_intensity=.5)
    context = WeatherForecastContext.from_schedule([{"lap": 3, "rain_intensity": 1}])
    clock = StrategyWeatherClock(tuple(index * 170. for index in range(6)),
                                 10., 90., 8, 7., 7.)
    args = (driver, car, track, weather, TIRE_COMPOUNDS[TireCompound.INTERMEDIATE], 5, 1, 2)
    options = dict(used_compounds=used, remaining_dry_stops=1, remaining_damp_stops=1,
                   forecast_context=context, weather_clock=clock, tire_warmup=warmup,
                   current_fit_pending=True)
    native = plan_rain_transition(*args, **options)
    monkeypatch.setattr(rain_strategy, "shared_forecast_available", lambda: False)
    assert plan_rain_transition(*args, **options) == native


def test_timed_native_candidate_work_is_bounded_by_surfaces_and_custom_clock_keeps_hooks():
    driver, car, track = models(laps=8)
    weather = Weather(track_wetness=.5, rain_intensity=.5)
    fields = (tuple(index * 170. for index in range(8)), 10., 90., 8, 7., 7.)
    clock = StrategyWeatherClock(*fields)
    args = (driver, car, track, weather, TIRE_COMPOUNDS[TireCompound.INTERMEDIATE], 5, 1, 3)
    profile = Profile()
    native = profile.runcall(plan_rain_transition, *args, weather_clock=clock)
    code = rain_strategy.paid_compound_candidates.__code__
    native_calls = sum(entry.callcount for entry in profile.getstats() if entry.code is code)
    assert 0 < native_calls <= clock.max_updates + 1
    queries = []

    class ObservedClock(StrategyWeatherClock):
        def updates(self, *args, **kwargs):
            queries.append((args, kwargs))
            return super().updates(*args, **kwargs)

    profile = Profile()
    custom = profile.runcall(plan_rain_transition, *args, weather_clock=ObservedClock(*fields))
    custom_calls = sum(entry.callcount for entry in profile.getstats() if entry.code is code)
    assert custom == native
    assert queries and custom_calls > native_calls * 10


@pytest.mark.parametrize("used", [(), (TireCompound.SOFT,), (TireCompound.INTERMEDIATE,),
                                  (TireCompound.SOFT, TireCompound.HARD, TireCompound.WET)])
@pytest.mark.parametrize("warmup", [{}, {"intermediate": 35., "wet": 7.}])
def test_native_recursive_and_low_capacity_iterative_costs_match_exactly(monkeypatch, used, warmup):
    driver, car, track = models(laps=6)
    weather = Weather(track_wetness=.5, rain_intensity=.5)
    context = WeatherForecastContext.from_schedule([{"lap": 3, "rain_intensity": 1}])
    clock = StrategyWeatherClock(tuple(index * 170. for index in range(6)),
                                 10., 90., 8, 7., 7.)
    args = (driver, car, track, weather, TIRE_COMPOUNDS[TireCompound.INTERMEDIATE], 5, 1, 2)
    options = dict(used_compounds=used, remaining_dry_stops=1, remaining_damp_stops=1,
                   forecast_context=context, weather_clock=clock, tire_warmup=warmup,
                   current_fit_pending=True)
    recursive = plan_rain_transition(*args, **options)
    # Report low remaining capacity without reducing pytest's own interpreter
    # limit; dispatch must retain the complete graph on its explicit stack.
    monkeypatch.setattr(sys, "getrecursionlimit", lambda: 64)
    assert plan_rain_transition(*args, **options) == recursive


def test_long_native_clock_horizon_uses_iterative_fallback_without_truncation():
    driver, car, track = models(laps=500)
    weather = Weather(track_wetness=.5, rain_intensity=.5)
    clock = StrategyWeatherClock(tuple(index * 90. for index in range(500)),
                                 10., 90., 0, 7., 7.)
    result = plan_rain_transition(
        driver, car, track, weather, TIRE_COMPOUNDS[TireCompound.INTERMEDIATE], 0, 1, 0,
        weather_clock=clock, used_compounds=(TireCompound.INTERMEDIATE,),
    )
    simulator = LapSimulator(np.random.default_rng(0))
    expected = 0.
    # Fold backwards to match the graph's scalar addition order exactly.
    for offset in reversed(range(500)):
        driver.current_tire_laps = offset
        expected = simulator.calculate_lap_time(
            driver, car, track, TIRE_COMPOUNDS[TireCompound.INTERMEDIATE], weather,
            offset + 1, 500, sample_variation=False,
        ) + expected
    assert result.wait_cost == expected
    assert result.pit_now_cost == inf


@pytest.mark.parametrize("current,future,cap", [(0., 7., 6), (95., 5., 6), (7., 27., 3),
                                               (7., 7., 0)])
def test_clock_quotient_preserves_every_remaining_fit_and_retain_path(current, future, cap):
    horizon = 6
    clock = StrategyWeatherClock(tuple(index * 90. for index in range(horizon)),
                                 10., 90., cap, current, future)
    rows, _ = rain_strategy._equivalent_clock_branches(clock, horizon)

    def path(offset, paid, first, fits):
        counts = []
        for index, fitted in zip(range(offset, horizon), fits, strict=True):
            counts.append(clock.updates(index, paid, first))
            paid += fitted
            counts.append(clock.updates(index, paid, first))
        return counts

    for offset in range(1, horizon):
        for (paid, first), (representative_paid, representative_first) in rows[offset].items():
            for fits in product((False, True), repeat=horizon - offset):
                assert path(offset, paid, first, fits) == path(
                    offset, representative_paid, representative_first, fits,
                )
    if cap == 0:
        assert all(len(set(row.values())) == 1 for row in rows[1:])


@pytest.mark.parametrize("current_delay", [7., 97.])
def test_clock_quotient_keeps_over_budget_repairs_and_current_costs(current_delay):
    driver, car, track = models(laps=6)
    weather = Weather(track_wetness=.1, rain_intensity=0.)
    context = WeatherForecastContext.from_schedule(
        [{"lap": 3, "rain_intensity": 1.}, {"lap": 6, "rain_intensity": 0.}],
    )
    clock = StrategyWeatherClock(tuple(index * 90. for index in range(6)),
                                 10., 90., 8, current_delay, 7.)
    expected = exhaustive_safe_actions(
        driver, car, track, weather, TIRE_COMPOUNDS[TireCompound.WET], 5, 1, 0, context,
        clock=clock, dry=0, damp=0, physical=8,
        lane=.5, queue=17., modifier=1.3, aero=False, gaps=(.5, 2.),
    )
    actual = plan_rain_transition(
        driver, car, track, weather, TIRE_COMPOUNDS[TireCompound.WET], 5, 1, 0,
        forecast_context=context, weather_clock=clock,
        remaining_dry_stops=0, remaining_damp_stops=0, physical_total_laps=8,
        used_compounds=(),
        pit_lane_factor=.5, additional_current_stop_cost=17.,
        current_lap_time_modifier=1.3, active_aero_enabled=False,
        current_traffic_gaps=(.5, 2.),
    )
    assert actual.pit_now_cost == pytest.approx(expected[0])
    assert actual.wait_cost == expected[1] == inf
    assert actual.compound == expected[2]


def test_patched_native_clock_bypasses_clock_quotient_and_observes_new_counts(monkeypatch):
    driver, car, track = models(laps=6)
    weather = Weather(track_wetness=.5, rain_intensity=.5)
    context = WeatherForecastContext.from_schedule([{"lap": 3, "rain_intensity": 1.}])
    fields = (tuple(index * 90. for index in range(6)), 10., 90.)
    args = (driver, car, track, weather, TIRE_COMPOUNDS[TireCompound.INTERMEDIATE], 5, 1, 2)
    options = dict(forecast_context=context, used_compounds=(TireCompound.INTERMEDIATE,))
    expected = plan_rain_transition(*args, weather_clock=StrategyWeatherClock(*fields, 0, 7., 7.),
                                    **options)
    plan_rain_transition(*args, weather_clock=StrategyWeatherClock(*fields, 8, 7., 7.), **options)
    queries = []

    def no_updates(self, *args, **kwargs):
        queries.append((args, kwargs))
        return 0

    monkeypatch.setattr(StrategyWeatherClock, "updates", no_updates)
    profile = Profile()
    actual = profile.runcall(plan_rain_transition, *args,
                             weather_clock=StrategyWeatherClock(*fields, 8, 7., 7.), **options)
    assert actual == expected
    assert queries
    quotient = rain_strategy._equivalent_clock_branches.__code__
    assert not any(entry.code is quotient for entry in profile.getstats())


class NoSharedRefits(OrderedDict):
    def get(self, key, default=None):
        return default


@pytest.mark.parametrize("change", ["equivalent_clock", "boundary", "driver", "car", "track",
                                   "physical_distance", "fresh_tire", "weather", "schedule",
                                   "budget", "mask", "allowance", "retained_tire", "age"])
def test_shared_future_refits_match_native_search_without_shared_hits(monkeypatch, change):
    driver, car, track = models(laps=6)
    weather = Weather(track_wetness=.5, rain_intensity=.5)
    context = WeatherForecastContext.from_schedule([{"lap": 3, "rain_intensity": 1.}])
    clock = StrategyWeatherClock(tuple(index * 90. for index in range(6)),
                                 20., 90., 3, 5., 5.)
    monkeypatch.setattr(rain_strategy, "_refit_costs", OrderedDict())
    options = dict(forecast_context=context, weather_clock=clock,
                   used_compounds=(TireCompound.INTERMEDIATE,), remaining_dry_stops=1,
                   remaining_damp_stops=1, physical_total_laps=8,
                   current_lap_time_modifier=1.3, additional_current_stop_cost=11.,
                   pit_lane_factor=.5, active_aero_enabled=False,
                   current_traffic_gaps=(.5, 1.8))
    budget = 2
    retained = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE]
    age = 5
    plan_rain_transition(driver, car, track, weather, TIRE_COMPOUNDS[TireCompound.INTERMEDIATE],
                         5, 1, budget, **options)
    if change == "equivalent_clock":
        options["weather_clock"] = StrategyWeatherClock(clock.lap_start_offsets,
                                                        30., 90., 3, 6., 5.)
        driver = driver.model_copy(update={"id": "Other", "name": "Other", "team_id": "Other"})
        car = car.model_copy(update={"team_id": "Other", "team_name": "Other"})
    elif change == "boundary":
        options["weather_clock"] = StrategyWeatherClock(clock.lap_start_offsets,
                                                        5., 90., 3, 95., 5.)
    elif change == "driver":
        driver = driver.model_copy(update={"tire_management": .6})
    elif change == "car":
        car = car.model_copy(update={"tire_degradation_factor": 1.4, "pit_stop_avg": 4.})
    elif change == "track":
        track = track.model_copy(update={"base_lap_time": 97., "pit_lane_delta": 19.})
    elif change == "physical_distance":
        options["physical_total_laps"] = 12
    elif change == "fresh_tire":
        monkeypatch.setitem(
            TIRE_COMPOUNDS, TireCompound.WET,
            TIRE_COMPOUNDS[TireCompound.WET].model_copy(update={"initial_grip": .7}),
        )
    elif change == "weather":
        weather = weather.model_copy(update={"track_wetness": .68, "rain_intensity": .8})
    elif change == "schedule":
        options["forecast_context"] = WeatherForecastContext.from_schedule(
            [{"lap": 3, "rain_intensity": 0.}],
        )
    elif change == "budget":
        budget = 0
    elif change == "mask":
        options["used_compounds"] = ()
    elif change == "allowance":
        options["remaining_damp_stops"] = 0
    elif change == "retained_tire":
        retained = retained.model_copy(update={"initial_grip": .7, "degradation_rate": .05})
    elif change == "age":
        age = 19
    actual = plan_rain_transition(
        driver, car, track, weather, retained, age, 1, budget,
        **options,
    )
    monkeypatch.setattr(rain_strategy, "_refit_costs", NoSharedRefits())
    expected = plan_rain_transition(
        driver, car, track, weather, retained, age, 1, budget,
        **options,
    )
    assert actual == expected


def test_equivalent_raw_clocks_share_only_future_refits(monkeypatch):
    class CountHits(OrderedDict):
        hits = 0

        def get(self, key, default=None):
            value = super().get(key, default)
            if value is not None:
                self.hits += 1
            return value

    cache = CountHits()
    monkeypatch.setattr(rain_strategy, "_refit_costs", cache)
    driver, car, track = models(laps=6)
    weather = Weather(track_wetness=.5, rain_intensity=.5)
    fields = (tuple(index * 90. for index in range(6)),)
    args = (driver, car, track, weather, TIRE_COMPOUNDS[TireCompound.INTERMEDIATE], 5, 1, 2)
    options = dict(used_compounds=(TireCompound.INTERMEDIATE,))
    first = plan_rain_transition(*args, weather_clock=StrategyWeatherClock(*fields, 20., 90., 3,
                                                                          5., 5.), **options)
    old_hits = cache.hits
    second = plan_rain_transition(*args, weather_clock=StrategyWeatherClock(*fields, 30., 90., 3,
                                                                           6., 5.),
                                 current_lap_time_modifier=1.3, additional_current_stop_cost=11.,
                                 pit_lane_factor=.5, active_aero_enabled=False,
                                 current_traffic_gaps=(.5, 1.8), **options)
    assert cache.hits > old_hits
    assert second != first
    monkeypatch.setattr(rain_strategy, "_refit_costs", NoSharedRefits())
    assert plan_rain_transition(*args, weather_clock=StrategyWeatherClock(*fields, 30., 90., 3,
                                                                       6., 5.),
                               current_lap_time_modifier=1.3, additional_current_stop_cost=11.,
                               pit_lane_factor=.5, active_aero_enabled=False,
                               current_traffic_gaps=(.5, 1.8), **options) == second


def test_shared_clock_ids_are_bounded_and_never_recycled_after_eviction_or_reset(monkeypatch):
    monkeypatch.setattr(rain_strategy, "_clock_nodes", OrderedDict())
    monkeypatch.setattr(rain_strategy, "_refit_costs", OrderedDict())
    monkeypatch.setattr(rain_strategy, "_CLOCK_NODE_LIMIT", 3)
    monkeypatch.setattr(rain_strategy, "_REFIT_COST_LIMIT", 3)
    first = rain_strategy._shared_clock_node(("first",))
    for index in range(10):
        identity = rain_strategy._shared_clock_node((index,))
        rain_strategy._store_refit_cost((identity, index), float(index))
    assert len(rain_strategy._clock_nodes) == len(rain_strategy._refit_costs) == 3
    assert rain_strategy._shared_clock_node(("first",)) != first
    inherited = rain_strategy._shared_clock_node(("inherited",))
    rain_strategy._reset_green_cache_after_fork()
    assert not rain_strategy._clock_nodes and not rain_strategy._refit_costs
    assert rain_strategy._shared_clock_node(("inherited",)) != inherited


@pytest.mark.parametrize("used", [(), (TireCompound.SOFT,), (TireCompound.INTERMEDIATE,),
                                  (TireCompound.SOFT, TireCompound.HARD)])
@pytest.mark.parametrize("compound", [TireCompound.SOFT, TireCompound.INTERMEDIATE])
@pytest.mark.parametrize("budget", [0, 2])
@pytest.mark.parametrize("prescribed", [False, True])
def test_noncritical_budget_clock_matches_complete_action_oracle(
    used, compound, budget, prescribed,
):
    driver, car, track = models(laps=6)
    weather = Weather(track_wetness=.4, rain_intensity=.35)
    context = (WeatherForecastContext.from_schedule([{"lap": 3, "rain_intensity": .36}])
               if prescribed else None)
    clock = StrategyWeatherClock(tuple(index * 90. for index in range(6)),
                                 10., 90., 8, 97., 7.)
    expected = exhaustive_safe_actions(
        driver, car, track, weather, TIRE_COMPOUNDS[compound], 19, 1, budget, context,
        clock=clock, used=used, dry=0, damp=0, physical=8,
        lane=.75, queue=2., modifier=1.2, aero=False, gaps=(.6, 1.7),
    )
    profile = Profile()
    actual = profile.runcall(
        plan_rain_transition, driver, car, track, weather, TIRE_COMPOUNDS[compound], 19, 1,
        budget, forecast_context=context, weather_clock=clock, used_compounds=used,
        remaining_dry_stops=0, remaining_damp_stops=0, physical_total_laps=8,
        pit_lane_factor=.75, additional_current_stop_cost=2., current_lap_time_modifier=1.2,
        active_aero_enabled=False, current_traffic_gaps=(.6, 1.7),
    )
    assert actual.pit_now_cost == pytest.approx(expected[0])
    assert actual.wait_cost == pytest.approx(expected[1])
    assert actual.compound == expected[2]
    budget_code = rain_strategy._budget_clock_branches.__code__
    assert any(entry.code is budget_code for entry in profile.getstats())


def test_later_critical_rain_step_keeps_complete_compulsory_fit_clock():
    driver, car, track = models(laps=6)
    weather = Weather(track_wetness=.4, rain_intensity=.35)
    context = WeatherForecastContext.from_schedule([{"lap": 3, "rain_intensity": 1.}])
    clock = StrategyWeatherClock(tuple(index * 90. for index in range(6)), 10., 90., 8, 97., 7.)
    profile = Profile()
    actual = profile.runcall(
        plan_rain_transition, driver, car, track, weather, TIRE_COMPOUNDS[TireCompound.SOFT],
        19, 1, 0, forecast_context=context, weather_clock=clock,
        used_compounds=(TireCompound.SOFT, TireCompound.HARD),
    )
    expected = exhaustive_safe_actions(
        driver, car, track, weather, TIRE_COMPOUNDS[TireCompound.SOFT], 19, 1, 0, context,
        clock=clock, used=(TireCompound.SOFT, TireCompound.HARD),
    )
    assert actual.pit_now_cost == pytest.approx(expected[0])
    assert actual.wait_cost == pytest.approx(expected[1])
    assert actual.compound == expected[2]
    budget_code = rain_strategy._budget_clock_branches.__code__
    complete_code = rain_strategy._equivalent_clock_branches.__code__
    assert not any(entry.code is budget_code for entry in profile.getstats())
    assert any(entry.code is complete_code for entry in profile.getstats())


def models(laps=5, lane=5):
    return (Driver(id="D", name="D", team_id="T"), Car(team_id="T", team_name="T"),
            Track(id="T", name="T", country="T", total_laps=laps,
                  base_lap_time=90, pit_lane_delta=lane))


def exhaustive_safe_actions(driver, car, track, weather, tire, age, current_lap, budget,
                            context, *, intervals=None, clock=None, used=(), dry=None,
                            damp=None, warmup=0., pending=False, physical=None,
                            lane=1., queue=0., modifier=1., aero=True, gaps=(None, None)):
    """Enumerate every safe retain/paid-fit action and complete legal outcome.

    Leading event counts come from explicit event timestamps rather than the
    production clock method. Atmosphere changes are dispatched independently.
    Only a completed running lap grants compound-use credit.
    """
    driver = deepcopy(driver)
    context = context or WeatherForecastContext((), leading_lap=current_lap)
    physics = LapSimulator(np.random.default_rng(91))
    horizon = track.total_laps - current_lap + 1
    intervals = tuple(range(horizon)) if intervals is None else tuple(intervals)
    bits = {compound: 1 << index if index < 3 else 8
            for index, compound in enumerate(TireCompound)}
    mask = 0
    for compound in used:
        mask |= bits[compound]
    changes = {lap: rain for lap, rain, _ in context.schedule}

    @lru_cache(None)
    def surface(offset, paid, first_paid, fit_delay):
        count = intervals[offset]
        if clock is not None:
            elapsed = clock.lap_start_offsets[offset] + paid * clock.future_stop_delay
            if offset:
                elapsed += fit_delay
            if first_paid:
                elapsed += clock.current_stop_delay - clock.future_stop_delay
            count = (0 if offset == 0 and paid == 0 else sum(
                clock.first_update_after + index * clock.update_interval <= elapsed + 1e-12
                for index in range(clock.max_updates)
            ))
        value = weather.model_copy(deep=True)
        for shared_lap in range(context.leading_lap + 1, context.leading_lap + count + 1):
            if shared_lap in changes:
                value.rain_intensity = changes[shared_lap]
            value = value.project_surface()
        return value

    def legal(used):
        return bool(used & 8) or (used & 7).bit_count() >= 2

    def reduced(value):
        return None if value is None else max(0, value - 1)

    def running(offset, compound, age, retained, value, fitted, stop):
        driver.current_tire_laps = age
        value = physics.calculate_lap_time(
            driver, car, track, tire if retained else TIRE_COMPOUNDS[compound], value,
            current_lap + offset, track.total_laps if physical is None else physical,
            sample_variation=False, active_aero_enabled=aero if offset == 0 else True,
            gap_to_car_ahead=gaps[int(stop)] if offset == 0 else None,
        )
        return value * (modifier if offset == 0 else 1.) + (warmup if fitted else 0.)

    @lru_cache(None)
    def solve(offset, compound, age, retained, left, dry, damp, used, paid, first_paid, delay):
        if offset == horizon:
            return 0. if legal(used) else inf
        before = surface(offset, paid, first_paid, delay)
        critical = before.tire_mismatch(compound) == "critical"
        best = inf
        if not critical:
            fee = warmup if offset == 0 and retained and pending else 0.
            best = running(offset, compound, age, retained, before, bool(fee), False) + solve(
                offset + 1, compound, age + 1, retained, left, dry, damp,
                used | bits[compound], paid, first_paid, delay + fee,
            )
        limit = dry if before.track_wetness < .08 and before.rain_intensity < .15 else damp
        allowed = critical or (left > 0 and (
            compound in (TireCompound.INTERMEDIATE, TireCompound.WET)
            or before.track_wetness > .3 or limit is None or limit > 0
        ))
        for candidate in TireCompound:
            if before.tire_mismatch(candidate) == "critical":
                continue
            if not allowed and (legal(used) or used & bits[candidate]):
                continue
            after_first = first_paid or offset == 0
            after = surface(offset, paid + 1, after_first, delay)
            cost = (track.pit_lane_delta * (lane if offset == 0 else 1.)
                    + expected_stationary_time(car) + (queue if offset == 0 else 0.))
            cost += running(offset, candidate, 0, False, after, True, True)
            cost += solve(offset + 1, candidate, 1, False, max(0, left - 1),
                          reduced(dry), reduced(damp), used | bits[candidate],
                          paid + 1, after_first, delay + warmup)
            best = min(best, cost)
        return best

    before = surface(0, 0, False, 0.)
    wait = inf
    if before.tire_mismatch(tire.compound) != "critical":
        fee = warmup if pending else 0.
        wait = running(0, tire.compound, age, True, before, pending, False) + solve(
            1, tire.compound, age + 1, True, budget, dry, damp,
            mask | bits[tire.compound], 0, False, fee,
        )
    critical = before.tire_mismatch(tire.compound) == "critical"
    limit = dry if before.track_wetness < .08 and before.rain_intensity < .15 else damp
    allowed = critical or (budget > 0 and (
        tire.compound in (TireCompound.INTERMEDIATE, TireCompound.WET)
        or before.track_wetness > .3 or limit is None or limit > 0
    ))
    root_costs = {}
    for candidate in TireCompound:
        if before.tire_mismatch(candidate) == "critical":
            continue
        if not allowed and (legal(mask) or mask & bits[candidate]):
            continue
        after = surface(0, 1, True, 0.)
        cost = track.pit_lane_delta * lane + expected_stationary_time(car) + queue
        cost += running(0, candidate, 0, False, after, True, True)
        cost += solve(1, candidate, 1, False, max(0, budget - 1), reduced(dry), reduced(damp),
                      mask | bits[candidate], 1, True, warmup)
        root_costs[candidate] = cost
    selected = min(root_costs, key=root_costs.get) if root_costs else None
    pit = root_costs.get(selected, inf)
    return pit, wait, selected if pit < inf else None, root_costs


CASES = [
    (.50, .50, TireCompound.SOFT, [{"lap": 3, "rain_intensity": 1}]),
    (.19, .50, TireCompound.INTERMEDIATE, [{"lap": 3, "rain_intensity": 0}]),
    (.69, .90, TireCompound.WET, [{"lap": 3, "rain_intensity": 0}]),
    (.07, .20, TireCompound.MEDIUM,
     [{"lap": 3, "rain_intensity": .9}, {"lap": 5, "rain_intensity": 0}]),
]


@pytest.mark.parametrize("prescribed", [False, True])
@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("warmup,pending", [(0., False), (35., False), (35., True)])
@pytest.mark.parametrize("budget,limits,used", [
    (0, 0, ()), (2, 1, (TireCompound.SOFT,)),
    (2, 0, (TireCompound.SOFT, TireCompound.HARD)),
])
@pytest.mark.parametrize("water,rain,compound,entries", CASES)
def test_transition_matches_independent_safe_action_oracle(
    prescribed, external, warmup, pending, budget, limits, used, water, rain, compound, entries,
):
    driver, car, track = models()
    weather = Weather(track_wetness=water, rain_intensity=rain)
    context = (WeatherForecastContext.from_schedule(entries, leading_lap=2)
               if prescribed else None)
    horizon = track.total_laps - 1
    clock = (StrategyWeatherClock(tuple(index * 170. for index in range(horizon)),
                                  10., 90., 6, 7., 7.) if external else None)
    expected = exhaustive_safe_actions(
        driver, car, track, weather, TIRE_COMPOUNDS[compound], 19, 2, budget, context,
        clock=clock, used=used, dry=limits, damp=limits, warmup=warmup, pending=pending,
        physical=7, lane=.75, queue=2., modifier=1.2, aero=False, gaps=(.6, 1.7),
    )
    options = dict(used_compounds=used, remaining_dry_stops=limits,
                   remaining_damp_stops=limits, physical_total_laps=7,
                   pit_lane_factor=.75, additional_current_stop_cost=2.,
                   current_lap_time_modifier=1.2, active_aero_enabled=False,
                   current_traffic_gaps=(.6, 1.7),
                   tire_warmup={candidate.value: warmup for candidate in TireCompound},
                   current_fit_pending=pending)
    if clock is not None:
        options["weather_clock"] = clock
    actual = plan_rain_transition(
        driver, car, track, weather, TIRE_COMPOUNDS[compound], 19, 2, budget,
        forecast_context=context, **options,
    )
    assert actual.pit_now_cost == pytest.approx(expected[0])
    assert actual.wait_cost == pytest.approx(expected[1])
    assert actual.compound == expected[2]
    carried = normalize_weather_intervals(
        horizon, forecast_context=context or WeatherForecastContext(()),
    )
    assert plan_rain_transition(
        driver, car, track, weather, TIRE_COMPOUNDS[compound], 19, 2, budget,
        weather_intervals=carried, **options,
    ) == actual


@pytest.mark.parametrize("external", [False, True])
def test_safe_wet_paid_choice_beats_current_intermediate_recommendation(external):
    driver, car, track = models(laps=12)
    weather = Weather(track_wetness=.5, rain_intensity=.5)
    context = WeatherForecastContext.from_schedule(
        [{"lap": 3, "rain_intensity": 1}], leading_lap=2,
    )
    clock = (StrategyWeatherClock(tuple(index * 170. for index in range(11)),
                                  10., 90., 6, 7., 7.) if external else None)
    actual = plan_rain_transition(
        driver, car, track, weather, TIRE_COMPOUNDS[TireCompound.SOFT], 5, 2, 4,
        forecast_context=context, weather_clock=clock, used_compounds=(TireCompound.SOFT,),
    )
    assert actual.compound == TireCompound.WET
    assert weather.fresh_rain_compound() == TireCompound.INTERMEDIATE
    assert actual.pit_now_cost < (1215.516310 if external else 1212.190495)


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("warmup", [0., 35.])
def test_weather_stop_first_refit_bound_uses_all_safe_compounds(external, warmup):
    driver, car, track = models(laps=4)
    weather = Weather(track_wetness=.69, rain_intensity=.7)
    context = WeatherForecastContext.from_schedule([{"lap": 2, "rain_intensity": 1}])
    clock = (StrategyWeatherClock((0., 170., 340., 510.), 10., 90., 6, 7., 7.)
             if external else None)
    physics = LapSimulator(np.random.default_rng(9))
    changes = {lap: rain for lap, rain, _ in context.schedule}

    @lru_cache(None)
    def surface(offset, paid, first_paid, delay):
        count = offset
        if clock is not None:
            time = clock.lap_start_offsets[offset] + paid * clock.future_stop_delay
            if first_paid:
                time += clock.current_stop_delay - clock.future_stop_delay
            if offset:
                time += delay
            count = sum(clock.first_update_after + index * clock.update_interval <= time + 1e-12
                        for index in range(clock.max_updates))
        value = weather.model_copy(deep=True)
        for lap in range(2, count + 2):
            if lap in changes:
                value.rain_intensity = changes[lap]
            value = value.project_surface()
        return value

    def running(offset, compound, age, value):
        driver.current_tire_laps = age
        return physics.calculate_lap_time(
            driver, car, track, TIRE_COMPOUNDS[compound], value, offset + 1, 4,
            sample_variation=False,
        )

    @lru_cache(None)
    def future(offset, compound, age, paid, delay):
        if offset == 4:
            return 0.
        before = surface(offset, paid, True, delay)
        best = inf
        if before.tire_mismatch(compound) != "critical":
            best = running(offset, compound, age, before) + future(
                offset + 1, compound, age + 1, paid, delay,
            )
        for candidate in TireCompound:
            if before.tire_mismatch(candidate) == "critical":
                continue
            after = surface(offset, paid + 1, True, delay)
            cost = track.pit_lane_delta + expected_stationary_time(car) + warmup
            cost += running(offset, candidate, 0, after)
            cost += future(offset + 1, candidate, 1, paid + 1, delay + warmup)
            best = min(best, cost)
        return best

    root = {}
    for candidate in TireCompound:
        if weather.tire_mismatch(candidate) == "critical":
            continue
        cost = track.pit_lane_delta + expected_stationary_time(car) + warmup
        cost += running(0, candidate, 0, surface(0, 1, True, 0.))
        cost += future(1, candidate, 1, 1, warmup)
        root[candidate] = cost
    actual = weather_stop_costs(
        driver, car, track, weather, TIRE_COMPOUNDS[TireCompound.INTERMEDIATE], 5, 1,
        forecast_context=context, weather_clock=clock,
        tire_warmup={candidate.value: warmup for candidate in TireCompound},
    )
    assert actual.pit_now_cost == pytest.approx(min(root.values()))
    assert root[TireCompound.WET] < root[TireCompound.INTERMEDIATE]


@pytest.mark.parametrize("external", [False, True])
def test_empty_context_matches_no_context_and_same_compound_api(external):
    driver, car, track = models(laps=4)
    weather = Weather(track_wetness=.5, rain_intensity=.5)
    clock = (StrategyWeatherClock((0., 170., 340., 510.), 10., 90., 6, 7., 7.)
             if external else None)
    args = (driver, car, track, weather, TIRE_COMPOUNDS[TireCompound.INTERMEDIATE], 5, 1)
    options = {"weather_clock": clock}
    default = plan_rain_transition(*args, 2, **options)
    assert plan_rain_transition(*args, 2, forecast_context=WeatherForecastContext(()),
                                **options) == default
    assert plan_rain_stop(*args, 2, forecast_context=WeatherForecastContext(()),
                          **options) == plan_rain_stop(*args, 2, **options)
    assert weather_stop_costs(*args, forecast_context=WeatherForecastContext(()),
                               **options) == weather_stop_costs(*args, **options)


@pytest.mark.parametrize("external", [False, True])
def test_shared_candidate_helper_patch_invalidates_native_transition_suffixes(
    monkeypatch, external,
):
    driver, car, track = models(laps=12)
    weather = Weather(track_wetness=.5, rain_intensity=.5)
    context = WeatherForecastContext.from_schedule(
        [{"lap": 3, "rain_intensity": 1}], leading_lap=2,
    )
    args = (driver, car, track, weather, TIRE_COMPOUNDS[TireCompound.SOFT], 5, 2, 4)
    options = {"forecast_context": context, "used_compounds": (TireCompound.SOFT,)}
    if external:
        options["weather_clock"] = StrategyWeatherClock(
            tuple(index * 170. for index in range(11)), 10., 90., 8, 7., 7.,
        )
    original = plan_rain_transition(*args, **options)
    assert original.compound == TireCompound.WET
    monkeypatch.setattr(rain_strategy, "paid_compound_candidates",
                        lambda weather, context: (TireCompound.INTERMEDIATE,))
    changed = plan_rain_transition(*args, **options)
    assert changed.compound == TireCompound.INTERMEDIATE
    assert changed.pit_now_cost > original.pit_now_cost


def test_scheduled_same_rain_transition_keeps_general_safe_actions(monkeypatch):
    driver, car, track = models(laps=4)
    weather = Weather(track_wetness=.5, rain_intensity=.5)
    context = WeatherForecastContext.from_schedule([{"lap": 3, "rain_intensity": .5}])
    clock = StrategyWeatherClock((0., 170., 340., 510.), 10., 90., 6, 7., 7.)
    monkeypatch.setattr(rain_strategy, "_clock_rain_stop",
                        lambda *args, **kwargs: pytest.fail("same-compound specialization"))
    plan_rain_transition(
        driver, car, track, weather, TIRE_COMPOUNDS[TireCompound.INTERMEDIATE], 5, 1, 2,
        forecast_context=context, weather_clock=clock,
    )


def test_weather_method_patch_changes_safe_actions_without_stale_cached_costs(monkeypatch):
    driver, car, track = models(laps=12)
    weather = Weather(track_wetness=.5, rain_intensity=.5)
    context = WeatherForecastContext.from_schedule(
        [{"lap": 3, "rain_intensity": 1}], leading_lap=2,
    )
    args = (driver, car, track, weather, TIRE_COMPOUNDS[TireCompound.SOFT], 5, 2, 4)
    options = {"forecast_context": context, "used_compounds": (TireCompound.SOFT,)}
    assert plan_rain_transition(*args, **options).compound == TireCompound.WET
    original = Weather.tire_mismatch

    def mismatch(weather, compound):
        return "critical" if compound == TireCompound.WET else original(weather, compound)

    monkeypatch.setattr(Weather, "tire_mismatch", mismatch)
    assert plan_rain_transition(*args, **options).compound == TireCompound.INTERMEDIATE


def test_instance_weather_extension_keeps_intentional_base_projection_boundary():
    driver, car, track = models(laps=4)
    weather = Weather(track_wetness=.5, rain_intensity=.5)
    context = WeatherForecastContext.from_schedule([{"lap": 3, "rain_intensity": 1}])
    options = {"forecast_context": context, "used_compounds": (TireCompound.SOFT,)}
    expected = plan_rain_transition(
        driver, car, track, weather, TIRE_COMPOUNDS[TireCompound.SOFT], 5, 1, 2, **options,
    )
    weather.__dict__["tire_mismatch"] = lambda compound: "critical"
    actual = plan_rain_transition(
        driver, car, track, weather, TIRE_COMPOUNDS[TireCompound.SOFT], 5, 1, 2, **options,
    )
    assert actual == expected


@pytest.mark.parametrize("external", [False, True])
def test_ordinary_drying_prices_intermediate_before_wet_recommendation(external):
    driver, car, track = models(laps=12)
    weather = Weather(track_wetness=.76, rain_intensity=0)
    clock = (StrategyWeatherClock(tuple(index * 170. for index in range(11)),
                                  10., 90., 6, 7., 7.) if external else None)
    expected = exhaustive_safe_actions(
        driver, car, track, weather, TIRE_COMPOUNDS[TireCompound.SOFT], 5, 2, 4, None,
        clock=clock, used=(TireCompound.SOFT,),
    )
    actual = plan_rain_transition(
        driver, car, track, weather, TIRE_COMPOUNDS[TireCompound.SOFT], 5, 2, 4,
        weather_clock=clock, used_compounds=(TireCompound.SOFT,),
    )
    assert actual.pit_now_cost == pytest.approx(expected[0])
    assert actual.compound == expected[2] == TireCompound.INTERMEDIATE
    assert expected[3][TireCompound.INTERMEDIATE] < expected[3][TireCompound.WET]
    assert weather.fresh_rain_compound() == TireCompound.WET
