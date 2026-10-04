"""Native suffix pruning preserves physical schedules and exact cache values."""

import sys
from collections import Counter
from copy import deepcopy
from math import inf, nextafter

import pytest
from test_inventory_control_handoff import observed_context
from test_inventory_partial_strategies import assert_continuation, independent_schedules

from f1sim.models import Car, Driver, Track, Weather
from f1sim.models._native import native_physics
from f1sim.simulation.inventory_strategy import _bounded_inventory_suffix, plan_inventory_strategy
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.strategy_control_clock import ProjectedControlCost
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.tire_inventory import TireInventory
from f1sim.simulation.weather_schedule import WeatherForecastContext


def test_a_suffix_excluded_by_one_prefix_is_researched_with_a_looser_budget():
    # A first 50-second finish makes B too expensive after its 40-second entry.
    # C subsequently reaches B for free. Reusing B's truncated value as an exact
    # result would miss the independently evident 20-second A -> C -> B finish.
    graph = {"A": [("X", 50., 0.), ("P", 40., 0.), ("C", 0., 0.)],
             "X": [("Y", 0., 0.)], "Y": [("end-first", 0., 0.)],
             "P": [("B", 0., 0.)], "B": [("end-later", 20., 0.)],
             "C": [("B", 0., 0.)]}
    calls = Counter()

    def actions(state):
        calls[state] += 1
        return graph[state]

    solved, excluded = {}, {}
    result = _bounded_inventory_suffix(
        "A", solved, excluded, actions,
        lambda state: ProjectedControlCost(0, 0.) if state.startswith("end-") else None)
    assert result.finished and result.seconds == 20. and result.laps == 3
    assert calls["B"] == 2
    assert solved["B"].seconds == 20.
    # A later root may also reuse an exact value despite an earlier exclusion.
    assert _bounded_inventory_suffix("B", solved, excluded, actions, lambda _: None).seconds == 20.


def test_a_completion_bound_cannot_discard_a_longer_retirement():
    graph = {0: [(1, 10., inf), (2, 15., inf)], 1: [(2, 10., inf)]}
    result = _bounded_inventory_suffix(
        0, {}, {}, lambda state: graph[state],
        lambda state: ProjectedControlCost(0, 0., False) if state == 2 else None)
    assert not result.finished and result.laps == 2 and result.seconds == 20.


def test_prefix_cutoff_keeps_a_one_ulp_improvement():
    smaller = nextafter(100., -inf)
    graph = {"A": [("end", 100., 0.), ("B", 0., smaller)],
             "B": [("end", smaller, 0.)]}
    result = _bounded_inventory_suffix(
        "A", {}, {}, lambda state: graph[state],
        lambda state: ProjectedControlCost(0, 0.) if state == "end" else None)
    assert result.seconds == smaller


def models(water=.2, rain=.2, limited=False):
    records = [dict(id=key, compound=compound, age=age,
                    remaining_laps=1 if limited else None)
               for key, compound, age in (("S", "soft", 3), ("M", "medium", 0),
                                          ("I", "intermediate", 0),
                                          ("I-used", "intermediate", 4), ("W", "wet", 2))]
    stock = TireInventory.from_sets(records)
    stock.fit("I")
    return (Driver(id="A", name="Synthetic", team_id="A"),
            Car(team_id="A", team_name="Synthetic"),
            Track(id="T", name="Synthetic", country="Test", total_laps=4,
                  base_lap_time=90., pit_lane_delta=20.),
            Weather(track_wetness=water, rain_intensity=rain, change_probability=0.), stock)


@pytest.mark.parametrize("name", ["stationary", "returning_rain", "drying", "limited"])
@pytest.mark.parametrize("clocked", [False, True])
@pytest.mark.parametrize("fitting_fees", [False, True])
@pytest.mark.parametrize("used,require", [(("soft",), True), (("medium", "hard"), True),
                                       (("soft", "hard", "intermediate"), True), ((), False)])
def test_native_suffixes_match_every_physical_schedule(name, clocked, fitting_fees, used, require):
    args = models(.8 if name == "drying" else .2, 0. if name == "drying" else .2,
                  limited=name == "limited")
    options = dict(tire_age=0, used_compounds=used, require_compound_rule=require,
                   remaining_stops=2, remaining_dry_stops=1, remaining_damp_stops=2,
                   physical_total_laps=20, current_fit_pending=True)
    if fitting_fees:
        options["tire_warmup"] = {"soft": 2., "medium": 5., "intermediate": 60., "wet": 3.}
    if name in {"stationary", "returning_rain"}:
        options["forecast_context"] = WeatherForecastContext.from_schedule([
            dict(lap=2, rain_intensity=.85 if name == "returning_rain" else .2),
            dict(lap=4, rain_intensity=.2)])
    if clocked:
        stop = args[2].pit_lane_delta + expected_stationary_time(args[1])
        options["weather_clock"] = StrategyWeatherClock((0., 80., 160., 240.),
                                                        20., 20., 18, stop, stop)
    else:
        options["weather_intervals"] = (0, 1, 8, 18)
    before = deepcopy((args[:-1], args[-1].__dict__, options))
    assert native_physics(*args[:-1])
    expected, first_choices = independent_schedules(args, options)
    decision = plan_inventory_strategy(*args, 1, **options)
    assert_continuation(decision.continuation(False), expected[False])
    assert_continuation(decision.continuation(True), expected[True])
    if decision.set_id is not None:
        assert first_choices[decision.set_id] == pytest.approx(expected[True], abs=1.e-8)
    assert (args[:-1], args[-1].__dict__, options) == before


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("control", ["sc", "vsc"])
@pytest.mark.parametrize("used", [("soft",), ("medium", "hard"), ("intermediate",)])
def test_satisfied_compound_credit_survives_control_to_green_handoff(engine, control, used):
    args = models(0., 0.)
    options = dict(tire_age=0, used_compounds=used, remaining_stops=2,
                   remaining_dry_stops=2, remaining_damp_stops=2, physical_total_laps=20,
                   control_context=observed_context(args[2], args[1], engine, control, 2))
    before = deepcopy((args[:-1], args[-1].__dict__, options))
    expected, first_choices = independent_schedules(args, options)
    decision = plan_inventory_strategy(*args, 1, **options)
    assert_continuation(decision.continuation(False), expected[False])
    assert_continuation(decision.continuation(True), expected[True])
    if decision.set_id is not None:
        assert first_choices[decision.set_id] == pytest.approx(expected[True], abs=1.e-8)
    assert (args[:-1], args[-1].__dict__, options) == before


@pytest.mark.parametrize("clocked", [False, True])
def test_replaced_physics_bypasses_native_suffix_pruning(monkeypatch, clocked):
    args = models(.8, 0.)
    original = LapSimulator.calculate_lap_time

    def changed(self, driver, *args, **kwargs):
        return original(self, driver, *args, **kwargs) + .2 * (driver.current_tire_laps % 3)

    monkeypatch.setattr(LapSimulator, "calculate_lap_time", changed)
    assert not native_physics(*args[:-1])
    calls = []

    def profile(frame, event, argument):
        if event == "call" and frame.f_code.co_name == "_bounded_inventory_suffix":
            calls.append(frame.f_code.co_name)

    options = dict(tire_age=0, remaining_stops=2, used_compounds=("intermediate",),
                   physical_total_laps=20)
    if clocked:
        options["weather_clock"] = StrategyWeatherClock((0., 80., 160., 240.),
                                                        20., 20., 18, 24., 24.)
    expected, _ = independent_schedules(args, options)
    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        decision = plan_inventory_strategy(*args, 1, **options)
    finally:
        sys.setprofile(previous)
    assert_continuation(decision.continuation(False), expected[False])
    assert_continuation(decision.continuation(True), expected[True])
    assert not calls


@pytest.mark.parametrize("clocked", [False, True])
def test_seven_set_scheduled_race_does_not_expand_every_stint_combination(clocked):
    driver, car, track, weather, _ = models()
    track.total_laps = 30
    stock = TireInventory.from_sets([
        dict(id=key, compound=compound) for key, compound in (
            ("M1", "medium"), ("M2", "medium"), ("S", "soft"), ("H", "hard"),
            ("I1", "intermediate"), ("I2", "intermediate"), ("W", "wet"))])
    stock.fit("M1")
    options = dict(tire_age=1, remaining_stops=4, remaining_dry_stops=3,
                   remaining_damp_stops=2, used_compounds=("medium",),
                   forecast_context=WeatherForecastContext.from_schedule([
                       dict(lap=10, rain_intensity=.5), dict(lap=20, rain_intensity=0.)],
                       leading_lap=2))
    if clocked:
        stop = track.pit_lane_delta + expected_stationary_time(car)
        options["weather_clock"] = StrategyWeatherClock(
            tuple(94. * i for i in range(29)), 90., 94., 28, stop, stop)
    count = 0

    def profile(frame, event, argument):
        nonlocal count
        if (event == "call" and frame.f_code.co_name == "make_actions"
                and frame.f_code.co_filename.endswith("inventory_strategy.py")):
            count += 1
            assert count < 20_000, "Scheduled finite stock expanded too many suffixes"

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        decision = plan_inventory_strategy(driver, car, track, weather, stock, 2, **options)
    finally:
        sys.setprofile(previous)
    assert decision.wait_laps == decision.pit_now_laps == 29
    assert decision.wait_cost < inf and decision.pit_now_cost < inf
    assert decision.set_id in {"I1", "I2"}
