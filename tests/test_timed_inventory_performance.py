"""Bounded finite-inventory regressions for the timed strategy solver."""

import sys
from collections import Counter

import pytest

from f1sim.models import Car, Driver, Track, Weather
from f1sim.simulation.inventory_strategy import plan_inventory_strategy
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.tire_inventory import TireInventory


def _models_and_inventory(weather=None):
    models = (
        Driver(id="d", name="D", team_id="team"),
        Car(team_id="team", team_name="Team"),
        Track(id="track", name="Track", country="Test", total_laps=15,
              base_lap_time=90, pit_lane_delta=3),
        weather or Weather(),
    )
    # This is the original finite-pool drying probe: every physical choice is
    # distinct and the weather becomes progressively less restrictive.
    inventory = TireInventory.from_sets([
        {"compound": compound, "age": age}
        for compound, age in (
            ("soft", 0), ("medium", 1), ("hard", 2),
            ("intermediate", 3), ("wet", 4), ("soft", 5),
            ("medium", 6), ("hard", 7),
        )
    ])
    inventory.fit("set-1")
    return models, inventory


def _clock(track, *, first_update_after=10.0, max_updates=0,
           current_stop_delay=24.0, future_stop_delay=24.0):
    offsets = tuple(float(90 * index) for index in range(track.total_laps))
    return StrategyWeatherClock(
        offsets, first_update_after=first_update_after, update_interval=90.0,
        max_updates=max_updates, current_stop_delay=current_stop_delay,
        future_stop_delay=future_stop_delay,
    )


def _options(remaining_stops=2, physical_total_laps=20):
    return dict(
        tire_age=0,
        remaining_stops=remaining_stops,
        remaining_dry_stops=remaining_stops,
        remaining_damp_stops=remaining_stops,
        physical_total_laps=physical_total_laps,
        current_traffic_gaps=(None, None),
        require_compound_rule=True,
        used_compounds=("soft",),
    )


def test_timed_eight_set_fifteen_lap_search_is_bounded_and_exact(monkeypatch):
    """The timed branch stays compact and preserves the legacy exact result.

    With a zero-update clock, every projected surface is the decision surface,
    so the timed and legacy planners have the same objective.  This gives an
    independent result check while the call-count assertion protects the
    finite-pool branch-and-bound from silently regressing to an exhaustive
    eight-set schedule walk.
    """
    models, inventory = _models_and_inventory()
    clock = _clock(models[2])
    options = _options()
    original = LapSimulator.calculate_lap_time
    calls = []

    def counted(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(LapSimulator, "calculate_lap_time", counted)
    timed = plan_inventory_strategy(*models, inventory, 1,
                                    weather_clock=clock, **options)
    timed_calls = len(calls)

    calls.clear()
    legacy = plan_inventory_strategy(
        *models, inventory, 1,
        weather_intervals=(0,) * models[2].total_laps,
        **options,
    )

    assert timed.wait_cost == pytest.approx(legacy.wait_cost)
    assert timed.pit_now_cost == pytest.approx(legacy.pit_now_cost)
    assert timed.set_id == legacy.set_id
    assert timed.compound == legacy.compound
    # This threshold is deliberately above the current physical call count;
    # it catches a full schedule expansion without making wall time part of
    # the test contract.
    assert timed_calls < 20_000


def test_timed_drying_eight_set_search_keeps_state_expansion_bounded():
    """The original uncapped changing-surface probe does not explode the DAG."""
    models, inventory = _models_and_inventory(
        Weather(track_wetness=.25, rain_intensity=0.0),
    )
    clock = _clock(
        models[2], first_update_after=45.0, max_updates=15,
        current_stop_delay=30.0, future_stop_delay=25.0,
    )
    options = _options(remaining_stops=4, physical_total_laps=15)
    counts = Counter()

    def profile(frame, event, _argument):
        if event == "call" and frame.f_code.co_name in {"make_actions", "solve"}:
            counts[frame.f_code.co_name] += 1

    previous_profile = sys.getprofile()
    sys.setprofile(profile)
    try:
        result = plan_inventory_strategy(*models, inventory, 1,
                                          weather_clock=clock, **options)
    finally:
        sys.setprofile(previous_profile)

    assert result.wait_cost < float("inf")
    assert result.pit_now_cost < float("inf")
    # The current solver visits about 46k action states for this deliberately
    # branching case.  A broad ceiling catches an exhaustive schedule walk
    # while leaving room for small changes in the deterministic model.
    assert counts["make_actions"] < 80_000


def test_full_distance_wetting_search_prices_the_first_future_stop():
    """A normal three-set, full-distance decision must not expand every stint."""
    driver = Driver(id="d", name="D", team_id="team", tire_management=.99)
    car = Car(team_id="team", team_name="Team")
    track = Track(id="t", name="Track", country="Test", total_laps=53,
                  base_lap_time=83.5, pit_lane_delta=21, tire_stress=.38)
    weather = Weather(track_wetness=.214, rain_intensity=.35)
    inventory = TireInventory.from_sets([
        {"compound": compound, "age": age}
        for compound, age in (("soft", 6), ("hard", 0), ("intermediate", 4))
    ])
    inventory.fit("set-1")
    clock = StrategyWeatherClock(tuple(89.8 * i for i in range(52)),
                                 88.63, 89.19, 51, 23.75, 23.75)
    count = 0

    def profile(frame, event, _argument):
        nonlocal count
        if event == "call" and frame.f_code.co_name == "make_actions":
            count += 1
            assert count < 5_000, "Full-distance inventory search expanded too many states"

    previous_profile = sys.getprofile()
    sys.setprofile(profile)
    try:
        result = plan_inventory_strategy(
            driver, car, track, weather, inventory, 2, tire_age=6,
            remaining_stops=4, remaining_dry_stops=3, remaining_damp_stops=2,
            used_compounds=("soft",), weather_clock=clock,
        )
    finally:
        sys.setprofile(previous_profile)
    assert result.wait_cost < float("inf")
    assert result.pit_now_cost < float("inf")
