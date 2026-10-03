"""Exhausted-set history does not multiply otherwise identical continuations."""

import json
import runpy
import sys
from copy import deepcopy
from dataclasses import replace
from math import inf
from pathlib import Path

import pytest
from test_controlled_dry_strategy import single_car_context
from test_inventory_partial_strategies import assert_continuation, independent_schedules

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models._native import native_physics
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.chronological_finish import ChronologicalFinishContext
from f1sim.simulation.inventory_strategy import plan_inventory_strategy
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race_timing import RaceFinishTimeline
from f1sim.simulation.strategy_control_clock import StrategyControlContext
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.tire_inventory import TireInventory


def models(name="one_lap"):
    records, water, laps = {
        "one_lap": ([dict(id="H0", compound="hard", remaining_laps=1),
                     dict(id="H1", compound="hard", age=35, remaining_laps=1),
                     dict(id="H2", compound="hard", age=70, remaining_laps=1),
                     dict(id="S", compound="soft", age=3, remaining_laps=1)], 0., 6),
        "dry_rule": ([dict(id="H0", compound="hard", remaining_laps=1),
                      dict(id="H1", compound="hard", age=35, remaining_laps=1),
                      dict(id="H2", compound="hard", age=70, remaining_laps=1)], 0., 3),
        "equilibrium": ([dict(id="I0", compound="intermediate", remaining_laps=1),
                         dict(id="I1", compound="intermediate", age=35, remaining_laps=1),
                         dict(id="I2", compound="intermediate", age=70, remaining_laps=1),
                         dict(id="I3", compound="intermediate", age=3, remaining_laps=1)], .3, 6),
        "reusable": ([dict(id="I0", compound="intermediate", remaining_laps=1),
                      dict(id="I1", compound="intermediate", age=25, remaining_laps=3),
                      dict(id="W", compound="wet", age=2, remaining_laps=1),
                      dict(id="H", compound="hard", remaining_laps=2)], .24, 8),
    }[name]
    stock = TireInventory.from_sets(records)
    stock.fit(records[0]["id"])
    return (Driver(id="A", name="Synthetic", team_id="A"),
            Car(team_id="A", team_name="Synthetic"),
            Track(id="T", name="Synthetic", country="Test", total_laps=laps,
                  base_lap_time=90., pit_lane_delta=20.),
            Weather(track_wetness=water, rain_intensity=water if name == "equilibrium" else 0.,
                    change_probability=0), stock)


@pytest.mark.parametrize("name", ["one_lap", "dry_rule", "reusable", "equilibrium"])
@pytest.mark.parametrize("cadence", ["own", "external", "standard_sc", "chronological_vsc"])
@pytest.mark.parametrize("budget", [0, 2])
def test_expired_states_preserve_all_physical_choices_and_compound_credit(name, cadence, budget):
    driver, car, track, weather, stock = models(name)
    options = dict(tire_age=0, remaining_stops=budget, remaining_dry_stops=budget,
                   remaining_damp_stops=budget, physical_total_laps=20,
                   tire_warmup={"hard": 1., "soft": 3., "intermediate": 2., "wet": 4.})
    if cadence == "external":
        stop = track.pit_lane_delta + expected_stationary_time(car)
        options["weather_clock"] = StrategyWeatherClock(
            tuple(90. * i for i in range(track.total_laps)), 35., 100., 12, stop, stop)
    elif cadence != "own":
        observed = single_car_context(track, car, "sc" if cadence == "standard_sc" else "vsc", 2)
        field = (replace(observed.field, stop_delay=observed.current_stop_delay)
                 if cadence == "standard_sc" else ChronologicalFinishContext(
                     "A", RaceFinishTimeline(20, ["A"]), ("A",), (), 90., 1.2,
                     False, control_intervals=2))
        options["control_context"] = StrategyControlContext(
            field, 50., observed.current_stop_delay)
    snapshot = deepcopy((driver, car, track, weather, stock.__dict__, options))
    expected, costs = independent_schedules((driver, car, track, weather, stock), options)
    result = plan_inventory_strategy(driver, car, track, weather, stock, 1, **options)
    assert_continuation(result.continuation(False), expected[False])
    assert_continuation(result.continuation(True), expected[True])
    if result.set_id is not None:
        assert costs[result.set_id] == pytest.approx(expected[True], abs=1.e-8)
    assert (driver, car, track, weather, stock.__dict__, options) == snapshot
    if name == "dry_rule":
        assert result.wait_cost == result.pit_now_cost == inf
        assert result.wait_laps == 2


@pytest.mark.parametrize("clocked", [False, True])
@pytest.mark.parametrize("sets", [10, 14])
def test_distinct_one_lap_stock_shares_suffixes_after_expiry(clocked, sets):
    driver, car, track, weather, _ = models()
    track.total_laps = 30
    stock = TireInventory.from_sets([
        dict(id=f"H{i}", compound="hard", age=i, remaining_laps=1) for i in range(sets)])
    stock.fit("H0")
    options = dict(remaining_stops=0, remaining_dry_stops=0, remaining_damp_stops=0,
                   used_compounds=("soft", "hard"))
    if clocked:
        stop = track.pit_lane_delta + expected_stationary_time(car)
        options["weather_clock"] = StrategyWeatherClock(
            tuple(90. * i for i in range(track.total_laps)), 45., 90., 0, stop, stop)
    visited = set()

    def profile(frame, event, argument):
        if (event == "call" and frame.f_code.co_name in {"frame", "make_actions"}
                and frame.f_code.co_filename.endswith("inventory_strategy.py")):
            visited.add(frame.f_locals["state"])

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        result = plan_inventory_strategy(driver, car, track, weather, stock, 1, **options)
    finally:
        sys.setprofile(previous)
    # Every suffix is described by unused stock, rather than by the last
    # exhausted set as well. The broad ceiling catches the original expansion
    # without relying on a particular machine's elapsed time.
    assert len(visited) <= 2 ** (sets - 1) + sets
    assert result.wait_laps == sets and result.wait_cost == inf
    assert result.set_id is None and not result.should_pit()
    physics = LapSimulator()
    elapsed = 0.
    for number in range(sets):
        clean = driver.model_copy(deep=True)
        clean.current_tire_laps = number
        elapsed += physics.calculate_lap_time(
            clean, car, track, TIRE_COMPOUNDS[TireCompound.HARD], weather,
            number + 1, track.total_laps,
            sample_variation=False)
        if number:
            elapsed += track.pit_lane_delta + expected_stationary_time(car)
    assert result.wait_partial_time == pytest.approx(elapsed, abs=1.e-9)


@pytest.mark.parametrize("clocked", [False, True])
def test_changed_lap_physics_keeps_actual_age_dispatch_and_isolation(monkeypatch, clocked):
    driver, car, track, weather, stock = models()
    original = LapSimulator.calculate_lap_time

    def custom(self, driver, *args, **kwargs):
        return original(self, driver, *args, **kwargs) + .2 * (driver.current_tire_laps % 3)

    monkeypatch.setattr(LapSimulator, "calculate_lap_time", custom)
    options = dict(tire_age=0, remaining_stops=2, remaining_dry_stops=2,
                   remaining_damp_stops=2)
    if clocked:
        stop = track.pit_lane_delta + expected_stationary_time(car)
        options["weather_clock"] = StrategyWeatherClock(
            tuple(90. * i for i in range(track.total_laps)), 45., 90., 0, stop, stop)
    before = deepcopy((driver, car, track, weather, stock.__dict__))
    expected, _ = independent_schedules((driver, car, track, weather, stock), options)
    result = plan_inventory_strategy(driver, car, track, weather, stock, 1, **options)
    assert_continuation(result.continuation(False), expected[False])
    assert_continuation(result.continuation(True), expected[True])
    assert (driver, car, track, weather, stock.__dict__) == before


def test_equilibrium_surface_keeps_subclass_clock_dispatch():
    driver, car, track, weather, stock = models("equilibrium")
    calls = []

    class ObservedClock(StrategyWeatherClock):
        def updates(self, offset, paid_stops=0, stopped_first=False, *, fit_delay=0.):
            calls.append(offset)
            return super().updates(offset, paid_stops, stopped_first, fit_delay=fit_delay)

    stop = track.pit_lane_delta + expected_stationary_time(car)
    clock = ObservedClock(tuple(90. * i for i in range(track.total_laps)),
                          45., 90., 12, stop, stop)
    result = plan_inventory_strategy(driver, car, track, weather, stock, 1,
                                     weather_clock=clock, remaining_stops=0)
    assert result.wait_laps == 4
    assert 1 in calls and 2 in calls


def test_replaced_clock_update_method_bypasses_native_reductions(monkeypatch):
    driver, car, track, weather, stock = models("equilibrium")
    original = StrategyWeatherClock.updates
    calls = []

    def updates(self, *args, **kwargs):
        calls.append(args)
        return original(self, *args, **kwargs)

    assert native_physics(driver, car, track, weather)
    monkeypatch.setattr(StrategyWeatherClock, "updates", updates)
    assert not native_physics(driver, car, track, weather)
    stop = track.pit_lane_delta + expected_stationary_time(car)
    clock = StrategyWeatherClock(tuple(90. * i for i in range(track.total_laps)),
                                 45., 90., 12, stop, stop)
    result = plan_inventory_strategy(driver, car, track, weather, stock, 1,
                                     weather_clock=clock, remaining_stops=0)
    assert result.wait_laps == 4 and calls


@pytest.fixture
def diagnostic():
    return runpy.run_path(str(Path(__file__).resolve().parents[1]
                             / "examples" / "benchmark_inventory_retirement.py"))


@pytest.mark.parametrize("cadence", ["own", "external", "both"])
def test_retirement_benchmark_reports_reproducible_physical_outcomes(diagnostic, cadence):
    options = dict(sets=4, laps=8, trials=2, cadence=cadence)
    first = diagnostic["benchmark"](**options)
    second = diagnostic["benchmark"](**options)
    assert first["outcome_sha256"] == second["outcome_sha256"]
    assert first["native_forecasts"] and first["synthetic"]
    assert len(first["outcomes"]) == (2 if cadence == "both" else 1)
    assert len(first["tire_inventory"]) == 4
    assert len({row["age"] for row in first["tire_inventory"]}) == 4
    for outcome, timing in zip(first["outcomes"], first["timings"]):
        assert outcome["accepted_laps"] == 4
        assert not outcome["completion_possible"]
        assert outcome["retirement_time_seconds"] > 0
        assert outcome["pit_now_candidate"] in {"I1", "I2", "I3"}
        assert not outcome["should_pit"]
        assert outcome["pit_now_laps"] == outcome["accepted_laps"]
        assert outcome["pit_now_time_seconds"] > outcome["retirement_time_seconds"]
        assert len(timing["trial_seconds"]) == 2
        assert all(value >= 0 for value in timing["trial_seconds"])
    json.dumps(first, allow_nan=False)


@pytest.mark.parametrize("options", [dict(sets=True), dict(sets=1), dict(sets=21),
                                     dict(laps=2), dict(laps=101), dict(sets=5, laps=5),
                                     dict(trials=0), dict(trials=False), dict(trials=101),
                                     dict(cadence="unknown")])
def test_retirement_benchmark_rejects_invalid_or_finishable_inputs(diagnostic, options):
    with pytest.raises(ValueError):
        diagnostic["benchmark"](**options)


def test_retirement_benchmark_cli_reports_strict_json(diagnostic, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["benchmark_inventory_retirement.py", "--sets", "3",
                                     "--laps", "6", "--trials", "1", "--cadence", "both"])
    diagnostic["main"]()
    result = json.loads(capsys.readouterr().out)
    assert result["sets"] == 3 and result["laps"] == 6
    assert {row["cadence"] for row in result["outcomes"]} == {"own", "external"}


def test_retirement_benchmark_cli_rejects_a_finishable_pool(diagnostic, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["benchmark_inventory_retirement.py", "--sets", "6",
                                     "--laps", "6"])
    with pytest.raises(SystemExit) as error:
        diagnostic["main"]()
    assert error.value.code == 2
    assert "laps must exceed sets" in capsys.readouterr().err
