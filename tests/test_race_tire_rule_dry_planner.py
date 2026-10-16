"""Restricted dry plans match exhaustive short-race costs and physical rules."""

from itertools import product
from math import inf

import numpy as np
import pytest

from f1sim.models import Car, Driver, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS, TireCompound
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time, plan_dry_stop

SLICKS = (TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD)


def models(laps=6):
    return (Driver(id="A", name="A", team_id="T"), Car(team_id="T", team_name="T"),
            Track(id="test", name="Test", country="T", total_laps=laps, base_lap_time=74,
                  tire_stress=0.4, pit_lane_delta=19))


def partitions(total, count):
    if count == 1:
        yield (total,)
    else:
        for first in range(1, total - count + 2):
            for rest in partitions(total - first, count - 1):
                yield (first, *rest)


def exhaustive(driver, car, track, *, age, current_remaining, cap, new_sets, used_current):
    """Enumerate every legal stint partition/compound path, independent of DP."""
    result = {"pit": inf, "wait": inf}
    simulator = LapSimulator(np.random.default_rng(0))
    for first_action in result:
        for paid in range(4):
            if first_action == "pit" and paid == 0:
                continue
            count = paid if first_action == "pit" else paid + 1
            required = new_sets + int(first_action == "pit" and not used_current)
            if paid < required:
                continue
            for lengths in partitions(track.total_laps, count):
                limits = ([cap] * count if first_action == "pit" else
                          [current_remaining, *([cap] * paid)])
                if any(limit is not None and n > limit for n, limit in zip(lengths, limits)):
                    continue
                for replacements in product(SLICKS, repeat=paid):
                    compounds = replacements if first_action == "pit" else (
                        TireCompound.MEDIUM, *replacements,
                    )
                    used = {TireCompound.MEDIUM} if used_current else set()
                    if len(used | set(compounds)) < 2:
                        continue
                    cost = paid * (track.pit_lane_delta + expected_stationary_time(car))
                    lap = 1
                    for index, (compound, length) in enumerate(zip(compounds, lengths)):
                        initial_age = age if first_action == "wait" and index == 0 else 0
                        for offset in range(length):
                            running = driver.model_copy(deep=True)
                            running.current_tire_laps = initial_age + offset
                            cost += simulator.calculate_lap_time(
                                running, car.model_copy(deep=True), track.model_copy(deep=True),
                                TIRE_COMPOUNDS[compound].model_copy(deep=True), Weather(),
                                lap, track.total_laps, sample_variation=False,
                            )
                            lap += 1
                    result[first_action] = min(result[first_action], cost)
    return result


@pytest.mark.parametrize("age,remaining,cap,new_sets,used", [
    (2, 2, 2, 1, True),
    (0, 3, 3, 2, False),
    (4, 0, 3, 1, True),
    (1, None, None, 2, True),
])
def test_restricted_dry_costs_equal_exhaustive_legal_schedules(age, remaining, cap, new_sets, used):
    driver, car, track = models()
    actual = plan_dry_stop(
        driver, car, track, TIRE_COMPOUNDS[TireCompound.MEDIUM], age, 6, 3,
        {TireCompound.MEDIUM} if used else set(), current_set_used=used,
        maximum_set_laps=cap, current_remaining_set_laps=remaining, minimum_new_sets=new_sets,
    )
    expected = exhaustive(driver, car, track, age=age, current_remaining=remaining,
                          cap=cap, new_sets=new_sets, used_current=used)
    assert actual.pit_now_cost == pytest.approx(expected["pit"], abs=1e-9)
    assert actual.wait_cost == pytest.approx(expected["wait"], abs=1e-9)


def test_insufficient_stop_budget_cannot_price_an_illegal_finish_as_feasible():
    driver, car, track = models()
    decision = plan_dry_stop(
        driver, car, track, TIRE_COMPOUNDS[TireCompound.MEDIUM], 1, 6, 1,
        {TireCompound.MEDIUM}, maximum_set_laps=2, current_remaining_set_laps=2,
        minimum_new_sets=2,
    )
    assert decision.pit_now_cost == decision.wait_cost == inf
    assert decision.compound in SLICKS  # A hard guard still has an immediate safe fresh fit.


@pytest.mark.parametrize("cap,new_sets", [(2, 0), (None, 4)])
def test_enabled_rules_can_require_more_than_three_future_stops(cap, new_sets):
    driver, car, track = models(8)
    decision = plan_dry_stop(
        driver, car, track, TIRE_COMPOUNDS[TireCompound.MEDIUM], 1, 8, 4,
        {TireCompound.MEDIUM}, current_set_used=True,
        maximum_set_laps=cap, current_remaining_set_laps=1 if cap else None,
        minimum_new_sets=new_sets,
    )
    assert decision.wait_cost < inf
    assert decision.pit_now_cost < inf


def test_impossible_minimum_does_not_allocate_its_unbounded_requested_count():
    driver, car, track = models()
    decision = plan_dry_stop(
        driver, car, track, TIRE_COMPOUNDS[TireCompound.MEDIUM], 1, 6, 100000,
        {TireCompound.MEDIUM}, minimum_new_sets=100000,
    )
    assert decision.wait_cost == decision.pit_now_cost == inf


def test_three_set_rule_is_priced_before_the_final_two_laps():
    driver, car, track = models(12)
    current = TIRE_COMPOUNDS[TireCompound.MEDIUM]
    # Equal tyre compounds still accrue wear: two mandatory replacements have
    # a cost-optimal interior window rather than a last-two-laps correction.
    age, used, stops, pit_laps = 0, 0, 0, []
    for lap in range(1, 13):
        decision = plan_dry_stop(
            driver, car, track, current, age, 13 - lap, 3 - stops,
            {TireCompound.MEDIUM, current.compound} if used else set(),
            current_set_used=bool(age), minimum_new_sets=max(0, 3 - max(1, used)),
        )
        if decision.should_pit():
            current = TIRE_COMPOUNDS[decision.compound]
            age = 0
            pit_laps.append(lap)
            stops += 1
        if age == 0:
            used += 1
        age += 1
    assert used >= 3
    assert len(pit_laps) >= 2 and pit_laps[0] < 11


@pytest.mark.parametrize("options", [
    {"maximum_set_laps": 0}, {"maximum_set_laps": True},
    {"current_remaining_set_laps": -1}, {"current_remaining_set_laps": 2.0},
    {"minimum_new_sets": -1}, {"minimum_new_sets": None},
])
def test_restricted_options_are_validated(options):
    driver, car, track = models()
    with pytest.raises(ValueError, match="must be an integer"):
        plan_dry_stop(driver, car, track, TIRE_COMPOUNDS[TireCompound.MEDIUM], 0, 6, 3,
                      set(), **options)


def test_explicit_disabled_options_preserve_the_original_decision():
    driver, car, track = models()
    args = (driver, car, track, TIRE_COMPOUNDS[TireCompound.MEDIUM], 2, 6, 3,
            {TireCompound.MEDIUM})
    assert plan_dry_stop(*args) == plan_dry_stop(
        *args, maximum_set_laps=None, current_remaining_set_laps=None, minimum_new_sets=0,
    )
