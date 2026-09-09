"""Transition plans agree with exhaustive legal paid-stop schedules."""

import copy
from math import inf

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.rain_strategy import plan_rain_transition


def inputs(compound=TireCompound.INTERMEDIATE, wetness=.23, rain=0, laps=7):
    return (Driver(id="A", name="A", team_id="T"), Car(team_id="T", team_name="T"),
            Track(id="t", name="T", country="T", total_laps=laps, base_lap_time=90,
                  pit_lane_delta=5), Weather(track_wetness=wetness, rain_intensity=rain),
            TIRE_COMPOUNDS[compound].model_copy(deep=True))


def exhaustive(args, age, lap, budget, **options):
    """Enumerate complete legal schedules, charging actual mean lap physics."""
    driver, car, track, weather, tire = copy.deepcopy(args)
    sim = LapSimulator(np.random.default_rng(1))
    minima = [inf, inf]
    best_compound = None

    def visit(surface, fitted, tire_age, number, stops, total, first_stop, first_compound):
        nonlocal best_compound
        if number > track.total_laps:
            index = int(first_stop)
            if total < minima[index]:
                minima[index] = total
                if first_stop:
                    best_compound = first_compound
            return
        critical = surface.tire_mismatch(fitted.compound) == "critical"
        available = [] if critical else [(False, fitted)]
        limit = options.get("remaining_dry_stops" if surface.track_wetness < .08
                            and surface.rain_intensity < .15 else "remaining_damp_stops")
        elective = stops < budget and (
            fitted.compound in (TireCompound.INTERMEDIATE, TireCompound.WET)
            or limit is None or stops < limit
        )
        if elective or critical:
            rain = surface.fresh_rain_compound()
            compounds = [rain] if rain else [TireCompound.SOFT, TireCompound.MEDIUM,
                                           TireCompound.HARD]
            available += [(True, TIRE_COMPOUNDS[c]) for c in compounds
                          if surface.tire_mismatch(c) != "critical"]
        for stop, next_tire in available:
            next_age = 0 if stop else tire_age
            driver.current_tire_laps = next_age
            cost = sim.calculate_lap_time(
                driver, car, track, next_tire, surface, number,
                options.get("physical_total_laps", track.total_laps),
                sample_variation=False,
                active_aero_enabled=options.get("active_aero_enabled", True)
                if number == lap else True,
            )
            if number == lap:
                cost *= options.get("current_lap_time_modifier", 1)
            if stop:
                cost += track.pit_lane_delta * (
                    options.get("pit_lane_factor", 1) if number == lap else 1
                ) + expected_stationary_time(car)
                if number == lap:
                    cost += options.get("additional_current_stop_cost", 0)
            visit(surface.project_surface(), next_tire, next_age + 1, number + 1,
                  stops + int(stop), total + cost,
                  stop if number == lap else first_stop,
                  next_tire.compound if number == lap and stop else first_compound)

    visit(weather, tire, age, lap, 0, 0, False, None)
    return minima[1], minima[0], best_compound


@pytest.mark.parametrize("compound,wetness,rain", [
    (TireCompound.INTERMEDIATE, .23, 0),
    (TireCompound.WET, .23, 0),
    (TireCompound.INTERMEDIATE, .65, .9),
])
@pytest.mark.parametrize("budget", [0, 1, 3])
def test_exhaustive_transition_schedules(compound, wetness, rain, budget):
    args = inputs(compound, wetness, rain)
    expected = exhaustive(args, 12, 1, budget)
    result = plan_rain_transition(*args, 12, 1, budget)
    assert result.pit_now_cost == pytest.approx(expected[0])
    assert result.wait_cost == pytest.approx(expected[1])
    assert result.compound == expected[2]
    assert result.should_pit() == (expected[0] < expected[1])


@pytest.mark.parametrize("adjustment", [-15, 8])
def test_current_controls_queue_and_physical_fuel_distance(adjustment):
    args = inputs(laps=6)
    options = dict(pit_lane_factor=.4, additional_current_stop_cost=adjustment,
                   current_lap_time_modifier=1.3, active_aero_enabled=False,
                   physical_total_laps=50)
    expected = exhaustive(args, 18, 2, 2, **options)
    result = plan_rain_transition(*args, 18, 2, 2, **options)
    assert result.pit_now_cost == pytest.approx(expected[0])
    assert result.wait_cost == pytest.approx(expected[1])
    assert result.compound == expected[2]


def test_custom_tire_and_mutable_physics_invalidate_cache(monkeypatch):
    args = inputs(laps=4, wetness=.1)
    before = copy.deepcopy(args)
    original = plan_rain_transition(*args, 20, 1, 2)
    assert args == before
    soft = TIRE_COMPOUNDS[TireCompound.SOFT].model_copy(deep=True)
    soft.initial_grip = .6
    monkeypatch.setitem(TIRE_COMPOUNDS, TireCompound.SOFT, soft)
    args[4].initial_grip = .65
    args[1].base_pace -= .1
    result = plan_rain_transition(*args, 20, 1, 2)
    expected = exhaustive(args, 20, 1, 2)
    assert result != original
    assert result.pit_now_cost == pytest.approx(expected[0])
    assert result.wait_cost == pytest.approx(expected[1])


@pytest.mark.parametrize("options", [{"remaining_stops": -1}, {"tire_age": True},
                                      {"current_lap": 0}, {"physical_total_laps": 2},
                                      {"additional_current_stop_cost": float("nan")},
                                      {"active_aero_enabled": 1}])
def test_invalid_inputs(options):
    values = dict(tire_age=1, current_lap=1, remaining_stops=1)
    values.update(options)
    with pytest.raises(ValueError):
        plan_rain_transition(*inputs(), **values)


def test_ties_wait_and_no_elective_stop_with_empty_budget():
    result = plan_rain_transition(*inputs(laps=1, wetness=.4), 0, 1, 0)
    assert result.compound is None
    assert result.pit_now_cost == inf
    assert not result.should_pit()
    assert not type(result)(result.wait_cost, result.wait_cost, TireCompound.WET).should_pit()




@pytest.mark.parametrize("dry,damp", [(0, 0), (3, 1), (2, 0), (1, 3), (None, 1)])
def test_slick_allowances_match_exhaustive_schedules(dry, damp):
    args = inputs(laps=7, wetness=.19)
    options = dict(remaining_dry_stops=dry, remaining_damp_stops=damp)
    expected = exhaustive(args, 1, 1, 4, **options)
    result = plan_rain_transition(*args, 1, 1, 4, **options)
    assert result.pit_now_cost == pytest.approx(expected[0])
    assert result.wait_cost == pytest.approx(expected[1])
    assert result.compound == expected[2]


@pytest.mark.parametrize("name", ["remaining_dry_stops", "remaining_damp_stops"])
@pytest.mark.parametrize("value", [-1, True, 1.5, float("inf")])
def test_invalid_slick_allowances(name, value):
    with pytest.raises(ValueError, match=name):
        plan_rain_transition(*inputs(), 1, 1, 4, **{name: value})


def test_long_valid_distance_uses_full_horizon_without_recursion(monkeypatch, request):
    from f1sim.simulation import rain_strategy

    def clear():
        rain_strategy._transition_plan.cache_clear()
        rain_strategy._running_row.cache_clear()
        rain_strategy._transition_suffixes.clear()

    clear()
    request.addfinalizer(clear)
    args = inputs(laps=400, wetness=.19)
    args[2].base_lap_time = 10
    seen = set()

    def running(self, driver, car, track, tire, weather, lap, total_laps, **options):
        seen.add(lap)
        return 10.0

    monkeypatch.setattr(LapSimulator, "calculate_lap_time", running)
    result = plan_rain_transition(*args, 1, 2, 4, remaining_dry_stops=3,
                                  remaining_damp_stops=1)
    stop = args[2].pit_lane_delta + expected_stationary_time(args[1])
    assert result.wait_cost == pytest.approx(3990 + stop)
    assert result.pit_now_cost == pytest.approx(3990 + stop)
    assert max(seen) == 400
    assert not result.should_pit()


def test_used_intermediate_does_not_forecast_forbidden_fourth_slick_stop():
    args = inputs(laps=25, wetness=.19)
    args[1].tire_degradation_factor = 1.5
    args[2].base_lap_time = 200
    args[2].pit_lane_delta = 1
    args[2].tire_stress = 1
    unrestricted = plan_rain_transition(*args, 1, 2, 4)
    restricted = plan_rain_transition(*args, 1, 2, 4,
                                      remaining_dry_stops=3, remaining_damp_stops=1)
    assert unrestricted.should_pit()
    assert not restricted.should_pit()
    assert restricted.pit_now_cost > unrestricted.pit_now_cost


def test_rain_fit_remains_allowed_after_slick_allowance_exhausted():
    args = inputs(laps=3, wetness=.1)
    result = plan_rain_transition(*args, 1, 1, 1,
                                  remaining_dry_stops=0, remaining_damp_stops=0)
    expected = exhaustive(args, 1, 1, 1, remaining_dry_stops=0, remaining_damp_stops=0)
    assert result.pit_now_cost < inf
    assert result.pit_now_cost == pytest.approx(expected[0])
    assert result.wait_cost == pytest.approx(expected[1])


def test_current_queue_changes_reuse_green_physics(monkeypatch):
    from f1sim.simulation import rain_strategy

    rain_strategy._transition_plan.cache_clear()
    rain_strategy._transition_suffixes.clear()
    args = inputs(laps=10, wetness=.19)
    initial = plan_rain_transition(*args, 1, 2, 3)

    def unexpected(*args, **kwargs):
        pytest.fail("A queue-only change must reuse the existing green stint physics")

    monkeypatch.setattr(LapSimulator, "calculate_lap_time", unexpected)
    changed = plan_rain_transition(*args, 1, 2, 3, additional_current_stop_cost=7)
    assert changed.wait_cost == initial.wait_cost
    assert changed.pit_now_cost == pytest.approx(initial.pit_now_cost + 7)


def test_suffix_cache_eviction_preserves_costs(monkeypatch):
    from f1sim.simulation import rain_strategy

    args = inputs(laps=7, wetness=.19)
    expected = exhaustive(args, 2, 1, 3)
    rain_strategy._transition_plan.cache_clear()
    rain_strategy._transition_suffixes.clear()
    monkeypatch.setattr(rain_strategy, "_TRANSITION_SUFFIX_LIMIT", 3)
    result = plan_rain_transition(*args, 2, 1, 3)
    assert len(rain_strategy._transition_suffixes) <= 3
    assert result.pit_now_cost == pytest.approx(expected[0])
    assert result.wait_cost == pytest.approx(expected[1])
