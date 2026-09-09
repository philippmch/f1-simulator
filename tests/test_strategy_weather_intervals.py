"""Own-lap projections can share or skip leading weather intervals."""

import copy
from math import inf

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.rain_strategy import plan_rain_stop, plan_rain_transition
from f1sim.simulation.surface_projection import projected_surfaces, suffix_weather_intervals
from f1sim.simulation.weather_strategy import weather_stop_costs


def args():
    return (Driver(id="A", name="A", team_id="T"), Car(team_id="T", team_name="T"),
            Track(id="T", name="T", country="T", total_laps=5, base_lap_time=90,
                  pit_lane_delta=3), Weather(track_wetness=.25, rain_intensity=0),
            TIRE_COMPOUNDS[TireCompound.INTERMEDIATE].model_copy(deep=True))


def oracle(models, intervals, mode, budget=2, queue=0, modifier=1, physical=None, aero=True):
    driver, car, track, initial, old = copy.deepcopy(models)
    # Independently advance each requested absolute interval from the input.
    surfaces = []
    for updates in intervals:
        surface = initial.model_copy(deep=True)
        for _ in range(updates):
            surface = surface.project_surface()
        surfaces.append(surface)
    sim = LapSimulator(np.random.default_rng(1))
    costs = [inf, inf]

    def visit(offset, tire, age, left, cost, stopped_first):
        if offset == len(intervals):
            costs[int(stopped_first)] = min(costs[int(stopped_first)], cost)
            return
        surface = surfaces[offset]
        critical = surface.tire_mismatch(tire.compound) == "critical"
        fits = []
        if mode == "same" or not critical:
            fits.append((False, tire))
        if mode == "same":
            compounds = [old.compound] if left else []
        elif mode == "weather" and offset:
            compounds = list(TireCompound)
        else:
            rain = surface.fresh_rain_compound()
            compounds = [rain] if rain else [TireCompound.SOFT, TireCompound.MEDIUM,
                                           TireCompound.HARD]
        if mode != "weather" and not left and not critical:
            compounds = []
        if mode == "retained" and not critical:
            compounds = []
        fits += [(True, TIRE_COMPOUNDS[c]) for c in compounds
                 if mode == "same" or surface.tire_mismatch(c) != "critical"]
        for stop, fitted in fits:
            driver.current_tire_laps = 0 if stop else age
            running = sim.calculate_lap_time(driver, car, track, fitted, surface,
                                             offset + 1, physical or track.total_laps,
                                             sample_variation=False,
                                             active_aero_enabled=aero if offset == 0 else True)
            if offset == 0:
                running *= modifier
            paid = track.pit_lane_delta + expected_stationary_time(car) if stop else 0
            if stop and offset == 0:
                paid += queue
            visit(offset + 1, fitted, driver.current_tire_laps + 1,
                  max(0, left - int(stop)), cost + running + paid,
                  stop if offset == 0 else stopped_first)

    visit(0, old, 8, budget, 0, False)
    return costs[1], costs[0]


@pytest.mark.parametrize("intervals", [(0, 0, 1, 3, 7), (0, 2, 4, 6, 9), (0, 0, 0, 0, 0)])
@pytest.mark.parametrize("mode", ["same", "transition", "weather"])
def test_irregular_costs_against_exhaustive_schedules(intervals, mode):
    models = args()
    expected = oracle(models, intervals, mode, queue=7, modifier=1.2)
    options = dict(weather_intervals=intervals, additional_current_stop_cost=7,
                   current_lap_time_modifier=1.2)
    if mode == "weather":
        result = weather_stop_costs(*models, 8, 1, traffic_possible=False, **options)
        assert result.pit_now_cost == pytest.approx(expected[0])
        retained = oracle(models, intervals, "retained", queue=7, modifier=1.2)
        assert result.stay_cost == pytest.approx(retained[1])
    else:
        planner = plan_rain_stop if mode == "same" else plan_rain_transition
        result = planner(*models, 8, 1, 2, **options)
        assert result.pit_now_cost == pytest.approx(expected[0])
        assert result.wait_cost == pytest.approx(expected[1])


@pytest.mark.parametrize("planner", [plan_rain_stop, plan_rain_transition, weather_stop_costs])
def test_default_cadence_exact_parity(planner):
    models = args()
    positional = (*models, 8, 1) + (() if planner is weather_stop_costs else (2,))
    assert planner(*positional) == planner(*positional, weather_intervals=(0, 1, 2, 3, 4))


@pytest.mark.parametrize("intervals", [(0, 1), (1, 2, 3, 4, 5), (0, 2, 1, 3, 4),
                                      (0, True, 2, 3, 4), (0, -1, 2, 3, 4),
                                      (0, .5, 2, 3, 4), [0, 1, 2, 3, 4]])
@pytest.mark.parametrize("planner", [plan_rain_stop, plan_rain_transition, weather_stop_costs])
def test_invalid_intervals(planner, intervals):
    positional = (*args(), 8, 1) + (() if planner is weather_stop_costs else (2,))
    with pytest.raises(ValueError, match="weather_intervals"):
        planner(*positional, weather_intervals=intervals)


def test_projection_and_suffix_are_isolated_and_pure():
    weather = args()[3]
    before = weather.model_copy(deep=True)
    intervals = (0, 0, 2, 2, 7)
    surfaces = projected_surfaces(weather, 5, intervals)
    assert surfaces[0] == surfaces[1] == before
    assert surfaces[2] == surfaces[3]
    suffix = projected_surfaces(surfaces[2], 3, suffix_weather_intervals(intervals, 2))
    assert suffix == surfaces[2:]
    surfaces[0].track_wetness = .9
    assert surfaces[1] == before
    assert weather == before


def test_cache_cadence_and_queue_changes_preserve_inputs():
    models = args()
    before = copy.deepcopy(models)
    slow = plan_rain_transition(*models, 8, 1, 2, weather_intervals=(0, 0, 0, 0, 0))
    fast = plan_rain_transition(*models, 8, 1, 2, weather_intervals=(0, 2, 4, 6, 9))
    shifted = plan_rain_transition(*models, 8, 1, 2, weather_intervals=(0, 2, 4, 6, 9),
                                   additional_current_stop_cost=-4)
    assert slow != fast
    assert shifted.wait_cost == fast.wait_cost
    assert shifted.pit_now_cost == pytest.approx(fast.pit_now_cost - 4)
    assert models == before


def test_irregular_current_aero_and_original_fuel_distance():
    models = args()
    intervals = (0, 0, 3, 3, 8)
    expected = oracle(models, intervals, "transition", physical=50, aero=False,
                      modifier=1.4, queue=-2)
    result = plan_rain_transition(
        *models, 8, 1, 2, weather_intervals=intervals, physical_total_laps=50,
        active_aero_enabled=False, current_lap_time_modifier=1.4,
        additional_current_stop_cost=-2,
    )
    assert result.pit_now_cost == pytest.approx(expected[0])
    assert result.wait_cost == pytest.approx(expected[1])


@pytest.mark.parametrize("wetness,rain,intervals", [
    (.95, .95, (0, 0, 3, 7, 10)),
    (0, 0, (0, 2, 5, 5, 9)),
    (.03, 0, (0, 1, 5, 7, 9)),
])
def test_physically_equivalent_clocks_share_cached_plans(wetness, rain, intervals):
    from f1sim.simulation import rain_strategy

    models = args()
    models[3].track_wetness = wetness
    models[3].rain_intensity = rain
    rain_strategy._transition_plan.cache_clear()
    ordinary = plan_rain_transition(*models, 8, 1, 2)
    before = rain_strategy._transition_plan.cache_info()
    changed = plan_rain_transition(*models, 8, 1, 2, weather_intervals=intervals)
    after = rain_strategy._transition_plan.cache_info()
    assert changed == ordinary
    assert after.hits == before.hits + 1


def test_repeated_surface_cache_results_remain_independent():
    weather = args()[3]
    intervals = (0, 0, 3, 7, 9)
    initial = projected_surfaces(weather, 5, intervals)
    repeated = projected_surfaces(weather, 5, intervals)
    assert initial == repeated
    initial[0].track_wetness = .99
    assert repeated[0] == weather
    assert initial[1] == weather


def test_drying_canonicalization_preserves_float_boundary():
    weather = args()[3]
    weather.track_wetness = .09
    intervals = (0, 1, 2, 3, 7)
    actual = projected_surfaces(weather, 5, intervals)
    for count, surface in zip(intervals, actual):
        expected = weather.model_copy(deep=True)
        for _ in range(count):
            expected = expected.project_surface()
        assert surface == expected
