from itertools import product
from math import inf

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.rain_strategy import RainStopDecision, plan_rain_stop


def fixture(laps):
    return (Driver(id='A', name='A', team_id='A'), Car(team_id='A', team_name='A'),
            Track(id='t', name='T', country='T', total_laps=laps, base_lap_time=90,
                  pit_lane_delta=1), Weather(track_wetness=0.73))


@pytest.mark.parametrize('laps', range(1, 7))
@pytest.mark.parametrize('budget', range(3))
@pytest.mark.parametrize('modifier,lane,queue', [(1, 1, 0), (1.2, .75, 2.3), (1.4, .55, -.38)])
def test_matches_complete_stop_schedule_enumeration(laps, budget, modifier, lane, queue):
    driver, car, track, weather = fixture(laps)
    tire = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE].model_copy(update={'degradation_rate': .08})
    before = [x.model_dump() for x in (driver, car, track, weather, tire)]
    actual = plan_rain_stop(driver, car, track, weather, tire, 8, 1, budget,
                           pit_lane_factor=lane, additional_current_stop_cost=queue,
                           current_lap_time_modifier=modifier, active_aero_enabled=False,
                           physical_total_laps=50)
    simulator = LapSimulator(np.random.default_rng(33))
    rng_before = simulator.rng.bit_generator.state
    expected = [inf, inf]
    for actions in product((False, True), repeat=laps):
        if sum(actions) > budget:
            continue
        surface, fitted, age, cost = weather, tire, 8, 0.0
        for offset, stop in enumerate(actions):
            if stop:
                fitted, age = TIRE_COMPOUNDS[tire.compound], 0
                cost += expected_stationary_time(car)
                cost += track.pit_lane_delta * (lane if offset == 0 else 1)
                cost += queue if offset == 0 else 0
            isolated = driver.model_copy(update={'current_tire_laps': age})
            cost += simulator.calculate_lap_time(
                isolated, car, track, fitted, surface, offset + 1, 50,
                active_aero_enabled=offset > 0, sample_variation=False,
            ) * (modifier if offset == 0 else 1)
            age += 1
            surface = surface.project_surface()
        expected[int(actions[0])] = min(expected[int(actions[0])], cost)
    assert actual.wait_cost == pytest.approx(expected[0])
    assert actual.pit_now_cost == pytest.approx(expected[1])
    assert before == [x.model_dump() for x in (driver, car, track, weather, tire)]
    assert simulator.rng.bit_generator.state == rng_before


def test_ties_wait_and_decision_is_immutable():
    from dataclasses import FrozenInstanceError
    assert not RainStopDecision(10, 10).should_pit()
    assert RainStopDecision(9, 10).should_pit()
    assert not RainStopDecision(9, 10).should_pit(tolerance=1)
    with pytest.raises(FrozenInstanceError):
        RainStopDecision(9, 10).wait_cost = 11


@pytest.mark.parametrize('kwargs', [dict(tire_age=-1), dict(current_lap=0),
    dict(current_lap=5), dict(remaining_stops=True), dict(pit_lane_factor=float('nan')),
    dict(additional_current_stop_cost=float('inf')), dict(current_lap_time_modifier=0),
    dict(physical_total_laps=2), dict(active_aero_enabled=1)])
def test_rejects_invalid_parameters(kwargs):
    driver, car, track, weather = fixture(4)
    args = dict(tire_age=0, current_lap=1, remaining_stops=2)
    args.update(kwargs)
    with pytest.raises(ValueError):
        plan_rain_stop(driver, car, track, weather, TIRE_COMPOUNDS[TireCompound.WET], **args)


def test_fresh_set_waits_while_worn_set_can_pay_for_refit():
    driver, car, track, weather = fixture(6)
    tire = TIRE_COMPOUNDS[TireCompound.WET]
    assert not plan_rain_stop(driver, car, track, weather, tire, 0, 1, 2).should_pit()
    worn = tire.model_copy(update={'degradation_rate': .1})
    assert plan_rain_stop(driver, car, track, weather, worn, 15, 1, 2).should_pit()


def test_mid_race_no_budget_preserves_original_fuel_without_rng_draws(monkeypatch):
    import f1sim.simulation.rain_strategy as module

    driver, car, track, weather = fixture(6)
    tire = TIRE_COMPOUNDS[TireCompound.WET]
    class NoRandom:
        def normal(self, *args, **kwargs):
            raise AssertionError('planning must not draw lap variation')

    module._plan.cache_clear()
    monkeypatch.setattr(module.np.random, 'default_rng', lambda *args: NoRandom())
    actual = plan_rain_stop(driver, car, track, weather, tire, 3, 4, 0,
                           physical_total_laps=50)
    surface, expected = weather, 0.0
    simulator = LapSimulator(NoRandom())
    for offset in range(3):
        expected += simulator.calculate_lap_time(
            driver.model_copy(update={'current_tire_laps': 3 + offset}), car, track,
            tire, surface, 4 + offset, 50, sample_variation=False,
        )
        surface = surface.project_surface()
    assert actual.wait_cost == pytest.approx(expected)
    assert actual.pit_now_cost == inf


def test_future_cache_reuses_current_adjustments_and_successive_laps():
    import f1sim.simulation.rain_strategy as module

    for cache in (module._plan, module._fresh_future, module._running_row, module._surfaces):
        cache.cache_clear()
    driver, car, track, weather = fixture(6)
    tire = TIRE_COMPOUNDS[TireCompound.WET]
    baseline = plan_rain_stop(driver, car, track, weather, tire, 8, 1, 2)
    misses = module._fresh_future.cache_info().misses
    adjusted = plan_rain_stop(driver, car, track, weather, tire, 8, 1, 2,
                             additional_current_stop_cost=3, pit_lane_factor=.55,
                             current_lap_time_modifier=1.4)
    assert module._fresh_future.cache_info().misses == misses
    assert adjusted != baseline
    # Matching identities/performance and arbitrary mutable race state share plans.
    other = driver.model_copy(update={'id': 'B', 'name': 'B', 'position': 9, 'pit_stops': 2})
    assert plan_rain_stop(other, car, track, weather, tire, 8, 1, 2) == baseline
    assert module._fresh_future.cache_info().misses == misses
    hits = module._fresh_future.cache_info().hits
    plan_rain_stop(driver, car, track, weather.project_surface(), tire, 9, 2, 2)
    assert module._fresh_future.cache_info().hits > hits
    # Changing physics cannot reuse stale future costs.
    faster = driver.model_copy(update={'skill_rating': 1.0})
    changed = plan_rain_stop(faster, car, track, weather, tire, 8, 1, 2)
    assert changed.wait_cost < baseline.wait_cost
    assert module._fresh_future.cache_info().misses > misses
    changed_fuel = plan_rain_stop(driver, car, track, weather, tire, 8, 1, 2,
                                  physical_total_laps=50)
    assert changed_fuel.wait_cost > baseline.wait_cost
