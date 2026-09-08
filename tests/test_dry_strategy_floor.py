from itertools import combinations, product
from math import inf

import numpy as np
import pytest

from f1sim.models import Car, Driver, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS, TireCompound
from f1sim.models.track import ActiveAeroZone
from f1sim.simulation import pit_strategy
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import SLICKS, expected_stationary_time, plan_dry_stop


def models(laps=5):
    return (
        Driver(id="d", name="Driver", team_id="t", skill_rating=1, tire_management=0.8),
        Car(team_id="t", team_name="Team", base_pace=1, straight_line_speed=1,
            downforce_level=0.8, tire_degradation_factor=1.5),
        Track(id="floor", name="Floor", country="Test", total_laps=laps,
              base_lap_time=200, tire_stress=1, pit_lane_delta=20,
              active_aero_zones=[ActiveAeroZone(zone_id=i + 1, sector=1, time_gain=1)
                                 for i in range(16)]),
    )


class ScaledWeather(Weather):
    scale: float = 1.0

    def lap_time_multiplier(self):
        return self.scale


def oracle(driver, car, track, tire, age, remaining, budget, used, exempt,
           physical, scale, aero, modifier, lane, queue):
    simulator = LapSimulator(np.random.default_rng(0))
    start = track.total_laps - remaining + 1
    best = {True: inf, False: inf}
    for count in range(budget + 1):
        for stops in combinations(range(remaining), count):
            for sets in product(SLICKS, repeat=count):
                held, tire_age = tire, age
                driven = set(used)
                elapsed = 0
                for offset in range(remaining):
                    if offset in stops:
                        held = TIRE_COMPOUNDS[sets[stops.index(offset)]]
                        tire_age = 0
                        elapsed += (track.pit_lane_delta * (lane if offset == 0 else 1)
                                    + expected_stationary_time(car)
                                    + (queue if offset == 0 else 0))
                    driven.add(held.compound)
                    projection = driver.model_copy(deep=True)
                    projection.current_tire_laps = tire_age
                    elapsed += simulator.calculate_lap_time(
                        projection, car, track, held, ScaledWeather(scale=scale),
                        start + offset, physical, sample_variation=False,
                        active_aero_enabled=aero if offset == 0 else True,
                    ) * (modifier if offset == 0 else 1)
                    tire_age += 1
                if exempt or len(driven) >= 2:
                    pit = 0 in stops
                    best[pit] = min(best[pit], elapsed)
    return best


@pytest.mark.parametrize("budget", range(4))
@pytest.mark.parametrize("remaining", [1, 3, 5])
@pytest.mark.parametrize("controls", [(1, True, 1, 0, 1), (1.35, False, .55, 2, 1.002)])
@pytest.mark.parametrize("exempt", [False, True])
def test_floor_plans_match_exhaustive_full_lap_schedules(budget, remaining, controls, exempt):
    driver, car, track = models(8)
    track.active_aero_zones.extend([
        ActiveAeroZone(zone_id=i + 17, sector=1, time_gain=1) for i in range(4)
    ])
    modifier, aero, lane, queue, scale = controls
    tire = TIRE_COMPOUNDS[TireCompound.SOFT].model_copy(deep=True)
    tire.degradation_rate *= 1.1
    before = [obj.model_dump() for obj in (driver, car, track, tire)]
    actual = plan_dry_stop(driver, car, track, tire, 25, remaining, budget,
                           {TireCompound.SOFT}, exempt, lane, queue, modifier, scale,
                           physical_total_laps=30, active_aero_enabled=aero)
    expected = oracle(driver, car, track, tire, 25, remaining, budget,
                      {TireCompound.SOFT}, exempt, 30, scale, aero, modifier, lane, queue)
    assert actual.pit_now_cost == pytest.approx(expected[True])
    assert actual.wait_cost == pytest.approx(expected[False])
    assert [obj.model_dump() for obj in (driver, car, track, tire)] == before


def test_clipped_fresh_pace_does_not_justify_a_losing_stop():
    driver, car, track = models(30)
    tire = TIRE_COMPOUNDS[TireCompound.SOFT]
    decision = plan_dry_stop(driver, car, track, tire, 25, 5, 1,
                             {TireCompound.SOFT, TireCompound.MEDIUM})
    expected = oracle(driver, car, track, tire, 25, 5, 1,
                      {TireCompound.SOFT, TireCompound.MEDIUM}, False,
                      30, 1, True, 1, 1, 0)
    assert expected[True] > expected[False]
    assert not decision.should_pit()
    assert decision.wait_cost == pytest.approx(expected[False])


def test_full_tables_reuse_physics_and_invalidate_changed_inputs(monkeypatch):
    driver, car, track = models(6)
    pit_strategy._floor_tables.cache_clear()
    calculate = LapSimulator.calculate_lap_time

    def deterministic(self, *args, **kwargs):
        assert kwargs["sample_variation"] is False
        return calculate(self, *args, **kwargs)

    monkeypatch.setattr(LapSimulator, "calculate_lap_time", deterministic)
    def decide():
        return plan_dry_stop(driver, car, track, TIRE_COMPOUNDS[TireCompound.SOFT],
                             25, 4, 2, {TireCompound.SOFT, TireCompound.MEDIUM})
    try:
        original = decide()
        driver.id = "other"
        car.team_name = "Other"
        assert decide() == original
        assert pit_strategy._floor_tables.cache_info().hits == 1
        car.tire_degradation_factor = 1.3
        decide()
        assert pit_strategy._floor_tables.cache_info().misses == 2
        replacement = TIRE_COMPOUNDS[TireCompound.MEDIUM].model_copy(deep=True)
        replacement.degradation_rate *= 1.1
        monkeypatch.setitem(TIRE_COMPOUNDS, TireCompound.MEDIUM, replacement)
        decide()
        assert pit_strategy._floor_tables.cache_info().misses == 3
    finally:
        pit_strategy._floor_tables.cache_clear()


def test_unclipped_tracks_keep_fast_tables(monkeypatch):
    driver, car, track = models()
    track.active_aero_zones = []
    def unexpected(*args, **kwargs):
        raise AssertionError("Unclipped physics should use the existing fast planner")
    monkeypatch.setattr(pit_strategy, "_floor_tables", unexpected)
    plan_dry_stop(driver, car, track, TIRE_COMPOUNDS[TireCompound.SOFT],
                  2, 5, 2, {TireCompound.SOFT})
