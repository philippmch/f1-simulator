"""Weather-stop lower bounds charge every refit and use actual lap physics."""

import copy
from itertools import product
from math import inf

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.models.track import ActiveAeroZone
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.weather_strategy import _fresh_plan_costs, weather_stop_costs


def fixture(laps=4):
    return (Driver(id="A", name="A", team_id="A"), Car(team_id="A", team_name="A"),
            Track(id="t", name="T", country="T", total_laps=laps, base_lap_time=90,
                  pit_lane_delta=1), Weather(track_wetness=0.73))


@pytest.mark.parametrize("modifier,lane", [(1, 1), (1.2, 0.75), (1.4, 0.55)])
def test_paid_refit_dp_matches_independent_action_enumeration(modifier, lane):
    driver, car, track, weather = fixture()
    current = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE].model_copy(
        update={"degradation_rate": 0.08}
    )
    queue = 2.3
    actual = weather_stop_costs(driver, car, track, weather, current, 5, 1,
                               pit_lane_factor=lane, additional_current_stop_cost=queue,
                               current_lap_time_modifier=modifier, active_aero_enabled=False,
                               traffic_possible=False)
    surfaces = [weather]
    for _ in range(3):
        surfaces.append(surfaces[-1].project_surface())
    simulator = LapSimulator(np.random.default_rng(0))

    def running(tire, age, lap):
        return simulator.calculate_lap_time(
            driver.model_copy(update={"current_tire_laps": age}), car, track, tire,
            surfaces[lap], lap + 1, 4, active_aero_enabled=lap > 0, sample_variation=False,
        ) * (modifier if lap == 0 else 1)

    expected = inf
    # Stop now must fit wets; each subsequent action either stays or pays
    # another stop. Even a same-compound replacement consumes service time.
    for actions in product((None, *TireCompound), repeat=3):
        tire, age = TIRE_COMPOUNDS[TireCompound.WET], 0
        cost = track.pit_lane_delta * lane + expected_stationary_time(car) + queue
        for lap, action in enumerate((None, *actions)):
            if action is not None:
                tire, age = TIRE_COMPOUNDS[action], 0
                cost += track.pit_lane_delta + expected_stationary_time(car)
            if surfaces[lap].tire_mismatch(tire.compound) == "critical":
                cost = inf
                break
            cost += running(tire, age, lap)
            age += 1
        expected = min(expected, cost)
    assert actual.pit_now_cost == pytest.approx(expected)
    assert actual.stay_cost == pytest.approx(
        sum(running(current, 5 + lap, lap) for lap in range(4))
    )


def test_lone_car_rejects_losing_wet_stop_in_retention_window():
    from f1sim.simulation.race import DriverRaceState, RaceSimulator

    driver, car, track, weather = fixture(10)
    result = weather_stop_costs(driver, car, track, weather,
                                TIRE_COMPOUNDS[TireCompound.INTERMEDIATE], 1, 2,
                                traffic_possible=False)
    assert result.pit_now_cost > result.stay_cost
    state = DriverRaceState(driver, car, 1,
                            current_tire=TIRE_COMPOUNDS[TireCompound.INTERMEDIATE], tire_laps=1)
    assert not RaceSimulator(np.random.default_rng(2))._should_pit(
        state, [state], track, 2, False, weather,
    )


def test_projected_critical_old_set_bypasses_veto():
    driver, car, track, _ = fixture()
    result = weather_stop_costs(driver, car, track, Weather(track_wetness=0.34, rain_intensity=0.9),
                                TIRE_COMPOUNDS[TireCompound.SOFT], 5, 1)
    assert result.pit_now_cost == 0 and result.stay_cost == inf


def test_cache_config_and_current_adjustments_are_isolated_and_inputs_unchanged(monkeypatch):
    driver, car, track, weather = fixture()
    current = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE].model_copy()
    before = copy.deepcopy((driver, car, track, weather, current))
    _fresh_plan_costs.cache_clear()
    plain = weather_stop_costs(driver, car, track, weather, current, 1, 1)
    adjusted = weather_stop_costs(driver, car, track, weather, current, 9, 1,
                                  additional_current_stop_cost=4, traffic_possible=False)
    assert _fresh_plan_costs.cache_info().misses == track.total_laps
    weather_stop_costs(driver, car, track, weather.project_surface(), current, 2, 2,
                       traffic_possible=False)
    assert _fresh_plan_costs.cache_info().misses == track.total_laps
    assert _fresh_plan_costs.cache_info().maxsize == 4096
    assert adjusted.pit_now_cost == pytest.approx(plain.pit_now_cost + 4)
    assert (driver, car, track, weather, current) == before
    changed_wet = TIRE_COMPOUNDS[TireCompound.WET].model_copy(update={"degradation_rate": 0.1})
    monkeypatch.setitem(TIRE_COMPOUNDS, TireCompound.WET, changed_wet)
    changed = weather_stop_costs(driver, car, track, weather, current, 1, 1)
    assert _fresh_plan_costs.cache_info().misses == 2 * track.total_laps
    assert changed.pit_now_cost != plain.pit_now_cost


def test_no_pace_noise_draws_and_actual_lap_floor_preserved(monkeypatch):
    driver, car, track, weather = fixture(1)
    car.base_pace = 1.0
    track.base_lap_time = 1
    track.active_aero_zones = [ActiveAeroZone(zone_id=1, sector=1, time_gain=1)]

    class NoDraws:
        def normal(self, *args):
            pytest.fail("Projection sampled pace noise")

        def random(self):
            pytest.fail("Projection sampled a reaction")

    monkeypatch.setattr(np.random, "default_rng", lambda *args: NoDraws())
    result = weather_stop_costs(driver, car, track, weather,
                                TIRE_COMPOUNDS[TireCompound.INTERMEDIATE], 0, 1,
                                traffic_possible=False)
    assert result.stay_cost == track.base_lap_time * 0.95
    assert result.pit_now_cost == pytest.approx(
        track.base_lap_time * 0.95 + track.pit_lane_delta + expected_stationary_time(car)
    )
    disabled = weather_stop_costs(driver, car, track, weather,
                                  TIRE_COMPOUNDS[TireCompound.INTERMEDIATE], 0, 1,
                                  traffic_possible=False, active_aero_enabled=False)
    assert disabled.stay_cost > result.stay_cost
    assert disabled.pit_now_cost > result.pit_now_cost
    expected_fresh = LapSimulator(NoDraws()).calculate_lap_time(
        driver, car, track, TIRE_COMPOUNDS[TireCompound.WET], weather, 1, 1,
        active_aero_enabled=False, sample_variation=False,
    )
    assert disabled.pit_now_cost == pytest.approx(
        expected_fresh + track.pit_lane_delta + expected_stationary_time(car)
    )
