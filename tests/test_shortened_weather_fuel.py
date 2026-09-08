"""Weather planning retains original fuel loading after a timed finish is announced."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.models.track import ActiveAeroZone
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import DriverRaceState, RaceSimulator
from f1sim.simulation.weather_strategy import _fresh_plan_costs, _retained_costs, weather_stop_costs


def fixture(final=82):
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A", tire_degradation_factor=1.5)
    track = Track(id="t", name="T", country="T", total_laps=final,
                  base_lap_time=90, pit_lane_delta=0.1,
                  active_aero_zones=[ActiveAeroZone(zone_id=i + 1, sector=1, time_gain=0.7)
                                     for i in range(15)])
    weather = Weather(track_wetness=0.3, rain_intensity=0.3)
    return driver, car, track, weather


def test_announced_finish_retains_fuel_and_reverses_wrong_veto():
    driver, car, track, weather = fixture()
    tire = TIRE_COMPOUNDS[TireCompound.SOFT]
    old = weather_stop_costs(driver, car, track, weather, tire, 1, 82, traffic_possible=False)
    corrected = weather_stop_costs(driver, car, track, weather, tire, 1, 82,
                                  traffic_possible=False, physical_total_laps=100)
    assert old.pit_now_cost > old.stay_cost
    assert corrected.pit_now_cost < corrected.stay_cost
    assert corrected.stay_cost - corrected.pit_now_cost == pytest.approx(0.255021, abs=1e-6)
    simulator = RaceSimulator(np.random.default_rng(1))  # First reaction is below 70%.
    state = DriverRaceState(driver, car, 1, current_tire=tire, tire_laps=1,
                            tire_compound_history=[TireCompound.MEDIUM, TireCompound.SOFT])
    assert simulator._should_pit(state, [state], track, 82, False, weather,
                                 physical_total_laps=100)


@pytest.mark.parametrize("modifier,aero", [(1, True), (1.4, True), (1.15, False)])
def test_two_lap_paths_match_shared_physics_with_original_fuel(modifier, aero):
    driver, car, track, weather = fixture(83)
    simulator = LapSimulator(np.random.default_rng(9))
    surfaces = [weather, weather.project_surface()]

    def lap(compound, age, offset):
        local = driver.model_copy(update={"current_tire_laps": age})
        return simulator.calculate_lap_time(
            local, car, track, TIRE_COMPOUNDS[compound], surfaces[offset], 82 + offset, 100,
            active_aero_enabled=aero if offset == 0 else True, sample_variation=False,
        ) * (modifier if offset == 0 else 1)

    service = expected_stationary_time(car)
    first = lap(TireCompound.INTERMEDIATE, 0, 0)
    future = [lap(TireCompound.INTERMEDIATE, 1, 1)]
    future.extend(track.pit_lane_delta + service + lap(compound, 0, 1)
                  for compound in TireCompound
                  if surfaces[1].tire_mismatch(compound) != "critical")
    expected_pit = first + min(future) + track.pit_lane_delta * 0.55 + service + 2
    expected_stay = sum(lap(TireCompound.SOFT, 1 + offset, offset) for offset in range(2))
    result = weather_stop_costs(driver, car, track, weather, TIRE_COMPOUNDS[TireCompound.SOFT],
                                1, 82, physical_total_laps=100, traffic_possible=False,
                                current_lap_time_modifier=modifier, active_aero_enabled=aero,
                                pit_lane_factor=0.55, additional_current_stop_cost=2)
    assert result.pit_now_cost == pytest.approx(expected_pit)
    assert result.stay_cost == pytest.approx(expected_stay)


def test_original_fuel_denominator_is_part_of_both_cache_keys():
    driver, car, track, weather = fixture()
    args = driver, car, track, weather, TIRE_COMPOUNDS[TireCompound.SOFT], 1, 82
    _fresh_plan_costs.cache_clear()
    _retained_costs.cache_clear()
    default = weather_stop_costs(*args)
    assert weather_stop_costs(*args, physical_total_laps=82) == default
    assert _fresh_plan_costs.cache_info().misses == _retained_costs.cache_info().misses == 1
    weather_stop_costs(*args, physical_total_laps=100)
    assert _fresh_plan_costs.cache_info().misses == _retained_costs.cache_info().misses == 2


@pytest.mark.parametrize("projection", [False, True])
def test_live_and_opening_policy_forward_original_distance_when_forecast_shortens(
    monkeypatch, projection,
):
    from f1sim.simulation.opening_strategy import _policy_path_cost
    from f1sim.simulation.race import TeamStrategyArchetype

    driver, car, track, weather = fixture(10)
    simulator = RaceSimulator(np.random.default_rng(1))
    calls = []

    def should(self, state, states, planning_track, lap, *args, **kwargs):
        calls.append((lap, planning_track.total_laps, kwargs.get("physical_total_laps")))
        return False

    monkeypatch.setattr(RaceSimulator, "_should_pit", should)
    monkeypatch.setattr(LapSimulator, "calculate_lap_time", lambda *args, **kwargs: 1800)
    if projection:
        _policy_path_cost(driver, car, track, weather, TeamStrategyArchetype.BALANCED,
                          simulator.strategy_tuning, simulator.strategy_profiles,
                          TireCompound.INTERMEDIATE, 1)
    else:
        monkeypatch.setattr(simulator.event_manager, "process_lap", lambda **kwargs: [])
        monkeypatch.setattr(Weather, "evolve", lambda self, rng: self.project_surface())
        simulator.simulate_race([driver], {"A": car}, track, weather, ["A"],
                                starting_tires={"A": TireCompound.INTERMEDIATE})
    assert calls == [(1, 10, None)] + [(lap, 5, 10) for lap in range(2, 6)]
