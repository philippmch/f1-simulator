"""Reuse deterministic weather and stint physics without changing the forecast."""

from copy import deepcopy

import pytest
from test_strategy_pit_weather import clock_for, exhaustive_same, models

from f1sim.models import TireCompound, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.inventory_strategy import plan_inventory_strategy
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.rain_strategy import _running_row, plan_rain_stop, plan_rain_transition
from f1sim.simulation.tire_inventory import TireInventory
from f1sim.simulation.weather_strategy import weather_stop_costs


@pytest.mark.parametrize("planner", ["transition", "weather", "inventory"])
def test_timed_surface_cache_advances_each_update_once(monkeypatch, planner):
    driver, car, track, _ = models(laps=5)
    weather = Weather(track_wetness=.19, rain_intensity=.39)
    tire = TIRE_COMPOUNDS[TireCompound.MEDIUM]
    clock = clock_for(track)
    calls = 0
    original = Weather.project_surface

    def counted(self):
        nonlocal calls
        calls += 1
        return original(self)

    monkeypatch.setattr(Weather, "project_surface", counted)
    if planner == "inventory":
        inventory = TireInventory.from_sets([
            {"compound": "medium", "age": 3}, {"compound": "intermediate"},
            {"compound": "hard"},
        ])
        inventory.fit("set-1")
        plan_inventory_strategy(driver, car, track, weather, inventory, 1,
                                remaining_stops=2, weather_clock=clock)
    elif planner == "transition":
        plan_rain_transition(driver, car, track, weather, tire, 3, 1, 2,
                             used_compounds=("medium",), weather_clock=clock)
    else:
        weather_stop_costs(driver, car, track, weather, tire, 3, 1, weather_clock=clock)
    assert 0 < calls <= clock.max_updates


def test_clocked_stints_reuse_physics_but_include_changed_car_performance(monkeypatch, request):
    _running_row.cache_clear()
    request.addfinalizer(_running_row.cache_clear)
    driver, car, track, weather = models()
    tire = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE]
    clock = clock_for(track)
    calls = 0
    original = LapSimulator.calculate_lap_time

    def counted(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(LapSimulator, "calculate_lap_time", counted)
    first = plan_rain_stop(driver, car, track, weather, tire, 39, 1, 2,
                           weather_clock=clock)
    initial_calls = calls
    renamed_driver = driver.model_copy(update={"id": "other", "team_id": "other"})
    renamed_car = car.model_copy(update={"team_id": "other", "team_name": "Other"})
    same = plan_rain_stop(renamed_driver, renamed_car, track, weather, tire, 39, 1, 2,
                          weather_clock=clock)
    assert same == first
    assert calls - initial_calls == 2  # Current traffic/control laps stay outside the cache.

    changed_car = renamed_car.model_copy(update={"tire_degradation_factor": 1.5})
    before = calls
    changed = plan_rain_stop(renamed_driver, changed_car, track, weather, tire, 39, 1, 2,
                             weather_clock=clock)
    assert calls > before + 2
    expected = exhaustive_same(deepcopy(driver), changed_car, track, weather,
                               tire, 39, 1, 2, clock)
    assert changed.pit_now_cost == pytest.approx(expected[0])
    assert changed.wait_cost == pytest.approx(expected[1])


@pytest.mark.parametrize("budget", [0, 2])
def test_clocked_stints_match_control_traffic_and_physical_fuel(budget):
    driver, car, track, weather = models()
    tire = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE].model_copy(
        update={"degradation_rate": .08},
    )
    clock = clock_for(track, current=27, future=13)
    expected = exhaustive_same(
        deepcopy(driver), car, track, weather, tire, 12, 1, budget, clock,
        queue=-2, modifier=1.2, aero=False, lane=.75,
        current_gaps=(.2, 1.1), physical_total_laps=10,
    )
    actual = plan_rain_stop(
        driver, car, track, weather, tire, 12, 1, budget, weather_clock=clock,
        additional_current_stop_cost=-2, current_lap_time_modifier=1.2,
        active_aero_enabled=False, pit_lane_factor=.75,
        current_traffic_gaps=(.2, 1.1), physical_total_laps=10,
    )
    assert actual.pit_now_cost == pytest.approx(expected[0])
    assert actual.wait_cost == pytest.approx(expected[1])
