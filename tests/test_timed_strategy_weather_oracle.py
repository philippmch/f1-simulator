"""Clock-aware planners agree with explicit short event/stop schedules."""

from copy import deepcopy

import pytest
from test_strategy_pit_weather import (
    clock_for,
    exhaustive_transition,
    exhaustive_weather_bound,
    models,
)

from f1sim.models import TireCompound, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.rain_strategy import plan_rain_transition
from f1sim.simulation.weather_strategy import weather_stop_costs

CASES = [
    (.25, 0, TireCompound.WET),
    (.44, .6, TireCompound.INTERMEDIATE),
    (.19, .39, TireCompound.MEDIUM),
    (.07, .2, TireCompound.SOFT),
]


@pytest.mark.parametrize("wetness,rain,compound", [
    (.44, .6, TireCompound.INTERMEDIATE),
    (.8, .9, TireCompound.WET),
])
@pytest.mark.parametrize("age", [0, 40])
def test_unused_rain_fit_preserves_compulsory_stop_with_no_budget(
    wetness, rain, compound, age,
):
    driver, car, track, _ = models(laps=5, lane=3)
    weather = Weather(track_wetness=wetness, rain_intensity=rain)
    tire = TIRE_COMPOUNDS[compound].model_copy()
    clock = clock_for(track)
    expected = exhaustive_transition(
        deepcopy(driver), car, track, weather, tire, age, 1, 0, clock, used=(),
    )
    actual = plan_rain_transition(
        driver, car, track, weather, tire, age, 1, 0, weather_clock=clock,
        used_compounds=(), remaining_dry_stops=0, remaining_damp_stops=0,
    )
    assert actual.pit_now_cost == pytest.approx(expected[0])
    assert actual.wait_cost == pytest.approx(expected[1])
    assert actual.should_pit() == (expected[0] < expected[1])


@pytest.mark.parametrize("wetness,rain,compound", CASES)
@pytest.mark.parametrize("budget", [0, 2])
def test_transition_matches_event_oracle_across_thresholds(wetness, rain, compound, budget):
    driver, car, track, _ = models(laps=4, lane=3)
    weather = Weather(track_wetness=wetness, rain_intensity=rain)
    tire = TIRE_COMPOUNDS[compound].model_copy()
    clock = clock_for(track, current=27, future=13)
    expected = exhaustive_transition(
        deepcopy(driver), car, track, weather, tire, 19, 1, budget, clock,
        lane=.75, queue=-2, modifier=1.2, aero=False, used=(compound,),
    )
    actual = plan_rain_transition(
        driver, car, track, weather, tire, 19, 1, budget,
        weather_clock=clock, pit_lane_factor=.75, additional_current_stop_cost=-2,
        current_lap_time_modifier=1.2, active_aero_enabled=False,
        used_compounds=(compound,),
    )
    assert actual.pit_now_cost == pytest.approx(expected[0])
    assert actual.wait_cost == pytest.approx(expected[1])


@pytest.mark.parametrize("wetness,rain,compound", CASES)
@pytest.mark.parametrize("traffic", [False, True])
def test_weather_bound_matches_event_oracle_with_control_and_traffic(
    wetness, rain, compound, traffic,
):
    driver, car, track, _ = models(laps=4, lane=3)
    weather = Weather(track_wetness=wetness, rain_intensity=rain)
    tire = TIRE_COMPOUNDS[compound].model_copy()
    clock = clock_for(track, current=27, future=13)
    expected = exhaustive_weather_bound(
        deepcopy(driver), car, track, weather, tire, 19, 1, clock,
        lane=.75, queue=-2, modifier=1.2, aero=False, traffic_possible=traffic,
    )
    actual = weather_stop_costs(
        driver, car, track, weather, tire, 19, 1, weather_clock=clock,
        pit_lane_factor=.75, additional_current_stop_cost=-2,
        current_lap_time_modifier=1.2, active_aero_enabled=False,
        traffic_possible=traffic,
    )
    assert actual.pit_now_cost == pytest.approx(expected[0])
    assert actual.stay_cost == pytest.approx(expected[1])
