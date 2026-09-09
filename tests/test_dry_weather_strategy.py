"""Dry strategy uses the lap model's weather-scaled tyre seconds."""

import copy
from itertools import combinations, product

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather, WeatherCondition
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import SLICKS, expected_stationary_time, plan_dry_stop
from f1sim.simulation.race import DriverRaceState, RaceSimulator, TeamStrategyArchetype


def fixture():
    state = DriverRaceState(Driver(id="A", name="A", team_id="A"),
                            Car(team_id="A", team_name="A"), 1,
                            current_tire=TIRE_COMPOUNDS[TireCompound.SOFT], tire_laps=20,
                            tire_compound_history=["medium", "soft"])
    track = Track(id="t", name="T", country="T", total_laps=30,
                  base_lap_time=90, pit_lane_delta=10.8)
    return state, track


def test_cloudy_scaling_changes_near_tie_and_keeps_default_cache_isolated():
    state, track = fixture()
    args = (state.driver, state.car, track, state.current_tire, 20, 10, 1,
            {TireCompound.SOFT, TireCompound.MEDIUM})
    plain = plan_dry_stop(*args)
    cloudy = plan_dry_stop(*args, tire_pace_multiplier=1.01)
    assert not plain.should_pit()
    assert cloudy.should_pit()
    assert plan_dry_stop(*args) == plain
    assert plan_dry_stop(*args, tire_pace_multiplier=1) == plain
    assert track.base_lap_time == 90


@pytest.mark.parametrize("modifier,lane", [(1, 1), (1.2, 0.75), (1.4, 0.55)])
def test_scaled_dp_matches_independent_shared_pace_schedule_enumeration(modifier, lane):
    state, track = fixture()
    track.pit_lane_delta = 0.5
    weather = Weather(condition=WeatherCondition.CLOUDY, track_wetness=0.07, rain_intensity=0.1)
    factor = LapSimulator.weather_pace_multiplier(state.driver, state.car, weather)
    actual = plan_dry_stop(state.driver, state.car, track, state.current_tire, 20, 4, 2,
                          {TireCompound.SOFT, TireCompound.MEDIUM},
                          pit_lane_factor=lane, current_lap_time_modifier=modifier,
                          tire_pace_multiplier=factor)
    best = {True: float("inf"), False: float("inf")}
    for stops in range(3):
        for schedule in combinations(range(4), stops):
            for path in product(SLICKS, repeat=stops):
                tire, age, cost = state.current_tire, 20, 0
                for lap in range(4):
                    if lap in schedule:
                        tire, age = TIRE_COMPOUNDS[path[schedule.index(lap)]], 0
                        cost += expected_stationary_time(state.car)
                        cost += track.pit_lane_delta * (lane if lap == 0 else 1)
                    cost += LapSimulator.tire_pace_contribution(
                        state.driver, state.car, track, tire, age
                    ) * factor * (modifier if lap == 0 else 1)
                    age += 1
                now = bool(schedule and schedule[0] == 0)
                best[now] = min(best[now], cost)
    assert actual.pit_now_cost == pytest.approx(best[True])
    assert actual.wait_cost == pytest.approx(best[False])


@pytest.mark.parametrize("compound", SLICKS)
@pytest.mark.parametrize("age", [0, 10, 40])
def test_private_base_scaling_equals_actual_weather_scaled_tyre_physics(compound, age):
    state, track = fixture()
    factor = 1.037
    scaled = track.model_copy(update={"base_lap_time": track.base_lap_time * factor})
    assert LapSimulator.tire_pace_contribution(
        state.driver, state.car, scaled, TIRE_COMPOUNDS[compound], age
    ) == pytest.approx(LapSimulator.tire_pace_contribution(
        state.driver, state.car, track, TIRE_COMPOUNDS[compound], age
    ) * factor)


def test_all_race_planners_forward_weather_and_keep_traffic_separate_from_queue(monkeypatch):
    import f1sim.simulation.race as race_module

    state, track = fixture()
    state.car.wet_performance = 0.5
    weather = Weather(condition=WeatherCondition.CLOUDY, track_wetness=0.07, rain_intensity=0.1)
    factor = LapSimulator.weather_pace_multiplier(state.driver, state.car, weather)
    simulator = RaceSimulator(np.random.default_rng(42))
    observed = []

    def capture(*args, **kwargs):
        observed.append((args, kwargs))
        return plan_dry_stop(*args, **kwargs)

    monkeypatch.setattr(race_module, "plan_dry_stop", capture)
    monkeypatch.setattr(simulator, "_pit_rejoin_traffic_gaps", lambda *args: (None, 1))
    before = copy.deepcopy(simulator.rng.bit_generator.state)
    simulator._should_pit(state, [state], track, 20, False, weather, 2)
    args, options = observed[-1]
    assert args[10] == 2
    assert options["current_traffic_gaps"] == (None, 1)
    dirty = plan_dry_stop(*args, **options)
    clean = plan_dry_stop(*args, **{**options, "current_traffic_gaps": None})
    assert dirty.pit_now_cost - clean.pit_now_cost == pytest.approx(.25 * factor)
    assert dirty.wait_cost == clean.wait_cost
    simulator._choose_committed_dry_compound(state, track, 20, weather)
    simulator._choose_red_flag_tire(state, weather, track, 20)
    assert simulator.rng.bit_generator.state == before
    assert len(observed) == 7  # One decision, three paid and three free set choices.
    assert all(kwargs["tire_pace_multiplier"] == factor for _, kwargs in observed)
    opening_inputs = []
    original_opening = race_module.dry_opening_policy_costs

    def capture_opening(*args):
        opening_inputs.append(args)
        return original_opening(*args)

    monkeypatch.setattr(race_module, "dry_opening_policy_costs", capture_opening)
    simulator._choose_starting_compound(TeamStrategyArchetype.BALANCED, track, weather,
                                        state.driver, state.car)
    assert len(opening_inputs) == 1
    assert opening_inputs[0][:4] == (state.driver, state.car, track, weather)
