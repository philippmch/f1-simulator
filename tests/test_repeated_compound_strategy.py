"""A repeated compound is legal if a later stint completes the rule."""

import copy
from itertools import product

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import SLICKS, expected_stationary_time, plan_dry_stop
from f1sim.simulation.race import DriverRaceState, RaceSimulator


def fixture():
    state = DriverRaceState(Driver(id="A", name="A", team_id="A"),
                            Car(team_id="A", team_name="A"), 1,
                            current_tire=TIRE_COMPOUNDS[TireCompound.SOFT], tire_laps=25,
                            pit_stops=1, tire_compound_history=["soft", "soft"])
    track = Track(id="t", name="T", country="T", total_laps=60,
                  base_lap_time=90, pit_lane_delta=5)
    return state, track


@pytest.mark.parametrize("modifier,lane", [(1, 1), (1.2, 0.75), (1.4, 0.55)])
def test_two_stop_solver_matches_legal_repeat_enumeration(modifier, lane):
    state, track = fixture()
    outcomes = []
    for first, second in product(SLICKS, repeat=2):
        if len({TireCompound.SOFT, first, second}) < 2:
            continue
        for split in range(1, 22):
            cost = (track.pit_lane_delta * (lane + 1) + 2 * expected_stationary_time(state.car))
            cost += sum(LapSimulator.tire_pace_contribution(
                state.driver, state.car, track, TIRE_COMPOUNDS[first], age
            ) * (modifier if age == 0 else 1) for age in range(split))
            cost += sum(LapSimulator.tire_pace_contribution(
                state.driver, state.car, track, TIRE_COMPOUNDS[second], age
            ) for age in range(22 - split))
            outcomes.append((cost, first))
    for first in (TireCompound.MEDIUM, TireCompound.HARD):
        cost = track.pit_lane_delta * lane + expected_stationary_time(state.car)
        cost += sum(LapSimulator.tire_pace_contribution(
            state.driver, state.car, track, TIRE_COMPOUNDS[first], age
        ) * (modifier if age == 0 else 1) for age in range(22))
        outcomes.append((cost, first))
    decision = plan_dry_stop(state.driver, state.car, track, state.current_tire, 25, 22, 2,
                             {TireCompound.SOFT}, pit_lane_factor=lane,
                             current_lap_time_modifier=modifier)
    assert decision.pit_now_cost == pytest.approx(min(cost for cost, _ in outcomes))
    assert any(compound == decision.compound and abs(cost - decision.pit_now_cost) < 1e-9
               for cost, compound in outcomes)
    if modifier == 1.4:
        assert decision.compound == TireCompound.SOFT
        assert decision.pit_now_cost == pytest.approx(16.39932066113589)
        assert decision.pit_now_cost < 16.561320661135888 - 0.16


def test_execution_can_repeat_now_and_still_requires_distinct_later(monkeypatch):
    state, track = fixture()
    simulator = RaceSimulator(np.random.default_rng(42))
    simulator.event_manager.safety_car_active = True
    before = copy.deepcopy(simulator.rng.bit_generator.state)
    assert simulator._choose_committed_dry_compound(state, track, 39) == TireCompound.SOFT
    assert simulator.rng.bit_generator.state == before
    assert simulator._should_pit(state, [state], track, 39, True, Weather())
    assert state.dry_pit_proposal == (39, TireCompound.SOFT)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_pit_stop_time", lambda *args: 3)
    simulator._execute_pit_stop(state, track, Weather(), 39)
    assert state.current_tire.compound == TireCompound.SOFT
    state.pit_stops += 1
    state.tire_laps = 20
    assert simulator._used_slick_compounds(state) == {TireCompound.SOFT}
    assert simulator._should_pit(state, [state], track, 59, False, Weather())
    simulator._execute_pit_stop(state, track, Weather(), 59)
    assert state.current_tire.compound in {TireCompound.MEDIUM, TireCompound.HARD}
    state.tire_laps = 1  # Complete running on the distinct replacement.
    assert len(simulator._used_slick_compounds(state)) == 2


@pytest.mark.parametrize("remaining,budget", [(1, 2), (22, 1)])
def test_no_legal_future_means_repeated_final_compound_cannot_be_selected(remaining, budget):
    state, track = fixture()
    decision = plan_dry_stop(state.driver, state.car, track, state.current_tire, 25,
                             remaining, budget, {TireCompound.SOFT},
                             current_lap_time_modifier=1.4)
    assert decision.compound in {TireCompound.MEDIUM, TireCompound.HARD}
