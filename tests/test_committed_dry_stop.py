"""A committed stop chooses its set over all remaining paid dry stints."""

import copy
from itertools import combinations, product

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import SLICKS, expected_stationary_time, plan_dry_stop
from f1sim.simulation.race import DriverRaceState, RaceSimulator


def fixture():
    state = DriverRaceState(
        Driver(id="A", name="A", team_id="A"), Car(team_id="A", team_name="A"), 1,
        current_tire=TIRE_COMPOUNDS[TireCompound.MEDIUM], tire_laps=20,
    )
    track = Track(id="test", name="Test", country="Test", total_laps=30,
                  base_lap_time=90, pit_lane_delta=1)
    return state, track


@pytest.mark.parametrize("stops,budget", [(0, 2), (1, 1), (2, 0), (4, 0)])
@pytest.mark.parametrize("history,candidates", [
    (["medium"], {TireCompound.SOFT, TireCompound.HARD}),
    (["medium", "soft"], set(SLICKS)),
    (["wet", "medium"], set(SLICKS)),
])
def test_current_paid_stop_consumes_budget_and_requires_legal_finish(
    monkeypatch, stops, budget, history, candidates,
):
    import f1sim.simulation.race as race_module

    state, track = fixture()
    state.pit_stops = stops
    state.tire_compound_history = history
    simulator = RaceSimulator(np.random.default_rng(42))
    simulator.event_manager.safety_car_active = True
    observed = []

    def capture(*args, **kwargs):
        observed.append((args, kwargs))
        return plan_dry_stop(*args, **kwargs)

    monkeypatch.setattr(race_module, "plan_dry_stop", capture)
    before = copy.deepcopy(state)
    rng_before = copy.deepcopy(simulator.rng.bit_generator.state)
    chosen = simulator._choose_committed_dry_compound(state, track, 26)
    assert chosen in (candidates if budget == 0 else set(SLICKS))
    assert {args[3].compound for args, _ in observed} == set(SLICKS)
    for args, kwargs in observed:
        assert args[4:7] == (0, 5, budget)
        assert args[3].compound in args[7]
        assert args[8] == ("wet" in history)
        assert kwargs["current_lap_time_modifier"] == 1.4
    assert state == before
    assert simulator.rng.bit_generator.state == rng_before


@pytest.mark.parametrize("flag,modifier", [(None, 1), ("safety_car_active", 1.4),
                                         ("vsc_active", 1.2)])
def test_committed_choice_matches_exhaustive_future_stints(flag, modifier):
    state, track = fixture()
    simulator = RaceSimulator()
    if flag:
        setattr(simulator.event_manager, flag, True)
    costs = {}
    for first in SLICKS:
        alternatives = []
        for stops in range(3):
            for schedule in combinations(range(1, 5), stops):
                for path in product(SLICKS, repeat=stops):
                    if len({TireCompound.MEDIUM, first, *path}) < 2:
                        continue
                    tire, age, cost = TIRE_COMPOUNDS[first], 0, 0
                    for lap in range(5):
                        if lap in schedule:
                            tire, age = TIRE_COMPOUNDS[path[schedule.index(lap)]], 0
                            cost += track.pit_lane_delta + expected_stationary_time(state.car)
                        cost += LapSimulator.tire_pace_contribution(
                            state.driver, state.car, track, tire, age
                        ) * (modifier if lap == 0 else 1)
                        age += 1
                    alternatives.append(cost)
        costs[first] = min(alternatives)
    assert simulator._choose_committed_dry_compound(state, track, 26) == min(costs, key=costs.get)


@pytest.mark.parametrize("weather,proposal,expected", [
    (Weather(), None, TireCompound.HARD),
    (Weather(), (20, TireCompound.SOFT), TireCompound.SOFT),
    (Weather(track_wetness=0.8), (20, TireCompound.SOFT), TireCompound.WET),
    (Weather(track_wetness=0.1), None, TireCompound.SOFT),
])
def test_execution_respects_weather_proposal_and_damp_fallback(
    monkeypatch, weather, proposal, expected,
):
    state, track = fixture()
    state.dry_pit_proposal = proposal
    simulator = RaceSimulator()
    calls = []

    def committed(*args):
        calls.append(args)
        return TireCompound.HARD

    monkeypatch.setattr(simulator, "_choose_committed_dry_compound", committed)
    monkeypatch.setattr(
        simulator, "_choose_distinct_dry_compound", lambda *args, **kwargs: TireCompound.SOFT,
    )
    monkeypatch.setattr(simulator.lap_simulator, "calculate_pit_stop_time", lambda *args: 3)
    simulator._execute_pit_stop(state, track, weather, 20)
    assert state.current_tire.compound == expected
    assert bool(calls) == (weather.track_wetness == 0 and proposal is None)
