"""Frozen rejoin traffic and bounded, deterministic immediate strategy pricing."""

import copy
from typing import Any

import numpy as np
import pytest

from f1sim.models import Car, Driver, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS, TireCompound
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import DryPitDecision, expected_stationary_time
from f1sim.simulation.race import DriverRaceState, RaceSimulator, TeamStrategyArchetype


def track(laps: int = 30) -> Track:
    return Track(id="test", name="Test", country="Test", total_laps=laps,
                 base_lap_time=90, pit_lane_delta=1, overtake_difficulty=1,
                 tire_stress=0.5, safety_car_probability=0)


def state(name: str, position: int, clock: float) -> DriverRaceState:
    return DriverRaceState(
        driver=Driver(id=name, name=name, team_id=name),
        car=Car(team_id=name, team_name=name), position=position, total_time=clock,
        current_tire=TIRE_COMPOUNDS[TireCompound.MEDIUM].model_copy(deep=True),
        strategy_archetype=TeamStrategyArchetype.BALANCED,
    )


@pytest.mark.parametrize("gap,expected", [(None, 0), (-1, .5), (0, .5), (1, .25), (2, 0), (25, 0)])
def test_shared_traffic_curve(gap: float | None, expected: float) -> None:
    assert LapSimulator.traffic_pace_contribution(gap) == expected


@pytest.mark.parametrize("losses,expected", [
    ({"A": 4}, {"A": 1, "B": None, "C": 2}),
    ({"B": 25}, {"A": None, "B": 22, "C": 6}),
    ({"A": 10, "B": 4}, {"A": 3, "B": 1, "C": None}),
])
@pytest.mark.parametrize("reverse_loop", [False, True])
def test_frozen_batch_gaps_and_pre_stop_detection(monkeypatch, losses, expected, reverse_loop):
    sim = RaceSimulator(rng=np.random.default_rng(5))
    initial = [state("A", 1, 0), state("B", 2, 0), state("C", 3, 0)]
    observed = {}
    detected = {}
    original_process = sim._process_pit_stops

    def process(states, *args):
        result = original_process(states, *args)
        if reverse_loop:
            states.reverse()
        return result

    def lap_time(**kwargs: Any) -> float:
        observed[kwargs["lap_number"], kwargs["driver"].id] = kwargs["gap_to_car_ahead"]
        return {"A": 90, "B": 93, "C": 96}[kwargs["driver"].id]

    def detect(driver_state, race_track, gap, allowed):
        detected[driver_state.driver.id] = gap
        return False

    monkeypatch.setattr(sim, "_process_pit_stops", process)
    monkeypatch.setattr(
        sim, "_should_pit",
        lambda s, states, t, lap, *a, **k: lap == 2 and s.driver.id in losses,
    )
    monkeypatch.setattr(sim, "_execute_pit_stop", lambda s, *a, **k: losses[s.driver.id])
    monkeypatch.setattr(sim.lap_simulator, "calculate_lap_time", lap_time)
    monkeypatch.setattr(sim, "_deploy_overtake_mode_if_eligible", detect)
    monkeypatch.setattr(sim.event_manager, "process_lap", lambda **k: [])
    sim.simulate_race([s.driver for s in initial], {s.car.team_id: s.car for s in initial},
                      track(2), Weather(), ["A", "B", "C"])
    assert {name: observed[2, name] for name in expected} == expected
    # Detection remains at the original line, even when the leader has pitted.
    for name in set(expected) - losses.keys():
        assert detected[name] == {"A": None, "B": 3, "C": 3}[name]


@pytest.mark.parametrize("escape", [False, True])
def test_expected_traffic_can_change_near_tie_without_rng_or_state_mutation(monkeypatch, escape):
    sim = RaceSimulator(rng=np.random.default_rng(6))
    own = state("A", 2 if escape else 1, 100)
    cost = track().pit_lane_delta + expected_stationary_time(own.car)
    rival = state("B", 1 if escape else 2, 99 if escape else 100 + cost - 1)
    field = [own, rival]
    before = [(s.total_time, s.position) for s in field]
    rng_before = copy.deepcopy(sim.rng.bit_generator.state)
    observed = []

    def plan(*args, **kwargs):
        stay, rejoin = kwargs["current_traffic_gaps"]
        observed.append((args[-1], stay, rejoin))
        return DryPitDecision(10 + LapSimulator.traffic_pace_contribution(rejoin),
                              10 + LapSimulator.traffic_pace_contribution(stay),
                              TireCompound.HARD)

    monkeypatch.setattr("f1sim.simulation.race.plan_dry_stop", plan)
    assert sim._should_pit(own, field, track(), 10, True, Weather()) is escape
    assert observed == [(0, 1 if escape else None, pytest.approx(cost + 1) if escape else 1)]
    assert [(s.total_time, s.position) for s in field] == before
    assert sim.rng.bit_generator.state == rng_before


@pytest.mark.parametrize("neutralization", ["safety_car_active", "vsc_active"])
def test_neutralized_planner_excludes_traffic_and_preserves_queue_cost(monkeypatch, neutralization):
    sim = RaceSimulator(rng=np.random.default_rng(8))
    own = state("A", 1, 100)
    observed = []
    setattr(sim.event_manager, neutralization, True)

    def unexpected(*args):
        raise AssertionError("No green traffic projection under neutralisation")

    def plan(*args, **kwargs):
        observed.append(args[-1])
        return DryPitDecision(10, 11, TireCompound.HARD)

    monkeypatch.setattr(sim, "_pit_rejoin_traffic_cost", unexpected)
    monkeypatch.setattr("f1sim.simulation.race.plan_dry_stop", plan)
    sim._should_pit(own, [own], track(), 10, True, Weather(), additional_current_stop_cost=2)
    assert observed == [2]


def test_queue_delay_changes_expected_rejoin_gap():
    sim = RaceSimulator()
    own = state("A", 1, 100)
    expected_stop = track().pit_lane_delta + expected_stationary_time(own.car)
    rival = state("B", 2, 100 + expected_stop + 1)
    assert sim._pit_rejoin_traffic_cost(own, [own, rival], track()) == 0
    assert sim._pit_rejoin_traffic_cost(own, [own, rival], track(), 2) == .25

@pytest.mark.parametrize("clocks", [(100, 101, 102), (100, 110, 102), (100, 100, 100)])
@pytest.mark.parametrize("own_index", [0, 1, 2])
@pytest.mark.parametrize("reverse", [False, True])
def test_scalar_projection_matches_batch_insertion(clocks, own_index, reverse):
    sim = RaceSimulator()
    field = [state(name, i + 1, clock) for i, (name, clock) in enumerate(zip("ABC", clocks))]
    own = field[own_index]
    race_track = track()
    projected = copy.deepcopy(field)
    projected_own = projected[own_index]
    projected_own.total_time += race_track.pit_lane_delta + expected_stationary_time(own.car)
    stay_gap = sim._get_gap_to_car_ahead(own, field)
    sim._handle_pit_batch_position_changes([projected_own], projected)
    expected = (
        LapSimulator.traffic_pace_contribution(sim._get_gap_to_car_ahead(projected_own, projected))
        - LapSimulator.traffic_pace_contribution(stay_gap)
    )
    if reverse:
        field.reverse()
    assert sim._pit_rejoin_traffic_cost(own, field, race_track) == expected
