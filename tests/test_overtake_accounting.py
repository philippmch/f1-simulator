"""Native engines count only calls made to the overtake model."""

from itertools import count

import numpy as np
import pytest

from f1sim.models import Car, Driver, Track, Weather
from f1sim.models.tire import TireCompound
from f1sim.simulation.chronological_race import ChronologicalRace, _PendingLap
from f1sim.simulation.events import EventType, RaceEvent
from f1sim.simulation.race import DriverRaceState, DriverStatus, RaceSimulator
from f1sim.simulation.race_timing import RaceFinishTimeline


def battle():
    drivers = [Driver(id=key, name=key, team_id=key) for key in ("A", "B")]
    cars = {key: Car(team_id=key, team_name=key) for key in ("A", "B")}
    track = Track(id="test", name="Test", country="Test", total_laps=4,
                  base_lap_time=90)
    states = [
        DriverRaceState(drivers[0], cars["A"], 1, total_time=90),
        DriverRaceState(drivers[1], cars["B"], 2, total_time=90.5),
    ]
    return drivers, cars, track, states


@pytest.mark.parametrize("outcome,expected", [
    ((False, False), (1, 0, 0)),
    ((True, False), (1, 1, 0)),
    ((False, True), (1, 0, 1)),
])
def test_standard_engine_counts_each_model_return(monkeypatch, outcome, expected):
    _, _, track, states = battle()
    simulator = RaceSimulator(np.random.default_rng(17))
    monkeypatch.setattr(simulator.overtaking_model, "should_attempt_overtake",
                        lambda *args: True)
    calls = []

    def attempt(*args, **kwargs):
        calls.append(kwargs["attacker"].id)
        return outcome

    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake", attempt)
    simulator._process_overtakes(states, track, Weather(), lap=1)

    assert calls == ["B"]
    assert (states[1].overtake_attempts, states[1].overtake_successes,
            states[1].overtake_contacts) == expected
    assert (states[0].overtake_attempts, states[0].overtake_successes,
            states[0].overtake_contacts) == (0, 0, 0)


def chronological_battle(monkeypatch, outcome=(False, False), *, lapping=False,
                         neutralized=False):
    drivers, cars, track, _ = battle()
    simulator = RaceSimulator(np.random.default_rng(31))
    weather = Weather(change_probability=0)
    engine = ChronologicalRace(simulator)
    state_a = DriverRaceState(drivers[0], cars["A"], 1)
    state_b = DriverRaceState(drivers[1], cars["B"], 2)
    if lapping:
        state_b.laps_completed = 1
        engine.timeline = RaceFinishTimeline(track.total_laps, ["A", "B"])
        engine.timeline.observe_crossing("B", 1, 90, is_leader=True)
        pending_a = _PendingLap(1, 0, 90, weather, False, state_a.current_tire, 0, 90)
        pending_b = _PendingLap(2, 90, 180, weather, False, state_b.current_tire, 0, 90)
    else:
        state_a.laps_completed = state_b.laps_completed = 1
        engine.timeline = RaceFinishTimeline(track.total_laps, ["A", "B"])
        engine.timeline.observe_crossing("A", 1, 90, is_leader=True)
        engine.timeline.observe_crossing("B", 1, 91)
        pending_a = _PendingLap(2, 90, 180, weather, False, state_a.current_tire, 0, 90)
        pending_b = _PendingLap(2, 91, 181, weather, neutralized,
                                state_b.current_tire, 0, 90)

    engine.track = track
    engine.weather = weather
    engine.states = {"A": state_a, "B": state_b}
    engine.order = ["A", "B"]
    engine.pending = {"A": pending_a, "B": pending_b}
    engine.queue = []
    engine.serial = count()
    engine.running_paces = {}
    engine.fastest = {}
    engine.regrouping = False
    engine.red_waiting = set()
    engine.incidents = 0
    engine.control_intervals = 0
    engine.green_streak = 0
    engine.has_two_green = False
    engine.crossings = []
    engine.pit_exits = []
    calls = []
    monkeypatch.setattr(simulator.event_manager, "_check_mechanical_failure",
                        lambda *args: None)
    monkeypatch.setattr(simulator.event_manager, "_check_random_incident",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(simulator.event_manager, "process_lap", lambda *args, **kwargs: [])
    monkeypatch.setattr(simulator.lap_simulator, "tire_weather_pace_contribution",
                        lambda *args, **kwargs: 0.0)

    def attempt(*args, **kwargs):
        calls.append(args[0].id)
        return outcome

    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake", attempt)
    return engine, calls


@pytest.mark.parametrize("outcome,expected", [
    ((False, False), (1, 0, 0)),
    ((True, False), (1, 1, 0)),
    ((False, True), (1, 0, 1)),
])
def test_chronological_engine_records_outcome_and_keeps_dnf_counts(
    monkeypatch, outcome, expected,
):
    engine, calls = chronological_battle(monkeypatch, outcome)
    engine._resolve_crossing("B", 181)
    state = engine.states["B"]
    state.status = DriverStatus.DNF
    state.dnf_reason = "test retirement"

    result = next(row for row in engine._results() if row.driver_id == "B")
    assert calls == ["B"]
    assert (state.overtake_attempts, state.overtake_successes,
            state.overtake_contacts) == expected
    assert (result.overtake_attempts, result.overtake_successes,
            result.overtake_contacts) == expected
    assert result.status == DriverStatus.DNF


def test_chronological_compliant_lapping_yield_is_not_an_attempt(monkeypatch):
    engine, calls = chronological_battle(monkeypatch, lapping=True)
    engine._resolve_crossing("B", 180)
    assert calls == []
    assert (engine.states["B"].overtake_attempts,
            engine.states["B"].overtake_successes,
            engine.states["B"].overtake_contacts) == (0, 0, 0)


def test_chronological_neutralized_crossing_is_not_an_attempt(monkeypatch):
    engine, calls = chronological_battle(monkeypatch, neutralized=True)
    engine._resolve_crossing("B", 181)
    assert calls == []
    assert engine.states["B"].overtake_attempts == 0


def test_chronological_failed_attempt_can_be_retried_on_a_later_lap(monkeypatch):
    engine, calls = chronological_battle(monkeypatch)
    engine._resolve_crossing("B", 181)
    assert engine.pending["B"].attempted == {"A"}

    # A new pending lap has a fresh encounter set; the same pair can reach the
    # model again, and the race-state total records both actual calls.
    old = engine.pending["B"]
    engine.pending["B"] = _PendingLap(
        3, 181, 271, old.weather, False, old.tire, old.tire_age, 90,
    )
    old_defender = engine.pending["A"]
    engine.pending["A"] = _PendingLap(
        3, 180, 270, old_defender.weather, False, old_defender.tire,
        old_defender.tire_age, 90,
    )
    engine.order = ["A", "B"]
    engine._resolve_crossing("B", 271)

    assert calls == ["B", "B"]
    assert engine.states["B"].overtake_attempts == 2


def test_standard_result_builder_retains_counts_for_dnf(monkeypatch):
    drivers, cars, track, _ = battle()
    drivers[0].current_tire_laps = drivers[1].current_tire_laps = 0
    simulator = RaceSimulator(np.random.default_rng(55))
    monkeypatch.setattr(simulator, "_should_pit", lambda *args, **kwargs: False)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time",
                        lambda *args, **kwargs: 90.0)
    monkeypatch.setattr(simulator.overtaking_model, "should_attempt_overtake",
                        lambda *args: True)
    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake",
                        lambda *args, **kwargs: (False, False))

    def retire_after_first_lap(lap, *args, **kwargs):
        if lap == 1:
            drivers[1].dnf = True
            drivers[1].dnf_reason = "test retirement"
            return [RaceEvent(EventType.MECHANICAL_FAILURE, lap, ["B"])]
        return []

    monkeypatch.setattr(simulator.event_manager, "process_lap", retire_after_first_lap)
    results = simulator.simulate_race(
        drivers, cars, track, Weather(change_probability=0), ["A", "B"],
        starting_tires={"A": TireCompound.SOFT, "B": TireCompound.SOFT},
    )
    retired = next(row for row in results if row.driver_id == "B")
    assert retired.status == DriverStatus.DNF
    assert (retired.overtake_attempts, retired.overtake_successes,
            retired.overtake_contacts) == (1, 0, 0)
