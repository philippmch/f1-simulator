"""Overtaking contact enters the ledger once, with already-applied losses."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from f1sim.analysis.montecarlo import _run_single_simulation
from f1sim.models import Car, Driver, Track, Weather
from f1sim.simulation.events import EventType, RaceEvent
from f1sim.simulation.overtaking import OvertakingModel
from f1sim.simulation.qualifying import QualifyingSimulator
from f1sim.simulation.race import DriverRaceState, RaceSimulator


def fixture():
    drivers = [Driver(id=name, name=name, team_id=name) for name in ("A", "B")]
    cars = {name: Car(team_id=name, team_name=name) for name in ("A", "B")}
    track = Track(id="test", name="Test", country="Test", total_laps=2, base_lap_time=90)
    states = [DriverRaceState(d, cars[d.id], i + 1, total_time=90 + i * 0.5,
                              last_lap_time=90) for i, d in enumerate(drivers)]
    return drivers, cars, track, states


@pytest.mark.parametrize("lap", [None, 7])
def test_contact_records_one_event_with_exact_applied_losses_and_no_extra_draw(monkeypatch, lap):
    _, _, track, states = fixture()
    simulator = RaceSimulator(np.random.default_rng(42))
    simulator.event_manager.current_lap = 99  # Must not leak into unknown helper lap.
    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake", lambda **kw: (False, True))
    expected_rng = np.random.default_rng(42)
    attack_loss, defend_loss = expected_rng.uniform(1, 3), expected_rng.uniform(0.5, 2)
    assert simulator._process_overtakes(states, track, Weather(), lap=lap) == 1
    event, = simulator.event_manager.events
    assert event.event_type == EventType.COLLISION
    assert event.lap == (0 if lap is None else lap)
    assert event.drivers_involved == ["B", "A"]
    assert event.applied_time_losses == {"B": attack_loss, "A": defend_loss}
    assert event.time_loss_seconds == 0 and not event.forces_pit_stop
    assert states[0].total_time == pytest.approx(90 + defend_loss)
    assert states[1].total_time == pytest.approx(90.5 + attack_loss)
    assert states[0].last_lap_time == pytest.approx(90 + defend_loss)
    assert states[1].last_lap_time == pytest.approx(90 + attack_loss)
    assert simulator.rng.random() == expected_rng.random()
    simulator.event_manager.reset()
    assert simulator.event_manager.events == []


@pytest.mark.parametrize("success", [False, True])
def test_clean_battle_has_no_ledger_event(monkeypatch, success):
    _, _, track, states = fixture()
    simulator = RaceSimulator(np.random.default_rng(42))
    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake",
                        lambda **kw: (success, False))
    assert simulator._process_overtakes(states, track, Weather(), lap=1) == 0
    assert simulator.event_manager.events == []
    assert [state.driver.id for state in sorted(states, key=lambda state: state.position)] == (
        ["B", "A"] if success else ["A", "B"]
    )


@pytest.mark.parametrize("restart,gap,eligible", [
    (False, 1.5, True),
    (False, 1.500001, False),
    (True, 1.75, True),
    (True, 2.0, True),
    (True, 2.000001, False),
])
def test_opportunity_window_gates_native_calls_and_attempt_counts(
    monkeypatch, restart, gap, eligible,
):
    _, _, track, states = fixture()
    states[1].total_time = states[0].total_time + gap
    simulator = RaceSimulator(np.random.default_rng(42))
    before_rng = deepcopy(simulator.rng.bit_generator.state)
    before_states = deepcopy(states)
    eligibility_calls, attempts = [], []
    native_attempt = simulator.overtaking_model.attempt_overtake

    def should_attempt(*args):
        eligibility_calls.append(args)
        return True

    def attempt(**kwargs):
        attempts.append(kwargs)
        return native_attempt(**kwargs)

    monkeypatch.setattr(simulator.overtaking_model, "should_attempt_overtake", should_attempt)
    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake", attempt)
    assert simulator._process_overtakes(states, track, Weather(),
                                       restart_lap=restart, lap=1) == 0
    assert len(attempts) == int(eligible)
    assert len(eligibility_calls) == int(eligible and not restart)
    assert [state.overtake_attempts for state in states] == [0, int(eligible)]
    assert [state.overtake_successes for state in states] == [0, 0]
    assert [state.overtake_contacts for state in states] == [0, 0]
    assert simulator.event_manager.events == []
    assert [state.position for state in states] == [1, 2]
    if eligible:
        assert attempts[0]["gap"] == pytest.approx(gap)
        assert attempts[0]["restart_boost"] is restart
        expected_rng = np.random.default_rng(42)
        expected_rng.random()  # One actual maneuver roll, including at the boundary.
        assert simulator.rng.bit_generator.state == expected_rng.bit_generator.state
    else:
        assert states == before_states
        assert simulator.rng.bit_generator.state == before_rng


def test_restart_window_supports_replacement_models_without_gap_helper():
    _, _, track, states = fixture()
    states[1].total_time = 91.75
    simulator = RaceSimulator(np.random.default_rng(42))
    attempts = []

    def attempt(**kwargs):
        attempts.append(kwargs)
        return True, False

    simulator.overtaking_model = SimpleNamespace(attempt_overtake=attempt)
    assert simulator._process_overtakes(states, track, Weather(), restart_lap=True) == 0
    assert len(attempts) == 1
    assert [state.position for state in states] == [2, 1]
    assert states[1].overtake_attempts == states[1].overtake_successes == 1


@pytest.mark.parametrize("seed", [8, 42])
@pytest.mark.parametrize("success", [False, True])
def test_contact_preserves_battle_outcome_for_either_loss_order(monkeypatch, seed, success):
    _, _, track, states = fixture()
    states[1].total_time = 89
    simulator = RaceSimulator(np.random.default_rng(seed))
    monkeypatch.setattr(simulator.overtaking_model, "should_attempt_overtake",
                        lambda *args: True)
    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake",
                        lambda **kwargs: (success, True))
    expected_rng = np.random.default_rng(seed)
    attacker_loss = expected_rng.uniform(1, 3)
    defender_loss = expected_rng.uniform(0.5, 2)

    assert simulator._process_overtakes(states, track, Weather(), lap=1) == 1
    assert [state.driver.id for state in sorted(states, key=lambda state: state.position)] == (
        ["B", "A"] if success else ["A", "B"]
    )
    assert simulator.event_manager.events[0].applied_time_losses == {
        "B": attacker_loss, "A": defender_loss,
    }
    assert [state.total_time for state in states] == pytest.approx(
        [90 + defender_loss, 89 + attacker_loss]
    )
    assert [state.last_lap_time for state in states] == pytest.approx(
        [90 + defender_loss, 90 + attacker_loss]
    )
    assert (states[1].overtake_attempts, states[1].overtake_successes,
            states[1].overtake_contacts) == (1, int(success), 1)
    assert simulator.rng.random() == expected_rng.random()


@pytest.mark.parametrize("first_success", [False, True])
@pytest.mark.parametrize("following_success", [False, True])
def test_following_car_battles_immediate_neighbor_after_contact(
    monkeypatch, first_success, following_success,
):
    _, _, track, states = fixture()
    driver = Driver(id="C", name="C", team_id="C")
    states.append(DriverRaceState(driver, Car(team_id="C", team_name="C"), 3,
                                  total_time=88, last_lap_time=88))
    states[1].total_time = 89
    simulator = RaceSimulator(np.random.default_rng(8))
    monkeypatch.setattr(simulator.overtaking_model, "should_attempt_overtake",
                        lambda *args: True)
    battles = []

    def attempt(**kwargs):
        attacker, defender = kwargs["attacker"].id, kwargs["defender"].id
        battles.append((attacker, defender))
        return (first_success, True) if attacker == "B" else (following_success, False)

    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake", attempt)
    assert simulator._process_overtakes(states, track, Weather(), lap=1) == 1
    assert battles == [("B", "A"), ("C", "A" if first_success else "B")]
    expected_order = ["B", "A", "C"] if first_success else ["A", "B", "C"]
    if following_success:
        expected_order[1], expected_order[2] = expected_order[2], expected_order[1]
    assert [state.driver.id for state in sorted(states, key=lambda state: state.position)] == (
        expected_order
    )
    assert len(simulator.event_manager.events) == 1
    assert [(state.overtake_attempts, state.overtake_successes, state.overtake_contacts)
            for state in states] == [
                (0, 0, 0), (1, int(first_success), 1), (1, int(following_success), 0),
            ]


def test_worker_counts_contact_once_and_race_control_follows_it(monkeypatch):
    drivers, cars, track, _ = fixture()
    monkeypatch.setattr(QualifyingSimulator, "simulate_qualifying", lambda *args: [])
    monkeypatch.setattr(QualifyingSimulator, "get_starting_grid", lambda *args: ["A", "B"])
    monkeypatch.setattr(OvertakingModel, "attempt_overtake", lambda *args, **kwargs: (False, True))
    original_init = RaceSimulator.__init__
    captured = []

    def initialize(simulator, *args, **kwargs):
        original_init(simulator, *args, **kwargs)
        captured.append(simulator)
        simulator._should_pit = lambda *args, **kwargs: False
        simulator.lap_simulator.calculate_lap_time = lambda *args, **kwargs: 90

        def process_lap(lap, incidents_this_lap, **kwargs):
            if lap == 1:
                assert incidents_this_lap == 1
                assert [event.event_type for event in simulator.event_manager.events] == [
                    EventType.COLLISION
                ]
                event = RaceEvent(EventType.SAFETY_CAR, lap, duration_laps=2)
                simulator.event_manager.events.append(event)
                simulator.event_manager.safety_car_active = True
                return [event]
            return []

        simulator.event_manager.process_lap = process_lap

    monkeypatch.setattr(RaceSimulator, "__init__", initialize)
    results, _, counts = _run_single_simulation((
        [driver.model_dump() for driver in drivers],
        {key: car.model_dump() for key, car in cars.items()}, track.model_dump(),
        Weather(change_probability=0).model_dump(), 42,
    ))
    assert counts.pop("weather_history") == captured[0].weather_history
    assert counts == {"incidents": 1, "safety_car": 1, "vsc": 0, "red_flag": 0,
                      "mechanical_failure_breakdown": {}}
    assert [event.event_type for event in captured[0].event_manager.events] == [
        EventType.COLLISION, EventType.SAFETY_CAR
    ]
    assert all(result.laps_completed == 2 for result in results)
    # Reusing the simulator resets the previous race's authoritative ledger.
    captured[0].simulate_race(drivers, cars, track, Weather(change_probability=0), ["A", "B"])
    assert [event.event_type for event in captured[0].event_manager.events] == [
        EventType.COLLISION, EventType.SAFETY_CAR
    ]
