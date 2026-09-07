"""Overtaking contact enters the ledger once, with already-applied losses."""

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
        simulator.lap_simulator.calculate_lap_time = lambda **kwargs: 90

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
