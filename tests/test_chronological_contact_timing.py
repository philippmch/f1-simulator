"""Native chronological crossing clocks after a passing contact."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.events import EventType
from f1sim.simulation.race import DriverStatus, RaceSimulator

EPSILON = 1e-9
TIME_TOLERANCE = 1e-7


class RecordingRng:
    """Keep the simulator's real seeded draws observable."""

    def __init__(self, seed):
        self.generator = np.random.default_rng(seed)
        self.uniform_calls = []

    def uniform(self, low=0.0, high=1.0, size=None):
        result = self.generator.uniform(low, high, size)
        self.uniform_calls.append((low, high, result))
        return result

    def __getattr__(self, name):
        return getattr(self.generator, name)


def _run_contact(monkeypatch, *, paces, laps, seed):
    driver_ids = list(paces)
    drivers = [Driver(id=key, name=key, team_id=key) for key in driver_ids]
    cars = {key: Car(team_id=key, team_name=key) for key in driver_ids}
    track = Track(id="contact", name="Contact", country="Test", total_laps=laps,
                  base_lap_time=90, pit_lane_delta=1)
    rng = RecordingRng(seed)
    simulator = RaceSimulator(rng)
    engine = ChronologicalRace(simulator)

    # Synthetic fixture controls are limited to pace, pit choice, the contact
    # model outcome, weather evolution, and unrelated stochastic hazards.
    monkeypatch.setattr(
        simulator.lap_simulator, "calculate_lap_time",
        lambda driver, *args, **kwargs: paces[driver.id],
    )
    monkeypatch.setattr(simulator, "_should_pit", lambda *args, **kwargs: False)
    monkeypatch.setattr(Weather, "evolve", lambda self, _rng: self.model_copy(deep=True))
    manager = simulator.event_manager
    monkeypatch.setattr(manager, "_check_mechanical_failure", lambda *args, **kwargs: None)
    monkeypatch.setattr(manager, "_check_random_incident", lambda *args, **kwargs: None)

    attempts = []
    def return_contact_for_first_battle(attacker, _car, defender, *_args, **_kwargs):
        lap = engine.pending[attacker.id].lap
        contact = attacker.id == "B" and defender.id == "A" and lap == 1
        attempts.append((attacker.id, defender.id, lap, contact))
        return False, contact

    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake",
                        return_contact_for_first_battle)

    race_control_incidents = []
    def observe_race_control(*args, **kwargs):
        race_control_incidents.append(kwargs["incidents_this_lap"])
        # Race-control interventions are unrelated background hazards here.
        return []

    monkeypatch.setattr(manager, "process_lap", observe_race_control)
    results = engine.run(
        drivers, cars, track, Weather(change_probability=0), driver_ids,
        starting_tires={key: TireCompound.SOFT for key in driver_ids},
    )
    return engine, results, rng, attempts, race_control_incidents, manager


@pytest.mark.parametrize(("paces", "laps", "seed", "expected"), [
    (
        {"A": 90.0, "B": 89.0}, 3, 0,
        [("A", 1, 90.9046800706458), ("B", 1, 91.2739233746429),
         ("A", 2, 180.904680070646), ("B", 2, 180.904680071646),
         ("A", 3, 270.904680070646), ("B", 3, 270.904680071646)],
    ),
    (
        {"A": 90.0, "B": 89.0}, 3, 8,
        [("A", 1, 91.9809152650069), ("B", 1, 91.9809152660069),
         ("A", 2, 181.980915265007), ("B", 2, 181.980915266007),
         ("A", 3, 271.980915265007), ("B", 3, 271.980915266007)],
    ),
    (
        {"A": 90.0, "B": 89.0, "C": 90.5}, 3, 0,
        [("A", 1, 90.9046800706458), ("B", 1, 91.2739233746429),
         ("C", 1, 91.2739233756429), ("A", 2, 180.904680070646),
         ("B", 2, 180.904680071646), ("C", 2, 181.773923375643),
         ("A", 3, 270.904680070646), ("B", 3, 270.904680071646),
         ("C", 3, 272.273923375643)],
    ),
    (
        {"A": 90.0, "B": 89.0}, 1, 0,
        [("A", 1, 90.9046800706458), ("B", 1, 91.2739233746429)],
    ),
])
def test_contact_loss_is_charged_once_on_native_crossing_clock(
    monkeypatch, paces, laps, seed, expected,
):
    engine, results, rng, attempts, control_incidents, manager = _run_contact(
        monkeypatch, paces=paces, laps=laps, seed=seed,
    )

    actual = engine.crossings
    assert [(driver, lap) for driver, lap, _ in actual] == [
        (driver, lap) for driver, lap, _ in expected
    ]
    assert [time for _, _, time in actual] == pytest.approx(
        [time for _, _, time in expected], rel=0, abs=TIME_TOLERANCE,
    )
    assert len(actual) == len(paces) * laps  # A stale pre-delay heap entry never crosses.
    assert all(row.laps_completed == laps for row in results)
    assert all(row.status == DriverStatus.FINISHED for row in results)

    attacker_loss, defender_loss = np.random.default_rng(seed).uniform(
        [1.0, 0.5], [3.0, 2.0],
    )
    assert [(low, high) for low, high, _ in rng.uniform_calls] == [(1, 3), (0.5, 2)]
    assert [value for _, _, value in rng.uniform_calls] == pytest.approx(
        [attacker_loss, defender_loss], abs=1e-12,
    )
    collision_events = [event for event in manager.events
                        if event.event_type == EventType.COLLISION]
    assert len(collision_events) == 1
    collision = collision_events[0]
    assert collision.lap == 1
    assert collision.drivers_involved == ["B", "A"]
    assert collision.applied_time_losses == pytest.approx(
        {"B": attacker_loss, "A": defender_loss}, abs=1e-12,
    )
    assert collision.time_loss_seconds == 0 and not collision.forces_pit_stop
    assert [row.overtake_contacts for row in results if row.driver_id == "B"] == [1]
    assert control_incidents == [1] + [0] * (laps - 1)

    # The sampled loss that does not dominate the preceding car's delayed clock
    # is subsumed by the one-nanosecond crossing-order clamp.
    if attacker_loss < defender_loss:
        a1, b1 = actual[:2]
        assert b1[2] - a1[2] == pytest.approx(EPSILON, rel=0, abs=1e-12)

    if "C" in paces:
        a1, b1, c1 = actual[:3]
        assert c1[2] - b1[2] == pytest.approx(EPSILON, rel=0, abs=1e-12)
        assert [row.overtake_contacts for row in results if row.driver_id == "C"] == [0]

    # A contact on the last crossing finishes at the configured distance.
    if laps == 1:
        assert [(row.driver_id, row.laps_completed) for row in results] == [("A", 1), ("B", 1)]

    # The same pair can be tried on a later pending lap after the contact lap.
    if laps > 1:
        b_vs_a_laps = [lap for attacker, defender, lap, _ in attempts
                       if (attacker, defender) == ("B", "A")]
        assert b_vs_a_laps == list(range(1, laps + 1))
