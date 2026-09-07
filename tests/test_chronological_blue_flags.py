"""Lapping yields differ from position battles and attempts to unlap."""

from math import ceil

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.race import RaceSimulator


def setup(monkeypatch, paces, laps=10):
    drivers = [Driver(id=key, name=key, team_id=key) for key in paces]
    cars = {key: Car(team_id=key, team_name=key) for key in paces}
    track = Track(id="t", name="T", country="T", total_laps=laps, base_lap_time=90)
    simulator = RaceSimulator(np.random.default_rng(4))
    monkeypatch.setattr(simulator, "_should_pit", lambda *a, **k: False)
    monkeypatch.setattr(simulator.event_manager, "process_lap", lambda *a, **k: [])
    monkeypatch.setattr(simulator.event_manager, "_check_mechanical_failure", lambda *a: None)
    monkeypatch.setattr(simulator.event_manager, "_check_random_incident", lambda *a: None)
    monkeypatch.setattr(Weather, "evolve", lambda self, rng: self.model_copy(deep=True))
    calls, attempts = [], []

    def lap(driver, car, track, tire, weather, lap, total_laps, **kwargs):
        calls.append((driver.id, lap))
        pace = paces[driver.id]
        return pace(lap) if callable(pace) else pace

    def fail(attacker, attacker_car, defender, *a, **k):
        attempts.append((attacker.id, defender.id))
        return False, False

    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", lap)
    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake", fail)
    engine = ChronologicalRace(simulator)

    def run():
        return engine.run(drivers, cars, track, Weather(), list(paces),
                          starting_tires={key: TireCompound.INTERMEDIATE for key in paces})

    return engine, run, calls, attempts


@pytest.mark.parametrize("paces", [
    {"A": 90, "B": 110}, {"A": 90, "B": 110, "C": 150, "D": 200},
])
@pytest.mark.parametrize("laps", [10, 20])
def test_compliant_lapping_needs_no_successful_battle(monkeypatch, paces, laps):
    engine, run, calls, attempts = setup(monkeypatch, paces, laps)
    results = run()
    flag_time = min(paces.values()) * laps
    for row in results:
        expected_laps = ceil(flag_time / paces[row.driver_id])
        assert row.laps_completed == expected_laps
        assert row.total_time == expected_laps * paces[row.driver_id]
        assert [lap for driver, lap in calls if driver == row.driver_id] == list(
            range(1, expected_laps + 1)
        )
    assert not attempts  # Lapping must not sample ordinary defense/contact risk.
    assert not engine.simulator.event_manager.events
    assert [time for _, _, time in engine.crossings] == sorted(
        time for _, _, time in engine.crossings
    )


def test_unlapping_still_requires_an_ordinary_pass(monkeypatch):
    # B loses a lap, then becomes much faster. A has no duty to let B unlap.
    engine, run, _, attempts = setup(monkeypatch, {"A": 90, "B": lambda lap:
                                                   200 if lap == 1 else 10}, laps=5)
    results = run()
    assert ("B", "A") in attempts
    crossing = {(driver, lap): time for driver, lap, time in engine.crossings}
    assert crossing["B", 2] > crossing["A", 3]  # B's free readiness was 210.
    assert results[0].driver_id == "A"


@pytest.mark.parametrize("attacker_neutral,defender_neutral", [
    (True, False), (False, True), (True, True),
])
def test_either_neutralized_snapshot_prevents_blue_flag_pass(
    monkeypatch, attacker_neutral, defender_neutral,
):
    engine, run, _, attempts = setup(monkeypatch, {"B": 110, "A": 90})
    # Retain initialized pending laps without executing the crossing queue.
    monkeypatch.setattr(engine, "_enqueue", lambda *a: None)
    run()
    attacker = engine.pending["A"]
    defender = engine.pending["B"]
    attacker.lap = 2
    attacker.neutralized = attacker_neutral
    defender.neutralized = defender_neutral
    assert not engine._resolve_crossing("A", 90)
    assert engine.order == ["B", "A"]
    assert attacker.ready > defender.ready
    assert not attempts
    assert not engine.crossings


def test_pit_lane_car_does_not_block_or_need_a_blue_flag(monkeypatch):
    engine, run, _, attempts = setup(monkeypatch, {"A": 90, "B": 100}, laps=4)
    monkeypatch.setattr(engine.simulator, "_should_pit", lambda state, states, track, lap,
                        *a, **k: state.driver.id == "B" and lap == 2)
    monkeypatch.setattr(engine.simulator.lap_simulator, "calculate_pit_stop_time", lambda car: 200)
    results = run()
    assert engine.pit_exits == [("B", 2, 320)]
    assert results[0].total_time == 360
    assert results[1].total_time == 420 and results[1].laps_completed == 2
    assert not attempts
