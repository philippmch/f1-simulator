"""An announced flag survives retirement without making the successor run extra laps."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.events import EventType, RaceEvent
from f1sim.simulation.race import DriverStatus, RaceSimulator


def setup(monkeypatch, scheduled, catching=False):
    simulator = RaceSimulator(np.random.default_rng(12))
    drivers = [Driver(id=key, name=key, team_id=key) for key in "ABC"]
    cars = {key: Car(team_id=key, team_name=key) for key in "ABC"}
    track = Track(id="t", name="T", country="T", total_laps=scheduled, base_lap_time=1800)
    calls, policies, horizons = [], [], {}

    def physics(driver, car, track, tire, weather, lap, total_laps, **kwargs):
        calls.append((driver.id, lap, total_laps))
        if driver.id == "A":
            return 50 if lap == 5 else 1800
        if catching:
            return 2000 if driver.id == "B" else (6800 if lap == 1 else 500)
        return {"B": 3650, "C": 3800}[driver.id]

    def failure(driver, car, track, lap, weather):
        if driver.id == "A" and lap == 5:
            driver.dnf = True
            driver.dnf_reason = "Controlled failure"
            return RaceEvent(EventType.MECHANICAL_FAILURE, lap, [driver.id])
        return None

    def should(state, states, track, lap, *a, **k):
        policies.append((state.driver.id, lap))
        horizons[state.driver.id, lap] = track.total_laps
        return False

    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", physics)
    monkeypatch.setattr(simulator, "_should_pit", should)
    monkeypatch.setattr(simulator.event_manager, "_check_mechanical_failure", failure)
    monkeypatch.setattr(simulator.event_manager, "_check_random_incident", lambda *a: None)
    monkeypatch.setattr(simulator.event_manager, "process_lap", lambda *a, **k: [])
    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake",
                        lambda *a, **k: (True, False))
    monkeypatch.setattr(Weather, "evolve", lambda self, rng: self.model_copy(deep=True))
    engine = ChronologicalRace(simulator)

    def run():
        return engine.run(drivers, cars, track, Weather(), list("ABC"),
                          starting_tires={key: TireCompound.INTERMEDIATE for key in "ABC"})

    return engine, run, calls, policies, horizons


@pytest.mark.parametrize("scheduled,points", [(5, [13, 10, 8]), (10, [6, 4, 3])])
def test_lapped_successor_takes_announced_flag_and_preserves_other_distances(
    monkeypatch, scheduled, points,
):
    engine, run, calls, policies, _ = setup(monkeypatch, scheduled)

    results = run()
    assert [(r.driver_id, r.laps_completed, r.total_time, r.status) for r in results] == [
        ("B", 2, 7300, DriverStatus.FINISHED),
        ("A", 4, 7200, DriverStatus.DNF),
        ("C", 2, 7600, DriverStatus.FINISHED),
    ]
    assert engine.timeline.chequered_time == 7300 and engine.timeline.final_lap == 5
    assert engine.timeline.states["A"].retirement_time == 7250
    assert all(row.race_time_limited and row.classified for row in results)
    assert [row.points_awarded for row in results] == points
    assert [lap for driver, lap, _ in calls if driver == "B"] == [1, 2]
    assert [lap for driver, lap in policies if driver == "B"] == [1, 2]
    assert all(fuel_laps == scheduled for _, _, fuel_laps in calls)
    assert run() == results  # No announcement or finish state survives reuse.


def test_matching_surviving_leaders_old_distance_cannot_receive_flag(monkeypatch):
    engine, run, calls, _, horizons = setup(monkeypatch, 10, catching=True)
    results = run()
    assert [(row.driver_id, row.laps_completed, row.total_time) for row in results] == [
        ("B", 4, 8000), ("A", 4, 7200), ("C", 4, 8300),
    ]
    assert engine.timeline.winner_id == "B"
    assert horizons["C", 3] == 4  # At 7300, B's next crossing is 8000, not 10000.
    assert horizons["C", 4] == 4
    assert [lap for driver, lap, _ in calls if driver == "C"] == [1, 2, 3, 4]
    assert [row.points_awarded for row in results] == [13, 10, 8]
