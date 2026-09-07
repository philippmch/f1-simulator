"""Full-SC catch-up changes future pace while VSC and pit clocks stay distinct."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.events import EventType, RaceEvent
from f1sim.simulation.race import RaceSimulator


def run_controlled(monkeypatch, mode, pit=False):
    drivers = [Driver(id=name, name=name, team_id=name) for name in "AB"]
    cars = {name: Car(team_id=name, team_name=name) for name in "AB"}
    track = Track(id="t", name="T", country="T", total_laps=8, base_lap_time=90)
    simulator = RaceSimulator(np.random.default_rng(1))
    control = simulator.event_manager
    monkeypatch.setattr(simulator, "_should_pit", lambda state, states, track, lap, *a, **k:
                        pit and state.driver.id == "B" and lap == 3)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time",
                        lambda driver, *a, **k: 90 if driver.id == "A" else 100)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_pit_stop_time", lambda *a, **k: 3)
    monkeypatch.setattr(control, "_check_mechanical_failure", lambda *a, **k: None)
    monkeypatch.setattr(control, "_check_random_incident", lambda *a, **k: None)
    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake",
                        lambda *a, **k: (True, False))
    monkeypatch.setattr(Weather, "evolve", lambda self, rng: self.project_surface())

    def events(lap, *args, **kwargs):
        setattr(control, "safety_car_active" if mode == "sc" else "vsc_active", 2 <= lap < 6)
        kind = EventType.SAFETY_CAR if mode == "sc" else EventType.VIRTUAL_SAFETY_CAR
        return [RaceEvent(kind, lap, duration_laps=4)] if lap == 2 else []

    monkeypatch.setattr(control, "process_lap", events)
    engine = ChronologicalRace(simulator)
    results = engine.run(
        drivers, cars, track, Weather(), list("AB"),
        starting_tires={name: TireCompound.INTERMEDIATE for name in "AB"},
    )
    crossings = {(driver, lap): time for driver, lap, time in engine.crossings}
    return engine, results, crossings


def test_sc_forms_queue_without_rewinding_previous_crossings(monkeypatch):
    engine, _, crossings = run_controlled(monkeypatch, "sc")
    assert crossings["A", 2] == 180 and crossings["B", 2] == 200
    assert crossings["A", 3] == 306
    assert crossings["B", 3] == 307
    assert crossings["B", 4] - crossings["A", 4] == pytest.approx(1)
    times = [time for _, _, time in engine.crossings]
    assert times == sorted(times)


def test_vsc_does_not_apply_full_sc_gap_compression(monkeypatch):
    _, _, crossings = run_controlled(monkeypatch, "vsc")
    assert crossings["A", 3] == 288
    assert crossings["B", 3] == 320
    assert crossings["B", 3] - crossings["A", 3] == 32


def test_sc_catchup_preserves_absolute_pit_exit_and_free_pace_bound(monkeypatch):
    engine, results, crossings = run_controlled(monkeypatch, "sc", pit=True)
    assert engine.pit_exits == [("B", 3, 214)]  # 20 * .55 lane + 3 service.
    assert crossings["B", 2] == 200
    assert crossings["B", 3] == 314  # Cannot run faster than the 100-second free lap.
    assert next(row for row in results if row.driver_id == "B").pit_laps == [3]
