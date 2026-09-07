"""Independent constant-pace oracle for chronological lapped finish boundaries."""

from math import ceil

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.race import RaceSimulator


@pytest.mark.parametrize("paces", [
    (90, 110), (90, 110, 150, 200), (90, 90, 90, 90), (90, 100, 120, 150),
])
@pytest.mark.parametrize("distance", [10, 20])
def test_free_running_crossings_match_analytic_finish(paces, distance, monkeypatch):
    drivers = [Driver(id=str(i), name=str(i), team_id=str(i)) for i in range(len(paces))]
    cars = {d.id: Car(team_id=d.id, team_name=d.id) for d in drivers}
    track = Track(id="t", name="T", country="T", total_laps=distance, base_lap_time=90)
    simulator = RaceSimulator(np.random.default_rng(42))
    monkeypatch.setattr(simulator, "_should_pit", lambda *args, **kwargs: False)
    control = simulator.event_manager
    monkeypatch.setattr(control, "_check_mechanical_failure", lambda *args, **kwargs: None)
    monkeypatch.setattr(control, "_check_random_incident", lambda *args, **kwargs: None)
    monkeypatch.setattr(control, "_deploy_safety_measure", lambda *args, **kwargs: None)
    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake",
                        lambda *args, **kwargs: (True, False))
    calls = []

    def lap(driver, car, track, tire, weather, lap_number, total_laps, **kwargs):
        calls.append((driver.id, lap_number, total_laps, driver.current_tire_laps))
        return paces[int(driver.id)]

    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", lap)
    engine = ChronologicalRace(simulator)
    rows = engine.run(
        drivers, cars, track, Weather(change_probability=0), [d.id for d in drivers],
        starting_tires={d.id: TireCompound.INTERMEDIATE for d in drivers},
    )
    flag_time = distance * paces[0]
    for row in rows:
        pace = paces[int(row.driver_id)]
        expected_laps = min(distance, ceil(flag_time / pace))
        assert row.laps_completed == expected_laps
        assert row.total_time == expected_laps * pace
        own_calls = [entry for entry in calls if entry[0] == row.driver_id]
        assert [entry[1] for entry in own_calls] == list(range(1, expected_laps + 1))
        assert [entry[2] for entry in own_calls] == [distance] * expected_laps
        assert [entry[3] for entry in own_calls] == list(range(expected_laps))
