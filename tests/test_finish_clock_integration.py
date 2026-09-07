"""The finish clock follows race leadership across retirements and repeated runs."""

import numpy as np

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.events import EventType, RaceEvent
from f1sim.simulation.race import RaceSimulator


def test_leader_changes_do_not_reset_the_time_limit_clock(monkeypatch):
    drivers = [Driver(id=name, name=name, team_id=name) for name in "ABC"]
    cars = {d.id: Car(team_id=d.id, team_name=d.id) for d in drivers}
    track = Track(id="t", name="T", country="T", total_laps=10, base_lap_time=90)
    simulator = RaceSimulator(np.random.default_rng(42))
    monkeypatch.setattr(simulator, "_should_pit", lambda *args, **kwargs: False)
    monkeypatch.setattr(simulator, "_process_overtakes", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        simulator.lap_simulator, "calculate_lap_time",
        lambda **kwargs: 1800 + 30 * "ABC".index(kwargs["driver"].id),
    )
    monkeypatch.setattr(Weather, "evolve", lambda self, rng: self.project_surface())

    def events(lap, drivers, **kwargs):
        victim = {3: "A", 4: "B"}.get(lap)
        if victim is None:
            return []
        next(driver for driver in drivers if driver.id == victim).dnf = True
        return [RaceEvent(EventType.MECHANICAL_FAILURE, lap, [victim])]

    monkeypatch.setattr(simulator.event_manager, "process_lap", events)
    # A fresh race gets a fresh clock even when the simulator and drivers are reused.
    for _ in range(2):
        results = simulator.simulate_race(
            drivers, cars, track, Weather(), list("ABC"),
            starting_tires={d.id: TireCompound.INTERMEDIATE for d in drivers},
        )
        winner = results[0]
        assert winner.driver_id == "C"
        assert winner.laps_completed == 5
        assert winner.total_time == 9300
        assert winner.race_time_limited and winner.classified
        assert winner.points_awarded == 19
        assert {r.driver_id: r.laps_completed for r in results} == {"A": 2, "B": 3, "C": 5}
