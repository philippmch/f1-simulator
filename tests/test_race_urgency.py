"""Race decisions must use the distance left, including the lap being simulated."""

import numpy as np

from f1sim.models import Car, Driver, Track, Weather
from f1sim.simulation.overtaking import OvertakingModel
from f1sim.simulation.race import RaceSimulator


def test_overtake_decisions_receive_remaining_laps(monkeypatch):
    simulator = RaceSimulator(rng=np.random.default_rng(7))
    drivers = [Driver(id=name, name=name, team_id=name) for name in ("A", "B")]
    cars = {driver.team_id: Car(team_id=driver.team_id, team_name=driver.name)
            for driver in drivers}
    track = Track(id="test", name="Test", country="Test", total_laps=4, base_lap_time=90)
    remaining = []

    def should_attempt(attacker, defender, gap, remaining_laps, position):
        remaining.append(remaining_laps)
        return False

    monkeypatch.setattr(simulator.overtaking_model, "should_attempt_overtake", should_attempt)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", lambda **kwargs: 90.0)
    monkeypatch.setattr(simulator.event_manager, "process_lap", lambda **kwargs: [])
    monkeypatch.setattr(simulator, "_should_pit", lambda *args, **kwargs: False)
    simulator.simulate_race(drivers, cars, track, Weather(change_probability=0), ["A", "B"])

    assert remaining == [4, 3, 2, 1]


def test_late_race_urgency_widens_attempt_window():
    model = OvertakingModel()
    attacker = Driver(id="A", name="A", team_id="A", overtaking_skill=0.5)
    defender = Driver(id="B", name="B", team_id="B")
    assert not model.should_attempt_overtake(attacker, defender, 1.1, 50, 5)
    assert model.should_attempt_overtake(attacker, defender, 1.1, 2, 5)
    assert not model.should_attempt_overtake(attacker, defender, 1.6, 1, 5)


def test_attacking_skill_and_points_battle_widen_attempt_window():
    model = OvertakingModel()
    cautious = Driver(id="A", name="A", team_id="A", overtaking_skill=0.2)
    confident = cautious.model_copy(update={"overtaking_skill": 0.9})
    defender = Driver(id="B", name="B", team_id="B")
    assert not model.should_attempt_overtake(cautious, defender, 0.95, 50, 15)
    assert model.should_attempt_overtake(confident, defender, 0.95, 50, 15)
    assert not model.should_attempt_overtake(cautious, defender, 0.83, 50, 15)
    assert model.should_attempt_overtake(cautious, defender, 0.83, 50, 5)
