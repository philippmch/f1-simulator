"""Sequential battles must follow the physical queue after earlier passes."""

import pytest

from f1sim.models import Car, Driver, Track, Weather
from f1sim.simulation.race import DriverRaceState, RaceSimulator


@pytest.mark.parametrize("first_success", [False, True])
def test_next_attacker_targets_current_neighbour_after_pass(monkeypatch, first_success):
    simulator = RaceSimulator()
    states = [DriverRaceState(
        Driver(id=name, name=name, team_id=name), Car(team_id=name, team_name=name),
        position, total_time=100 + position * 0.1,
    ) for position, name in enumerate(("A", "B", "C", "D"), 1)]
    track = Track(id="test", name="Test", country="Test", total_laps=50, base_lap_time=90)
    battles = []

    def attempt(**kwargs):
        attacker, defender = kwargs["attacker"].id, kwargs["defender"].id
        current = {state.driver.id: state.position for state in states}
        battles.append((attacker, defender, current[attacker] - current[defender]))
        return (first_success if attacker == "B" else True), False

    monkeypatch.setattr(simulator.overtaking_model, "should_attempt_overtake", lambda *a: True)
    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake", attempt)
    before = {state.driver.id: state.total_time for state in states}
    assert simulator._process_overtakes(states, track, Weather(), lap=20) == 0
    expected_pairs = [("B", "A"), ("C", "A"), ("D", "A")] if first_success else [
        ("B", "A"), ("C", "B"), ("D", "B"),
    ]
    assert [(attacker, defender) for attacker, defender, _ in battles] == expected_pairs
    assert all(distance == 1 for _, _, distance in battles)
    assert [state.driver.id for state in sorted(states, key=lambda s: s.position)] == (
        ["B", "C", "D", "A"] if first_success else ["A", "C", "D", "B"]
    )
    assert {state.driver.id: state.total_time for state in states} == before
