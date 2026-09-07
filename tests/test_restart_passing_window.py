"""The wider restart attempt window also permits a successful maneuver."""

import pytest

from f1sim.models import Car, Driver, Track, Weather
from f1sim.simulation.overtaking import OvertakingModel
from f1sim.simulation.race import DriverRaceState, RaceSimulator


class FixedRoll:
    def __init__(self, value):
        self.value = value
        self.draws = 0

    def random(self):
        self.draws += 1
        return self.value


@pytest.fixture
def battle():
    attacker = Driver(id="A", name="A", team_id="a")
    defender = Driver(id="B", name="B", team_id="b")
    car_a = Car(team_id="a", team_name="A")
    car_b = Car(team_id="b", team_name="B")
    track = Track(id="test", name="Test", country="Test", total_laps=50,
                  base_lap_time=90, overtake_difficulty=0.5)
    return attacker, car_a, defender, car_b, track


@pytest.mark.parametrize("wet", [False, True])
def test_restart_probability_decays_across_entire_attempt_window(battle, wet):
    model = OvertakingModel()
    chances = [model._calculate_probability(*battle, gap, False, wet, restart_boost=True)
               for gap in (0.5, 1.5, 1.75, 2.0)]
    assert chances[0] > chances[1] > chances[2] > chances[3] == 0
    assert model._calculate_probability(*battle, 1.75, False, wet) == 0


def test_extended_restart_window_can_pass_and_keeps_outer_gate(battle):
    rng = FixedRoll(0.02)
    model = OvertakingModel(rng)
    assert model.attempt_overtake(*battle, 1.75) == (False, False)
    assert rng.draws == 0
    assert model.attempt_overtake(*battle, 1.75, restart_boost=True) == (True, False)
    assert rng.draws == 1
    assert model.attempt_overtake(
        *battle, 2.01, overtake_mode_active=True, restart_boost=True,
        tire_pace_advantage_seconds=100,
    ) == (False, False)
    assert rng.draws == 1


def test_race_restart_pass_works_outside_normal_window(battle):
    attacker, car_a, defender, car_b, track = battle
    states = [DriverRaceState(defender, car_b, 1, total_time=100),
              DriverRaceState(attacker, car_a, 2, total_time=101.75)]
    simulator = RaceSimulator()
    rng = FixedRoll(0.02)
    simulator.overtaking_model.rng = rng
    incidents = simulator._process_overtakes(
        states, track, Weather(), restart_lap=True, lap=20, overtake_mode_allowed=False,
    )
    assert incidents == 0
    assert states[1].position == 1
    assert states[0].position == 2
    assert rng.draws == 1
