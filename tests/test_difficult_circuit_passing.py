"""Circuit difficulty reduces passing odds without disabling the race model."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.overtaking import OvertakingModel
from f1sim.simulation.race import RaceSimulator


class FixedRoll:
    def __init__(self, value=0.001):
        self.value = value
        self.draws = 0

    def random(self):
        self.draws += 1
        return self.value


def run_battle(monkeypatch, difficulty, gap=0.5, neutralization=None, roll=0.001, observed=None):
    simulator = RaceSimulator(np.random.default_rng(5))
    drivers = [Driver(id=name, name=name, team_id=name) for name in ("A", "B")]
    cars = {d.id: Car(team_id=d.id, team_name=d.id) for d in drivers}
    track = Track(id="test", name="Test", country="Test", total_laps=1,
                  base_lap_time=90, overtake_difficulty=difficulty)
    rng = FixedRoll(roll)
    simulator.overtaking_model.rng = rng
    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time",
                        lambda **kw: 90 + (gap if kw["driver"].id == "B" else 0))
    def process_lap(**kwargs):
        if observed is not None:
            observed["incidents"] = kwargs["incidents_this_lap"]
        return []

    monkeypatch.setattr(simulator.event_manager, "process_lap", process_lap)
    if neutralization:
        reset = simulator.event_manager.reset

        def neutralized_reset():
            reset()
            setattr(simulator.event_manager, neutralization, True)

        monkeypatch.setattr(simulator.event_manager, "reset", neutralized_reset)
    result = simulator.simulate_race(
        drivers, cars, track, Weather(change_probability=0), ["A", "B"],
        starting_tires={d.id: TireCompound.MEDIUM for d in drivers},
    )
    if observed is not None:
        observed["events"] = list(simulator.event_manager.events)
        observed["times"] = {r.driver_id: r.total_time for r in result}
    return [r.driver_id for r in result], rng.draws


@pytest.mark.parametrize("difficulty", [0.8999, 0.9, 0.95])
def test_green_race_honors_positive_maneuver_probability_above_old_cutoff(monkeypatch, difficulty):
    order, draws = run_battle(monkeypatch, difficulty)
    assert order == ["B", "A"]
    assert draws == 1


@pytest.mark.parametrize("neutralization", ["safety_car_active", "vsc_active", "red_flag_active"])
def test_difficult_circuit_still_prohibits_neutralized_passing(monkeypatch, neutralization):
    order, draws = run_battle(monkeypatch, 0.95, neutralization=neutralization)
    assert order == ["A", "B"]
    assert draws == 0


def test_large_gap_still_prevents_attempt_and_draw(monkeypatch):
    order, draws = run_battle(monkeypatch, 0.95, gap=2)
    assert order == ["A", "B"]
    assert draws == 0


def test_difficulty_still_reduces_success_at_the_same_roll(monkeypatch):
    assert run_battle(monkeypatch, 0.4, roll=0.1) == (["B", "A"], 1)
    assert run_battle(monkeypatch, 0.95, roll=0.1) == (["A", "B"], 1)


@pytest.mark.parametrize("difficulty", [0.95, 1.0])
def test_newly_enabled_contact_reaches_ledger_and_race_control_once(monkeypatch, difficulty):
    observed = {}
    _, draws = run_battle(monkeypatch, difficulty, roll=0.04, observed=observed)
    assert draws == 1
    assert observed["incidents"] == 1
    event, = observed["events"]
    assert event.event_type.value == "collision"
    assert event.lap == 1
    assert set(event.drivers_involved) == {"A", "B"}
    assert event.time_loss_seconds == 0  # Losses were already applied by the maneuver.
    for driver_id, running_time in (("A", 90), ("B", 90.5)):
        assert observed["times"][driver_id] == pytest.approx(
            running_time + event.applied_time_losses[driver_id]
        )


def test_maximum_difficulty_has_no_unboosted_pass(monkeypatch):
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="test", name="Test", country="Test", total_laps=1,
                  base_lap_time=90, overtake_difficulty=1)
    assert OvertakingModel()._calculate_probability(
        driver, car, driver, car, track, 0.5, False, False,
        tire_pace_advantage_seconds=100,
    ) == 0
    assert run_battle(monkeypatch, 1, roll=0.9) == (["A", "B"], 1)
