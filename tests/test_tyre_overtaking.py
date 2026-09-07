"""Tyre condition changes passing chances without bypassing passing gates."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.overtaking import OvertakingModel
from f1sim.simulation.race import DriverRaceState, RaceSimulator


@pytest.fixture
def battle():
    driver = Driver(id="A", name="A", team_id="team")
    car = Car(team_id="team", team_name="Team")
    track = Track(id="test", name="Test", country="Test", total_laps=50,
                  base_lap_time=90, overtake_difficulty=0.4)
    return driver, car, driver.model_copy(update={"id": "B"}), car, track


def probability(battle, advantage=0):
    return OvertakingModel()._calculate_probability(
        *battle, 0.5, False, False, tire_pace_advantage_seconds=advantage
    )


def tyre_advantage(battle, attacker_compound, attacker_age, defender_compound, defender_age):
    attacker, attacker_car, defender, defender_car, track = battle
    return LapSimulator.tire_pace_contribution(
        defender, defender_car, track, TIRE_COMPOUNDS[defender_compound], defender_age
    ) - LapSimulator.tire_pace_contribution(
        attacker, attacker_car, track, TIRE_COMPOUNDS[attacker_compound], attacker_age
    )


def test_equal_wear_preserves_default_and_positional_api(battle):
    advantage = tyre_advantage(battle, TireCompound.MEDIUM, 8, TireCompound.MEDIUM, 8)
    legacy = OvertakingModel()._calculate_probability(*battle, 0.5, False, False, False)
    assert advantage == 0
    assert probability(battle, advantage) == legacy


def test_direct_wet_probability_ignores_dry_tyre_advantage(battle):
    model = OvertakingModel(np.random.default_rng(23))
    baseline = model._calculate_probability(*battle, 0.5, False, True)
    assert model._calculate_probability(
        *battle, 0.5, False, True, tire_pace_advantage_seconds=100
    ) == baseline
    expected_rng = np.random.default_rng(23)
    expected_model = OvertakingModel(expected_rng)
    assert model.attempt_overtake(
        *battle, 0.5, is_wet=True, tire_pace_advantage_seconds=100
    ) == expected_model.attempt_overtake(*battle, 0.5, is_wet=True)
    assert model.rng.random() == expected_rng.random()


@pytest.mark.parametrize("compound", [TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD])
def test_fresh_set_advantage_reverses_when_wear_is_swapped(battle, compound):
    advantage = tyre_advantage(battle, compound, 1, compound, 30)
    reverse = tyre_advantage(battle, compound, 30, compound, 1)
    assert advantage > 0
    assert reverse == pytest.approx(-advantage)
    assert probability(battle, reverse) < probability(battle) < probability(battle, advantage)


def test_soft_advantage_disappears_after_wear_crossover(battle):
    fresh = tyre_advantage(battle, TireCompound.SOFT, 1, TireCompound.HARD, 1)
    worn = tyre_advantage(battle, TireCompound.SOFT, 30, TireCompound.HARD, 1)
    assert fresh > 0 > worn
    assert probability(battle, fresh) > probability(battle) > probability(battle, worn)


def test_extreme_advantages_are_bounded_and_circuit_difficulty_still_matters(battle):
    scale = battle[-1].base_lap_time * 0.03
    assert probability(battle, 1e9) == probability(battle, scale)
    assert probability(battle, -1e9) == probability(battle, -scale)
    assert 0 < probability(battle, -1e9) < probability(battle, 1e9) < 0.9
    difficult = (*battle[:-1], battle[-1].model_copy(update={"overtake_difficulty": 0.85}))
    assert probability(difficult, 1e9) < probability(battle, 1e9)


def test_fixed_roll_between_chances_flips_pass_with_one_draw(battle):
    class FixedRoll:
        draws = 0

        def random(self):
            self.draws += 1
            return (probability(battle, -1) + probability(battle, 1)) / 2

    rng = FixedRoll()
    model = OvertakingModel(rng)
    assert model.attempt_overtake(*battle, 0.5, tire_pace_advantage_seconds=1)[0]
    assert not model.attempt_overtake(*battle, 0.5, tire_pace_advantage_seconds=-1)[0]
    assert rng.draws == 2
    assert model.attempt_overtake(*battle, 1.6, tire_pace_advantage_seconds=1e9) == (False, False)
    assert rng.draws == 2


@pytest.mark.parametrize("wetness,rain,compound,enabled", [
    (0, 0, TireCompound.SOFT, True),
    (0.079, 0.149, TireCompound.SOFT, True),
    (0.08, 0, TireCompound.SOFT, False),
    (0, 0.15, TireCompound.SOFT, False),
    (0, 0, TireCompound.INTERMEDIATE, False),
    (0, 0, TireCompound.WET, False),
])
def test_race_forwards_current_set_condition_only_in_clear_dry(
    battle, monkeypatch, wetness, rain, compound, enabled,
):
    attacker, car, defender, _, track = battle
    attacker_car = car.model_copy(update={"tire_degradation_factor": 1.4})
    attacker = attacker.model_copy(update={"tire_management": 0.6, "current_tire_laps": 2})
    states = [
        DriverRaceState(defender, car, 1, total_time=90,
                        current_tire=TIRE_COMPOUNDS[TireCompound.HARD], tire_laps=7),
        DriverRaceState(attacker, attacker_car, 2, total_time=90.5,
                        current_tire=TIRE_COMPOUNDS[compound], tire_laps=24),
    ]
    simulator = RaceSimulator(np.random.default_rng(8))
    captured = []
    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake",
                        lambda **kwargs: (captured.append(kwargs) or False, False))
    before = repr(simulator.rng.bit_generator.state)
    simulator._process_overtakes(states, track, Weather(track_wetness=wetness, rain_intensity=rain),
                                lap=20, overtake_mode_allowed=False)
    expected = tyre_advantage((attacker, attacker_car, defender, car, track),
                             compound, 24, TireCompound.HARD, 7) if enabled else 0
    assert captured[0]["tire_pace_advantage_seconds"] == pytest.approx(expected)
    assert repr(simulator.rng.bit_generator.state) == before
    assert states[1].tire_laps == 24
    assert states[1].driver.current_tire_laps == 2
