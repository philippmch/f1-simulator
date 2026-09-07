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


def test_wet_tyre_advantage_changes_fixed_roll_with_one_draw(battle):
    model = OvertakingModel()
    baseline = model._calculate_probability(*battle, 0.5, False, True)
    advantage = model._calculate_probability(
        *battle, 0.5, False, True, tire_pace_advantage_seconds=1
    )
    assert advantage > baseline

    class FixedRoll:
        draws = 0

        def random(self):
            self.draws += 1
            return (baseline + advantage) / 2

    rng = FixedRoll()
    model = OvertakingModel(rng)
    assert not model.attempt_overtake(*battle, 0.5, is_wet=True)[0]
    assert model.attempt_overtake(
        *battle, 0.5, is_wet=True, tire_pace_advantage_seconds=1
    )[0]
    assert rng.draws == 2


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


@pytest.mark.parametrize("wetness,rain,compound", [
    (0, 0, TireCompound.SOFT),
    (0.079, 0.149, TireCompound.SOFT),
    (0.08, 0, TireCompound.SOFT),
    (0, 0.15, TireCompound.SOFT),
    (0, 0, TireCompound.INTERMEDIATE),
    (0, 0, TireCompound.WET),
])
def test_race_forwards_current_set_condition_in_all_weather(
    battle, monkeypatch, wetness, rain, compound,
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
    weather = Weather(track_wetness=wetness, rain_intensity=rain)
    expected = LapSimulator.tire_weather_pace_contribution(
        defender, car, track, TIRE_COMPOUNDS[TireCompound.HARD], 7, weather
    ) - LapSimulator.tire_weather_pace_contribution(
        attacker, attacker_car, track, TIRE_COMPOUNDS[compound], 24, weather
    )
    assert captured[0]["tire_pace_advantage_seconds"] == pytest.approx(expected)
    assert repr(simulator.rng.bit_generator.state) == before
    assert states[1].tire_laps == 24
    assert states[1].driver.current_tire_laps == 2


@pytest.mark.parametrize("wetness,attacking,defending", [
    (0.21, TireCompound.INTERMEDIATE, TireCompound.SOFT),
    (0.6, TireCompound.INTERMEDIATE, TireCompound.HARD),
    (0.9, TireCompound.WET, TireCompound.INTERMEDIATE),
    (0, TireCompound.SOFT, TireCompound.WET),
])
def test_weather_suitable_tyre_advantage_matches_actual_lap_swap(
    battle, wetness, attacking, defending,
):
    driver, car, _, _, track = battle
    weather = Weather(track_wetness=wetness, rain_intensity=wetness)
    simulator = LapSimulator()

    def contribution(compound):
        return simulator.tire_weather_pace_contribution(
            driver, car, track, TIRE_COMPOUNDS[compound], 4, weather
        )

    def actual(compound):
        return simulator.calculate_lap_time(
            driver.model_copy(update={"current_tire_laps": 4}), car, track,
            TIRE_COMPOUNDS[compound], weather, 10, track.total_laps, sample_variation=False,
        )

    advantage = contribution(defending) - contribution(attacking)
    assert advantage > 0
    assert contribution(attacking) - contribution(defending) == -advantage
    assert actual(defending) - actual(attacking) == pytest.approx(advantage)


@pytest.mark.parametrize("compound", list(TireCompound))
def test_shared_weather_term_equal_sets_zero_and_dry_slick_baseline(battle, compound):
    driver, car, _, _, track = battle
    tire = TIRE_COMPOUNDS[compound]
    weather = Weather(track_wetness=0.6)
    contribution = LapSimulator.tire_weather_pace_contribution
    assert contribution(driver, car, track, tire, 8, weather) == contribution(
        driver.model_copy(), car.model_copy(), track, tire, 8, weather
    )
    assert contribution(driver, car, track, tire, 25, weather) > contribution(
        driver, car, track, tire, 1, weather
    )
    if compound in (TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD):
        assert contribution(driver, car, track, tire, 8, Weather()) == (
            LapSimulator.tire_pace_contribution(driver, car, track, tire, 8)
        )


def test_weather_multiplier_uses_each_driver_and_car_without_base_pace(battle):
    driver, car, _, _, track = battle
    weather = Weather(track_wetness=0.7, rain_intensity=0.8)
    driver = driver.model_copy(update={"wet_skill_modifier": 0.8})
    car = car.model_copy(update={"wet_performance": 0.6})
    expected = weather.lap_time_multiplier() * (1 + (1 - 0.8) * 0.02)
    expected *= 1 + (1 - 0.6) * 0.7 * 0.06
    assert LapSimulator.weather_pace_multiplier(driver, car, weather) == pytest.approx(expected)
    tire = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE]
    contribution = LapSimulator.tire_weather_pace_contribution
    assert contribution(driver, car, track, tire, 8, weather) == pytest.approx(
        LapSimulator.tire_pace_contribution(driver, car, track, tire, 8) * expected
    )
    assert contribution(driver, car, track, tire, 8, weather) == contribution(
        driver.model_copy(update={"skill_rating": 0.5}),
        car.model_copy(update={"base_pace": 0.5}), track, tire, 8, weather,
    )
