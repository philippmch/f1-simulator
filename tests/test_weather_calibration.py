"""Regression checks for normalized weather and race-level interruption priors."""

import numpy as np
import pytest

from f1sim.models import Track, Weather, WeatherCondition
from f1sim.simulation.events import EventManager, EventType


class ConstantRng:
    def __init__(self, value: float):
        self.value = value

    def random(self) -> float:
        return self.value


def storm(wetness: float = 0.9, intensity: float = 0.9) -> Weather:
    return Weather(
        condition=WeatherCondition.HEAVY_RAIN,
        rain_intensity=intensity,
        track_wetness=wetness,
        change_probability=0,
    )


def check(manager: EventManager, weather: Weather, lap: int = 1):
    return manager._check_red_flag_conditions(lap, 0, weather, total_laps=50)


@pytest.mark.parametrize("initial_wetness", [0.0, 0.45, 1.0])
def test_persistent_light_rain_approaches_damp_equilibrium(initial_wetness):
    weather = Weather(
        condition=WeatherCondition.LIGHT_RAIN,
        rain_intensity=0.35,
        track_wetness=initial_wetness,
        change_probability=0,
    )
    rng = np.random.default_rng(1)
    for _ in range(100):
        previous = weather.track_wetness
        weather = weather.evolve(rng)
        assert abs(weather.track_wetness - 0.35) <= abs(previous - 0.35)
    assert weather.track_wetness == pytest.approx(0.35, abs=1e-8)
    assert weather.is_wet()
    assert not weather.requires_wet_tires()


def test_zero_change_probability_cannot_transition_on_zero_roll():
    weather = Weather(track_wetness=0.04, change_probability=0)
    evolved = weather.evolve(ConstantRng(0))
    assert evolved.condition == WeatherCondition.DRY
    assert evolved.track_wetness == pytest.approx(0.01)
    assert evolved.evolve(ConstantRng(0)).track_wetness == 0
    assert weather.track_wetness == 0.04


def test_sustained_heavy_rain_can_reach_severe_threshold():
    weather = storm(wetness=0)
    for _ in range(20):
        weather = weather.evolve(ConstantRng(0.5))
    assert 0.8 < weather.track_wetness < 0.9


@pytest.mark.parametrize("roll, expected", [(0.25, 1), (0.75, 0)])
def test_unchanged_storm_is_decided_once_even_when_first_decision_declines(roll, expected):
    manager = EventManager(ConstantRng(roll))
    flags = 0
    for lap in range(1, 51):
        flags += check(manager, storm(), lap) is not None
        manager.end_red_flag()
    assert flags == expected


def test_episode_hysteresis_and_reset_allow_new_storm():
    manager = EventManager(ConstantRng(0.25))
    assert check(manager, storm()) is not None
    manager.end_red_flag()
    assert check(manager, storm(wetness=0.79, intensity=0.79)) is None
    assert check(manager, storm()) is None
    assert check(manager, storm(wetness=0.79, intensity=0.64)) is None
    assert check(manager, storm()) is not None
    manager.reset()
    assert check(manager, storm()) is not None


def test_flooded_surface_qualifies_without_intense_current_rain():
    manager = EventManager(ConstantRng(0.25))
    assert check(manager, storm(wetness=0.95, intensity=0.2)) is not None


def test_background_incident_remains_possible_in_decided_storm():
    manager = EventManager(ConstantRng(0.25))
    assert check(manager, storm()) is not None
    manager.end_red_flag()
    manager.rng = ConstantRng(0)
    event = check(manager, storm(), 2)
    assert event is not None
    assert "Major incident" in event.description


def test_forced_flags_preserve_restart_and_survive_reset_in_same_storm():
    manager = EventManager(ConstantRng(0.25))
    track = Track(id="test", name="Test", country="Test", total_laps=50, base_lap_time=90)
    manager.set_forced_red_flag([1, 2])
    for _ in range(2):
        manager.reset()
        for lap in [1, 2]:
            events = manager.process_lap(lap, [], {}, track, storm())
            assert len(events) == 1
            assert events[0].event_type == EventType.RED_FLAG
            assert "Manual trigger" in events[0].description
            manager.end_red_flag()
            assert manager.is_restart_lap(lap + 1)
            assert not manager.is_overtake_mode_allowed(lap + 1)


@pytest.mark.parametrize("laps", [20, 50, 80])
def test_background_prior_is_stable_across_race_distances(laps):
    weather = Weather(change_probability=0)
    races_with_flags = 0
    for seed in range(1000):
        manager = EventManager(np.random.default_rng(seed))
        for lap in range(1, laps + 1):
            if manager._check_red_flag_conditions(lap, 0, weather, total_laps=laps):
                races_with_flags += 1
                break
    assert 0.07 <= races_with_flags / 1000 <= 0.13


def test_minor_contact_count_does_not_change_red_flag_probability():
    weather = Weather(change_probability=0)
    for seed in range(100):
        outcomes = []
        for contacts in [0, 1, 2, 3, 20]:
            manager = EventManager(np.random.default_rng(seed))
            outcomes.append([
                manager._check_red_flag_conditions(lap, contacts, weather, total_laps=50)
                is not None
                for lap in range(1, 51)
            ])
        assert all(outcome == outcomes[0] for outcome in outcomes)


def test_storm_episode_prior_matches_fixed_seed_sanity_target():
    flags = sum(
        event.description == "Red flag: Severe weather"
        for seed in range(2000)
        if (event := check(EventManager(np.random.default_rng(seed)), storm())) is not None
    )
    assert 0.46 <= flags / 2000 <= 0.54

def test_weather_episode_can_clear_while_safety_car_is_active():
    manager = EventManager(ConstantRng(0.25))
    assert check(manager, storm()) is not None
    manager.end_red_flag()
    manager.safety_car_active = True
    manager.safety_car_laps_remaining = 3
    track = Track(id="test", name="Test", country="Test", total_laps=50, base_lap_time=90)
    manager.process_lap(2, [], {}, track, storm(wetness=0.7, intensity=0.6))
    assert check(manager, storm(), 3) is not None
