"""Exactness and dispatch guards for prepared deterministic lap physics."""

from copy import deepcopy

import numpy as np
import pytest

from f1sim.models import ActiveAeroZone, Car, Driver, Sector, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS, TireCompound
from f1sim.simulation.lap import LapSimulator, minimum_lap_time


def _models():
    driver = Driver(
        id="driver",
        name="Driver",
        team_id="team",
        skill_rating=.77,
        consistency=.88,
        wet_skill_modifier=1.13,
        tire_management=.69,
    )
    car = Car(
        team_id="team",
        team_name="Team",
        base_pace=.73,
        downforce_level=.64,
        straight_line_speed=.91,
        tire_degradation_factor=1.21,
        wet_performance=.62,
    )
    track = Track(
        id="track",
        name="Track",
        country="Test",
        total_laps=15,
        base_lap_time=87.35,
        tire_stress=.83,
        sectors=[
            Sector(number=1, base_time=30.0, is_high_speed=True,
                   overtake_opportunity=.8),
            Sector(number=2, base_time=28.0, is_high_speed=False,
                   overtake_opportunity=.35),
            Sector(number=3, base_time=29.35, is_high_speed=True,
                   overtake_opportunity=.6),
        ],
        active_aero_zones=[
            ActiveAeroZone(zone_id=1, sector=1, time_gain=.31),
            ActiveAeroZone(zone_id=2, sector=3, time_gain=.18),
        ],
    )
    return driver, car, track


@pytest.mark.parametrize(
    "weather",
    [
        Weather(track_wetness=0.0, rain_intensity=0.0),
        Weather(track_wetness=.16, rain_intensity=.12),
        Weather(track_wetness=.34, rain_intensity=.3),
        Weather(track_wetness=.58, rain_intensity=.72),
        Weather(track_wetness=.91, rain_intensity=.95),
    ],
)
@pytest.mark.parametrize("gap", [None, .15, 1.25, 3.0])
@pytest.mark.parametrize("active_aero_enabled", [False, True])
@pytest.mark.parametrize("total_laps", [7, 20, 53])
def test_prepared_lap_is_bit_exact_across_native_inputs(
    weather, gap, active_aero_enabled, total_laps,
):
    """Prepared physics must select exactly the same deterministic lap value."""
    driver, car, track = _models()
    simulator = LapSimulator(np.random.default_rng(19))
    prepared = simulator.prepare_deterministic_lap_time(
        driver, car, track, total_laps,
    )
    assert prepared is not None

    for compound in TireCompound:
        tire = TIRE_COMPOUNDS[compound]
        for age in (0, 1, 19, 20, 31, 57):
            lap_number = 1 if age % 2 else max(1, total_laps - 1)
            driver.current_tire_laps = age
            expected = simulator.calculate_lap_time(
                driver, car, track, tire, weather, lap_number, total_laps,
                gap_to_car_ahead=gap,
                active_aero_enabled=active_aero_enabled,
                sample_variation=False,
            )

            # The prepared path receives age explicitly and must not alter the
            # mutable race-state field used by the public evaluator.
            driver.current_tire_laps = 101
            actual = prepared(
                tire, weather, lap_number, age, gap, active_aero_enabled,
            )
            assert actual == expected
            assert driver.current_tire_laps == 101


def test_prepared_path_does_not_draw_randomness():
    driver, car, track = _models()
    simulator = LapSimulator(np.random.default_rng(41))
    prepared = simulator.prepare_deterministic_lap_time(driver, car, track, 20)
    assert prepared is not None
    before = deepcopy(simulator.rng.bit_generator.state)

    prepared(
        TIRE_COMPOUNDS[TireCompound.MEDIUM],
        Weather(track_wetness=.27, rain_intensity=.2),
        4,
        12,
        .5,
        True,
    )

    assert simulator.rng.bit_generator.state == before


def test_prepared_path_preserves_an_active_lap_time_floor():
    driver, car, track = _models()
    driver.skill_rating = 1.0
    car.base_pace = car.straight_line_speed = 1.0
    track.base_lap_time = 200
    track.active_aero_zones = [
        ActiveAeroZone(zone_id=i + 1, sector=1, time_gain=1) for i in range(16)
    ]
    simulator = LapSimulator(np.random.default_rng(41))
    prepared = simulator.prepare_deterministic_lap_time(driver, car, track, 20)
    tire = TIRE_COMPOUNDS[TireCompound.SOFT]
    weather = Weather()
    expected = simulator.calculate_lap_time(
        driver, car, track, tire, weather, 20, 20, sample_variation=False,
    )
    assert expected == minimum_lap_time(track)
    assert prepared(tire, weather, 20, 0) == expected


@pytest.mark.parametrize("target", ["instance", "class"])
@pytest.mark.parametrize("method", [
    "weather_pace_multiplier", "tire_pace_contribution",
    "traffic_pace_contribution", "_track_car_delta",
])
def test_custom_physics_helpers_disable_preparation(monkeypatch, target, method):
    driver, car, track = _models()
    simulator = LapSimulator(np.random.default_rng(3))
    owner = simulator if target == "instance" else LapSimulator
    monkeypatch.setattr(owner, method, lambda *args: 2.0)

    assert simulator.prepare_deterministic_lap_time(driver, car, track, 20) is None


class _CustomLapSimulator(LapSimulator):
    def calculate_lap_time(self, *args, **kwargs):
        return super().calculate_lap_time(*args, **kwargs) + 0.25


class _CustomWeather(Weather):
    def lap_time_multiplier(self) -> float:
        return super().lap_time_multiplier() + .01


def test_custom_simulator_subclass_disables_preparation():
    driver, car, track = _models()
    simulator = _CustomLapSimulator(np.random.default_rng(2))

    assert simulator.prepare_deterministic_lap_time(driver, car, track, 20) is None


def test_monkeypatched_public_method_disables_preparation(monkeypatch):
    driver, car, track = _models()
    original = LapSimulator.calculate_lap_time

    def custom(self, *args, **kwargs):
        return original(self, *args, **kwargs) + .125

    monkeypatch.setattr(LapSimulator, "calculate_lap_time", custom)
    simulator = LapSimulator(np.random.default_rng(3))

    assert simulator.prepare_deterministic_lap_time(driver, car, track, 20) is None


def test_custom_weather_uses_public_method_fallback():
    driver, car, track = _models()
    simulator = LapSimulator(np.random.default_rng(5))
    prepared = simulator.prepare_deterministic_lap_time(driver, car, track, 20)
    assert prepared is not None
    weather = _CustomWeather(track_wetness=.2, rain_intensity=.1)
    tire = TIRE_COMPOUNDS[TireCompound.SOFT]

    driver.current_tire_laps = 9
    expected = simulator.calculate_lap_time(
        driver, car, track, tire, weather, 5, 20,
        gap_to_car_ahead=1.0, active_aero_enabled=False, sample_variation=False,
    )
    driver.current_tire_laps = 37
    actual = prepared(tire, weather, 5, 9, 1.0, False)

    assert actual == expected
    assert driver.current_tire_laps == 9
