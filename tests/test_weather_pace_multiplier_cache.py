"""Regression coverage for the scalar weather pace multiplier cache."""

import numpy as np
import pytest
from pydantic import PrivateAttr

from f1sim.models import Car, Driver, Weather
from f1sim.simulation.lap import (
    LapSimulator,
    _weather_pace_multiplier_from_values,
)


def _legacy_weather_pace_multiplier(
    weather_multiplier: float,
    wet_severity: float,
    wet_skill_modifier: float,
    wet_performance: float,
) -> float:
    """Keep the pre-cache scalar arithmetic as an independent oracle."""
    wet_adjustment = 1.0 + (1.0 - wet_skill_modifier) * 0.02 * wet_severity
    weather_multiplier *= wet_adjustment
    if wet_severity > 0.0:
        car_wet_penalty = (1.0 - wet_performance) * wet_severity * 0.06
        weather_multiplier *= 1.0 + float(np.clip(car_wet_penalty, 0.0, 0.06))
    return weather_multiplier


@pytest.fixture
def models():
    return (
        Driver(id="D", name="Driver", team_id="T", wet_skill_modifier=0.72),
        Car(team_id="T", team_name="Team", wet_performance=0.61),
        Weather(track_wetness=0.63, rain_intensity=0.4),
    )


@pytest.mark.parametrize(
    ("weather_multiplier", "wet_severity", "wet_skill", "wet_performance"),
    [(1.0, 0.0, 1.0, 0.8), (1.126, 0.63, 0.72, 0.61), (1.2, 1.0, 0.5, 0.0)],
)
def test_cached_weather_term_preserves_previous_scalar_arithmetic(
    weather_multiplier, wet_severity, wet_skill, wet_performance,
):
    _weather_pace_multiplier_from_values.cache_clear()
    expected = _legacy_weather_pace_multiplier(
        weather_multiplier, wet_severity, wet_skill, wet_performance,
    )
    actual = _weather_pace_multiplier_from_values(
        weather_multiplier, wet_severity, wet_skill, wet_performance,
    )
    assert actual == expected


def test_weather_wrapper_reuses_bounded_scalar_cache(models):
    driver, car, weather = models
    _weather_pace_multiplier_from_values.cache_clear()
    before = _weather_pace_multiplier_from_values.cache_info()

    first = LapSimulator.weather_pace_multiplier(driver, car, weather)
    after_first = _weather_pace_multiplier_from_values.cache_info()
    repeated = LapSimulator.weather_pace_multiplier(driver, car, weather)
    after_repeat = _weather_pace_multiplier_from_values.cache_info()

    assert repeated == first
    assert after_first.misses - before.misses == 1
    assert after_repeat.hits - after_first.hits == 1
    assert after_repeat.maxsize == 1024


def _mutate_driver(driver, car, weather):
    driver.wet_skill_modifier = 1.0


def _mutate_car(driver, car, weather):
    car.wet_performance = 1.0


def _mutate_track_wetness(driver, car, weather):
    weather.track_wetness = 0.2


def _mutate_rain_dominant_weather(driver, car, weather):
    weather.track_wetness = 0.05
    weather.rain_intensity = 0.8


@pytest.mark.parametrize(
    "mutation",
    [_mutate_driver, _mutate_car, _mutate_track_wetness, _mutate_rain_dominant_weather],
    ids=["driver", "car", "track_wetness", "rain_dominant"],
)
def test_weather_wrapper_refreshes_each_mutable_scalar_from_fresh_models(models, mutation):
    driver, car, weather = models
    _weather_pace_multiplier_from_values.cache_clear()
    before = LapSimulator.weather_pace_multiplier(driver, car, weather)

    mutation(driver, car, weather)
    actual = LapSimulator.weather_pace_multiplier(driver, car, weather)
    expected = _legacy_weather_pace_multiplier(
        weather.lap_time_multiplier(), weather.wet_severity(),
        driver.wet_skill_modifier, car.wet_performance,
    )

    assert actual == expected
    assert actual != before


class OverriddenWeather(Weather):
    """Weather implementation whose pace values are supplied by overrides."""

    multiplier: float = 1.17
    severity_override: float = 0.44
    _lap_calls: int = PrivateAttr(default=0)
    _severity_calls: int = PrivateAttr(default=0)

    def lap_time_multiplier(self) -> float:
        self._lap_calls += 1
        return self.multiplier

    def wet_severity(self) -> float:
        self._severity_calls += 1
        return self.severity_override


@pytest.mark.parametrize(
    ("attribute", "value"),
    [("multiplier", 1.03), ("severity_override", 0.12)],
    ids=["multiplier_only", "severity_only"],
)
def test_weather_helper_overrides_are_called_each_time_and_cached_by_values(
    attribute, value,
):
    driver = Driver(id="D", name="Driver", team_id="T", wet_skill_modifier=0.8)
    car = Car(team_id="T", team_name="Team", wet_performance=0.6)
    weather = OverriddenWeather()
    _weather_pace_multiplier_from_values.cache_clear()

    first = LapSimulator.weather_pace_multiplier(driver, car, weather)
    repeated = LapSimulator.weather_pace_multiplier(driver, car, weather)

    baseline_expected = _legacy_weather_pace_multiplier(
        weather.multiplier, weather.severity_override,
        driver.wet_skill_modifier, car.wet_performance,
    )
    assert first == baseline_expected
    assert repeated == baseline_expected
    assert weather._lap_calls == 2
    assert weather._severity_calls == 2

    setattr(weather, attribute, value)
    changed = LapSimulator.weather_pace_multiplier(driver, car, weather)
    assert changed == _legacy_weather_pace_multiplier(
        weather.multiplier, weather.severity_override,
        driver.wet_skill_modifier, car.wet_performance,
    )
    assert changed != first
    assert weather._lap_calls == 3
    assert weather._severity_calls == 3
