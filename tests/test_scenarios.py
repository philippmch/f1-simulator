import pytest

from f1sim.analysis import parse_scenario_labels, scenario_weather_from_label
from f1sim.models import Weather, WeatherCondition


def test_parse_scenario_labels_deduplicates_and_normalizes() -> None:
    labels = parse_scenario_labels("Dry, light_rain, dry,HEAVY_RAIN")
    assert labels == ["dry", "light_rain", "heavy_rain"]


def test_scenario_weather_light_rain_sets_wet_conditions() -> None:
    base = Weather(condition=WeatherCondition.DRY, track_wetness=0.0, rain_intensity=0.0)
    scenario = scenario_weather_from_label(base, "light_rain")

    assert scenario.name == "light_rain"
    assert scenario.weather.condition == WeatherCondition.LIGHT_RAIN
    assert scenario.weather.track_wetness >= 0.45
    assert scenario.weather.rain_intensity >= 0.35


def test_parse_scenario_labels_rejects_empty() -> None:
    with pytest.raises(ValueError, match="At least one scenario label"):
        parse_scenario_labels("  ,  ")
