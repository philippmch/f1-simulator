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


def test_parse_scenario_labels_rejects_unknown_before_simulation() -> None:
    with pytest.raises(ValueError, match="Unknown scenario label.*snow"):
        parse_scenario_labels("dry,snow")


@pytest.mark.parametrize("label", ["dry", "cloudy", "light_rain", "heavy_rain"])
def test_fixed_rainfall_changes_only_transition_probability(label: str) -> None:
    import numpy as np

    base = Weather(track_wetness=0.9, rain_intensity=0.2, change_probability=0.75)
    original = base.model_dump()
    default = scenario_weather_from_label(base, label).weather
    evolving = scenario_weather_from_label(base, label, weather_mode="evolving").weather
    fixed = scenario_weather_from_label(base, label, weather_mode="fixed_rainfall").weather
    assert evolving.model_dump() == default.model_dump()
    assert evolving.change_probability == 0.75
    assert fixed.model_dump() == {**default.model_dump(), "change_probability": 0.0}
    assert base.model_dump() == original
    rng = np.random.default_rng(42)
    current = fixed
    for _ in range(30):
        projected = current.project_surface()
        current = current.evolve(rng)
        assert current.condition == fixed.condition
        assert current.rain_intensity == fixed.rain_intensity
        assert current.track_wetness == projected.track_wetness
    if label != "dry":
        assert current.track_wetness != fixed.track_wetness


@pytest.mark.parametrize("mode", [None, True, 0, [], {}, "fixed", "EVOLVING", " evolving"])
def test_invalid_weather_mode_rejected(mode: object) -> None:
    with pytest.raises(ValueError, match="weather_mode"):
        scenario_weather_from_label(Weather(), "dry", weather_mode=mode)
