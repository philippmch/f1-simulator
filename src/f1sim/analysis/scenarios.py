"""Scenario comparison helpers for Monte Carlo runs."""

from dataclasses import dataclass

from f1sim.analysis.montecarlo import SimulationResults
from f1sim.models import Weather, WeatherCondition


@dataclass(frozen=True)
class WeatherScenario:
    """Named weather scenario configuration."""

    name: str
    weather: Weather


def scenario_weather_from_label(base_weather: Weather, label: str) -> WeatherScenario:
    """Build a weather scenario from a simple label.

    Supported labels: dry, cloudy, light_rain, heavy_rain.
    """
    normalized = label.strip().lower()

    if normalized == "dry":
        weather = base_weather.model_copy(
            update={
                "condition": WeatherCondition.DRY,
                "rain_intensity": 0.0,
                "track_wetness": 0.0,
            }
        )
    elif normalized == "cloudy":
        weather = base_weather.model_copy(
            update={
                "condition": WeatherCondition.CLOUDY,
                "rain_intensity": 0.0,
                "track_wetness": min(base_weather.track_wetness, 0.1),
            }
        )
    elif normalized == "light_rain":
        weather = base_weather.model_copy(
            update={
                "condition": WeatherCondition.LIGHT_RAIN,
                "rain_intensity": max(base_weather.rain_intensity, 0.35),
                "track_wetness": max(base_weather.track_wetness, 0.45),
            }
        )
    elif normalized == "heavy_rain":
        weather = base_weather.model_copy(
            update={
                "condition": WeatherCondition.HEAVY_RAIN,
                "rain_intensity": max(base_weather.rain_intensity, 0.85),
                "track_wetness": max(base_weather.track_wetness, 0.85),
            }
        )
    else:
        msg = f"Unknown scenario label: {label}"
        raise ValueError(msg)

    return WeatherScenario(name=normalized, weather=weather)


def parse_scenario_labels(raw: str) -> list[str]:
    """Parse comma-separated scenario labels, preserving order and uniqueness."""
    labels: list[str] = []
    seen: set[str] = set()

    for part in raw.split(","):
        label = part.strip().lower()
        if not label or label in seen:
            continue
        labels.append(label)
        seen.add(label)

    if not labels:
        msg = "At least one scenario label is required"
        raise ValueError(msg)

    return labels


def build_scenario_win_matrix(
    scenario_results: dict[str, SimulationResults],
) -> dict[str, dict[str, float]]:
    """Build per-driver win probability matrix by scenario."""
    matrix: dict[str, dict[str, float]] = {}

    for scenario_name, results in scenario_results.items():
        for driver_id, win_prob in results.get_win_probabilities().items():
            matrix.setdefault(driver_id, {})[scenario_name] = win_prob

    return matrix
