"""Describe qualifying-session conditions from the completed run's saved inputs."""

from f1sim.simulation.qualifying_weather import (
    effective_qualifying_weather,
    validate_qualifying_weather,
)


def qualifying_weather_context(snapshot) -> str:
    if not isinstance(snapshot, dict) or "qualifying_weather" not in snapshot:
        return ""
    try:
        overrides = validate_qualifying_weather(snapshot["qualifying_weather"])
        if not overrides:
            return ""
        if (type(snapshot.get("schema_version")) is not int
                or snapshot["schema_version"] not in (7, 8, 9, 10, 11)):
            return "Qualifying weather: unrecognized saved schema."
        effective = effective_qualifying_weather(snapshot.get("weather"), overrides)
    except (TypeError, ValueError):
        return "Qualifying weather: invalid saved conditions."
    phases = []
    for phase, weather in effective.items():
        inherited = " (race weather)" if phase not in overrides else ""
        phases.append(
            f"{phase}: {weather['condition'].replace('_', ' ')}, "
            f"rain {weather['rain_intensity'] * 100:g}%, "
            f"surface water {weather['track_wetness'] * 100:g}%{inherited}"
        )
    return "Qualifying weather (fixed within each session): " + "; ".join(phases) + "."
