"""Strict, fixed weather inputs for individual qualifying sessions."""

import json

from f1sim.models import Weather

QUALIFYING_PHASES = ("Q1", "Q2", "Q3")


def validate_qualifying_weather(value) -> dict[str, dict]:
    """Return isolated complete Weather schemas for explicitly supplied phases."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("qualifying_weather must be an object with Q1/Q2/Q3 keys")
    if any(type(phase) is not str or phase not in QUALIFYING_PHASES for phase in value):
        raise ValueError("qualifying_weather keys must be exactly Q1, Q2 or Q3")
    result = {}
    for phase in QUALIFYING_PHASES:
        if phase not in value:
            continue
        fields = value[phase]
        if isinstance(fields, Weather):
            fields = fields.model_dump(mode="json")
        if not isinstance(fields, dict):
            raise ValueError(f"qualifying_weather.{phase} must be a Weather object or field object")
        if any(type(name) is not str or name not in Weather.model_fields for name in fields):
            raise ValueError(f"qualifying_weather.{phase} contains unknown Weather fields")
        try:
            weather = Weather.model_validate_json(json.dumps(fields, allow_nan=False), strict=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid qualifying_weather.{phase}: {exc}") from exc
        result[phase] = weather.model_dump(mode="json")
    return result


def effective_qualifying_weather(weather, overrides) -> dict[str, dict]:
    """Complete session weather for comparisons, using race weather as fallback."""
    canonical = validate_qualifying_weather(overrides)
    race = validate_qualifying_weather({"Q1": weather})["Q1"]
    return {phase: canonical.get(phase, race).copy() for phase in QUALIFYING_PHASES}
