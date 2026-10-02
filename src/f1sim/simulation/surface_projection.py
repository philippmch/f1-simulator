"""Deterministic surface projections on a caller-supplied weather clock."""

from numbers import Integral

from f1sim.models import Weather
from f1sim.models._native import (
    native_forecast_cache,
    native_model,
    native_model_class,
    register_forecast_helpers,
)


def normalize_weather_intervals(horizon, weather_intervals=None, *, weather=None):
    """Validate cumulative update counts; None keeps the ordinary lap clock."""
    if weather_intervals is None:
        return None
    if not isinstance(weather_intervals, tuple) or len(weather_intervals) != horizon:
        raise ValueError("weather_intervals must be a tuple matching the planning horizon")
    if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0
           for value in weather_intervals):
        raise ValueError("weather_intervals must contain nonnegative integers")
    values = tuple(int(value) for value in weather_intervals)
    if not values or values[0] != 0 or any(a > b for a, b in zip(values, values[1:])):
        raise ValueError("weather_intervals must start at zero and be nondecreasing")
    return _canonical_intervals(values, weather)


@native_forecast_cache(maxsize=4096)
def _drying_steps(weather_json):
    surface = Weather.model_validate_json(weather_json)
    steps = 0
    while surface.track_wetness > 0:
        surface = surface.project_surface()
        steps += 1
    return steps


def _canonical_intervals(values, weather):
    ordinary = tuple(range(len(values)))
    if values == ordinary:
        return None
    if weather is not None and native_model(weather):
        if weather.track_wetness == weather.rain_intensity:
            return None
        if weather.rain_intensity == 0:
            # Once drainage reaches zero, further updates have no effect.
            # Use the actual surface model to find that boundary, including
            # its floating-point subtraction behavior.
            steps = _drying_steps(weather.model_dump_json())
            values = tuple(min(value, steps) for value in values)
            if values == tuple(min(value, steps) for value in ordinary):
                return None
    return values


def suffix_weather_intervals(weather_intervals, offset, weather=None):
    """Rebase a future own-lap suffix onto its already projected weather."""
    if weather_intervals is None or (weather is not None and native_model(weather)
                                      and weather.track_wetness == weather.rain_intensity):
        return None
    origin = weather_intervals[offset]
    values = tuple(value - origin for value in weather_intervals[offset:])
    return _canonical_intervals(values, weather)


@native_forecast_cache(maxsize=4096)
def _surface_snapshots(weather_json, intervals):
    surface = Weather.model_validate_json(weather_json)
    snapshots = []
    previous = 0
    for interval in intervals:
        for _ in range(interval - previous):
            surface = surface.project_surface()
        snapshots.append(surface.model_dump_json())
        previous = interval
    return tuple(snapshots)


def projected_surfaces(weather: Weather, horizon: int, weather_intervals=None):
    """Return isolated surfaces without randomness or changing rainfall/condition."""
    intervals = normalize_weather_intervals(horizon, weather_intervals, weather=weather)
    # This forecast boundary intentionally projects only the base Weather schema.
    # Callable extras and subclass methods are outside that documented policy.
    values = object.__getattribute__(weather, "__dict__")
    weather = Weather.model_validate({name: values[name] for name in Weather.model_fields})
    intervals = tuple(range(horizon)) if intervals is None else intervals
    # Serialize cached values rather than sharing mutable Weather objects.
    snapshots = (_surface_snapshots if native_model_class(Weather)
                 else _surface_snapshots.__wrapped__)(weather.model_dump_json(), intervals)
    return tuple(Weather.model_validate_json(snapshot) for snapshot in snapshots)


register_forecast_helpers(globals(), ('_surface_snapshots', '_drying_steps'))
