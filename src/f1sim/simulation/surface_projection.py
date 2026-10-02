"""Deterministic surface projections on a caller-supplied weather clock."""

from numbers import Integral

from f1sim.models import Weather
from f1sim.models._native import (
    native_forecast_cache,
    native_model,
    native_model_class,
    register_forecast_helpers,
)
from f1sim.simulation.weather_schedule import (
    ScheduledWeatherIntervals,
    WeatherForecastContext,
    project_next_surface,
)


def normalize_weather_intervals(horizon, weather_intervals=None, *, weather=None,
                                forecast_context=None):
    """Validate cumulative update counts; None keeps the ordinary lap clock."""
    if forecast_context is not None and type(forecast_context) is not WeatherForecastContext:
        raise ValueError("forecast_context must be a WeatherForecastContext")
    if isinstance(weather_intervals, ScheduledWeatherIntervals):
        if forecast_context is not None and forecast_context != weather_intervals.context:
            raise ValueError("conflicting forecast contexts")
        forecast_context = weather_intervals.context
        weather_intervals = weather_intervals.values
    if weather_intervals is None and forecast_context is not None:
        weather_intervals = tuple(range(horizon))
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
    if forecast_context is not None:
        return ScheduledWeatherIntervals(values, forecast_context)
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
    if isinstance(weather_intervals, ScheduledWeatherIntervals):
        origin = weather_intervals[offset]
        return ScheduledWeatherIntervals(
            tuple(value - origin for value in weather_intervals.values[offset:]),
            weather_intervals.context.advanced(origin),
        )
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
    context = intervals.context if isinstance(intervals, ScheduledWeatherIntervals) else None
    for interval in intervals:
        for update in range(previous, interval):
            surface = project_next_surface(surface, context, update)
        snapshots.append(surface.model_dump_json())
        previous = interval
    return tuple(snapshots)


def projected_surfaces(weather: Weather, horizon: int, weather_intervals=None,
                       *, forecast_context=None):
    """Return isolated surfaces on the explicit shared clock, without randomness."""
    intervals = normalize_weather_intervals(horizon, weather_intervals, weather=weather,
                                           forecast_context=forecast_context)
    # This forecast boundary intentionally projects only the base Weather schema.
    # Callable extras and subclass methods are outside that documented policy.
    values = object.__getattribute__(weather, "__dict__")
    weather = Weather.model_validate({name: values[name] for name in Weather.model_fields})
    intervals = tuple(range(horizon)) if intervals is None else intervals
    # Serialize cached values rather than sharing mutable Weather objects.
    snapshots = (_surface_snapshots if native_model_class(Weather)
                 else _surface_snapshots.__wrapped__)(weather.model_dump_json(), intervals)
    return tuple(Weather.model_validate_json(snapshot) for snapshot in snapshots)


register_forecast_helpers(globals(), (
    '_surface_snapshots', '_drying_steps', 'project_next_surface',
))
