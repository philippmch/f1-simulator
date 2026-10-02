"""Known race atmosphere changes on the shared leading-lap clock."""

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real

from f1sim.models._native import register_forecast_helpers
from f1sim.models.tire import TireCompound
from f1sim.models.weather import WeatherCondition


def validate_weather_schedule(value, *, total_laps=None):
    """Return isolated canonical entries; never coerce quoted numbers or booleans."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("weather_schedule must be a list")
    result = []
    previous = 1
    for item in value:
        if (not isinstance(item, dict) or set(item) - {"lap", "rain_intensity", "condition"}
                or not {"lap", "rain_intensity"} <= set(item)):
            raise ValueError("weather_schedule entries require lap and rain_intensity only, "
                             "with optional condition")
        lap, rain = item["lap"], item["rain_intensity"]
        if (isinstance(lap, bool) or not isinstance(lap, Integral) or lap < 2
                or lap <= previous or (total_laps is not None and lap > total_laps)):
            raise ValueError("weather_schedule laps must be increasing integers from 2 "
                             "through the scheduled distance")
        if (isinstance(rain, bool) or not isinstance(rain, Real)
                or not 0 <= rain <= 1 or not isfinite(rain)):
            raise ValueError("weather_schedule rain_intensity must be finite and from 0 to 1")
        entry = {"lap": int(lap), "rain_intensity": float(rain)}
        if "condition" in item:
            try:
                entry["condition"] = WeatherCondition(item["condition"]).value
            except (ValueError, TypeError) as exc:
                raise ValueError("weather_schedule condition must be an existing value") from exc
        result.append(entry)
        previous = lap
    return result


@dataclass(frozen=True)
class WeatherForecastContext:
    """Immutable prescribed atmosphere and current shared leading-lap ordinal.

    Surface updates, including externally timed paid-stop updates, advance this
    ordinal. An own-lap suffix must rebase by its cumulative shared updates.
    """

    schedule: tuple[tuple[int, float, str | None], ...]
    leading_lap: int = 1

    def __post_init__(self):
        if (type(self.leading_lap) is not int or self.leading_lap < 1
                or type(self.schedule) is not tuple):
            raise ValueError("forecast context requires immutable schedule and "
                             "positive leading_lap")
        previous = 1
        for entry in self.schedule:
            if type(entry) is not tuple or len(entry) != 3:
                raise ValueError("forecast context schedule entries must be immutable triples")
            lap, rain, condition = entry
            if (type(lap) is not int or lap <= previous or type(rain) is not float
                    or not isfinite(rain) or not 0 <= rain <= 1
                    or (condition is not None and (type(condition) is not str
                        or condition not in {item.value for item in WeatherCondition}))):
                raise ValueError("invalid forecast context schedule entry")
            previous = lap

    @classmethod
    def from_schedule(cls, value, *, total_laps=None, leading_lap=1):
        entries = validate_weather_schedule(value, total_laps=total_laps)
        return cls(tuple((item["lap"], item["rain_intensity"], item.get("condition"))
                         for item in entries), leading_lap)

    def advanced(self, updates=1):
        if type(updates) is not int or updates < 0:
            raise ValueError("forecast updates must be a nonnegative integer")
        return WeatherForecastContext(self.schedule, self.leading_lap + updates)

    def project_next(self, weather):
        """Apply atmosphere entering the next shared lap before ordinary drainage."""
        next_lap = self.leading_lap + 1
        for lap, rain, condition in self.schedule:
            if lap == next_lap:
                changes = {"rain_intensity": rain}
                if condition is not None:
                    changes["condition"] = WeatherCondition(condition)
                weather = weather.model_copy(update=changes, deep=True)
                break
        return weather.project_surface()


@dataclass(frozen=True)
class ScheduledWeatherIntervals:
    """Internal immutable cadence plus context, including native cache identity."""

    values: tuple[int, ...]
    context: WeatherForecastContext

    def __post_init__(self):
        if (type(self.context) is not WeatherForecastContext or type(self.values) is not tuple
                or not self.values or self.values[0] != 0
                or any(type(value) is not int or value < 0 for value in self.values)
                or any(a > b for a, b in zip(self.values, self.values[1:]))):
            raise ValueError("scheduled weather intervals require valid immutable cadence/context")

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, index):
        return self.values[index]


def project_next_surface(weather, context=None, updates=0):
    """Advance one surface update relative to a forecast's original context."""
    return (weather.project_surface() if context is None
            else context.advanced(updates).project_next(weather))


def validate_forecast_context(context):
    if context is not None and type(context) is not WeatherForecastContext:
        raise ValueError("forecast_context must be a WeatherForecastContext")
    return context


def has_prescribed_weather(context):
    """Empty contexts keep the unscheduled policy as well as its candidate set."""
    return context is not None and bool(context.schedule)


def paid_compound_candidates(weather, context=None):
    """Price every safe fresh fit, including currently suboptimal alternatives.

    Eligibility uses the observed commitment surface. A paid forecast's rejoin
    surface determines pace; future critical retention still ends that stint.
    The optional context keeps shared callers uniform; it never narrows safety.
    """
    return tuple(compound for compound in TireCompound
                 if weather.tire_mismatch(compound) != "critical")


register_forecast_helpers(globals(), (
    "project_next_surface", "has_prescribed_weather", "paid_compound_candidates",
))
register_forecast_helpers(vars(WeatherForecastContext), ("project_next", "advanced"))
