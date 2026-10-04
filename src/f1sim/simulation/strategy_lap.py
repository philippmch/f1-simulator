"""Isolated mean lap calls for forecast models with behavioral extensions."""

from collections import OrderedDict
from contextvars import ContextVar
from functools import wraps

from f1sim.models._native import forecast_json, native_physics, register_forecast_helpers
from f1sim.models.tire import TIRE_COMPOUNDS

_CONTROL_LAPS = ContextVar("controlled_strategy_laps", default=None)
_CONTROL_LAP_LIMIT = 65_536
_CONTROL_RELAXATION_LIMIT = 32


def control_lap_scope(function):
    """Share bounded scalar mean laps only within one field decision."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        token = _CONTROL_LAPS.set(({}, [0], {}, OrderedDict()))
        try:
            return function(*args, **kwargs)
        finally:
            _CONTROL_LAPS.reset(token)
    return wrapped


def _control_lap_package(driver, car, track, physical_total_laps):
    return (physical_total_laps, *(forecast_json(model) for model in (driver, car, track)),
            tuple(forecast_json(tire) for tire in TIRE_COMPOUNDS.values()))


def control_lap_memo(driver, car, track, physical_total_laps):
    """Separate every native model package before sharing green suffix laps."""
    memo = _CONTROL_LAPS.get()
    if memo is None or not native_physics(driver, car, track):
        return None
    package = _control_lap_package(driver, car, track, physical_total_laps)
    packages, count = memo[:2]
    return packages.setdefault(package, {}), packages, count


def control_relaxation_memo(cache, key):
    """Share completed fresh-service scalars within one native field decision.

    The lap memo owns its exact physics package for the scope's lifetime.
    Keep at most 32 forecast tables; callers bound each table's own horizon.
    No stock, model, callback or unfinished search frame enters this cache.
    """
    scope = _CONTROL_LAPS.get()
    if scope is None or cache is None:
        return None
    tables = scope[3]
    key = id(cache[0]), key
    if key not in tables:
        if len(tables) >= _CONTROL_RELAXATION_LIMIT:
            tables.popitem(last=False)
        tables[key] = {}
    else:
        tables.move_to_end(key)
    return tables[key]


def install_control_wear_bound(driver, car, track, physical, current_lap, calculate):
    """Keep an optimistic original-stock bound for this decision's suffixes."""
    scope = _CONTROL_LAPS.get()
    if scope is not None:
        scope[2].update(package=_control_lap_package(driver, car, track, physical),
                        first_lap=current_lap, final_lap=track.total_laps, calculate=calculate)


def control_wear_bound(driver, car, track, physical, current_lap):
    scope = _CONTROL_LAPS.get()
    if scope is None or not scope[2] or not native_physics(driver, car, track):
        return None
    bound = scope[2]
    if (track.total_laps != bound["final_lap"]
            or _control_lap_package(driver, car, track, physical) != bound["package"]):
        return None
    offset = current_lap - bound["first_lap"]
    if offset < 0:
        return None
    if "rows" not in bound:
        bound["rows"] = bound["calculate"]()
    return bound["rows"][offset:]


def memoized_control_lap(cache, prepared, tire, surface, lap, age, gap, aero):
    """Reuse identical native physics, keeping clock and stock choices separate."""
    memo, packages, count = cache
    weather = (surface.condition, surface.track_temperature, surface.air_temperature,
               surface.humidity, surface.rain_intensity, surface.track_wetness,
               surface.change_probability)
    key = tire.compound.value, weather, lap, age, gap, aero
    if key not in memo:
        if count[0] >= _CONTROL_LAP_LIMIT:
            for values in packages.values():
                values.clear()
            count[0] = 0
        memo[key] = prepared(tire, surface, lap, age, gap, aero)
        count[0] += 1
    return memo[key]


def strategy_projection_models(driver, car, *models):
    """Normalize only native metadata that cannot participate in lap physics."""
    driver, car = driver.model_copy(deep=True), car.model_copy(deep=True)
    if native_physics(driver, car, *models):
        driver.reset_race_state()
        driver.id = driver.name = driver.team_id = "projection"
        car.team_id = car.team_name = "projection"
    return driver, car


def isolated_strategy_lap(simulator, driver, car, track, tire, surface,
                          lap, physical, *, tire_age, **options):
    """Keep actual metadata while each hypothesis owns its mutable models."""
    clean = driver.model_copy(deep=True)
    clean.current_tire_laps = tire_age
    return simulator.calculate_lap_time(
        clean, car.model_copy(deep=True), track.model_copy(deep=True),
        tire.model_copy(deep=True), surface.model_copy(deep=True), lap, physical,
        sample_variation=False, **options)


register_forecast_helpers(globals(), (
    "isolated_strategy_lap", "strategy_projection_models", "native_physics",
    "control_lap_scope", "control_lap_memo", "memoized_control_lap", "forecast_json",
    "install_control_wear_bound", "control_wear_bound", "_control_lap_package",
    "control_relaxation_memo",
))
