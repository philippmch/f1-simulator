"""Paid rain stints and compound transitions under deterministic surface projection."""

import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from math import inf, isfinite
from numbers import Integral, Real
from threading import RLock

import numpy as np

from f1sim.models import Car, Driver, Tire, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.surface_projection import (
    normalize_weather_intervals,
    projected_surfaces,
    suffix_weather_intervals,
)


@dataclass(frozen=True)
class RainStopDecision:
    pit_now_cost: float
    wait_cost: float

    def should_pit(self, tolerance: float = 0.0) -> bool:
        _nonnegative(tolerance, "tolerance")
        return self.pit_now_cost + tolerance < self.wait_cost


def _nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


@lru_cache(maxsize=4096)
def _running_row(models, weather_json, tire_json, age, lap, physical, intervals=None):
    """Green stint beginning here; no stop budget or current-control key."""
    driver = Driver.model_validate_json(models[0])
    car = Car.model_construct(**json.loads(models[1]))
    track = Track.model_validate_json(models[2])
    surface = Weather.model_validate_json(weather_json)
    tire = Tire.model_validate_json(tire_json)
    simulator = LapSimulator(np.random.default_rng(0))
    row = []
    for offset, surface in enumerate(projected_surfaces(
        surface, track.total_laps - lap + 1, intervals,
    )):
        driver.current_tire_laps = age + offset
        row.append(simulator.calculate_lap_time(
            driver, car, track, tire, surface, lap + offset, physical,
            sample_variation=False,
        ))
    return tuple(row)


@lru_cache(maxsize=4096)
def _surfaces(weather_json, horizon, intervals=None):
    return tuple(surface.model_dump_json() for surface in projected_surfaces(
        Weather.model_validate_json(weather_json), horizon, intervals,
    ))


@lru_cache(maxsize=8192)
def _fresh_future(models, weather_json, fresh_json, lap, budget, physical, intervals=None):
    """Best green cost after a fresh set is fitted; its service is excluded."""
    row = _running_row(models, weather_json, fresh_json, 0, lap, physical, intervals)
    total = sum(row)
    if budget == 0:
        return total
    car = Car.model_construct(**json.loads(models[1]))
    track = Track.model_validate_json(models[2])
    stop = track.pit_lane_delta + expected_stationary_time(car)
    surfaces = _surfaces(weather_json, len(row), intervals)
    stint = 0.0
    for offset in range(1, len(row)):
        stint += row[offset - 1]
        total = min(total, stint + stop + _fresh_future(
            models, surfaces[offset], fresh_json, lap + offset, budget - 1, physical,
            suffix_weather_intervals(intervals, offset),
        ))
    return total


@lru_cache(maxsize=256)
def _plan(snapshots, tire_age, current_lap, budget, lane, queue, modifier, aero, physical,
          intervals=None):
    models = snapshots[:3]
    weather_json, retained_json, fresh_json = snapshots[3:]
    car = Car.model_construct(**json.loads(models[1]))
    track = Track.model_validate_json(models[2])
    row = _running_row(models, weather_json, retained_json, tire_age, current_lap,
                       physical, intervals)
    fresh_row = _running_row(models, weather_json, fresh_json, 0, current_lap, physical, intervals)
    surfaces = _surfaces(weather_json, len(row), intervals)
    service = expected_stationary_time(car)
    wait = sum(row)
    if budget:
        stint = 0.0
        for offset in range(1, len(row)):
            stint += row[offset - 1]
            wait = min(wait, stint + track.pit_lane_delta + service + _fresh_future(
                models, surfaces[offset], fresh_json, current_lap + offset, budget - 1, physical,
                suffix_weather_intervals(intervals, offset),
            ))
    pit = inf if budget == 0 else (
        track.pit_lane_delta * lane + service + queue + _fresh_future(
            models, weather_json, fresh_json, current_lap, budget - 1, physical, intervals,
        )
    )
    first_old, first_fresh = row[0], fresh_row[0]
    if not aero:
        driver = Driver.model_validate_json(models[0])
        simulator = LapSimulator(np.random.default_rng(0))
        weather = Weather.model_validate_json(weather_json)
        def first(tire_json, age):
            driver.current_tire_laps = age
            return simulator.calculate_lap_time(
                driver, car, track, Tire.model_validate_json(tire_json), weather,
                current_lap, physical, active_aero_enabled=False, sample_variation=False,
            )
        first_old, first_fresh = first(retained_json, tire_age), first(fresh_json, 0)
    return RainStopDecision(pit + first_fresh * modifier - fresh_row[0],
                            wait + first_old * modifier - row[0])


def plan_rain_stop(
    driver: Driver, car: Car, track: Track, weather: Weather, current_tire: Tire,
    tire_age: int, current_lap: int, remaining_stops: int, *, pit_lane_factor: float = 1.0,
    additional_current_stop_cost: float = 0.0, current_lap_time_modifier: float = 1.0,
    active_aero_enabled: bool = True, physical_total_laps: int | None = None,
    weather_intervals: tuple[int, ...] | None = None,
) -> RainStopDecision:
    """Compare stopping now with driving at least one lap before any stop.

    Every refit pays service and lane loss and consumes the bounded stop budget.
    Current neutralization and queue costs apply once; future laps assume green
    clean air. Rainfall stays fixed while surface wetness evolves. The caller
    must ensure the same rain compound remains appropriate over the horizon.
    Fuel follows physical_total_laps even when track bounds a shorter plan.
    weather_intervals optionally supplies cumulative surface-update counts for
    each remaining own lap, starting at zero; None uses one update per lap.
    """
    for name, value, minimum in (("tire_age", tire_age, 0), ("current_lap", current_lap, 1),
                                 ("remaining_stops", remaining_stops, 0)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if current_lap > track.total_laps:
        raise ValueError("current_lap must not exceed the planning distance")
    physical = track.total_laps if physical_total_laps is None else physical_total_laps
    if (isinstance(physical, bool) or not isinstance(physical, Integral)
            or physical < track.total_laps):
        raise ValueError("physical_total_laps must be an integer >= planning distance")
    if current_tire.compound not in (TireCompound.INTERMEDIATE, TireCompound.WET):
        raise ValueError("current_tire must be a rain compound")
    if (isinstance(additional_current_stop_cost, bool)
            or not isinstance(additional_current_stop_cost, Real)
            or not isfinite(additional_current_stop_cost)):
        raise ValueError("additional_current_stop_cost must be finite")
    for name, value in (("pit_lane_factor", pit_lane_factor),
                        ("current_lap_time_modifier", current_lap_time_modifier)):
        _nonnegative(value, name)
    if current_lap_time_modifier == 0:
        raise ValueError("current_lap_time_modifier must be positive")
    if not isinstance(active_aero_enabled, bool):
        raise ValueError("active_aero_enabled must be boolean")
    intervals = normalize_weather_intervals(
        track.total_laps - current_lap + 1, weather_intervals, weather=weather,
    )
    clean = driver.model_copy(deep=True)
    clean.reset_race_state()
    # Lap physics reads performance attributes, never names or identifiers.
    # Keep every performance field, including consistency used to compute std.
    clean.id = clean.name = clean.team_id = "projection"
    clean_car = car.model_copy(update={"team_id": "projection", "team_name": "projection"})
    snapshots = tuple(model.model_dump_json() for model in (
        clean, clean_car, track, weather, current_tire, TIRE_COMPOUNDS[current_tire.compound],
    ))
    return _plan(snapshots, int(tire_age), int(current_lap),
                 min(int(remaining_stops), track.total_laps - current_lap + 1),
                 float(pit_lane_factor), float(additional_current_stop_cost),
                 float(current_lap_time_modifier), active_aero_enabled, int(physical), intervals)


@dataclass(frozen=True)
class RainTransitionDecision(RainStopDecision):
    compound: TireCompound | None


# Completed immutable suffix costs are shared across later decisions. Accesses
# are synchronized; computing duplicate misses concurrently is harmless.
_TRANSITION_SUFFIX_LIMIT = 8192
_transition_suffixes = OrderedDict()
_transition_suffix_lock = RLock()


def _reset_transition_cache_after_fork():
    # Another parent thread may own the inherited lock. Replace it without
    # acquiring it; the child's cache starts independently of parent updates.
    global _transition_suffix_lock, _transition_suffixes
    _transition_suffix_lock = RLock()
    _transition_suffixes = OrderedDict()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_transition_cache_after_fork)


def _remember_transition(key, value):
    with _transition_suffix_lock:
        _transition_suffixes[key] = value
        _transition_suffixes.move_to_end(key)
        while len(_transition_suffixes) > _TRANSITION_SUFFIX_LIMIT:
            _transition_suffixes.popitem(last=False)


@lru_cache(maxsize=256)
def _transition_plan(snapshots, tire_age, current_lap, budget, lane, queue,
                     modifier, aero, physical, dry_budget, damp_budget, intervals=None):
    models = snapshots[:3]
    weather_json, retained_json, tires_json = snapshots[3:]
    track = Track.model_validate_json(models[2])
    car = Car.model_construct(**json.loads(models[1]))
    fresh = {TireCompound(key): Tire.model_validate(value).model_dump_json()
             for key, value in json.loads(tires_json).items()}
    retained = Tire.model_validate_json(retained_json)
    horizon = track.total_laps - current_lap + 1
    surface_json = _surfaces(weather_json, horizon, intervals)
    surfaces = [Weather.model_validate_json(value) for value in surface_json]
    cadence_suffixes = tuple(suffix_weather_intervals(intervals, offset, surface)
                             for offset, surface in enumerate(surfaces))
    service = expected_stationary_time(car)
    stop_cost = track.pit_lane_delta + service
    candidates = []
    critical = {}
    for compound in TireCompound:
        critical[compound] = [s.tire_mismatch(compound) == "critical" for s in surfaces]
    for surface in surfaces:
        rain = surface.fresh_rain_compound()
        candidates.append((rain,) if rain is not None else (
            TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD,
        ))

    def reduced(limit):
        return None if limit is None else max(0, limit - 1)

    def may_stop(offset, compound, left, dry, damp):
        if critical[compound][offset]:
            return True
        limit = dry if (surfaces[offset].track_wetness < .08
                        and surfaces[offset].rain_intensity < .15) else damp
        return left > 0 and (compound in (TireCompound.INTERMEDIATE, TireCompound.WET)
                             or limit is None or limit > 0)

    def cache_key(state):
        offset, compound, left, dry, damp = state
        return (models, surface_json[offset], tires_json, current_lap + offset,
                compound, left, dry, damp, physical, cadence_suffixes[offset])

    solved = {}
    dependencies = []

    def suffix(state):
        if state not in solved:
            key = cache_key(state)
            with _transition_suffix_lock:
                value = _transition_suffixes.get(key)
                if value is not None:
                    _transition_suffixes.move_to_end(key)
            if value is not None:
                solved[state] = value
            else:
                dependencies.append(state)
                return 0.0
        return solved[state]

    def stint(start, compound, age, tire_json, left, dry, damp):
        row = _running_row(models, surface_json[start], tire_json, age,
                           current_lap + start, physical,
                           cadence_suffixes[start])
        total, best_cost = 0.0, inf
        # This set must run its fitting/current lap before any future stop.
        for offset in range(start, horizon):
            if offset > start and may_stop(offset, compound, left, dry, damp):
                for candidate in candidates[offset]:
                    if not critical[candidate][offset]:
                        best_cost = min(best_cost, total + stop_cost + suffix((
                            offset, candidate, max(0, left - 1), reduced(dry), reduced(damp),
                        )))
            if critical[compound][offset]:
                return best_cost
            total += row[offset - start]
        return min(best_cost, total)

    def solve(initial):
        pending = [initial]
        while pending:
            state = pending[-1]
            if state in solved:
                pending.pop()
                continue
            dependencies.clear()
            offset, compound, left, dry, damp = state
            value = stint(offset, compound, 0, fresh[compound], left, dry, damp)
            if dependencies:
                pending.extend(dict.fromkeys(dependencies))
            else:
                solved[state] = value
                _remember_transition(cache_key(state), value)
                pending.pop()
        return solved[initial]

    # First discover and solve all paid-fit suffixes needed by retaining the
    # current set. Subsequent stop-now candidates reuse that same green cache.
    while True:
        dependencies.clear()
        wait = stint(0, retained.compound, tire_age, retained_json,
                     budget, dry_budget, damp_budget)
        missing = tuple(dict.fromkeys(dependencies))
        if not missing:
            break
        for state in missing:
            solve(state)

    def first(tire_json, age):
        row = _running_row(models, weather_json, tire_json, age, current_lap, physical, intervals)
        if aero:
            return row[0] * modifier - row[0]
        driver = Driver.model_validate_json(models[0])
        driver.current_tire_laps = age
        simulator = LapSimulator(np.random.default_rng(0))
        actual = simulator.calculate_lap_time(
            driver, car, track, Tire.model_validate_json(tire_json), surfaces[0],
            current_lap, physical, active_aero_enabled=False, sample_variation=False,
        )
        return actual * modifier - row[0]

    wait += first(retained_json, tire_age)
    pit, compound = inf, None
    if may_stop(0, retained.compound, budget, dry_budget, damp_budget):
        for candidate in candidates[0]:
            if critical[candidate][0]:
                continue
            state = (0, candidate, max(0, budget - 1),
                     reduced(dry_budget), reduced(damp_budget))
            # Consult shared cache before scheduling an uncached suffix.
            dependencies.clear()
            value = suffix(state)
            if dependencies:
                value = solve(state)
            cost = (track.pit_lane_delta * lane + service + queue + value
                    + first(fresh[candidate], 0))
            if cost < pit:
                pit, compound = cost, candidate
    return RainTransitionDecision(pit, wait, compound)


def plan_rain_transition(
    driver: Driver, car: Car, track: Track, weather: Weather, current_tire: Tire,
    tire_age: int, current_lap: int, remaining_stops: int, *, pit_lane_factor: float = 1.0,
    additional_current_stop_cost: float = 0.0, current_lap_time_modifier: float = 1.0,
    active_aero_enabled: bool = True, physical_total_laps: int | None = None,
    weather_intervals: tuple[int, ...] | None = None,
    remaining_dry_stops: int | None = None, remaining_damp_stops: int | None = None,
) -> RainTransitionDecision:
    """Plan bounded paid stops across rain/slick transitions under fixed rainfall.

    weather_intervals optionally supplies cumulative surface-update counts for
    each remaining own lap, starting at zero; None uses one update per lap.

    A retained set may run while noncritical. Fresh fits follow the surface's
    rain-compound recommendation, or consider all slicks when it recommends
    none. Only critical-set replacements may exceed the elective stop budget.
    Every fit runs its fitting lap; future costs assume green clean air.
    Optional slick-state allowances count all paid fits from this decision,
    including earlier rain fits; they never reset on a compound transition.
    """
    for name, value, minimum in (("tire_age", tire_age, 0), ("current_lap", current_lap, 1),
                                 ("remaining_stops", remaining_stops, 0)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if current_lap > track.total_laps:
        raise ValueError("current_lap must not exceed the planning distance")
    physical = track.total_laps if physical_total_laps is None else physical_total_laps
    if (isinstance(physical, bool) or not isinstance(physical, Integral)
            or physical < track.total_laps):
        raise ValueError("physical_total_laps must be an integer >= planning distance")
    if current_tire.compound not in (TireCompound.INTERMEDIATE, TireCompound.WET):
        raise ValueError("current_tire must be a rain compound")
    if (isinstance(additional_current_stop_cost, bool)
            or not isinstance(additional_current_stop_cost, Real)
            or not isfinite(additional_current_stop_cost)):
        raise ValueError("additional_current_stop_cost must be finite")
    for name, value in (("pit_lane_factor", pit_lane_factor),
                        ("current_lap_time_modifier", current_lap_time_modifier)):
        _nonnegative(value, name)
    if current_lap_time_modifier == 0:
        raise ValueError("current_lap_time_modifier must be positive")
    if not isinstance(active_aero_enabled, bool):
        raise ValueError("active_aero_enabled must be boolean")
    for name, value in (("remaining_dry_stops", remaining_dry_stops),
                        ("remaining_damp_stops", remaining_damp_stops)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, Integral) or value < 0
        ):
            raise ValueError(f"{name} must be a nonnegative integer or None")
    intervals = normalize_weather_intervals(
        track.total_laps - current_lap + 1, weather_intervals, weather=weather,
    )
    clean = driver.model_copy(deep=True)
    clean.reset_race_state()
    clean.id = clean.name = clean.team_id = "projection"
    clean_car = car.model_copy(update={"team_id": "projection", "team_name": "projection"})
    snapshots = tuple(model.model_dump_json() for model in (
        clean, clean_car, track, weather, current_tire,
    )) + (json.dumps({compound.value: tire.model_dump(mode="json")
                     for compound, tire in TIRE_COMPOUNDS.items()}, sort_keys=True),)
    return _transition_plan(
        snapshots, int(tire_age), int(current_lap),
        min(int(remaining_stops), track.total_laps - current_lap + 1),
        float(pit_lane_factor), float(additional_current_stop_cost),
        float(current_lap_time_modifier), active_aero_enabled, int(physical),
        None if remaining_dry_stops is None else int(remaining_dry_stops),
        None if remaining_damp_stops is None else int(remaining_damp_stops), intervals,
    )
