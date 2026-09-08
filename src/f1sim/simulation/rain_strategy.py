"""Paid same-compound rain stints under a deterministic surface projection."""

import json
from dataclasses import dataclass
from functools import lru_cache
from math import inf, isfinite
from numbers import Integral, Real

import numpy as np

from f1sim.models import Car, Driver, Tire, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time


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
def _running_row(models, weather_json, tire_json, age, lap, physical):
    """Green stint beginning here; no stop budget or current-control key."""
    driver = Driver.model_validate_json(models[0])
    car = Car.model_construct(**json.loads(models[1]))
    track = Track.model_validate_json(models[2])
    surface = Weather.model_validate_json(weather_json)
    tire = Tire.model_validate_json(tire_json)
    simulator = LapSimulator(np.random.default_rng(0))
    row = []
    for offset in range(track.total_laps - lap + 1):
        driver.current_tire_laps = age + offset
        row.append(simulator.calculate_lap_time(
            driver, car, track, tire, surface, lap + offset, physical,
            sample_variation=False,
        ))
        surface = surface.project_surface()
    return tuple(row)


@lru_cache(maxsize=4096)
def _surfaces(weather_json, horizon):
    surface = Weather.model_validate_json(weather_json)
    result = []
    for _ in range(horizon):
        result.append(surface.model_dump_json())
        surface = surface.project_surface()
    return tuple(result)


@lru_cache(maxsize=8192)
def _fresh_future(models, weather_json, fresh_json, lap, budget, physical):
    """Best green cost after a fresh set is fitted; its service is excluded."""
    row = _running_row(models, weather_json, fresh_json, 0, lap, physical)
    total = sum(row)
    if budget == 0:
        return total
    car = Car.model_construct(**json.loads(models[1]))
    track = Track.model_validate_json(models[2])
    stop = track.pit_lane_delta + expected_stationary_time(car)
    surfaces = _surfaces(weather_json, len(row))
    stint = 0.0
    for offset in range(1, len(row)):
        stint += row[offset - 1]
        total = min(total, stint + stop + _fresh_future(
            models, surfaces[offset], fresh_json, lap + offset, budget - 1, physical,
        ))
    return total


@lru_cache(maxsize=256)
def _plan(snapshots, tire_age, current_lap, budget, lane, queue, modifier, aero, physical):
    models = snapshots[:3]
    weather_json, retained_json, fresh_json = snapshots[3:]
    car = Car.model_construct(**json.loads(models[1]))
    track = Track.model_validate_json(models[2])
    row = _running_row(models, weather_json, retained_json, tire_age, current_lap, physical)
    fresh_row = _running_row(models, weather_json, fresh_json, 0, current_lap, physical)
    surfaces = _surfaces(weather_json, len(row))
    service = expected_stationary_time(car)
    wait = sum(row)
    if budget:
        stint = 0.0
        for offset in range(1, len(row)):
            stint += row[offset - 1]
            wait = min(wait, stint + track.pit_lane_delta + service + _fresh_future(
                models, surfaces[offset], fresh_json, current_lap + offset, budget - 1, physical,
            ))
    pit = inf if budget == 0 else (
        track.pit_lane_delta * lane + service + queue + _fresh_future(
            models, weather_json, fresh_json, current_lap, budget - 1, physical,
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
) -> RainStopDecision:
    """Compare stopping now with driving at least one lap before any stop.

    Every refit pays service and lane loss and consumes the bounded stop budget.
    Current neutralization and queue costs apply once; future laps assume green
    clean air. Rainfall stays fixed while surface wetness evolves. The caller
    must ensure the same rain compound remains appropriate over the horizon.
    Fuel follows physical_total_laps even when track bounds a shorter plan.
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
                 float(current_lap_time_modifier), active_aero_enabled, int(physical))
