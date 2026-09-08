"""Optimistic paid weather-stop plans under a fixed rainfall surface projection."""

import json
from dataclasses import dataclass
from functools import lru_cache
from math import inf

import numpy as np

from f1sim.models import Car, Driver, Tire, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time


@dataclass(frozen=True)
class WeatherStopCosts:
    pit_now_cost: float
    stay_cost: float


def _surface_path(weather: Weather, laps: int) -> list[Weather]:
    path = [weather]
    for _ in range(1, laps):
        path.append(path[-1].project_surface())
    return path


def _running(simulator, driver, car, track, tire, weather, lap, age, *, gap=None, aero=True,
             physical_total_laps=None):
    driver.current_tire_laps = age  # Projection owns this isolated driver.
    total_laps = track.total_laps if physical_total_laps is None else physical_total_laps
    return simulator.calculate_lap_time(
        driver, car, track, tire, weather, lap, total_laps,
        gap_to_car_ahead=gap, active_aero_enabled=aero, sample_variation=False,
    )


@lru_cache(maxsize=4096)
def _fresh_plan_costs(driver_json: str, car_json: str, track_json: str,
                      weather_json: str, tires_json: str, current_lap: int, physical_total_laps: int
                      ) -> tuple[tuple[TireCompound, float, float], ...]:
    """Immutable first-set costs and first-lap times; cached futures assume green."""
    driver = Driver.model_validate_json(driver_json)
    # Snapshots originate from an existing model, whose supported mutable
    # overrides (for example deterministic pit_stop_std=0) must survive here.
    car = Car.model_construct(**json.loads(car_json))
    track = Track.model_validate_json(track_json)
    weather = Weather.model_validate_json(weather_json)
    tires = {TireCompound(key): Tire.model_validate(value)
             for key, value in json.loads(tires_json).items()}
    simulator = LapSimulator(np.random.default_rng(0))
    horizon = track.total_laps - current_lap + 1
    surfaces = _surface_path(weather, horizon)
    service = track.pit_lane_delta + expected_stationary_time(car)
    # Each cached row represents one possible future stop. Reusing rows at
    # their actual lap/weather lets later decisions share the same suffixes.
    future = [0.0] * (horizon + 1)
    for offset in range(horizon - 1, 0, -1):
        suffix = _fresh_plan_costs(
            driver_json, car_json, track_json,
            json.dumps(surfaces[offset].model_dump(), sort_keys=True),
            tires_json, current_lap + offset, physical_total_laps,
        )
        future[offset] = service + min(cost for _, cost, _ in suffix)
    first = []
    for compound, tire in tires.items():
        stint, candidate, first_lap = 0.0, inf, 0.0
        for offset, surface in enumerate(surfaces):
            if surface.tire_mismatch(compound) == "critical":
                break
            running = _running(simulator, driver, car, track, tire, surface,
                               current_lap + offset, offset,
                               physical_total_laps=physical_total_laps)
            if offset == 0:
                first_lap = running
            stint += running
            candidate = min(candidate, stint + future[offset + 1])
        first.append((compound, candidate, first_lap))
    return tuple(first)


@lru_cache(maxsize=4096)
def _retained_costs(driver_json, car_json, track_json, weather_json, tire_json,
                    tire_age, current_lap, traffic_possible, physical_total_laps, tires_json):
    """Retain while safe, paying for required future changes before running.

    This is a feasible waiting policy, not an optimal delayed-stop plan. Its
    future stops receive no current queue or neutralization discount.
    """
    driver = Driver.model_validate_json(driver_json)
    car = Car.model_construct(**json.loads(car_json))
    track = Track.model_validate_json(track_json)
    weather = Weather.model_validate_json(weather_json)
    tire = Tire.model_validate_json(tire_json)
    simulator = LapSimulator(np.random.default_rng(0))
    total, first = 0.0, 0.0
    for offset, surface in enumerate(_surface_path(weather, track.total_laps - current_lap + 1)):
        if surface.tire_mismatch(tire.compound) == "critical":
            # A currently critical set is handled before this projection.
            # Each recursive replacement is noncritical, so its next stop
            # must occur at a strictly later lap.
            required = surface.fresh_rain_compound()
            candidates = {required} if required is not None else {
                TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD,
            }
            fresh = {TireCompound(key): value for key, value in json.loads(tires_json).items()}
            suffix = min((
                _retained_costs(
                    driver_json, car_json, track_json,
                    json.dumps(surface.model_dump(), sort_keys=True),
                    json.dumps(fresh[compound], sort_keys=True), 0, current_lap + offset,
                    traffic_possible, physical_total_laps, tires_json,
                )[0] for compound in candidates
                if surface.tire_mismatch(compound) != "critical"
            ), default=inf)
            total += track.pit_lane_delta + expected_stationary_time(car) + suffix
            break
        running = _running(simulator, driver, car, track, tire, surface,
                           current_lap + offset, tire_age + offset,
                           gap=0.0 if traffic_possible else None,
                           physical_total_laps=physical_total_laps)
        if offset == 0:
            first = running
        total += running
    return total, first


def weather_stop_costs(
    driver: Driver, car: Car, track: Track, weather: Weather, current_tire: Tire,
    tire_age: int, current_lap: int, *, pit_lane_factor: float = 1.0,
    additional_current_stop_cost: float = 0.0, current_lap_time_modifier: float = 1.0,
    active_aero_enabled: bool = True, traffic_possible: bool = True,
    physical_total_laps: int | None = None,
) -> WeatherStopCosts:
    """Compare retaining while safe with an optimistic schedule of paid refits.

    Rainfall/condition stay fixed while the shared surface model evolves. Later
    refits may use any noncritical fresh set without budget or compound-rule
    constraints, but pay the full expected stop cost. The waiting alternative
    retains each set until it becomes critical, then pays for an appropriate
    fresh set before running that lap. Only a currently critical set bypasses
    the veto. This is a cost bound under projected weather, not a forecast.
    The track bounds the planning horizon; physical_total_laps preserves the
    original fuel schedule when a time limit shortens that horizon.
    """
    physical_total_laps = track.total_laps if physical_total_laps is None else physical_total_laps
    horizon = track.total_laps - current_lap + 1
    if horizon <= 0:
        raise ValueError("current_lap must not exceed the race distance")
    weather_json = json.dumps(weather.model_dump(), sort_keys=True)
    if weather.tire_mismatch(current_tire.compound) == "critical":
        return WeatherStopCosts(0.0, inf)
    clean = driver.model_copy(deep=True)
    clean.reset_race_state()
    # Names and identifiers do not enter lap or service physics. Normalize
    # only those fields so equivalent entrants can share immutable plans.
    clean.id = clean.name = clean.team_id = "projection"
    clean_car = car.model_copy(update={"team_id": "projection", "team_name": "projection"})
    snapshots = (clean.model_dump(), clean_car.model_dump(), track.model_dump(),
                 {compound.value: tire.model_dump() for compound, tire in TIRE_COMPOUNDS.items()})
    driver_json, car_json, track_json, tires_json = (
        json.dumps(value, sort_keys=True) for value in snapshots
    )
    fresh = _fresh_plan_costs(
        driver_json, car_json, track_json, weather_json,
        tires_json, current_lap, physical_total_laps,
    )
    simulator = LapSimulator(np.random.default_rng(0))
    stay, stay_first = _retained_costs(
        driver_json, car_json, track_json, weather_json,
        json.dumps(current_tire.model_dump(), sort_keys=True), tire_age, current_lap,
        traffic_possible, physical_total_laps, tires_json,
    )
    if current_lap_time_modifier != 1.0 or not active_aero_enabled:
        actual_stay_first = _running(
            simulator, clean, car, track, current_tire, weather, current_lap, tire_age,
            gap=0.0 if traffic_possible else None, aero=active_aero_enabled,
            physical_total_laps=physical_total_laps,
        )
        stay += actual_stay_first * current_lap_time_modifier - stay_first
    required = weather.fresh_rain_compound()
    candidates = {required} if required is not None else {
        TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD,
    }
    pit = inf
    for compound, cost, baseline_first in fresh:
        if compound not in candidates:
            continue
        actual_first = baseline_first if active_aero_enabled else _running(
            simulator, clean, car, track, TIRE_COMPOUNDS[compound],
            weather, current_lap, 0, aero=False, physical_total_laps=physical_total_laps,
        )
        pit = min(pit, cost - baseline_first + actual_first * current_lap_time_modifier)
    pit += (track.pit_lane_delta * pit_lane_factor + expected_stationary_time(car)
            + additional_current_stop_cost)
    return WeatherStopCosts(pit, stay)
