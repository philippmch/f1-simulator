"""Deterministic remaining-race dry tyre and pit-cost planning.

Future stops assume green running. Each action includes this lap, and at
least one lap must be driven on a set before another stop. Only the current
lap uses supplied traffic gaps; future traffic, weather and tyre inventory
are deliberately outside this projection.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from math import erf, exp, inf, pi, sqrt
from types import SimpleNamespace

import numpy as np

from f1sim.models import Car, Driver, Tire, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS, TireCompound
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.strategy_traffic import normalize_current_traffic_gaps

SLICKS = (TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD)


class _ScaledDryWeather(Weather):
    pace_scale: float = 1.0

    def lap_time_multiplier(self) -> float:
        return self.pace_scale


def _full_row(driver, car, track, tire, age, lap, physical, scale, aero=True, gap=None):
    driver = driver.model_copy(deep=True)
    simulator = LapSimulator(np.random.default_rng(0))
    weather = _ScaledDryWeather(pace_scale=scale)
    result = []
    for number in range(lap, track.total_laps + 1):
        driver.current_tire_laps = age + number - lap
        result.append(simulator.calculate_lap_time(
            driver, car, track, tire, weather, number, physical,
            active_aero_enabled=aero, sample_variation=False,
            gap_to_car_ahead=gap if number == lap else None,
        ))
    return result


@lru_cache(maxsize=32)
def _floor_tables(models, fresh, physical, scale):
    """Absolute clean-air costs for suffixes with lap-dependent fuel and clipping."""
    driver = Driver.model_validate_json(models[0])
    car = Car.model_construct(**json.loads(models[1]))
    track = Track.model_validate_json(models[2])
    end = track.total_laps
    prefixes = {}
    for lap in range(1, end + 1):
        for c, tire_json in enumerate(fresh):
            prefixes[lap, c] = np.cumsum(_full_row(
                driver, car, track, Tire.model_validate_json(tire_json),
                0, lap, physical, scale,
            ))
    costs = np.full((4, 8, end + 2), inf)
    green = track.pit_lane_delta + expected_stationary_time(car)
    for budget in range(1, 4):
        for lap in range(end, 0, -1):
            for mask in range(8):
                for c in range(3):
                    next_mask = mask | (1 << c)
                    prefix = prefixes[lap, c]
                    best = float(prefix[-1]) if next_mask.bit_count() >= 2 else inf
                    if budget > 1 and lap < end:
                        best = min(best, float(np.min(
                            prefix[:-1] + costs[budget - 1, next_mask, lap + 1:end + 1]
                        )))
                    costs[budget, mask, lap] = min(costs[budget, mask, lap], green + best)
    costs.flags.writeable = False
    for prefix in prefixes.values():
        prefix.flags.writeable = False
    return costs, prefixes


def _floor_plan(driver, car, track, tire, age, lap, budget, mask,
                physical, scale, aero, modifier, lane, queue, gaps=None):
    projection = driver.model_copy(deep=True)
    projection.reset_race_state()
    projection.id = projection.name = projection.team_id = "projection"
    package = car.model_copy(deep=True)
    package.team_id = package.team_name = "projection"
    models = (projection.model_dump_json(), package.model_dump_json(), track.model_dump_json())
    fresh = tuple(TIRE_COMPOUNDS[c].model_dump_json() for c in SLICKS)
    costs, prefixes = _floor_tables(models, fresh, physical, scale)
    wait_mask = mask | (1 << SLICKS.index(tire.compound)) if tire.compound in SLICKS else mask
    row = _full_row(projection, car, track, tire, age, lap, physical, scale)
    old = np.cumsum(row)
    wait = float(old[-1]) if wait_mask.bit_count() >= 2 else inf
    if budget and lap < track.total_laps:
        wait = min(wait, float(np.min(
            old[:-1] + costs[budget, wait_mask, lap + 1:track.total_laps + 1]
        )))
    def first(set_tire, set_age, gap):
        return _full_row(projection, car, track, set_tire, set_age,
                         lap, physical, scale, aero, gap)[0] * modifier
    wait += first(tire, age, gaps[0] if gaps else None) - row[0]
    pit, selected = inf, None
    if budget:
        for c, compound in enumerate(SLICKS):
            next_mask = mask | (1 << c)
            prefix = prefixes[lap, c]
            best = float(prefix[-1]) if next_mask.bit_count() >= 2 else inf
            if budget > 1 and lap < track.total_laps:
                best = min(best, float(np.min(
                    prefix[:-1] + costs[budget - 1, next_mask, lap + 1:track.total_laps + 1]
                )))
            candidate = (best + first(TIRE_COMPOUNDS[compound], 0, gaps[1] if gaps else None)
                         - prefix[0]
                         + track.pit_lane_delta * lane + expected_stationary_time(car) + queue)
            if candidate < pit:
                pit, selected = candidate, compound
    return DryPitDecision(pit, wait, selected)


def expected_stationary_time(car: Car) -> float:
    """Exact mean of execution's clipped normal plus 5% uniform slow stop."""
    return _expected_service(car.pit_stop_avg, car.pit_stop_std)


@lru_cache(maxsize=128)
def _expected_service(mean: float, std: float) -> float:
    floor = 1.8

    def positive_mean(x: float) -> float:
        if std == 0:
            return max(0.0, x)
        z = x / std
        return x * (1 + erf(z / sqrt(2))) / 2 + std * exp(-z * z / 2) / sqrt(2 * pi)

    def primitive(x: float) -> float:
        if std == 0:
            return max(0.0, x) ** 2 / 2
        z = x / std
        return ((x * x + std * std) * (1 + erf(z / sqrt(2))) / 2
                + x * std * exp(-z * z / 2) / sqrt(2 * pi)) / 2

    x = mean - floor
    return floor + 0.95 * positive_mean(x) + 0.05 * (primitive(x + 8) - primitive(x + 2)) / 6


def _tire_key(tire: Tire) -> tuple:
    """Every tyre field used by the shared dry pace calculation."""
    return (tire.compound, tire.initial_grip, tire.degradation_rate,
            tire.cliff_threshold, tire.cliff_multiplier)


@lru_cache(maxsize=512)
def _pace_curve(base: float, management: float, degradation: float, stress: float,
                tire_key: tuple, horizon: int) -> tuple[float, ...]:
    compound, grip, rate, cliff, multiplier = tire_key
    tire = Tire(compound=compound, initial_grip=grip, degradation_rate=rate,
                cliff_threshold=cliff, cliff_multiplier=multiplier)
    driver = SimpleNamespace(tire_management=management)
    car = SimpleNamespace(tire_degradation_factor=degradation)
    track = SimpleNamespace(base_lap_time=base, tire_stress=stress)
    return tuple(LapSimulator.tire_pace_contribution(driver, car, track, tire, age)
                 for age in range(horizon))


@lru_cache(maxsize=64)
def _fresh_tables(physics: tuple, tire_keys: tuple, horizon: int,
                  green_cost: float) -> tuple[np.ndarray, np.ndarray]:
    """Cost/compound when buying a fresh set now, indexed stops, used mask, laps.

    A mask with two bits satisfies the dry rule. Mask 7 also represents a
    weather exemption. Intermediate repeats are legal if the eventual finish
    uses two compounds; terminal feasibility enforces that requirement.
    The cache is bounded and its keys contain every used tyre/pace parameter.
    No simulation RNG is consulted.
    """
    curves = tuple(_pace_curve(*physics, key, horizon) for key in tire_keys)
    max_stops = 3
    prefix = np.pad(np.cumsum(np.asarray(curves), axis=1), ((0, 0), (1, 0)))
    costs = np.full((max_stops + 1, 8, horizon + 1), inf)
    compounds = np.full(costs.shape, -1, dtype=np.int8)
    for stops in range(1, max_stops + 1):
        for mask in range(8):
            for c in range(3):
                next_mask = mask | (1 << c)
                for laps in range(1, horizon + 1):
                    best = prefix[c, laps] if next_mask.bit_count() >= 2 else inf
                    if stops > 1 and laps > 1:
                        best = min(best, float(np.min(
                            prefix[c, 1:laps] + costs[stops - 1, next_mask, laps - 1:0:-1]
                        )))
                    candidate = green_cost + best
                    if candidate < costs[stops, mask, laps]:
                        costs[stops, mask, laps] = candidate
                        compounds[stops, mask, laps] = c
    costs.flags.writeable = False
    compounds.flags.writeable = False
    return costs, compounds


@dataclass(frozen=True)
class DryPitDecision:
    pit_now_cost: float
    wait_cost: float
    compound: TireCompound | None

    def should_pit(self, timing_bias: float = 0.0) -> bool:
        """Style can move a near tie by at most 0.1 seconds total."""
        return self.compound is not None and self.pit_now_cost < (
            self.wait_cost + max(-0.1, min(0.1, timing_bias))
        )


def plan_dry_stop(driver: Driver, car: Car, track: Track, current_tire: Tire,
                  tire_age: int, remaining_laps: int, remaining_stops: int,
                  used_compounds: set[TireCompound], wet_exemption: bool = False,
                  pit_lane_factor: float = 1.0,
                  additional_current_stop_cost: float = 0.0,
                  current_lap_time_modifier: float = 1.0,
                  tire_pace_multiplier: float = 1.0,
                  physical_total_laps: int | None = None,
                  active_aero_enabled: bool = True, *,
                  current_traffic_gaps: tuple[float | None, float | None] | None = None,
                  ) -> DryPitDecision:
    """Compare legal plans using tyre-relative, or floor-clipped absolute, costs."""
    gaps = normalize_current_traffic_gaps(current_traffic_gaps)
    if remaining_laps < 1 or remaining_stops < 0 or remaining_stops > 3:
        raise ValueError("Positive remaining laps and zero to three stops are required")
    physical = track.total_laps if physical_total_laps is None else physical_total_laps
    lap = track.total_laps - remaining_laps + 1
    if lap < 1 or physical < track.total_laps:
        raise ValueError("Planning laps must fit the original physical race distance")
    mask = 7 if wet_exemption else sum(1 << i for i, c in enumerate(SLICKS)
                                     if c in used_compounds
                                     or (tire_age > 0 and c == current_tire.compound))
    # Fresh soft at the lightest projected fuel load bounds all slick pace.
    # If even it stays above the floor, common full-lap terms still cancel.
    fastest = _full_row(driver, car, track, TIRE_COMPOUNDS[TireCompound.SOFT],
                        0, track.total_laps, physical, tire_pace_multiplier)[0]
    if fastest <= track.base_lap_time * 0.95:
        return _floor_plan(driver, car, track, current_tire, tire_age, lap,
                           remaining_stops, mask, physical, tire_pace_multiplier,
                           active_aero_enabled, current_lap_time_modifier,
                           pit_lane_factor, additional_current_stop_cost, gaps)
    # Both compound pace and degradation are linear in reference lap time.
    # Scale only the private tyre-physics key, never the actual track or pit
    # costs. This also isolates differently scaled curves in existing caches.
    physics = (track.base_lap_time * tire_pace_multiplier, driver.tire_management,
               car.tire_degradation_factor, track.tire_stress)
    horizon = max(track.total_laps, remaining_laps)
    tire_keys = tuple(_tire_key(TIRE_COMPOUNDS[c]) for c in SLICKS)
    green_cost = track.pit_lane_delta + expected_stationary_time(car)
    # Keep all ordinary budgets together so a driver reuses one table after
    # each stop, rather than evicting other drivers with separate budget keys.
    costs, compounds = _fresh_tables(physics, tire_keys, horizon, green_cost)
    mask = 7 if wet_exemption else sum(1 << i for i, c in enumerate(SLICKS)
                                     if c in used_compounds
                                     or (tire_age > 0 and c == current_tire.compound))
    wait_mask = mask
    if current_tire.compound in SLICKS:
        wait_mask |= 1 << SLICKS.index(current_tire.compound)
    # Preserve custom current sets; fresh replacements use the configured sets.
    current_curve = _pace_curve(*physics, _tire_key(current_tire),
                                max(horizon, tire_age + remaining_laps))
    old_cost = np.cumsum(current_curve[tire_age:tire_age + remaining_laps])
    wait_cost = float(old_cost[-1]) if wait_mask.bit_count() >= 2 else inf
    if remaining_laps > 1 and remaining_stops:
        wait_cost = min(wait_cost, float(np.min(
            old_cost[:-1] + costs[remaining_stops, wait_mask, remaining_laps - 1:0:-1]
        )))
    c = int(compounds[remaining_stops, mask, remaining_laps])
    pit_now_cost = float(costs[remaining_stops, mask, remaining_laps])
    if current_lap_time_modifier != 1.0:
        # Every wait path drives this same old-set lap before green futures.
        wait_cost += (current_lap_time_modifier - 1) * current_curve[tire_age]
        # Re-rank each possible first set before selecting it: the best green
        # compound need not remain best when only its first lap is slowed.
        pit_now_cost, c = inf, -1
        if remaining_stops:
            for candidate, key in enumerate(tire_keys):
                next_mask = mask | (1 << candidate)
                curve = _pace_curve(*physics, key, horizon)
                prefix = np.cumsum(curve[:remaining_laps])
                best = float(prefix[-1]) if next_mask.bit_count() >= 2 else inf
                if remaining_stops > 1 and remaining_laps > 1:
                    best = min(best, float(np.min(
                        prefix[:-1]
                        + costs[remaining_stops - 1, next_mask, remaining_laps - 1:0:-1]
                    )))
                adjusted = green_cost + best + (current_lap_time_modifier - 1) * curve[0]
                if adjusted < pit_now_cost:
                    pit_now_cost, c = adjusted, candidate
    if gaps is not None:
        wait_cost += (LapSimulator.traffic_pace_contribution(gaps[0])
                      * tire_pace_multiplier * current_lap_time_modifier)
        pit_now_cost += (LapSimulator.traffic_pace_contribution(gaps[1])
                         * tire_pace_multiplier * current_lap_time_modifier)
    # A committed teammate affects only this stop, never cached future plans.
    pit_now_cost += track.pit_lane_delta * (pit_lane_factor - 1) + additional_current_stop_cost
    return DryPitDecision(pit_now_cost, wait_cost, SLICKS[c] if c >= 0 else None)
