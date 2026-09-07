"""Deterministic remaining-race dry tyre and pit-cost planning.

Future stops assume green running. Each action includes this lap, and at
least one lap must be driven on a set before another stop. Traffic, future
weather and tyre inventory are deliberately outside this projection.
"""

from dataclasses import dataclass
from functools import lru_cache
from math import erf, exp, inf, pi, sqrt
from types import SimpleNamespace

import numpy as np

from f1sim.models import Car, Driver, Tire, Track
from f1sim.models.tire import TIRE_COMPOUNDS, TireCompound
from f1sim.simulation.lap import LapSimulator

SLICKS = (TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD)


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
    weather exemption. Until compliance, every dry stop adds an unused set.
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
                if mask.bit_count() < 2 and mask & (1 << c):
                    continue
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
                  current_lap_time_modifier: float = 1.0) -> DryPitDecision:
    """Compare legal stop/wait plans, neutralizing only this lap's running cost."""
    if remaining_laps < 1 or remaining_stops < 0 or remaining_stops > 3:
        raise ValueError("Positive remaining laps and zero to three stops are required")
    physics = (track.base_lap_time, driver.tire_management,
               car.tire_degradation_factor, track.tire_stress)
    horizon = max(track.total_laps, remaining_laps)
    tire_keys = tuple(_tire_key(TIRE_COMPOUNDS[c]) for c in SLICKS)
    green_cost = track.pit_lane_delta + expected_stationary_time(car)
    # Keep all ordinary budgets together so a driver reuses one table after
    # each stop, rather than evicting other drivers with separate budget keys.
    costs, compounds = _fresh_tables(physics, tire_keys, horizon, green_cost)
    mask = 7 if wet_exemption else sum(1 << i for i, c in enumerate(SLICKS)
                                     if c in used_compounds or c == current_tire.compound)
    # Preserve custom current sets; fresh replacements use the configured sets.
    current_curve = _pace_curve(*physics, _tire_key(current_tire),
                                max(horizon, tire_age + remaining_laps))
    old_cost = np.cumsum(current_curve[tire_age:tire_age + remaining_laps])
    wait_cost = float(old_cost[-1]) if mask.bit_count() >= 2 else inf
    if remaining_laps > 1 and remaining_stops:
        wait_cost = min(wait_cost, float(np.min(
            old_cost[:-1] + costs[remaining_stops, mask, remaining_laps - 1:0:-1]
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
                if mask.bit_count() < 2 and mask & (1 << candidate):
                    continue
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
    # A committed teammate affects only this stop, never cached future plans.
    pit_now_cost += track.pit_lane_delta * (pit_lane_factor - 1) + additional_current_stop_cost
    return DryPitDecision(pit_now_cost, wait_cost, SLICKS[c] if c >= 0 else None)
