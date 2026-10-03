"""Measure exact retirement planning over distinct one-lap physical tyre sets.

This synthetic decision starts on intermediates in fixed surface water 0.3,
with one permitted lap on each set and a longer scheduled distance. All paths
retire. Compare accepted distance and elapsed time before interpreting speed.
No live data, random laps or sampled pit service enter the benchmark.
"""

import argparse
import hashlib
import json
from math import isfinite
from time import perf_counter

from f1sim.models import Car, Driver, Track, Weather
from f1sim.models._native import native_physics
from f1sim.simulation.inventory_strategy import plan_inventory_strategy
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.tire_inventory import TireInventory


def benchmark(sets=12, laps=30, trials=2, cadence="both"):
    """Return timings and conditional continuation evidence for each clock."""
    for value, low, high, name in ((sets, 2, 20, "sets"), (laps, 3, 100, "laps"),
                                  (trials, 1, 100, "trials")):
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name} must be an integer from {low} through {high}")
    if laps <= sets:
        raise ValueError("laps must exceed sets so every continuation retires")
    if cadence not in {"own", "external", "both"}:
        raise ValueError("cadence must be own, external or both")
    driver = Driver(id="A", name="Synthetic", team_id="A")
    car = Car(team_id="A", team_name="Synthetic")
    track = Track(id="T", name="Synthetic", country="Test", total_laps=laps,
                  base_lap_time=90., pit_lane_delta=20.)
    weather = Weather(track_wetness=.3, rain_intensity=.3, change_probability=0)
    records = [dict(id=f"I{i}", compound="intermediate", age=i, remaining_laps=1)
               for i in range(sets)]
    inventory = TireInventory.from_sets(records)
    inventory.fit("I0")
    stop = track.pit_lane_delta + expected_stationary_time(car)
    clock = StrategyWeatherClock(tuple(90. * i for i in range(laps)), 45., 90., laps, stop, stop)
    outcomes, timings = [], []
    for kind in ("own", "external") if cadence == "both" else (cadence,):
        elapsed, previous = [], None
        for _ in range(trials):
            started = perf_counter()
            result = plan_inventory_strategy(
                driver, car, track, weather, inventory, 1, remaining_stops=0,
                remaining_dry_stops=0, remaining_damp_stops=0,
                weather_clock=clock if kind == "external" else None)
            elapsed.append(perf_counter() - started)
            if previous is not None and result != previous:
                raise RuntimeError("Repeated retirement decisions changed")
            previous = result
        if result.wait_laps != sets or isfinite(result.wait_cost) or result.should_pit():
            raise RuntimeError("The synthetic pool did not preserve its expected retirement")
        outcomes.append(dict(cadence=kind, accepted_laps=result.wait_laps,
                             completion_possible=isfinite(result.wait_cost),
                             retirement_time_seconds=result.wait_partial_time,
                             pit_now_candidate=result.set_id,
                             pit_now_laps=result.pit_now_laps,
                             pit_now_time_seconds=result.pit_now_partial_time,
                             should_pit=result.should_pit()))
        timings.append(dict(cadence=kind, trial_seconds=elapsed))
    encoded = json.dumps(outcomes, sort_keys=True, allow_nan=False).encode("utf-8")
    return dict(benchmark_version=1, synthetic=True, sets=sets, laps=laps, trials=trials,
                native_forecasts=native_physics(driver, car, track, weather),
                tire_inventory=records, outcomes=outcomes, timings=timings,
                outcome_sha256=hashlib.sha256(encoded).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sets", type=int, default=12)
    parser.add_argument("--laps", type=int, default=30)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--cadence", choices=("own", "external", "both"), default="both")
    try:
        result = benchmark(**vars(parser.parse_args()))
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
