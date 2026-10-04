"""Measure one native finite-stock forecast through known SC/VSC intervals.

The synthetic field holds observed rival pace and has no future incidents or
unannounced tyre decisions. This isolates the strategy search from race setup
and RNG draws. Compare outcome digests before interpreting timings. --profile
also records green suffix evaluations for comparisons independent of hardware.
"""

import argparse
import cProfile
import hashlib
import json
from dataclasses import asdict
from math import floor, isfinite
from time import perf_counter

from f1sim.models import Car, Driver, Track, Weather
from f1sim.models._native import native_physics
from f1sim.simulation.chronological_finish import ChronologicalFinishCar, ChronologicalFinishContext
from f1sim.simulation.inventory_strategy import plan_inventory_strategy
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race_timing import RaceFinishTimeline
from f1sim.simulation.strategy_control_clock import StrategyControlContext
from f1sim.simulation.tire_inventory import TireInventory
from f1sim.simulation.weather_schedule import WeatherForecastContext


def benchmark(control="sc", intervals=5, laps=53, drivers=22, profile=False):
    """Return the immutable scenario, exact decision and search measurements."""
    for value, low, high, name in ((intervals, 1, 8, "intervals"), (laps, 8, 100, "laps"),
                                  (drivers, 2, 22, "drivers")):
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name} must be an integer from {low} through {high}")
    if control not in ("sc", "vsc"):
        raise ValueError("control must be sc or vsc")
    if type(profile) is not bool:
        raise ValueError("profile must be a boolean")
    driver = Driver(id="A", name="Synthetic", team_id="T", skill_rating=.832,
                    tire_management=.832)
    car = Car(team_id="T", team_name="Synthetic", base_pace=.83, tire_degradation_factor=.98)
    track = Track(id="audit", name="Synthetic", country="Synthetic", total_laps=laps,
                  base_lap_time=90., pit_lane_delta=22.)
    current_lap = max(3, laps * 2 // 5)
    now = (current_lap - 1) * 105.
    paces = {"A": 105., **{f"B{i:02}": 100. + i * .17 for i in range(drivers - 1)}}
    ledger = RaceFinishTimeline(laps, paces)
    observations = sorted((lap * pace, -lap, identifier)
                          for identifier, pace in paces.items()
                          for lap in range(1, floor(now / pace) + 1))
    for time, negative_lap, identifier in observations:
        lap = -negative_lap
        leading = lap > max(row.completed_laps for row in ledger.states.values())
        ledger.observe_crossing(identifier, lap, time, is_leader=leading)
    rivals = tuple(ChronologicalFinishCar(
        identifier, ledger.states[identifier].completed_laps, pace,
        ledger.states[identifier].last_crossing_time + pace,
        ledger.states[identifier].last_crossing_time, False, index,
    ) for index, (identifier, pace) in enumerate(paces.items()) if identifier != "A")
    order = tuple(sorted(paces, key=lambda identifier: (
        -ledger.states[identifier].completed_laps,
        ledger.states[identifier].last_crossing_time, identifier)))
    leading_lap = max(row.completed_laps for row in ledger.states.values()) + 1
    schedule = [{"lap": leading_lap + 2, "rain_intensity": .5},
                {"lap": max(leading_lap + 3, laps * 2 // 3), "rain_intensity": 0.}]
    forecast = WeatherForecastContext.from_schedule(schedule, leading_lap=leading_lap)
    weather = Weather(track_wetness=.3955712, rain_intensity=.5, change_probability=0.)
    factor, modifier = (.55, 1.4) if control == "sc" else (.75, 1.2)
    context = StrategyControlContext(ChronologicalFinishContext(
        "A", ledger, order, rivals, paces["A"], modifier, control == "sc",
        forecast_context=forecast, control_intervals=intervals,
    ), now, track.pit_lane_delta * factor + expected_stationary_time(car))
    records = [{"id": identifier, "compound": compound} for identifier, compound in (
        ("M1", "medium"), ("M2", "medium"), ("S", "soft"), ("H", "hard"),
        ("I1", "intermediate"), ("I2", "intermediate"), ("W", "wet"),
    )]
    inventory = TireInventory.from_sets(records)
    inventory.fit("M1")
    native = native_physics(driver, car, track, weather)
    profiler = cProfile.Profile() if profile else None
    started = perf_counter()
    if profiler is not None:
        profiler.enable()
    try:
        result = plan_inventory_strategy(
            driver, car, track, weather, inventory, current_lap, tire_age=current_lap - 1,
            remaining_stops=4, remaining_dry_stops=3, remaining_damp_stops=2,
            used_compounds=("medium",), physical_total_laps=laps,
            forecast_context=forecast, control_context=context)
    finally:
        if profiler is not None:
            profiler.disable()
    seconds = perf_counter() - started
    decision = {key: (None if isinstance(value, float) and not isfinite(value) else value)
                for key, value in asdict(result).items()}
    decision["should_pit"] = result.should_pit()
    encoded = json.dumps(decision, sort_keys=True, allow_nan=False).encode()
    green_suffixes = None
    if profiler is not None:
        profiler.create_stats()
        green_suffixes = sum(value[1] for key, value in profiler.stats.items()
                             if key[2] == "plan_inventory_strategy") - 1
    return dict(
        benchmark_version=1, native=native, control=control, intervals=intervals,
        laps=laps, drivers=drivers, current_lap=current_lap, now=now,
        driver=driver.model_dump(mode="json"), car=car.model_dump(mode="json"),
        track=track.model_dump(mode="json"), weather=weather.model_dump(mode="json"),
        weather_schedule=schedule, leading_lap=leading_lap, paces=paces, order=order,
        rivals=[asdict(row) for row in rivals], tire_inventory=records,
        seconds=seconds, profiled=profile, green_suffix_evaluations=green_suffixes,
        decision=decision, outcome_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", choices=("sc", "vsc"), default="sc")
    parser.add_argument("--intervals", type=int, default=5)
    parser.add_argument("--laps", type=int, default=53)
    parser.add_argument("--drivers", type=int, default=22)
    parser.add_argument("--profile", action="store_true")
    try:
        result = benchmark(**vars(parser.parse_args()))
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
