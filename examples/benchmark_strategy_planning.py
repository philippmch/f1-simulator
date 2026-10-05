"""Measure full-race strategy workloads with reproducible synthetic inputs.

Print timings and a digest of race, qualifying, weather and event results.
Compare digests before interpreting speed changes. First-trial timing includes
cold strategy caches; later trials reuse the same process with consecutive seeds.
Use --inventory finite for reusable soft@5, hard@0 and intermediate@4 sets,
and --opening automatic to include native opening selection in the workload.
Use --inventory expanded for seven fresh physical sets and --scenario scheduled
for prescribed rain arriving and clearing during the race.
Use --saved-inputs PATH to benchmark validated replay inputs with their saved
seed, engine and trial count instead of the synthetic options.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from math import isfinite
from numbers import Real
from pathlib import Path
from statistics import mean
from time import perf_counter

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.analysis.provenance import simulation_runtime
from f1sim.analysis.replay import _load_saved_runner
from f1sim.models import Car, Driver, Track, Weather, WeatherCondition
from f1sim.models._native import native_physics

SCENARIOS = {
    "dry": (0, 0, "soft"),
    "steady_damp": (.1, .1, "soft"),
    "drying": (.19, 0, "soft"),
    "wetting": (.18, .35, "soft"),
    "rain_transition": (.23, 0, "intermediate"),
    "scheduled": (.2, .2, "soft"),
}


def _time_trials(runner, trials):
    seed = runner.base_seed
    times, outputs = [], []
    for trial in range(trials):
        runner.base_seed = seed + trial
        start = perf_counter()
        result = runner.run(1, parallel=False)
        times.append(perf_counter() - start)
        outputs.append({
            "race": [asdict(row) for row in result.race_results[0]],
            "qualifying": [asdict(row) for row in result.qualifying_results[0]],
            "weather": result.weather_histories,
            "events": asdict(result.event_stats),
        })
    encoded = json.dumps(outputs, sort_keys=True, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    return {
        "trial_seconds": times, "first_trial_seconds": times[0],
        "later_trial_mean_seconds": mean(times[1:]) if trials > 1 else None,
        "total_simulation_seconds": sum(times),
        "outcome_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def benchmark_saved(path):
    """Validate a saved replay and measure its complete consecutive trial outputs."""
    path = Path(path)
    before = path.read_bytes()
    runner, trials = _load_saved_runner(path)
    if trials > 100:
        raise ValueError("Saved benchmark must contain at most 100 trials")
    models = (*runner.drivers, *runner.cars.values(), runner.track, runner.weather)
    seed = runner.base_seed
    native_before = native_physics(*models)
    measurements = _time_trials(runner, trials)
    if path.read_bytes() != before:
        raise ValueError("Saved benchmark inputs changed while running")
    return {
        "benchmark_version": 1, "input_basis": "validated_saved_replay",
        "input_sha256": hashlib.sha256(before).hexdigest(),
        "engine": runner.race_engine, "trials": trials, "seed": seed,
        "drivers": len(runner.drivers), "laps": runner.track.total_laps,
        "runtime": simulation_runtime(),
        "native_physics_before": native_before,
        "native_physics_after": native_physics(*models),
        **measurements,
    }


def benchmark(engine="chronological", scenario="steady_damp", trials=3, drivers=22,
              laps=53, seed=42, inventory="unlimited", opening="explicit",
              change_probability=0.0):
    """Run controlled synthetic races without network access or writing files."""
    for value, low, high, name in ((trials, 1, 100, "trials"), (drivers, 1, 22, "drivers"),
                                  (laps, 2, 100, "laps"), (seed, 0, 2**32 - 1, "seed")):
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name} must be an integer from {low} through {high}")
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {tuple(SCENARIOS)}")
    if inventory not in ("unlimited", "finite", "expanded"):
        raise ValueError("inventory must be unlimited, finite or expanded")
    if opening not in ("explicit", "automatic"):
        raise ValueError("opening must be explicit or automatic")
    if scenario == "scheduled" and laps < 3:
        raise ValueError("scheduled weather requires at least three laps")
    if (isinstance(change_probability, bool) or not isinstance(change_probability, Real)
            or not 0 <= change_probability <= 1 or not isfinite(change_probability)):
        raise ValueError("change_probability must be a finite real number from 0 through 1")
    change_probability = float(change_probability)
    roster = [Driver(id=f"D{i:02}", name=f"Driver {i}", team_id=f"T{i // 2}",
                     skill_rating=.8 + i * .008, tire_management=.8 + i * .008)
              for i in range(drivers)]
    cars = {f"T{i}": Car(team_id=f"T{i}", team_name=f"Team {i}",
                         base_pace=.8 + i * .015, tire_degradation_factor=.9 + i * .04)
            for i in range((drivers + 1) // 2)}
    wetness, rain, compound = SCENARIOS[scenario]
    records = [{"id": "S", "compound": "soft", "age": 5},
               {"id": "H", "compound": "hard", "age": 0},
               {"id": "I", "compound": "intermediate", "age": 4}]
    if inventory == "expanded":
        records = [{"id": key, "compound": name, "age": 0} for key, name in (
            ("M1", "medium"), ("M2", "medium"), ("S", "soft"), ("H", "hard"),
            ("I1", "intermediate"), ("I2", "intermediate"), ("W", "wet"),
        )]
    pools = ({driver.id: [dict(item) for item in records] for driver in roster}
             if inventory != "unlimited" else None)
    weather_schedule = ([{"lap": max(2, laps // 3), "rain_intensity": .5},
                         {"lap": max(3, laps * 2 // 3), "rain_intensity": 0.}]
                        if scenario == "scheduled" else None)
    starting_tires = starting_ages = None
    if opening == "explicit":
        starting_tires = {driver.id: compound for driver in roster}
        age = next(item["age"] for item in records if item["compound"] == compound)
        starting_ages = {driver.id: age if pools else (i % 3) * 6
                         for i, driver in enumerate(roster)}
    runner = MonteCarloRunner(
        roster, cars, Track(id="benchmark", name="Synthetic", country="Synthetic",
                            total_laps=laps, base_lap_time=90),
        Weather(condition=WeatherCondition.CLOUDY, track_wetness=wetness,
                rain_intensity=rain, change_probability=change_probability),
        race_engine=engine, seed=seed, starting_tires=starting_tires,
        starting_tire_ages=starting_ages, tire_inventory=pools,
        weather_schedule=weather_schedule,
    )
    measurements = _time_trials(runner, trials)
    return {
        "benchmark_version": 4, "engine": engine, "scenario": scenario,
        "change_probability": change_probability,
        "weather_schedule": weather_schedule,
        "inventory": inventory, "opening": opening, "tire_inventory": pools,
        "starting_tires": starting_tires, "starting_tire_ages": starting_ages,
        "drivers": drivers, "laps": laps, "trials": trials, "seed": seed,
        **measurements,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("standard", "chronological"),
                        default="chronological")
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), default="steady_damp")
    parser.add_argument("--inventory", choices=("unlimited", "finite", "expanded"),
                        default="unlimited")
    parser.add_argument("--opening", choices=("explicit", "automatic"), default="explicit")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--drivers", type=int, default=22)
    parser.add_argument("--laps", type=int, default=53)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--change-probability", type=float, default=0.0)
    parser.add_argument("--saved-inputs", type=Path,
                        help="Use the saved replay seed, engine and trial count")
    args = parser.parse_args()
    try:
        options = vars(args)
        path = options.pop("saved_inputs")
        result = benchmark_saved(path) if path is not None else benchmark(**options)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
