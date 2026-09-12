"""Measure full-race strategy workloads with reproducible synthetic inputs.

Print timings and a digest of race, qualifying, weather and event results.
Compare digests before interpreting speed changes. First-trial timing includes
cold strategy caches; later trials reuse the same process with consecutive seeds.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from statistics import mean
from time import perf_counter

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.models import Car, Driver, Track, Weather, WeatherCondition

SCENARIOS = {
    "steady_damp": (.1, .1, "soft"),
    "drying": (.19, 0, "soft"),
    "wetting": (.18, .35, "soft"),
    "rain_transition": (.23, 0, "intermediate"),
}


def benchmark(engine="chronological", scenario="steady_damp", trials=3, drivers=22,
              laps=53, seed=42):
    """Run controlled synthetic races without network access or writing files."""
    for value, low, high, name in ((trials, 1, 100, "trials"), (drivers, 1, 22, "drivers"),
                                  (laps, 2, 100, "laps"), (seed, 0, 2**32 - 1, "seed")):
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name} must be an integer from {low} through {high}")
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {tuple(SCENARIOS)}")
    roster = [Driver(id=f"D{i:02}", name=f"Driver {i}", team_id=f"T{i // 2}",
                     skill_rating=.8 + i * .008, tire_management=.8 + i * .008)
              for i in range(drivers)]
    cars = {f"T{i}": Car(team_id=f"T{i}", team_name=f"Team {i}",
                         base_pace=.8 + i * .015, tire_degradation_factor=.9 + i * .04)
            for i in range((drivers + 1) // 2)}
    wetness, rain, compound = SCENARIOS[scenario]
    runner = MonteCarloRunner(
        roster, cars, Track(id="benchmark", name="Synthetic", country="Synthetic",
                            total_laps=laps, base_lap_time=90),
        Weather(condition=WeatherCondition.CLOUDY, track_wetness=wetness,
                rain_intensity=rain, change_probability=0),
        race_engine=engine, seed=seed, starting_tires={driver.id: compound for driver in roster},
        starting_tire_ages={driver.id: (i % 3) * 6 for i, driver in enumerate(roster)},
    )
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
        "benchmark_version": 1, "engine": engine, "scenario": scenario,
        "drivers": drivers, "laps": laps, "trials": trials, "seed": seed,
        "trial_seconds": times, "first_trial_seconds": times[0],
        "later_trial_mean_seconds": mean(times[1:]) if trials > 1 else None,
        "total_simulation_seconds": sum(times),
        "outcome_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("standard", "chronological"),
                        default="chronological")
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), default="steady_damp")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--drivers", type=int, default=22)
    parser.add_argument("--laps", type=int, default=53)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        result = benchmark(**vars(args))
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
