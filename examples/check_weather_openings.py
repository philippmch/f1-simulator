"""Compare automatic weather openings with every safe explicit opening.

Synthetic single-car races use fixed rainfall, evolving surface water, mean
lap pace, expected service and no incidents. Each explicit opening runs the
actual later pit policy for the same eight private reaction seeds used by the
opening comparison. This validates the conditional model, not real-race pace
or globally optimal strategy under traffic and unknown future weather.
"""

import argparse
import json

import numpy as np

from f1sim.models import Car, Driver, TireCompound, Track, Weather, WeatherCondition
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import DriverStatus, RaceSimulator, TeamStrategyArchetype

ENGINES = ("standard", "chronological")
SEEDS = tuple(range(8))
CASES = (
    dict(name="drying_wet_surface", water=.75, rain=0., laps=12, base=90., lane=22.),
    dict(name="wet_boundary", water=.72, rain=.7, laps=12, base=90., lane=22.),
    dict(name="light_rain", water=.45, rain=.35, laps=12, base=90., lane=22.),
    dict(name="damp_surface", water=.1, rain=.1, laps=10, base=90., lane=22.),
    dict(name="heavy_rain", water=.85, rain=.85, laps=30, base=90., lane=22.),
    dict(name="warmed_drying_surface", water=.75, rain=0., laps=12, base=90., lane=22.,
         warmup={"intermediate": 8., "wet": 35., "soft": 2.}),
    dict(name="timed_wet_race", water=.45, rain=.35, laps=10, base=1800., lane=22.),
)


def weather_for(case):
    condition = (WeatherCondition.CLOUDY if not case["rain"] else
                 WeatherCondition.HEAVY_RAIN if case["rain"] > .7 else WeatherCondition.LIGHT_RAIN)
    return Weather(condition=condition, track_wetness=case["water"],
                   rain_intensity=case["rain"], change_probability=0)


def run_race(case, engine, compound=None, seed=0):
    if engine not in ENGINES:
        raise ValueError(f"engine must be drawn from {ENGINES}")
    driver = Driver(id="A", name="Synthetic", team_id="A")
    car = Car(team_id="A", team_name="Synthetic")
    track = Track(id="opening", name="Synthetic", country="Synthetic",
                  total_laps=case["laps"], base_lap_time=case["base"], pit_lane_delta=case["lane"])
    simulator = RaceSimulator(np.random.default_rng(seed), tire_warmup=case.get("warmup"))
    simulator._infer_team_strategy = lambda *args: TeamStrategyArchetype.BALANCED
    simulator.event_manager.process_lap = lambda *args, **kwargs: []
    simulator.event_manager._check_mechanical_failure = lambda *args, **kwargs: None
    simulator.event_manager._check_random_incident = lambda *args, **kwargs: None
    calculate = simulator.lap_simulator.calculate_lap_time
    fuel_distances = []
    warmup_laps = []
    consume_warmup = simulator._consume_tire_warmup

    def record_warmup(state):
        seconds = consume_warmup(state)
        if seconds > 0:
            warmup_laps.append(dict(lap=state.laps_completed + 1, seconds=seconds))
        return seconds

    simulator._consume_tire_warmup = record_warmup

    def mean_lap(*args, **kwargs):
        tire = kwargs.get("tire", args[3] if args else None)
        weather = kwargs.get("weather", args[4] if args else None)
        if weather.tire_mismatch(tire.compound) == "critical":
            raise AssertionError("Automatic policy ran a critically unsuitable set")
        fuel_distances.append(kwargs.get("total_laps", args[6] if args else None))
        kwargs["sample_variation"] = False
        return calculate(*args, **kwargs)

    simulator.lap_simulator.calculate_lap_time = mean_lap
    simulator.lap_simulator.calculate_pit_stop_time = expected_stationary_time
    execute = (simulator.simulate_race if engine == "standard"
               else ChronologicalRace(simulator).run)
    options = {"starting_tires": {"A": compound}} if compound is not None else {}
    result, = execute([driver], {"A": car}, track, weather_for(case), ["A"], **options)
    if result.status != DriverStatus.FINISHED:
        raise AssertionError("Opening policy did not finish legally")
    if set(fuel_distances) != {track.total_laps}:
        raise AssertionError("Opening comparison changed the physical fuel distance")
    return dict(total_seconds=result.total_time, laps_completed=result.laps_completed,
                pit_laps=result.pit_laps, compounds=result.strategy,
                race_time_limited=result.race_time_limited, warmup_laps=warmup_laps)


def compare_openings(cases=CASES, engines=ENGINES):
    rows = []
    for case in cases:
        weather = weather_for(case)
        candidates = tuple(compound for compound in TireCompound
                           if weather.tire_mismatch(compound) != "critical")
        for engine in engines:
            selected = run_race(case, engine)
            scores = {}
            for compound in candidates:
                outcomes = [run_race(case, engine, compound, seed) for seed in SEEDS]
                scores[compound.value] = dict(
                    mean_laps=sum(row["laps_completed"] for row in outcomes) / len(outcomes),
                    mean_seconds=sum(row["total_seconds"] for row in outcomes) / len(outcomes),
                )
            best = min(scores, key=lambda compound: (-scores[compound]["mean_laps"],
                                                    scores[compound]["mean_seconds"]))
            chosen = selected["compounds"][0]
            if chosen not in scores:
                raise AssertionError("Automatic opening was not currently noncritical")
            rows.append(dict(
                case=dict(case), engine=engine, reaction_seeds=list(SEEDS),
                selected=selected, alternatives=scores, best_opening=best,
                mean_distance_gap=scores[chosen]["mean_laps"] - scores[best]["mean_laps"],
                mean_time_gap_seconds=scores[chosen]["mean_seconds"] - scores[best]["mean_seconds"],
            ))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=(*ENGINES, "both"), default="both")
    args = parser.parse_args()
    engines = ENGINES if args.engine == "both" else (args.engine,)
    rows = compare_openings(engines=engines)
    print(json.dumps(rows, indent=2, allow_nan=False))
    return 1 if any(row["mean_distance_gap"] != 0 or
                    abs(row["mean_time_gap_seconds"]) > 1.e-7 for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
