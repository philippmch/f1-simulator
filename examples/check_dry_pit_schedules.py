"""Exhaustive short dry-race diagnostic using actual execution in both engines.

Synthetic eight-lap cases use the balanced policy and start on medium.
Alternatives cover every legal slick
sequence with one to three paid stops after lap one. This is a bounded model
check, not calibrated venue strategy or a global optimum beyond that search.
The deliberately long 600-second reference case exercises the three-stop
budget; it does not represent a real venue. No network access or output files
are needed.
"""

import argparse
import json
from itertools import combinations, product

import numpy as np

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import DriverStatus, RaceSimulator, TeamStrategyArchetype

ENGINES = ("standard", "chronological")
SLICKS = (TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD)
CASES = (
    {"name": "normal_lane", "laps": 8, "base_lap_time": 90,
     "pit_lane_delta": 22, "tire_stress": .7, "degradation": 1.0},
    {"name": "low_lane_high_wear", "laps": 8, "base_lap_time": 300,
     "pit_lane_delta": 1, "tire_stress": 1.0, "degradation": 1.5},
    {"name": "long_reference_high_wear", "laps": 8, "base_lap_time": 600,
     "pit_lane_delta": 1, "tire_stress": 1.0, "degradation": 1.5},
)


def schedules(laps=8):
    """Yield legal (own lap, fresh compound) schedules with up to three stops."""
    if isinstance(laps, bool) or not isinstance(laps, int) or laps < 2:
        raise ValueError("laps must be an integer of at least two")
    for count in range(1, min(3, laps - 1) + 1):
        for stop_laps in combinations(range(2, laps + 1), count):
            for compounds in product(SLICKS, repeat=count):
                if any(compound != TireCompound.MEDIUM for compound in compounds):
                    yield tuple(zip(stop_laps, compounds, strict=True))


def run_race(case, engine, schedule=None):
    """Run the adaptive policy or an explicit schedule with actual lap physics."""
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}")
    driver = Driver(id="synthetic", name="Synthetic", team_id="synthetic")
    car = Car(team_id="synthetic", team_name="Synthetic",
              tire_degradation_factor=case["degradation"])
    track = Track(id="synthetic", name=case["name"], country="Synthetic",
                  total_laps=case["laps"], base_lap_time=case["base_lap_time"],
                  pit_lane_delta=case["pit_lane_delta"], tire_stress=case["tire_stress"])
    simulator = RaceSimulator(rng=np.random.default_rng(42))
    calculate = simulator.lap_simulator.calculate_lap_time

    def mean_lap(*args, **kwargs):
        kwargs["sample_variation"] = False
        return calculate(*args, **kwargs)

    simulator.lap_simulator.calculate_lap_time = mean_lap
    simulator.lap_simulator.calculate_pit_stop_time = expected_stationary_time
    simulator.event_manager.process_lap = lambda *args, **kwargs: []
    simulator.event_manager._check_mechanical_failure = lambda *args, **kwargs: None
    simulator.event_manager._check_random_incident = lambda *args, **kwargs: None
    simulator._infer_team_strategy = lambda *args: TeamStrategyArchetype.BALANCED
    if schedule is not None:
        stops = dict(schedule)
        if (len(stops) != len(schedule) or not 1 <= len(stops) <= 3
                or any(type(lap) is not int or not 2 <= lap <= track.total_laps
                       or compound not in SLICKS for lap, compound in schedule)
                or all(compound == TireCompound.MEDIUM for compound in stops.values())):
            raise ValueError("schedule must contain one to three distinct legal dry stops")
        stops = {lap: TireCompound(compound) for lap, compound in stops.items()}
        simulator._should_pit = lambda state, states, track, lap, *args, **kwargs: lap in stops
        # An explicit proposal is consumed by the normal paid-stop execution.
        execute = simulator._execute_pit_stop

        def forced_stop(state, track, weather, current_lap, **kwargs):
            state.dry_pit_proposal = (current_lap, stops[current_lap])
            return execute(state, track, weather, current_lap, **kwargs)

        simulator._execute_pit_stop = forced_stop
    run = (simulator.simulate_race if engine == "standard"
           else ChronologicalRace(simulator).run)
    result = run([driver], {car.team_id: car}, track, Weather(change_probability=0),
                 [driver.id], starting_tires={driver.id: TireCompound.MEDIUM})[0]
    if (result.status != DriverStatus.FINISHED or result.laps_completed != track.total_laps
            or len(set(result.strategy)) < 2):
        raise AssertionError("Diagnostic race did not complete its legal dry distance")
    return {"laps_completed": result.laps_completed, "pit_laps": result.pit_laps,
            "compounds": result.strategy, "paid_stops": result.pit_stops,
            "total_seconds": result.total_time}


def compare_schedules(cases=CASES, engines=ENGINES):
    """Return adaptive results and the fastest actually executed bounded schedule."""
    engines = tuple(engines)
    if any(engine not in ENGINES for engine in engines):
        raise ValueError(f"engines must be drawn from {ENGINES}")
    rows = []
    for case in cases:
        for engine in engines:
            selected = run_race(case, engine)
            best = None
            checked = 0
            for schedule in schedules(case["laps"]):
                alternative = run_race(case, engine, schedule)
                checked += 1
                if best is None or alternative["total_seconds"] < best["total_seconds"]:
                    best = alternative
            rows.append({"case": dict(case), "engine": engine, "selected": selected,
                         "best_schedule": best, "schedules_checked": checked,
                         "gap_seconds": selected["total_seconds"] - best["total_seconds"]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=(*ENGINES, "both"), default="both")
    args = parser.parse_args()
    engines = ENGINES if args.engine == "both" else (args.engine,)
    print(json.dumps(compare_schedules(engines=engines), indent=2))


if __name__ == "__main__":
    main()
