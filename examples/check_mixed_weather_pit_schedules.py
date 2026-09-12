"""Compare adaptive mixed-weather decisions with executed short-race schedules.

Synthetic single-car races start on slicks under fixed rainfall. Surface water
still evolves. Both engines use mean pace and service, with incidents disabled.
This checks strategy against the simulator's own physics, not real race pace.
"""

import argparse
import json

import numpy as np

from f1sim.models import Car, Driver, TireCompound, Track, Weather, WeatherCondition
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import DriverStatus, RaceSimulator, TeamStrategyArchetype

ENGINES = ("standard", "chronological")
SLICKS = (TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD)
CASES = (
    {"name": "used_soft_steady_damp", "laps": 6, "base_lap_time": 250,
     "pit_lane_delta": 1, "wetness": .1, "rain": .1, "opening_age": 12},
    {"name": "drying_slick_start", "laps": 6, "base_lap_time": 90,
     "pit_lane_delta": 20, "wetness": .17, "rain": 0, "opening_age": 0},
    {"name": "wetting_slick_start", "laps": 6, "base_lap_time": 90,
     "pit_lane_delta": 10, "wetness": .18, "rain": .35, "opening_age": 0},
    {"name": "drying_with_later_dry_stops", "laps": 6, "base_lap_time": 600,
     "pit_lane_delta": 1, "wetness": .17, "rain": 0, "opening_age": 12},
)


def initial_weather(case):
    return Weather(condition=WeatherCondition.CLOUDY, track_wetness=case["wetness"],
                   rain_intensity=case["rain"], change_probability=0)


def _legal_compounds(used):
    return bool(set(used) - set(SLICKS)) or len(set(used)) >= 2


def schedules(case):
    """Enumerate survivable, compound-compliant schedules after the opening lap.

    Balanced policy allows three dry stops, one damp stop in these short races,
    and four when on rain tyres or surface water exceeds .3. Every paid stop
    counts; a required final compound correction remains possible after that.
    """
    surfaces = [initial_weather(case)]
    for _ in range(1, case["laps"]):
        surfaces.append(surfaces[-1].project_surface())

    def visit(lap, compound, used, stops):
        if lap > case["laps"]:
            if _legal_compounds(used):
                yield stops
            return
        weather = surfaces[lap - 1]
        critical = weather.tire_mismatch(compound) == "critical"
        if not critical:
            yield from visit(lap + 1, compound, used | {compound}, stops)
        if lap == 1:
            return
        clearly_dry = weather.track_wetness < .08 and weather.rain_intensity < .15
        limit = 3 if clearly_dry else 1
        if weather.track_wetness > .3 or compound not in SLICKS:
            limit = 4
        rain = weather.fresh_rain_compound()
        candidates = (rain,) if rain else SLICKS
        for candidate in candidates:
            correction = not _legal_compounds(used) and _legal_compounds(used | {candidate})
            if (critical or len(stops) < limit or correction) and (
                weather.tire_mismatch(candidate) != "critical"
            ):
                yield from visit(lap + 1, candidate, used | {candidate},
                                 stops + ((lap, candidate.value),))

    yield from visit(1, TireCompound.SOFT, set(), ())


def run_race(case, engine, schedule=None):
    """Use real lap, ageing, surface and pit execution for either policy."""
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}")
    driver = Driver(id="synthetic", name="Synthetic", team_id="synthetic")
    car = Car(team_id="synthetic", team_name="Synthetic", tire_degradation_factor=1.5)
    track = Track(id="synthetic", name=case["name"], country="Synthetic",
                  total_laps=case["laps"], base_lap_time=case["base_lap_time"],
                  pit_lane_delta=case["pit_lane_delta"], tire_stress=1)
    simulator = RaceSimulator(np.random.default_rng(42))
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
        stops = {lap: TireCompound(compound) for lap, compound in schedule}
        selected = None

        def should_pit(state, states, track, lap, *args, **kwargs):
            nonlocal selected
            selected = stops.get(lap)
            return selected is not None

        simulator._should_pit = should_pit
        # Override choices, retaining the actual stop accounting and fresh fit.
        simulator._choose_distinct_dry_compound = lambda *args, **kwargs: selected
        simulator._choose_committed_dry_compound = lambda *args, **kwargs: selected
        simulator._choose_compound_for_next_stint = lambda *args, **kwargs: selected
    execute = (simulator.simulate_race if engine == "standard"
               else ChronologicalRace(simulator).run)
    result = execute([driver], {car.team_id: car}, track, initial_weather(case), [driver.id],
                     starting_tires={driver.id: TireCompound.SOFT},
                     starting_tire_ages={driver.id: case["opening_age"]})[0]
    assert result.status == DriverStatus.FINISHED and result.laps_completed == track.total_laps
    assert _legal_compounds({TireCompound(compound) for compound in result.strategy})
    if schedule is not None:
        assert result.pit_laps == [lap for lap, _ in schedule]
        assert result.strategy[1:] == [compound for _, compound in schedule]
    return {"laps_completed": result.laps_completed, "pit_laps": result.pit_laps,
            "compounds": result.strategy, "paid_stops": result.pit_stops,
            "total_seconds": result.total_time}


def compare_schedules(cases=CASES, engines=ENGINES):
    rows = []
    for case in cases:
        alternatives = tuple(schedules(case))
        for engine in engines:
            selected = run_race(case, engine)
            best = min((run_race(case, engine, schedule) for schedule in alternatives),
                       key=lambda result: result["total_seconds"])
            rows.append({"case": dict(case), "engine": engine, "selected": selected,
                         "best_schedule": best, "schedules_checked": len(alternatives),
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
