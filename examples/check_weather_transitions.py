"""Compare rain-tyre transitions with bounded schedules executed in both engines.

Synthetic single-car races use fixed rainfall, evolving surface wetness, mean
pace and expected service. All eligible schedules within the stated stop budget
are executed, including compulsory replacements after that budget is exhausted.
This checks the simulator's own physics, not calibrated real-world strategy.
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
    dict(name="drying_intermediates", compound="intermediate", wetness=.26, rain=0,
         laps=8, lane=22, base=200, stops=2),
    dict(name="cheap_drying_stop", compound="intermediate", wetness=.26, rain=0,
         laps=8, lane=1, base=200, stops=2),
    dict(name="wet_to_intermediate_to_slick", compound="wet", wetness=.74, rain=0,
         laps=24, lane=22, base=90, stops=2),
    dict(name="increasing_rain", compound="intermediate", wetness=.68, rain=.9,
         laps=8, lane=22, base=90, stops=2),
)


def weather_for(case):
    condition = WeatherCondition.HEAVY_RAIN if case["rain"] else WeatherCondition.CLOUDY
    return Weather(condition=condition,
                   track_wetness=case["wetness"], rain_intensity=case["rain"], change_probability=0)


def schedules(case):
    """Enumerate safe schedules, using the same fresh-set eligibility as execution."""
    surfaces = [weather_for(case)]
    for _ in range(1, case["laps"]):
        surfaces.append(surfaces[-1].project_surface())

    def visit(lap, compound, budget, stops):
        if lap > case["laps"]:
            yield stops
            return
        surface = surfaces[lap - 1]
        critical = surface.tire_mismatch(compound) == "critical"
        if not critical:
            yield from visit(lap + 1, compound, budget, stops)
        if lap > 1 and (budget or critical):
            required = surface.fresh_rain_compound()
            for fresh in (required,) if required is not None else SLICKS:
                yield from visit(lap + 1, fresh, max(0, budget - 1), stops + ((lap, fresh),))

    yield from visit(1, TireCompound(case["compound"]), case["stops"], ())


def run_race(case, engine, schedule=None):
    if engine not in ENGINES:
        raise ValueError(f"engine must be drawn from {ENGINES}")
    driver = Driver(id="A", name="Synthetic", team_id="A")
    car = Car(team_id="A", team_name="Synthetic", tire_degradation_factor=1.5)
    track = Track(id="t", name="Synthetic", country="Synthetic", total_laps=case["laps"],
                  base_lap_time=case["base"], pit_lane_delta=case["lane"], tire_stress=1)
    simulator = RaceSimulator(np.random.default_rng(42))
    simulator._infer_team_strategy = lambda *a: TeamStrategyArchetype.BALANCED
    simulator.event_manager.process_lap = lambda *a, **k: []
    simulator.event_manager._check_mechanical_failure = lambda *a, **k: None
    simulator.event_manager._check_random_incident = lambda *a, **k: None
    calculate = simulator.lap_simulator.calculate_lap_time

    def mean_lap(*args, **kwargs):
        tire = kwargs.get("tire", args[3] if len(args) > 3 else None)
        weather = kwargs.get("weather", args[4] if len(args) > 4 else None)
        if weather.tire_mismatch(tire.compound) == "critical":
            raise AssertionError("Diagnostic schedule runs a critically unsuitable set")
        kwargs["sample_variation"] = False
        return calculate(*args, **kwargs)

    simulator.lap_simulator.calculate_lap_time = mean_lap
    simulator.lap_simulator.calculate_pit_stop_time = expected_stationary_time
    if schedule is not None:
        stops = dict(schedule)
        simulator._should_pit = lambda state, states, track, lap, *a, **k: lap in stops
        execute_stop = simulator._execute_pit_stop

        def forced_stop(state, track, weather, current_lap, **kwargs):
            requested = stops[current_lap]
            required = weather.fresh_rain_compound()
            if requested not in ((required,) if required is not None else SLICKS):
                raise AssertionError("Schedule requests an ineligible fresh compound")
            state.weather_pit_proposal = (current_lap, requested)
            loss = execute_stop(state, track, weather, current_lap, **kwargs)
            if state.current_tire.compound != requested:
                raise AssertionError("Paid execution did not fit the requested compound")
            return loss

        simulator._execute_pit_stop = forced_stop
    execute = (simulator.simulate_race if engine == "standard"
               else ChronologicalRace(simulator).run)
    result = execute([driver], {"A": car}, track, weather_for(case), ["A"],
                     starting_tires={"A": TireCompound(case["compound"])})[0]
    if result.status != DriverStatus.FINISHED or result.laps_completed != track.total_laps:
        raise AssertionError("Diagnostic race did not complete its scheduled distance")
    return dict(total_seconds=result.total_time, laps_completed=result.laps_completed,
                pit_laps=result.pit_laps, compounds=result.strategy, paid_stops=result.pit_stops)


def compare_schedules(cases=CASES, engines=ENGINES):
    rows = []
    for case in cases:
        for engine in engines:
            selected = run_race(case, engine)
            checked, best = 0, None
            for schedule in schedules(case):
                result = run_race(case, engine, schedule)
                checked += 1
                if best is None or result["total_seconds"] < best["total_seconds"]:
                    best = result
            rows.append(dict(case=dict(case), engine=engine, selected=selected,
                             best_schedule=best, schedules_checked=checked,
                             gap_seconds=selected["total_seconds"] - best["total_seconds"]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=(*ENGINES, "both"), default="both")
    args = parser.parse_args()
    engines = ENGINES if args.engine == "both" else (args.engine,)
    print(json.dumps(compare_schedules(engines=engines), indent=2))


if __name__ == "__main__":
    main()
