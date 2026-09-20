"""Compare free restart tyres with controlled native remaining-race policies.

Single-car races use fixed rainfall, evolving surface water, mean lap physics,
expected service and one red flag after lap one. Each usable fresh compound
is fitted for free in a separate run, then native strategy handles later stops.
This measures consistency within the model, not calibrated real-race gains or
global schedule optimality. No provider feeds or simulation RNG are inspected.
"""

import json

import numpy as np

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import RaceSimulator, TeamStrategyArchetype

CASES = {
    "wetting": (.1375, .35, 15),
    "drying": (.25, 0, 15),
    "steady_damp": (.1, .1, 10),
    "worsening_rain": (.68, .9, 12),
    "steady_wet": (.9, .9, 12),
    "dry_restart": (.09, 0, 10),
}


def run_restart(name, engine, forced_compound=None):
    water, rain, laps = CASES[name]
    weather = Weather(track_wetness=water, rain_intensity=rain, change_probability=0)
    driver = Driver(id="A", name="Synthetic", team_id="A")
    car = Car(team_id="A", team_name="Synthetic")
    track = Track(id="T", name="Synthetic", country="Synthetic", total_laps=laps,
                  base_lap_time=90, pit_lane_delta=20)
    simulator = RaceSimulator(np.random.default_rng(42))
    simulator._infer_team_strategy = lambda *args: TeamStrategyArchetype.BALANCED
    calculate = simulator.lap_simulator.calculate_lap_time

    def mean_lap(*args, **kwargs):
        return calculate(*args, **dict(kwargs, sample_variation=False))

    simulator.lap_simulator.calculate_lap_time = mean_lap
    simulator.lap_simulator.calculate_pit_stop_time = expected_stationary_time
    simulator.event_manager._check_mechanical_failure = lambda *args, **kwargs: None
    simulator.event_manager._check_random_incident = lambda *args, **kwargs: None

    def control(lap, *args, **kwargs):
        simulator.event_manager.current_lap = lap
        return ([simulator.event_manager.deploy_red_flag(lap, "Synthetic diagnostic")]
                if lap == 1 else [])

    simulator.event_manager.process_lap = control
    choose = simulator._choose_red_flag_tire
    selections = []

    def refit(state, surface, planning, lap, **kwargs):
        compound = (choose(state, surface, planning, lap, **kwargs)
                    if forced_compound is None else TireCompound(forced_compound))
        if surface.tire_mismatch(compound) == "critical":
            raise AssertionError("Diagnostic attempted a critically mismatched free set")
        selections.append(compound.value)
        return compound

    simulator._choose_red_flag_tire = refit
    execute = simulator.simulate_race if engine == "standard" else ChronologicalRace(simulator).run
    result, = execute(
        [driver], {"A": car}, track, weather, ["A"],
        starting_tires={"A": weather.fresh_rain_compound() or TireCompound.MEDIUM},
    )
    if result.laps_completed != laps or result.status.value != "finished":
        raise AssertionError("Diagnostic did not complete its scheduled distance")
    return dict(restart_compound=selections[0], total_seconds=result.total_time,
                paid_stops=result.pit_stops, pit_laps=result.pit_laps,
                compounds=result.strategy, laps_completed=result.laps_completed)


def compare_restart_choices():
    rows = []
    for name, (water, rain, laps) in CASES.items():
        surface = Weather(track_wetness=water, rain_intensity=rain).project_surface()
        candidates = [compound for compound in TireCompound
                      if surface.tire_mismatch(compound) != "critical"]
        for engine in ("standard", "chronological"):
            selected = run_restart(name, engine)
            alternatives = [run_restart(name, engine, compound) for compound in candidates]
            best = min(alternatives, key=lambda row: row["total_seconds"])
            rows.append(dict(
                case=name, engine=engine,
                inputs=dict(scheduled_laps=laps, suspension_lap=1, base_lap_time=90,
                            pit_lane_delta=20, initial_wetness=water, rainfall=rain,
                            restart_wetness=surface.track_wetness),
                selected=selected, alternatives=alternatives, best_alternative=best,
                cost_vs_best_seconds=selected["total_seconds"] - best["total_seconds"],
            ))
    return rows


if __name__ == "__main__":
    print(json.dumps(compare_restart_choices(), indent=2, allow_nan=False))
