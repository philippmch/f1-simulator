"""Compare isolated opening-policy forecasts with controlled native execution.

Seven synthetic cases cover dry, drying, increasing rain and timed finishes.
Rainfall stays fixed while surface water evolves; laps use mean physics,
expected service and no incidents. This verifies execution fidelity, not
real-race optimality or calibration. JSON goes to stdout without file writes.
"""

import argparse
import json
from itertools import product
from math import inf, isfinite
from unittest.mock import patch

import numpy as np

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.opening_strategy import _policy_path_outcome
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import RaceSimulator, TeamStrategyArchetype

CASES = {
    "dry": (6, 90, 22, 0, 0),
    "drying": (12, 90, 1, .26, 0),
    "timed_damp": (20, 600, 22, .1, .1),
    "timed_drying": (20, 600, 22, .26, 0),
    "wet_to_dry": (20, 350, 8, .74, 0),
    "wetting": (12, 90, 22, .18, .35),
    "heavy_rain": (12, 90, 22, .68, .9),
}
ENGINES = ("standard", "chronological")
INVENTORIES = ("unlimited", "finite")


def _conserved(result, records):
    ages = {item["id"]: item["age"] for item in records}
    for stint in result.tire_set_history:
        if (stint["age_at_fit"] != ages.get(stint["set_id"])
                or stint["age_at_end"] != stint["age_at_fit"] + stint["laps_used"]):
            return False
        ages[stint["set_id"]] = stint["age_at_end"]
    return (ages == {item["id"]: item["age"] for item in result.tire_inventory}
            and sum(item["laps_used"] for item in result.tire_set_history) == result.laps_completed)


def run_case(name, compound, seed, engine, inventory):
    if engine not in ENGINES or inventory not in INVENTORIES:
        raise ValueError("Unknown engine or inventory mode")
    compound = TireCompound(compound)
    laps, base, lane, water, rain = CASES[name]
    driver = Driver(id="A", name="Synthetic", team_id="A")
    car = Car(team_id="A", team_name="Synthetic", tire_degradation_factor=1.5)
    track = Track(id="T", name="Synthetic", country="Synthetic", total_laps=laps,
                  base_lap_time=base, pit_lane_delta=lane, tire_stress=1)
    weather = Weather(track_wetness=water, rain_intensity=rain, change_probability=0)
    records = [dict(id=c.value, compound=c.value, age=5 if c == TireCompound.SOFT else 0)
               for c in TireCompound] if inventory == "finite" else None
    simulator = RaceSimulator(np.random.default_rng(seed))
    predicted_laps, predicted_time = _policy_path_outcome(
        driver, car, track, weather, TeamStrategyArchetype.BALANCED,
        simulator.strategy_tuning, simulator.strategy_profiles, compound, seed,
        **(dict(tire_inventory=records, opening_set_id=compound.value) if records else {}),
    )
    simulator._infer_team_strategy = lambda *a: TeamStrategyArchetype.BALANCED
    simulator.event_manager.process_lap = lambda *a, **kw: []
    simulator.event_manager._check_mechanical_failure = lambda *a, **kw: None
    simulator.event_manager._check_random_incident = lambda *a, **kw: None
    original = simulator.lap_simulator.calculate_lap_time
    fuel_distances = []

    def mean_lap(*args, **kwargs):
        fuel_distances.append(kwargs.get("total_laps", args[6] if args else None))
        kwargs["sample_variation"] = False
        return original(*args, **kwargs)

    simulator.lap_simulator.calculate_lap_time = mean_lap
    simulator.lap_simulator.calculate_pit_stop_time = expected_stationary_time
    execute = simulator.simulate_race if engine == "standard" else ChronologicalRace(simulator).run
    with patch.object(Weather, "evolve", lambda self, rng: self.project_surface()):
        result, = execute(
            [driver], {"A": car}, track, weather, ["A"], starting_tires={"A": compound},
            starting_tire_ages=({"A": 5 if compound == TireCompound.SOFT else 0}
                                if records else None),
            tire_inventory={"A": records} if records else None,
        )
    predicted_complete = isfinite(predicted_time)
    predicted_valid = predicted_complete or predicted_time == inf
    actual_valid = isfinite(result.total_time)
    actual_complete = result.status.value == "finished"
    gap = (result.total_time - predicted_time
           if predicted_complete and actual_complete and actual_valid else None)
    errors = []
    if not predicted_valid:
        errors.append("invalid_predicted_time")
    if not actual_valid:
        errors.append("invalid_executed_time")
    if predicted_laps != result.laps_completed:
        errors.append("completed_distance")
    if predicted_complete != actual_complete or (gap is not None and abs(gap) > 1e-7):
        errors.append("policy_cost")
    fuel_ok = all(value == laps for value in fuel_distances)
    conserved = _conserved(result, records) if records else None
    if not fuel_ok:
        errors.append("physical_fuel_distance")
    if conserved is False:
        errors.append("physical_wear_conservation")
    return dict(case=name, engine=engine, inventory=inventory, seed=seed,
                inputs=dict(scheduled_laps=laps, base_lap_time=base, pit_lane_delta=lane,
                            water=water, rain=rain, opening_compound=compound.value,
                            opening_age=5 if records and compound == TireCompound.SOFT else 0,
                            physical_sets=records, tire_degradation_factor=1.5, tire_stress=1),
                predicted=dict(laps=predicted_laps,
                               seconds=predicted_time if predicted_complete else None,
                               status=("finished" if predicted_complete else "infeasible"
                                       if predicted_valid else "invalid")),
                executed=dict(laps=result.laps_completed,
                              seconds=result.total_time if actual_valid else None,
                              status=result.status.value, dnf_reason=result.dnf_reason,
                              time_limited=result.race_time_limited, compounds=result.strategy,
                              stops=result.pit_laps, tire_set_history=result.tire_set_history,
                              tire_inventory=result.tire_inventory),
                gap_seconds=gap, physical_fuel_distances=sorted(set(fuel_distances)),
                wear_conserved=conserved, errors=errors)


def check_cases(cases=CASES, compounds=tuple(TireCompound), seeds=(0, 3),
                engines=ENGINES, inventories=INVENTORIES):
    rows = [run_case(*args) for args in product(cases, compounds, seeds, engines, inventories)]
    return dict(comparisons=len(rows), mismatches=sum(bool(row["errors"]) for row in rows),
                assumptions=dict(lap_variation=False, incidents=False, service="expected",
                                 rainfall="fixed", surface="projected", control="green"), rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=(*ENGINES, "both"), default="both")
    parser.add_argument("--inventory", choices=(*INVENTORIES, "both"), default="both")
    args = parser.parse_args()
    report = check_cases(engines=ENGINES if args.engine == "both" else (args.engine,),
                         inventories=INVENTORIES if args.inventory == "both" else (args.inventory,))
    print(json.dumps(report, indent=2, allow_nan=False))
    return 1 if report["mismatches"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
