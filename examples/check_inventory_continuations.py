"""Compare compulsory finite-pool fits with separately executed alternatives.

An explicit one-lap hard opening expires on a drying surface. Each eligible
replacement is committed in its own race, then normal strategy resumes. Repeat
with a free red-flag fit and custom instructions. Mean physics, expected service
and no incidents isolate the conditional policy; these are synthetic checks,
not calibrated race forecasts or an exhaustive search over elective schedules.
With --elective, an intermediate opening must use wets before their safe
window closes, then refit the removed intermediates with their accumulated
wear and remaining allowance.
"""

import argparse
import json
from dataclasses import replace
from math import inf

import numpy as np

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import DriverStatus, RaceSimulator, TeamStrategyArchetype

POOL = (dict(id="H", compound="hard", remaining_laps=1),
        dict(id="I", compound="intermediate", remaining_laps=4),
        dict(id="W", compound="wet", remaining_laps=1))
ELECTIVE_POOL = POOL[1:]
PLANS = {"automatic": None, "empty": [],
         "later_intermediate": [dict(lap=4, compound="intermediate")],
         "unreached_wet": [dict(lap=7, compound="wet")]}


def run_continuation(engine, *, free=False, plan=None, forced_set=None, reverse=False,
                     warmup=None, elective=False):
    if engine not in {"standard", "chronological"}:
        raise ValueError("engine must be standard or chronological")
    driver = Driver(id="A", name="Synthetic", team_id="A")
    car = Car(team_id="A", team_name="Synthetic")
    track = Track(id="continuation", name="Synthetic", country="Synthetic", total_laps=8,
                  base_lap_time=90., pit_lane_delta=20.)
    simulator = RaceSimulator(np.random.default_rng(7), tire_warmup=warmup)
    simulator._infer_team_strategy = lambda *args: TeamStrategyArchetype.BALANCED
    simulator.event_manager._check_mechanical_failure = lambda *args, **kwargs: None
    simulator.event_manager._check_random_incident = lambda *args, **kwargs: None
    simulator.event_manager._deploy_safety_measure = lambda *args, **kwargs: None
    if free:
        simulator.event_manager.set_forced_red_flag(1)
    else:
        simulator.event_manager.process_lap = lambda *args, **kwargs: []
    calculate = simulator.lap_simulator.calculate_lap_time
    fuel_distances = []

    def mean(*args, **kwargs):
        tire = kwargs.get("tire", args[3] if args else None)
        surface = kwargs.get("weather", args[4] if args else None)
        if surface.tire_mismatch(tire.compound) == "critical":
            raise AssertionError("Continuation ran a critically unsuitable set")
        fuel_distances.append(kwargs.get("total_laps", args[6] if args else None))
        return calculate(*args, **dict(kwargs, sample_variation=False))

    simulator.lap_simulator.calculate_lap_time = mean
    simulator.lap_simulator.calculate_pit_stop_time = expected_stationary_time
    if forced_set is not None:
        permitted = {"hold", "W"} if elective else {"I", "W"}
        if forced_set not in permitted:
            raise ValueError(f"forced_set must be one of {sorted(permitted)}")
        original = simulator._plan_inventory

        def commit_first(state, track, weather, lap, **options):
            decision = original(state, track, weather, lap, **options)
            if lap == 2:
                if forced_set == "hold":
                    return replace(decision, pit_now_cost=inf, wait_cost=0., set_id=None,
                                   compound=None, pit_now_laps=None, wait_laps=None)
                item = state.tire_inventory.sets[forced_set]
                if weather.tire_mismatch(item.compound) == "critical":
                    raise AssertionError("Forced diagnostic choice was ineligible")
                if elective:
                    return replace(decision, pit_now_cost=0., wait_cost=inf,
                                   set_id=item.id, compound=item.compound,
                                   pit_now_laps=None, wait_laps=None)
                return replace(decision, set_id=item.id, compound=item.compound)
            return decision

        simulator._plan_inventory = commit_first
    pool = ELECTIVE_POOL if elective else POOL
    records = list(reversed(pool)) if reverse else list(pool)
    run = (simulator.simulate_race if engine == "standard" else
           ChronologicalRace(simulator, red_flag_pause_seconds=0.).run)
    result, = run([driver], {"A": car}, track,
                  Weather(track_wetness=.24, change_probability=0), ["A"],
                  starting_tires={"A": (TireCompound.INTERMEDIATE if elective
                                         else TireCompound.HARD)},
                  tire_inventory={"A": records},
                  **({"pit_plans": {"A": plan}} if plan is not None else {}))
    if set(fuel_distances) != {8}:
        raise AssertionError("Continuation changed physical fuel distance")
    return dict(status=result.status.value, laps_completed=result.laps_completed,
                total_seconds=result.total_time, paid_stops=result.pit_stops,
                pit_laps=result.pit_laps, tire_set_history=result.tire_set_history,
                tire_inventory=result.tire_inventory, pit_plan_history=result.pit_plan_history,
                pit_stop_details=result.pit_stop_details, dnf_reason=result.dnf_reason)


def compare_continuations():
    rows = []
    for name, plan in PLANS.items():
        for free in (False, True):
            for engine in ("standard", "chronological"):
                options = dict(free=free, plan=plan)
                selected = run_continuation(engine, **options)
                alternatives = {identifier: run_continuation(engine, forced_set=identifier,
                                                              **options)
                                for identifier in ("I", "W")}
                best = min(alternatives, key=lambda key: (
                    alternatives[key]["status"] != DriverStatus.FINISHED.value,
                    -alternatives[key]["laps_completed"], alternatives[key]["total_seconds"]))
                distance_gap = selected["laps_completed"] - alternatives[best]["laps_completed"]
                time_gap = selected["total_seconds"] - alternatives[best]["total_seconds"]
                if distance_gap != 0 or abs(time_gap) > 1.e-7:
                    raise AssertionError("Automatic compulsory choice lost distance or time")
                rows.append(dict(engine=engine, policy=name, free_fit=free,
                                 inputs=dict(scheduled_laps=8, initial_water=.24, rainfall=0.,
                                             inventory=POOL, pit_plan=plan),
                                 selected=selected, alternatives=alternatives,
                                 best_alternative=best, distance_gap=distance_gap,
                                 time_gap_seconds=time_gap))
    return rows


def compare_elective_continuations():
    """Commit or defer the early wet window, then resume normal execution."""
    rows = []
    for engine in ("standard", "chronological"):
        selected = run_continuation(engine, elective=True)
        alternatives = {key: run_continuation(engine, elective=True, forced_set=key)
                        for key in ("W", "hold")}
        best = min(alternatives, key=lambda key: (
            alternatives[key]["status"] != DriverStatus.FINISHED.value,
            -alternatives[key]["laps_completed"], alternatives[key]["total_seconds"]))
        distance_gap = selected["laps_completed"] - alternatives[best]["laps_completed"]
        time_gap = selected["total_seconds"] - alternatives[best]["total_seconds"]
        if distance_gap != 0 or abs(time_gap) > 1.e-7:
            raise AssertionError("Automatic elective choice lost distance or time")
        rows.append(dict(engine=engine, policy="automatic", free_fit=False,
                         inputs=dict(scheduled_laps=8, initial_water=.24, rainfall=0.,
                                     inventory=ELECTIVE_POOL,
                                     opening_compound="intermediate"),
                         selected=selected, alternatives=alternatives,
                         best_alternative=best, distance_gap=distance_gap,
                         time_gap_seconds=time_gap))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elective", action="store_true",
                        help="Check a paid switch before the intermediate opening expires")
    args = parser.parse_args()
    rows = compare_elective_continuations() if args.elective else compare_continuations()
    print(json.dumps(rows, indent=2, allow_nan=False))
