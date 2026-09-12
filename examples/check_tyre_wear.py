"""Compare native tyre policies with short, executed schedule enumerations.

Used opening tyres put six remaining laps across the configured wear cliff.
Every candidate uses native pit execution, original race fuel, mean lap physics
and expected service. The oracle enumerates zero, one or two paid stops after
the opening lap; it is a bounded diagnostic, not a global strategy proof.
Run directly to print JSON. No files or network access are used.
"""

import itertools
import json
from unittest.mock import patch

import numpy as np

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import RaceSimulator

CASES = {
    "slick_cliff": dict(compound="soft", age=18, wetness=0., compounds=("soft", "medium", "hard")),
    "rain_cliff": dict(compound="wet", age=38, wetness=.8, compounds=("wet",)),
}
LAPS = 6


def _run(case, engine, finite, schedule=None):
    sim = RaceSimulator(np.random.default_rng(7))
    driver = Driver(id="A", name="Synthetic", team_id="A", tire_management=.8)
    car = Car(team_id="A", team_name="Synthetic")
    track = Track(id="T", name="Synthetic", country="Synthetic", total_laps=LAPS,
                  base_lap_time=180, pit_lane_delta=8)
    weather = Weather(track_wetness=case["wetness"], rain_intensity=case["wetness"],
                      condition="heavy_rain" if case["wetness"] else "dry", change_probability=0)
    records = [{"id": "opening", "compound": case["compound"], "age": case["age"]}]
    records.extend({"id": name, "compound": name} for name in case["compounds"])
    calculate = sim.lap_simulator.calculate_lap_time

    def mean_lap(*args, **kwargs):
        kwargs["sample_variation"] = False
        return calculate(*args, **kwargs)

    sim.lap_simulator.calculate_lap_time = mean_lap
    sim.lap_simulator.calculate_pit_stop_time = expected_stationary_time
    sim.event_manager.process_lap = lambda *a, **kw: []
    sim.event_manager._check_mechanical_failure = lambda *a, **kw: None
    sim.event_manager._check_random_incident = lambda *a, **kw: None
    if schedule is not None:
        stops = dict(schedule)

        def should_pit(state, states, track, lap, *args, **kwargs):
            target = stops.get(lap)
            if target is None:
                return False
            if finite:
                state.inventory_pit_proposal = (lap, target)
            else:
                state.dry_pit_proposal = (lap, TireCompound(target))
            return True

        sim._should_pit = should_pit
    execute = sim.simulate_race if engine == "standard" else ChronologicalRace(sim).run
    with patch.object(Weather, "evolve", lambda self, rng: self.model_copy(deep=True)):
        result = execute([driver], {"A": car}, track, weather, ["A"],
                         starting_tires={"A": case["compound"]},
                         starting_tire_ages={"A": case["age"]},
                         tire_inventory={"A": records} if finite else None)[0]
    return dict(total_time=result.total_time, stops=result.pit_laps,
                compounds=result.strategy, sets=result.tire_set_history,
                laps_completed=result.laps_completed, status=result.status.value)


def _schedules(case, finite):
    """Enumerate physical identities independently; removed sets can return."""
    choices = (("opening",) if finite else ()) + case["compounds"]
    for count in range(3):
        for laps in itertools.combinations(range(2, LAPS + 1), count):
            for targets in itertools.product(choices, repeat=count):
                current = "opening"
                used = {case["compound"]}
                valid = True
                for target in targets:
                    if finite and target == current:
                        valid = False
                        break
                    current = target
                    used.add(case["compound"] if target == "opening" else target)
                if valid and (case["wetness"] or len(used) >= 2):
                    yield tuple(zip(laps, targets, strict=True))


def run_case(name, engine, finite):
    case = CASES[name]
    selected = _run(case, engine, finite)
    candidates = [_run(case, engine, finite, schedule) for schedule in _schedules(case, finite)]
    completed = [row for row in candidates if row["laps_completed"] == LAPS
                 and row["status"] == "finished"]
    best = min(completed, key=lambda row: row["total_time"])
    tire = TIRE_COMPOUNDS[TireCompound(case["compound"])]
    return dict(case=name, engine=engine, inventory="finite" if finite else "unlimited",
                inputs=dict(case, remaining_laps=LAPS, base_lap_time=180, pit_lane_delta=8,
                            tire_management=.8, max_enumerated_stops=2),
                wear_samples=[dict(age=age, grip=tire.grip_at_lap(age, .8),
                                   penalty=tire.time_penalty_per_lap(age, 180, .8))
                              for age in range(case["age"], case["age"] + LAPS)],
                schedules_checked=len(candidates), completed_schedules=len(completed),
                selected=selected, best=best,
                difference_seconds=selected["total_time"] - best["total_time"])


def check_cases():
    return [run_case(name, engine, finite) for name in CASES
            for engine in ("standard", "chronological") for finite in (False, True)]


if __name__ == "__main__":
    print(json.dumps(check_cases(), indent=2, allow_nan=False))
