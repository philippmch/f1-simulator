"""Compare red-flag tyre choices with controlled full-race alternatives.

Synthetic 60-lap races use mean pace, expected pit service and a forced lap-two
medium-to-hard stop. A later suspension offers each fresh slick for free.
Each alternative then uses the actual remaining dry pit strategy. This checks
model consistency, not accuracy against real races or future weather.
"""

import json

import numpy as np

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.pit_strategy import SLICKS, expected_stationary_time
from f1sim.simulation.race import RaceSimulator, TeamStrategyArchetype


class MeanPace:
    def normal(self, mean, _std):
        return mean


def run_restart(suspension_lap, stop_budget, forced_compound=None, degradation=1.0):
    driver = Driver(id="synthetic", name="Synthetic", team_id="synthetic")
    car = Car(team_id="synthetic", team_name="Synthetic", tire_degradation_factor=degradation)
    track = Track(id="synthetic", name="Synthetic", country="Synthetic",
                  total_laps=60, base_lap_time=90, tire_stress=0.9)
    simulator = RaceSimulator(rng=np.random.default_rng(42))
    simulator.lap_simulator.rng = MeanPace()
    simulator.lap_simulator.calculate_pit_stop_time = lambda car: expected_stationary_time(car)
    simulator._infer_team_strategy = lambda *args: TeamStrategyArchetype.BALANCED
    simulator._ordinary_stop_budget = lambda *args: stop_budget
    simulator._dry_stop_budget = lambda *args: stop_budget
    original_should_pit = simulator._should_pit

    def should_pit(state, states, track, lap, *args, **kwargs):
        if lap <= suspension_lap:
            return lap == 2
        return original_should_pit(state, states, track, lap, *args, **kwargs)

    def events(lap, **kwargs):
        simulator.event_manager.current_lap = lap
        if lap == suspension_lap:
            return [simulator.event_manager.deploy_red_flag(lap, "Synthetic diagnostic")]
        return []

    simulator._should_pit = should_pit
    simulator.event_manager.process_lap = events
    simulator._choose_committed_dry_compound = lambda *args: TireCompound.HARD
    selected = []
    original_choose = simulator._choose_red_flag_tire

    def choose(*args, **kwargs):
        compound = forced_compound or original_choose(*args, **kwargs)
        selected.append(compound.value)
        return compound

    simulator._choose_red_flag_tire = choose
    result = simulator.simulate_race(
        [driver], {car.team_id: car}, track, Weather(change_probability=0),
        [driver.id], starting_tires={driver.id: TireCompound.MEDIUM},
    )[0]
    return {"restart_compound": selected[0], "total_seconds": result.total_time,
            "paid_stops": result.pit_stops, "compounds": result.strategy}


def compare_restart_choices():
    rows = []
    for suspension_lap, stop_budget, degradation in ((55, 1, 1.0), (15, 1, 1.0), (10, 2, 1.5)):
        selected = run_restart(suspension_lap, stop_budget, degradation=degradation)
        alternatives = [run_restart(suspension_lap, stop_budget, compound, degradation)
                        for compound in SLICKS]
        best = min(alternatives, key=lambda result: result["total_seconds"])
        rows.append({
            "remaining_laps": 60 - suspension_lap,
            "remaining_paid_stops": stop_budget - 1,
            "tire_degradation_factor": degradation,
            "selected": selected,
            "best_alternative": best,
            "cost_vs_best_seconds": round(selected["total_seconds"] - best["total_seconds"], 6),
        })
    return rows


if __name__ == "__main__":
    print(json.dumps(compare_restart_choices(), indent=2))
