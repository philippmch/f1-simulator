"""Check dry pit timing against exhaustive, controlled full-race alternatives.

Synthetic 30-lap races, one permitted stop, a medium start, no incidents,
mean lap variation and fixed 2.75-second stationary service. Every permitted
stop lap from 6 through 30 and both unused slick compounds are simulated.
This is an offline model check, not validation against observed race strategy.
"""

import json

import numpy as np

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.race import RaceSimulator, TeamStrategyArchetype


class MeanPace:
    def normal(self, mean, _std):
        return mean


def run_race(stress, forced_lap=None, forced_compound=None):
    driver = Driver(id="synthetic", name="Synthetic", team_id="synthetic")
    car = Car(team_id="synthetic", team_name="Synthetic")
    track = Track(id="synthetic", name="Synthetic", country="Synthetic",
                  total_laps=30, base_lap_time=90, tire_stress=stress)
    simulator = RaceSimulator(rng=np.random.default_rng(42))
    simulator.event_manager.process_lap = lambda **kwargs: []
    simulator.lap_simulator.rng = MeanPace()
    simulator.lap_simulator.calculate_pit_stop_time = lambda *args, **kwargs: 2.75
    simulator._infer_team_strategy = lambda *args: TeamStrategyArchetype.BALANCED
    if forced_lap is not None:
        simulator._should_pit = lambda state, states, track, lap, *args, **kwargs: lap == forced_lap
        simulator._choose_distinct_dry_compound = lambda *args: forced_compound
    stops = []
    execute = simulator._execute_pit_stop

    def record(state, track, weather, current_lap):
        loss = execute(state, track, weather, current_lap)
        stops.append(current_lap)
        return loss

    simulator._execute_pit_stop = record
    result = simulator.simulate_race(
        [driver], {car.team_id: car}, track, Weather(change_probability=0),
        [driver.id], starting_tires={driver.id: TireCompound.MEDIUM},
    )[0]
    return {"total_seconds": result.total_time, "pit_laps": stops, "compounds": result.strategy}


def compare_pit_timing():
    rows = []
    for stress in (0.3, 0.9):
        selected = run_race(stress)
        alternatives = [run_race(stress, lap, compound)
                        for lap in range(6, 31)
                        for compound in (TireCompound.SOFT, TireCompound.HARD)]
        best = min(alternatives, key=lambda result: result["total_seconds"])
        rows.append({
            "tire_stress": stress, "selected": selected, "best_one_stop": best,
            "cost_vs_best_seconds": round(selected["total_seconds"] - best["total_seconds"], 6),
        })
    return rows


if __name__ == "__main__":
    print(json.dumps(compare_pit_timing(), indent=2))
