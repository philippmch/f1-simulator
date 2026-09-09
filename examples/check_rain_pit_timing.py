"""Exhaustive short wet-race strategy check using both execution engines.

Single-car synthetic eight/twelve-lap races use fixed rainfall/surface, mean pace,
expected service, and no incidents. Enumerate every schedule of up to four
paid stops after the opening lap. Tyres, ageing, fuel and stop execution use
the actual engine. This checks model consistency, not observed strategy.
"""

import json
from itertools import combinations

import numpy as np

from f1sim.models import Car, Driver, TireCompound, Track, Weather, WeatherCondition
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import RaceSimulator


def run_race(engine, compound, lane_loss, forced_stops=None, *, laps=8, base_lap_time=200):
    driver = Driver(id="synthetic", name="Synthetic", team_id="synthetic")
    car = Car(team_id="synthetic", team_name="Synthetic", tire_degradation_factor=1.5)
    track = Track(id="synthetic", name="Synthetic", country="Synthetic",
                  total_laps=laps, base_lap_time=base_lap_time,
                  tire_stress=1, pit_lane_delta=lane_loss)
    wet = compound == TireCompound.WET
    intensity = 0.85 if wet else 0.35
    weather = Weather(condition=WeatherCondition.HEAVY_RAIN if wet else WeatherCondition.LIGHT_RAIN,
                      rain_intensity=intensity, track_wetness=intensity, change_probability=0)
    simulator = RaceSimulator(np.random.default_rng(42))
    simulator.event_manager.process_lap = lambda *args, **kwargs: []
    simulator.event_manager._check_mechanical_failure = lambda *args: None
    simulator.event_manager._check_random_incident = lambda *args, **kwargs: None
    actual_lap = simulator.lap_simulator.calculate_lap_time

    def mean_lap(*args, **kwargs):
        kwargs["sample_variation"] = False
        return actual_lap(*args, **kwargs)

    simulator.lap_simulator.calculate_lap_time = mean_lap
    simulator.lap_simulator.calculate_pit_stop_time = expected_stationary_time
    if forced_stops is not None:
        simulator._should_pit = lambda state, states, track, lap, *a, **k: lap in forced_stops
    execute = (simulator.simulate_race if engine == "standard"
               else ChronologicalRace(simulator).run)
    result = execute([driver], {car.team_id: car}, track, weather, [driver.id],
                     starting_tires={driver.id: compound})[0]
    return {"total_seconds": result.total_time, "pit_laps": result.pit_laps,
            "laps_completed": result.laps_completed, "compounds": result.strategy}


def compare_rain_pit_timing():
    rows = []
    for engine in ("standard", "chronological"):
        for compound, lane, laps, base in ((TireCompound.INTERMEDIATE, 1, 8, 200),
                                          (TireCompound.WET, 0.1, 8, 200),
                                          (TireCompound.INTERMEDIATE, 20, 8, 200),
                                          (TireCompound.INTERMEDIATE, 1, 12, 300)):
            selected = run_race(engine, compound, lane, laps=laps, base_lap_time=base)
            alternatives = [run_race(engine, compound, lane, stops,
                                     laps=laps, base_lap_time=base)
                            for count in range(5)
                            for stops in combinations(range(2, laps + 1), count)]
            best = min(alternatives, key=lambda result: result["total_seconds"])
            rows.append({"race_engine": engine, "compound": compound.value,
                         "pit_lane_delta": lane, "race_laps": laps, "base_lap_time": base,
                         "schedules_checked": len(alternatives),
                         "selected": selected, "best_schedule": best,
                         "cost_vs_best_seconds": selected["total_seconds"] - best["total_seconds"]})
    return rows


if __name__ == "__main__":
    print(json.dumps(compare_rain_pit_timing(), indent=2))
