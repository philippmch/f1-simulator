"""Inspect numeric weather pace and actual policies without noise or network.

Condition labels are varied independently of fixed water and rain. Short races
compare explicit soft@18 and automatic damp openings, with six physical fuel
laps and mean service.
The tiny sweeps evaluate actual race and qualifying physics around former tyre
mismatch boundaries; they do not change the simulator's safety thresholds.
Print JSON to stdout; no files are written by this diagnostic.
"""

import json
from dataclasses import asdict
from unittest.mock import patch

import numpy as np

from f1sim.models import Car, Driver, TireCompound, Track, Weather, WeatherCondition
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import RaceSimulator

LAPS = 6
CASES = {"explicit_damp": dict(water=.35, rain=.35, opening="explicit", pit_lane_delta=8,
                               starting_compound="soft", starting_age=18),
         "automatic_mild_damp": dict(water=.1, rain=.1, opening="automatic", pit_lane_delta=22)}
RECORDS = ({"id": "S", "compound": "soft", "age": 18},
           {"id": "I", "compound": "intermediate", "age": 0},
           {"id": "H", "compound": "hard", "age": 0})


def models():
    return (Driver(id="A", name="Synthetic", team_id="A", wet_skill_modifier=1.2),
            Car(team_id="A", team_name="Synthetic", wet_performance=.9),
            Track(id="T", name="Synthetic", country="Synthetic", total_laps=LAPS,
                  base_lap_time=90, pit_lane_delta=8))


def pace(water, rain, condition, compound):
    driver, car, track = models()
    driver.current_tire_laps = 18
    weather = Weather(condition=condition, track_wetness=water, rain_intensity=rain)
    simulator = LapSimulator(np.random.default_rng(0))
    tire = TIRE_COMPOUNDS[TireCompound(compound)]
    return dict(race=simulator.calculate_lap_time(driver, car, track, tire, weather, 1, LAPS,
                                                sample_variation=False),
                qualifying=simulator.calculate_qualifying_lap(driver, car, track, tire, weather,
                                                              sample_variation=False))


def run_race(condition, engine, finite, case="explicit_damp"):
    settings = CASES[case]
    driver, car, track = models()
    track.pit_lane_delta = settings["pit_lane_delta"]
    weather = Weather(condition=condition, track_wetness=settings["water"],
                      rain_intensity=settings["rain"],
                      change_probability=0)
    simulator = RaceSimulator(np.random.default_rng(7))
    original = simulator.lap_simulator.calculate_lap_time
    physical_distances = []

    def mean_lap(*args, **kwargs):
        physical_distances.append(kwargs.get("total_laps", args[6] if args else None))
        kwargs["sample_variation"] = False
        return original(*args, **kwargs)

    simulator.lap_simulator.calculate_lap_time = mean_lap
    simulator.lap_simulator.calculate_pit_stop_time = expected_stationary_time
    simulator.event_manager.process_lap = lambda *a, **kw: []
    simulator.event_manager._check_mechanical_failure = lambda *a, **kw: None
    simulator.event_manager._check_random_incident = lambda *a, **kw: None
    execute = simulator.simulate_race if engine == "standard" else ChronologicalRace(simulator).run
    # Equal numeric rainfall and water also keep strategy surface projections
    # at equilibrium; changing a label cannot alter future physical inputs.
    with patch.object(Weather, "evolve", lambda self, rng: self.model_copy(deep=True)):
        explicit = settings["opening"] == "explicit"
        result, = execute([driver], {"A": car}, track, weather, ["A"],
                          starting_tires={"A": "soft"} if explicit else None,
                          starting_tire_ages={"A": 18} if explicit else None,
                          tire_inventory=({"A": [dict(item) for item in RECORDS]}
                                          if finite else None))
    return dict(case=case, condition=condition.value, engine=engine,
                inventory="finite" if finite else "unlimited",
                final_rng_state=simulator.rng.bit_generator.state,
                physical_fuel_distances=sorted(set(physical_distances)), result=asdict(result))


def check_cases():
    labels = [dict(case=name, condition=condition.value, compound=compound.value,
                   **pace(settings["water"], settings["rain"], condition, compound))
              for name, settings in CASES.items() for compound in TireCompound
              for condition in WeatherCondition]
    sweeps = [dict(compound=compound.value, boundary=boundary,
                   samples=[dict(water=boundary + delta,
                                 **pace(boundary + delta, .35, WeatherCondition.CLOUDY, compound))
                            for delta in (-1e-6, 0., 1e-6)])
              for compound in TireCompound for boundary in (.08, .15, .2, .3, .5, .8)]
    races = [run_race(condition, engine, finite, case) for case in CASES
             for engine in ("standard", "chronological")
             for finite in (False, True) for condition in WeatherCondition]
    return dict(inputs=dict(cases=CASES, race_laps=LAPS, base_lap_time=90,
                            finite_sets=RECORDS, wet_skill_modifier=1.2, wet_performance=.9,
                            sweep_rain=.35, sweep_epsilon=1e-6,
                            lap_variation=False, incidents=False, service="expected"),
                label_paces=labels, boundary_sweeps=sweeps, races=races)


if __name__ == "__main__":
    print(json.dumps(check_cases(), indent=2, allow_nan=False))
