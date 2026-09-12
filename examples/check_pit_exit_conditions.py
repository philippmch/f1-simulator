"""Check actual pit-exit running conditions in controlled chronological races.

Two synthetic cars run four laps with mean physics and no incidents. One car
makes a scripted opening stop with deliberately extended service, so a leading
weather/control update occurs before it rejoins. This is a scheduler diagnostic,
not a realistic service-time estimate or an optimized pit strategy.
"""

import json

import numpy as np

from f1sim.models import Car, Driver, Track, Weather
from f1sim.models.track import ActiveAeroZone
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.race import RaceSimulator

CASES = (
    ("drying", "green", "green"),
    ("safety_car_deploys", "green", "sc"),
    ("safety_car_clears", "sc", "green"),
    ("vsc_deploys", "green", "vsc"),
    ("vsc_clears", "vsc", "green"),
)


def run_case(name, initial_control, later_control):
    simulator = RaceSimulator(np.random.default_rng(42))
    engine = ChronologicalRace(simulator)
    drivers = [Driver(id=key, name=key, team_id=key) for key in "AB"]
    cars = {key: Car(team_id=key, team_name=key) for key in "AB"}
    track = Track(id="synthetic", name="Synthetic", country="Synthetic",
                  total_laps=4, base_lap_time=90, pit_lane_delta=22,
                  active_aero_zones=[ActiveAeroZone(zone_id=1, sector=1, time_gain=.5)])
    weather = Weather(track_wetness=.26, rain_intensity=0, change_probability=0)
    control = simulator.event_manager

    def set_control(value):
        control.safety_car_active = value == "sc"
        control.vsc_active = value == "vsc"

    reset = control.reset

    def reset_control():
        reset()
        set_control(initial_control)

    def process(lap, *args, **kwargs):
        if lap == 1:
            set_control(later_control)
            if initial_control == "sc" and later_control == "green":
                control.sc_restart_lap_number = 2
        return []

    control.reset = reset_control
    control.process_lap = process
    control._check_mechanical_failure = lambda *a, **kw: None
    control._check_random_incident = lambda *a, **kw: None
    simulator.overtaking_model.attempt_overtake = lambda *a, **kw: (True, False)
    calculate = simulator.lap_simulator.calculate_lap_time

    def mean_lap(*args, **kwargs):
        kwargs["sample_variation"] = False
        return calculate(*args, **kwargs)

    simulator.lap_simulator.calculate_lap_time = mean_lap
    simulator.lap_simulator.calculate_pit_stop_time = lambda car: 160.0
    entries, pit_entry = [], {}

    def observed():
        return dict(wetness=engine.weather.track_wetness,
                    neutralized=not control.is_active_aero_allowed(),
                    lap_time_modifier=control.get_lap_time_modifier(),
                    active_aero_enabled=control.is_active_aero_allowed())

    def should_pit(state, states, planning, lap, *args, **kwargs):
        stop = state.driver.id == "B" and lap == 1
        if stop:
            pit_entry.update(observed())
        return stop

    simulator._should_pit = should_pit
    begin = engine._begin_running

    def record_entry(state, pending, now):
        actual = observed()
        begin(state, pending, now)
        snapshot = dict(wetness=pending.weather.track_wetness,
                        neutralized=pending.neutralized,
                        lap_time_modifier=pending.lap_time_modifier,
                        active_aero_enabled=pending.active_aero_enabled)
        entries.append(dict(driver=state.driver.id, lap=pending.lap, time=now,
                            observed=actual, running_snapshot=snapshot,
                            free_running_seconds=pending.running,
                            conditions_match=actual == snapshot))

    engine._begin_running = record_entry
    results = engine.run(drivers, cars, track, weather, list("AB"),
                         starting_tires={key: "intermediate" for key in "AB"})
    pitter = next(row for row in results if row.driver_id == "B")
    exit_entry = next(row for row in entries if row["driver"] == "B" and row["lap"] == 1)
    return dict(case=name, initial_control=initial_control, later_control=later_control,
                pit_entry=pit_entry, track_entry=exit_entry,
                paid_stops=pitter.pit_stops, pit_stop=pitter.pit_stop_details[0],
                laps_completed=pitter.laps_completed, finish_time=pitter.total_time,
                running_calls=sum(row["driver"] == "B" for row in entries))


def check_cases():
    return [run_case(*case) for case in CASES]


if __name__ == "__main__":
    print(json.dumps(check_cases(), indent=2, allow_nan=False))
