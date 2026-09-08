"""Projected traffic leaves at each rival's own anticipated finish crossing."""

import pickle
from copy import deepcopy

import numpy as np

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.chronological_race import ChronologicalRace, _PendingLap
from f1sim.simulation.race import DriverRaceState, RaceSimulator
from f1sim.simulation.race_timing import RaceFinishTimeline


def fixture(scheduled=90):
    engine = ChronologicalRace(RaceSimulator(np.random.default_rng(4)))
    engine.states = {
        key: DriverRaceState(Driver(id=key, name=key, team_id=key),
                             Car(team_id=key, team_name=key), index,
                             laps_completed=78 if key == "A" else 64)
        for index, key in enumerate("AB", 1)
    }
    engine.timeline = RaceFinishTimeline(scheduled, engine.states)
    engine.running_paces = {"A": 90, "B": 110}
    engine.order = list("AB")
    engine.pending = {}
    for key, start, ready in [("A", 7020, 7110), ("B", 7040, 7150)]:
        state = engine.states[key]
        engine.pending[key] = _PendingLap(
            state.laps_completed + 1, start, ready, Weather(), False,
            state.current_tire, 0, engine.running_paces[key], running_start=start,
        )
    return engine


def test_leader_forecast_bounds_future_traffic_without_mutation():
    engine = fixture()
    before = pickle.dumps(engine.timeline), deepcopy(engine.simulator.rng.bit_generator.state)
    assert engine._projected_flag_time(7040) == 7290
    assert engine._projected_progress("A", 7289, 65, now=7040) is not None
    assert engine._projected_progress("A", 7290, 65, now=7040) is None
    assert engine._projected_progress("A", 7291, 65, now=7040) is None
    assert (pickle.dumps(engine.timeline), engine.simulator.rng.bit_generator.state) == before


def test_ordinary_lane_after_announcement_keeps_lapped_rival_until_crossing():
    engine = fixture()
    for lap in range(1, 81):
        engine.timeline.observe_crossing("A", lap, lap * 90, is_leader=True)
    engine.states["A"].laps_completed = 80
    engine.pending["A"].lap = 81
    engine.pending["A"].running_start = 7200
    engine.pending["A"].ready = 7290
    assert engine.timeline.time_limit_announced
    exit_time = 7270 + 25 + 3
    assert engine._projected_progress("A", exit_time, 66, now=7270) is None
    assert engine._projected_progress("B", exit_time, 66, now=7270) is not None


def test_lapped_rival_remains_until_own_final_crossing_and_preserves_ties():
    engine = fixture()
    # B crosses at 7150, 7260, then 7370: still racing after A's 7290 flag.
    assert engine._projected_progress("B", 7300, 81, now=7040) is not None
    assert engine._projected_progress("B", 7370, 81, now=7040) == 1
    assert engine._projected_progress("B", 7370, 60, now=7040) is None
    assert engine._projected_progress("B", 7371, 81, now=7040) is None
    assert engine._projected_progress("B", 7480, 81, now=7040) is None


def test_scheduled_finish_and_unavailable_forecast():
    engine = fixture(scheduled=79)
    assert engine._projected_flag_time(7040) == 7110
    assert engine._projected_progress("A", 7111, 65, now=7040) is None
    engine = fixture()
    engine.running_paces.pop("A")
    engine.pending["A"].running = 0
    assert engine._projected_flag_time(7040) is None
    assert engine._projected_progress("B", 7371, 81, now=7040) is not None


def test_long_pit_exit_does_not_project_finished_leader_back_into_traffic(monkeypatch):
    simulator = RaceSimulator(np.random.default_rng(4))
    engine = ChronologicalRace(simulator)
    drivers = [Driver(id=key, name=key, team_id=key) for key in "AB"]
    cars = {key: Car(team_id=key, team_name=key) for key in "AB"}
    track = Track(id="t", name="T", country="T", total_laps=90,
                  base_lap_time=90, pit_lane_delta=248)
    snapshots = []

    def decide(state, states, planning, lap, *args, **kwargs):
        if state.driver.id == "B" and lap == 65:
            snapshots.append(kwargs["traffic_snapshot"])
            assert not engine.timeline.time_limit_announced
            assert engine._projected_flag_time(7040) == 7290
            return True
        return False

    monkeypatch.setattr(simulator, "_should_pit", decide)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time",
                        lambda driver, *a, **kw: 90 if driver.id == "A" else 110)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_pit_stop_time", lambda car: 3)
    monkeypatch.setattr("f1sim.simulation.chronological_race.expected_stationary_time",
                        lambda car: 3)
    monkeypatch.setattr(simulator.event_manager, "process_lap", lambda *a, **kw: [])
    monkeypatch.setattr(simulator.event_manager, "_check_mechanical_failure", lambda *a: None)
    monkeypatch.setattr(simulator.event_manager, "_check_random_incident", lambda *a: None)
    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake",
                        lambda *a, **kw: (True, False))
    monkeypatch.setattr(Weather, "evolve", lambda self, rng: self.model_copy(deep=True))
    results = engine.run(drivers, cars, track, Weather(track_wetness=.3), list("AB"),
                         starting_tires={key: TireCompound.INTERMEDIATE for key in "AB"})
    assert results[0].driver_id == "A" and results[0].laps_completed == 81
    assert engine.timeline.chequered_time == 7290
    assert snapshots[0].rejoin_traffic_cost == 0
