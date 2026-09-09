"""Chronological pit decisions see physical traffic and future pit-exit clocks."""

import copy

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.race import DriverStatus, RaceSimulator


def setup(monkeypatch, *, pit_lane=18, stop=False, queue_delay=0):
    drivers = [Driver(id=key, name=key, team_id=key) for key in "AB"]
    cars = {key: Car(team_id=key, team_name=key) for key in "AB"}
    track = Track(id="t", name="T", country="T", total_laps=10,
                  base_lap_time=90, pit_lane_delta=pit_lane)
    simulator = RaceSimulator(np.random.default_rng(4))
    engine = ChronologicalRace(simulator)
    snapshots, running = {}, {}

    def should(state, states, track, lap, *args, **kwargs):
        snapshots[state.driver.id, lap] = kwargs["traffic_snapshot"]
        return stop and state.driver.id == "A" and lap == 2

    def physics(driver, car, track, tire, weather, lap, *a, **kwargs):
        running[driver.id, lap] = kwargs["gap_to_car_ahead"]
        return 90 if driver.id == "A" else 110

    monkeypatch.setattr(simulator, "_should_pit", should)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", physics)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_pit_stop_time", lambda car: 2.5)
    monkeypatch.setattr("f1sim.simulation.chronological_race.expected_stationary_time",
                        lambda car: 2.5)
    monkeypatch.setattr(simulator.event_manager, "_check_mechanical_failure", lambda *a: None)
    monkeypatch.setattr(simulator.event_manager, "_check_random_incident", lambda *a, **kw: None)
    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake",
                        lambda *a, **k: (True, False))
    monkeypatch.setattr(Weather, "evolve", lambda self, rng: self.model_copy(deep=True))

    def control(lap, *a, **k):
        if lap == 1 and queue_delay:
            engine.box_releases["A"] = 90 + queue_delay
            engine.expected_box_releases["A"] = 90 + queue_delay
        return []

    monkeypatch.setattr(simulator.event_manager, "process_lap", control)

    def run():
        return engine.run(drivers, cars, track, Weather(track_wetness=.3), list("AB"),
                          starting_tires={key: TireCompound.INTERMEDIATE for key in "AB"})

    return engine, run, snapshots, running


def test_live_strategy_gets_physical_gaps_instead_of_stale_crossings(monkeypatch):
    _, run, snapshots, _ = setup(monkeypatch)
    run()
    assert snapshots["A", 2].gap_ahead == pytest.approx(90 / 110 * 90)
    assert snapshots["A", 2].gap_behind == 20
    assert snapshots["A", 6].gap_ahead == pytest.approx(10 / 110 * 90)
    assert snapshots["A", 6].gap_behind == 100
    assert snapshots["B", 2].gap_ahead == pytest.approx(20 / 90 * 110)


@pytest.mark.parametrize("pit_lane,queue_delay", [(18, 0), (16, 2), (10, 0), (100, 0)])
def test_rejoin_forecast_matches_actual_exit_across_rival_crossings(
    monkeypatch, pit_lane, queue_delay,
):
    engine, run, snapshots, running = setup(
        monkeypatch, pit_lane=pit_lane, stop=True, queue_delay=queue_delay,
    )
    run()
    traffic = engine.simulator.lap_simulator.traffic_pace_contribution
    snapshot = snapshots["A", 2]
    assert snapshot.rejoin_traffic_cost == pytest.approx(
        traffic(running["A", 2]) - traffic(snapshot.gap_ahead)
    )
    assert engine.pit_exits[0] == ("A", 2, 90 + pit_lane + 2.5 + queue_delay)
    if pit_lane + queue_delay == 18:
        assert running["A", 2] == pytest.approx(.5 / 110 * 90)
        assert snapshot.rejoin_traffic_cost > .39


@pytest.mark.parametrize("pit_lane", [17.5, 127.5])
def test_exact_exit_crossing_tie_uses_scheduler_distance_priority(monkeypatch, pit_lane):
    # At 110 A's lap-2 exit precedes B's lap-1 crossing by distance. At 220,
    # A's already-enqueued exit precedes B's future lap-2 crossing by serial.
    engine, run, snapshots, running = setup(monkeypatch, pit_lane=pit_lane, stop=True)
    run()
    traffic = engine.simulator.lap_simulator.traffic_pace_contribution
    assert snapshots["A", 2].rejoin_traffic_cost == pytest.approx(
        traffic(running["A", 2]) - traffic(snapshots["A", 2].gap_ahead)
    )
    assert running["A", 2] == 90


def pending_fixture(monkeypatch):
    engine, run, _, _ = setup(monkeypatch)
    monkeypatch.setattr(engine, "_enqueue", lambda *a: None)
    run()  # Keep the initialized cars pending, without consuming a crossing.
    engine.order = ["B", "A"]
    engine.running_paces = {"A": 90, "B": 110}
    del engine.pending["A"]
    engine.states["A"].laps_completed = 1
    return engine


def test_snapshot_is_pure_and_independent_of_race_rank_and_old_clocks(monkeypatch):
    engine = pending_fixture(monkeypatch)
    own = engine.states["A"]
    before = copy.deepcopy((engine.order, engine.states, engine.pending, engine.box_releases,
                            engine.running_paces, engine.simulator.rng.bit_generator.state))
    first = engine._strategy_traffic(own, 90, 0)
    assert before == (engine.order, engine.states, engine.pending, engine.box_releases,
                      engine.running_paces, engine.simulator.rng.bit_generator.state)
    own.position, own.total_time = 12, 8000
    engine.states["B"].position, engine.states["B"].total_time = 1, 10
    engine.states["B"].laps_completed = 7
    assert engine._strategy_traffic(own, 90, 0) == first


def test_pit_lane_rival_is_not_a_current_gap_but_can_rejoin_before_us(monkeypatch):
    engine = pending_fixture(monkeypatch)
    engine.order = ["A"]
    rival = engine.pending["B"]
    rival.on_track = False
    rival.running_start = None
    rival.ready = 110
    rival.running = 0
    snapshot = engine._strategy_traffic(engine.states["A"], 90, 0)
    assert snapshot.gap_ahead is None and snapshot.gap_behind is None
    traffic = engine.simulator.lap_simulator.traffic_pace_contribution
    assert snapshot.rejoin_traffic_cost == pytest.approx(traffic(.5 / 110 * 90))
    rival.ready = 120  # Still stationary when we rejoin.
    assert engine._strategy_traffic(engine.states["A"], 90, 0).rejoin_traffic_cost == 0


@pytest.mark.parametrize("flag", ["safety_car_active", "vsc_active", "red_flag_active"])
def test_neutralization_has_no_green_rejoin_penalty(monkeypatch, flag):
    engine = pending_fixture(monkeypatch)
    setattr(engine.simulator.event_manager, flag, True)
    assert engine._strategy_traffic(engine.states["A"], 90, 0).rejoin_traffic_cost == 0


def test_finished_retired_and_scheduled_terminal_cars_are_not_future_traffic(monkeypatch):
    engine = pending_fixture(monkeypatch)
    for status in [DriverStatus.FINISHED, DriverStatus.DNF]:
        engine.states["B"].status = status
        assert engine._projected_progress("B", 111, 2) is None
    engine.states["B"].status = DriverStatus.RACING
    engine.pending["B"].lap = 10
    assert engine._projected_progress("B", 109, 2) == pytest.approx(109 / 110)
    assert engine._projected_progress("B", 110, 2) is None
    engine.pending["B"].lap = 1
    assert engine._projected_progress("B", 1200, 2) is None


def test_known_flag_does_not_project_lapped_car_into_an_extra_lap(monkeypatch):
    engine = pending_fixture(monkeypatch)
    from f1sim.simulation.race_timing import RaceFinishTimeline

    engine.timeline = RaceFinishTimeline(1, ["A", "B"])
    engine.timeline.observe_crossing("A", 1, 100, is_leader=True)
    assert engine._projected_progress("B", 109, 2) == pytest.approx(109 / 110)
    assert engine._projected_progress("B", 111, 2) is None


def test_extrapolation_preserves_pending_sc_delay_then_uses_current_green_pace(monkeypatch):
    engine = pending_fixture(monkeypatch)
    rival = engine.pending["B"]
    rival.neutralized = True
    rival.lap_time_modifier = 1.4
    rival.ready = 154
    assert engine._projected_progress("B", 153, 2) == pytest.approx(153 / 154)
    assert engine._projected_progress("B", 155, 2) == pytest.approx(1 / 110)


def test_existing_same_distance_crossing_wins_tie_against_new_pit_exit(monkeypatch):
    engine = pending_fixture(monkeypatch)
    engine.pending["B"].lap = 2
    assert engine._projected_progress("B", 110, 2) == 0
