"""Chronological pit decisions see physical traffic and future pit-exit clocks."""

import copy

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import DriverStatus, RaceSimulator


def setup(monkeypatch, *, pit_lane=18, stop=False, queue_delay=0, driver_ids="AB",
          total_laps=10, lap_times=None, native_strategy=False):
    drivers = [Driver(id=key, name=key, team_id=key) for key in driver_ids]
    cars = {key: Car(team_id=key, team_name=key) for key in driver_ids}
    track = Track(id="t", name="T", country="T", total_laps=total_laps,
                  base_lap_time=90, pit_lane_delta=pit_lane)
    simulator = RaceSimulator(np.random.default_rng(4))
    engine = ChronologicalRace(simulator)
    snapshots, running = {}, {}
    native_should_pit = simulator._should_pit

    def should(state, states, track, lap, *args, **kwargs):
        snapshots[state.driver.id, lap] = kwargs["traffic_snapshot"]
        if native_strategy:
            return native_should_pit(state, states, track, lap, *args, **kwargs)
        return stop and state.driver.id == "A" and lap == 2

    def physics(driver, car, track, tire, weather, lap, *a, **kwargs):
        running[driver.id, lap] = kwargs["gap_to_car_ahead"]
        if lap_times is not None:
            return lap_times[driver.id]
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
        return engine.run(drivers, cars, track, Weather(track_wetness=.3), list(driver_ids),
                          starting_tires={key: TireCompound.INTERMEDIATE for key in driver_ids})

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
    assert snapshot.current_traffic_gaps == pytest.approx(
        (snapshot.gap_ahead, running["A", 2]),
    )
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


def test_native_weather_stop_rejoins_behind_lapped_rival_at_two_crossing_ties(monkeypatch):
    engine, run, snapshots, running = setup(
        monkeypatch, pit_lane=17.5, driver_ids="ABCD", total_laps=2,
        lap_times={"A": 90, "B": 110, "C": 110, "D": 130}, native_strategy=True,
    )
    monkeypatch.setattr(
        Weather, "evolve",
        lambda self, rng: self.model_copy(
            update={"track_wetness": 0.0, "rain_intensity": 0.0}, deep=True,
        ),
    )
    calculate_lap_time = engine.simulator.lap_simulator.calculate_lap_time
    entry = {}

    def record_entry(driver, car, track, tire, weather, lap, *args, **kwargs):
        if driver.id == "A" and lap == 2:
            entry.update(
                time=engine.pending["A"].running_start,
                order=tuple(engine.order),
                predecessor_lap=engine.pending["D"].lap,
                compound=tire.compound,
            )
        return calculate_lap_time(driver, car, track, tire, weather, lap, *args, **kwargs)

    monkeypatch.setattr(engine.simulator.lap_simulator, "calculate_lap_time", record_entry)
    run()

    snapshot = snapshots["A", 2]
    assert snapshot.current_traffic_gaps == pytest.approx((62.3076923, 76.1538462))
    assert snapshot.current_traffic_gaps[1] == pytest.approx(running["A", 2])
    assert ("A", 2, 110) in engine.pit_exits
    assert entry == {
        "time": 110,
        "order": ("B", "C", "D", "A"),
        "predecessor_lap": 1,
        "compound": TireCompound.SOFT,
    }
    assert [crossing for crossing in engine.crossings if crossing[2] == 110] == [
        ("B", 1, 110), ("C", 1, 110),
    ]
    stop = engine.states["A"].pit_stop_details[0]
    assert (stop["lap"], stop["from_compound"], stop["to_compound"]) == (
        2, "intermediate", "soft",
    )
    assert stop["decision_reason"] == "critical_weather"
    assert stop["queue_time"] == 0


def test_native_weather_stop_forecasts_rival_pending_fit_fee_before_service_exit(monkeypatch):
    fit_fee = 20
    simulator = RaceSimulator(np.random.default_rng(23), tire_warmup={"wet": fit_fee})
    engine = ChronologicalRace(simulator)
    drivers = [Driver(id=key, name=key, team_id=key) for key in "AB"]
    cars = {key: Car(team_id=key, team_name=key) for key in "AB"}
    track = Track(id="t", name="T", country="T", total_laps=2,
                  base_lap_time=1, pit_lane_delta=.1)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time",
                        lambda *a, **k: track.base_lap_time)
    # Align actual service with its native mean to isolate the pending fitting fee.
    monkeypatch.setattr(simulator.lap_simulator, "calculate_pit_stop_time",
                        expected_stationary_time)
    monkeypatch.setattr(simulator.event_manager, "process_lap", lambda *a, **k: [])
    monkeypatch.setattr(simulator.event_manager, "_check_mechanical_failure", lambda *a: None)
    monkeypatch.setattr(simulator.event_manager, "_check_random_incident", lambda *a, **k: None)
    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake",
                        lambda *a, **k: (True, False))
    monkeypatch.setattr(
        Weather, "evolve", lambda self, rng: self.model_copy(
            update={"track_wetness": 0, "rain_intensity": 0}, deep=True,
        ),
    )
    native_start = engine._start_lap

    def force_rival_service(state, now, **kwargs):
        if state.driver.id == "B" and state.laps_completed == 0:
            # Forced execution bypasses the elective finish-distance protection.
            state.force_pit_next_lap = True
            state.pit_plan_target = TireCompound.WET
        return native_start(state, now, **kwargs)

    monkeypatch.setattr(engine, "_start_lap", force_rival_service)
    native_should_pit = simulator._should_pit
    decision = {}

    def record_decision(state, states, planning, lap, *args, **kwargs):
        should_pit = native_should_pit(state, states, planning, lap, *args, **kwargs)
        if state.driver.id == "A" and lap == 2:
            rival = engine.pending["B"]
            decision.update(
                should_pit=should_pit, snapshot=kwargs["traffic_snapshot"],
                rival_on_track=rival.on_track,
                rival_in_service=any(
                    record.driver_id == "B"
                    and record.service_start <= state.total_time < record.service_end
                    for record in engine.pit_service_records
                ),
                rival_fit_pending=engine.states["B"].fit_lap_pending,
                rival_compound=engine.states["B"].current_tire.compound,
                rival_exit=engine._pending_service_exit("B", rival, state.total_time),
                candidate_exit=(state.total_time + track.pit_lane_delta
                                * simulator._pit_lane_factor()
                                + expected_stationary_time(state.car)),
            )
        return should_pit

    monkeypatch.setattr(simulator, "_should_pit", record_decision)
    native_begin = engine._begin_running
    entry, rival_run, fees = {}, {}, []

    def record_begin(state, pending, now):
        native_begin(state, pending, now)
        if state.driver.id == "B" and pending.lap == 1:
            rival_run.update(start=now, duration=pending.ready - now,
                             fit_pending=state.fit_lap_pending)
        if state.driver.id == "A" and pending.lap == 2:
            entry.update(time=now, gap=pending.detected_gap, order=tuple(engine.order))

    monkeypatch.setattr(engine, "_begin_running", record_begin)
    native_consume = simulator._consume_tire_warmup

    def record_fee(state):
        fee = native_consume(state)
        if state.driver.id == "B" and fee:
            fees.append(fee)
        return fee

    monkeypatch.setattr(simulator, "_consume_tire_warmup", record_fee)
    engine.run(
        drivers, cars, track, Weather(track_wetness=.5, rain_intensity=.1), list("AB"),
        starting_tires={key: TireCompound.INTERMEDIATE for key in "AB"},
    )

    assert decision["should_pit"] is True
    assert decision["rival_on_track"] is False and decision["rival_fit_pending"] is True
    assert decision["rival_in_service"] is True
    assert decision["rival_compound"] == TireCompound.WET
    assert decision["rival_exit"] == pytest.approx(
        track.pit_lane_delta + expected_stationary_time(cars["B"]),
    )
    assert decision["candidate_exit"] - decision["rival_exit"] == pytest.approx(
        track.base_lap_time,
    )
    expected_gap = track.base_lap_time ** 2 / (track.base_lap_time + fit_fee)
    snapshot = decision["snapshot"]
    assert snapshot.current_traffic_gaps[0] is None
    assert snapshot.current_traffic_gaps[1] == pytest.approx(expected_gap)
    assert entry["gap"] == pytest.approx(snapshot.current_traffic_gaps[1])
    assert entry["time"] == pytest.approx(decision["candidate_exit"])
    assert entry["order"] == ("B", "A")
    assert rival_run["start"] == pytest.approx(decision["rival_exit"])
    assert rival_run["duration"] == pytest.approx(track.base_lap_time + fit_fee)
    rival_crossing, = [time for driver, lap, time in engine.crossings if driver == "B" and lap == 1]
    assert rival_crossing - rival_run["start"] == pytest.approx(track.base_lap_time + fit_fee)
    assert rival_run["fit_pending"] is False
    assert fees == [fit_fee]
    stop = engine.states["A"].pit_stop_details[0]
    assert (stop["from_compound"], stop["to_compound"], stop["decision_reason"]) == (
        "intermediate", "soft", "critical_weather",
    )


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
