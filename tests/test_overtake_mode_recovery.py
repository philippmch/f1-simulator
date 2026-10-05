"""SC recovery waits for actual field crossings, including unfinished paid laps."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models._native import native_physics
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.events import EventType, RaceEvent
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.race import DriverStatus, RaceSimulator
from f1sim.simulation.strategy_traffic import StrategyTrafficSnapshot


def recovery_race(monkeypatch, *, stop=True, service=600., retire=False, red=False,
                  control_kind="sc", deploy_laps=(1,), running_delays=None, stop_laps=(2,)):
    simulator = RaceSimulator(np.random.default_rng(53))
    engine = ChronologicalRace(simulator, red_flag_pause_seconds=10)
    drivers = [Driver(id=key, name=key, team_id=key) for key in "ABC"]
    cars = {key: Car(team_id=key, team_name=key, reliability=1) for key in "ABC"}
    track = Track(id="test", name="Test", country="Test", total_laps=12,
                  base_lap_time=90, pit_lane_delta=20, safety_car_probability=0)
    manager = simulator.event_manager
    starts, decisions, releases = [], [], []
    actual_physics = simulator.lap_simulator.calculate_lap_time
    actual_begin = engine._begin_running
    actual_control = manager.process_lap

    def physics(*args, **kwargs):
        kwargs["sample_variation"] = False
        delay = (running_delays or {}).get((args[0].id, args[5]), 0.)
        return actual_physics(*args, **kwargs) + delay

    def begin(state, pending, now):
        before = state.overtake_mode_energy
        actual_begin(state, pending, now)
        starts.append(dict(driver=state.driver.id, lap=pending.lap, time=now,
                           interval=engine.control_intervals + 1,
                           allowed=pending.mode_allowed, active=pending.mode_active,
                           gap=pending.detected_gap, before=before,
                           after=state.overtake_mode_energy, paid=pending.paid_stop))

    def decision(state, states, planning, lap, *args, **kwargs):
        decisions.append(dict(driver=state.driver.id, lap=lap, time=state.total_time,
                              allowed=kwargs["current_overtake_mode_allowed"]))
        return stop and state.driver.id == "C" and lap in stop_laps

    def control(lap, *args, **kwargs):
        before = manager.safety_car_active or manager.vsc_active
        events = actual_control(lap, *args, **kwargs)
        if control_kind == "sc" and lap in deploy_laps:
            manager.safety_car_laps_remaining = 1
        elif control_kind == "vsc" and lap in deploy_laps:
            manager.vsc_active = True
            manager.vsc_laps_remaining = 1
            event = RaceEvent(EventType.VIRTUAL_SAFETY_CAR, lap)
            manager.events.append(event)
            events.append(event)
        elif control_kind == "red" and lap in deploy_laps:
            events.append(manager.deploy_red_flag(lap, "Controlled interruption"))
        if before and not (manager.safety_car_active or manager.vsc_active):
            releases.append(lap)
        if red and lap == 4:
            events.append(manager.deploy_red_flag(lap, "Controlled interruption"))
        return events

    def failure(driver, car, track, lap, weather):
        if retire and driver.id == "C" and lap == 2:
            driver.dnf = True
            driver.dnf_reason = "Controlled retirement"
            return RaceEvent(EventType.MECHANICAL_FAILURE, lap, [driver.id])
        return None

    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", physics)
    monkeypatch.setattr(engine, "_begin_running", begin)
    monkeypatch.setattr(simulator, "_should_pit", decision)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_pit_stop_time",
                        lambda car: service if car.team_id == "C" else 2.5)
    monkeypatch.setattr(manager, "process_lap", control)
    monkeypatch.setattr(manager, "_check_mechanical_failure", failure)
    monkeypatch.setattr(manager, "_check_random_incident", lambda *a, **kw: None)
    monkeypatch.setattr(manager, "_deploy_safety_measure", lambda *a, **kw: None)
    monkeypatch.setattr(manager, "_check_severe_weather_red_flag", lambda *a: None)
    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake",
                        lambda *a, **kw: (True, False))
    if control_kind == "sc":
        manager.set_forced_safety_car(list(deploy_laps))
    assert native_physics(), "Forecast definitions and actual lap formula stay native"

    def run():
        starts.clear()
        decisions.clear()
        releases.clear()
        return engine.run(drivers, cars, track, Weather(change_probability=0), list("ABC"),
                          starting_tires={key: TireCompound.MEDIUM for key in "ABC"})

    return engine, run, starts, decisions, releases


def recovery_window(engine, releases):
    assert releases == [2]
    release = next(time for _, lap, time in engine.crossings if lap == releases[0])
    leader_restart = next(time for _, lap, time in engine.crossings
                          if lap == releases[0] + 1)
    tail_crossing = next(time for driver, _, time in engine.crossings
                         if driver == "C" and time > release)
    assert tail_crossing > leader_restart
    return leader_restart, tail_crossing


@pytest.mark.parametrize("service", [400., 600.])
def test_restart_deployment_waits_for_the_paid_car_to_cross(monkeypatch, service):
    engine, run, starts, _, releases = recovery_race(monkeypatch, service=service)
    results = run()
    begin, end = recovery_window(engine, releases)
    blocked = [row for row in starts if begin <= row["time"] < end and not row["paid"]]
    assert blocked and any(row["gap"] is not None and row["gap"] < 1 for row in blocked)
    assert all(not row["allowed"] and not row["active"] for row in blocked)
    assert all(row["before"] == row["after"] for row in blocked)
    assert any(row["time"] >= end and row["allowed"] for row in starts)
    assert all(result.status == DriverStatus.FINISHED for result in results)


def test_pit_decisions_use_the_same_unfinished_field_gate(monkeypatch):
    engine, run, _, decisions, releases = recovery_race(monkeypatch)
    run()
    begin, end = recovery_window(engine, releases)
    blocked = [row for row in decisions if begin <= row["time"] < end]
    assert blocked and all(not row["allowed"] for row in blocked)
    assert any(row["time"] >= end and row["allowed"] for row in decisions)


def test_retirement_releases_the_waiting_field_before_another_car_crosses(monkeypatch):
    engine, run, starts, _, releases = recovery_race(monkeypatch, retire=True, service=400)
    results = run()
    tail = next(result for result in results if result.driver_id == "C")
    assert tail.status == DriverStatus.DNF and tail.laps_completed == 1
    retired = engine.timeline.states["C"].retirement_time
    before = [row for row in starts if row["time"] < retired and row["interval"] > 3]
    after = [row for row in starts if row["time"] >= retired and row["interval"] > 3]
    assert releases == [2] and before and after
    assert all(not row["allowed"] for row in before)
    assert any(row["allowed"] for row in after)


def test_reusing_the_engine_does_not_retain_a_safety_car_recovery_gate(monkeypatch):
    engine, run, starts, _, _ = recovery_race(monkeypatch)
    first = run()
    timeline = [(row["driver"], row["lap"], row["allowed"], row["active"]) for row in starts]
    assert run() == first
    assert [(row["driver"], row["lap"], row["allowed"], row["active"])
            for row in starts] == timeline


def test_red_flag_collection_supersedes_the_previous_safety_car_gate(monkeypatch):
    engine, run, starts, _, releases = recovery_race(monkeypatch, red=True, service=400)
    run()
    assert releases == [2, 5] and engine.suspensions
    resumed = engine.suspensions[0][1]
    # Collection clears the old recovery gate; the resumption SC has its own
    # return at leading interval five and prohibits every initial burst.
    resumption_starts = [row for row in starts if row["time"] == resumed]
    assert resumption_starts
    assert all(not row["allowed"] and not row["active"] for row in resumption_starts)
    assert any(row["time"] >= resumed and row["allowed"] for row in starts)
    assert any(row["time"] >= resumed and row["active"] for row in starts)


@pytest.mark.parametrize("control_kind", ["sc", "vsc", "red"])
def test_control_disables_a_previously_active_pending_burst_without_refunding_energy(
    monkeypatch, control_kind,
):
    engine, run, starts, _, releases = recovery_race(
        monkeypatch, control_kind=control_kind, deploy_laps=(2,), stop=False,
        running_delays={("B", 2): 300.},
    )
    cancellations = []
    actual_interval = engine._leader_interval

    def interval(pending):
        before = {key: (entry.ready, engine.states[key].overtake_mode_energy,
                        engine.states[key].overtake_mode_deployments)
                  for key, entry in engine.pending.items() if entry.mode_active}
        red = actual_interval(pending)
        if not engine.simulator.event_manager.is_active_aero_allowed():
            for key, value in before.items():
                entry = engine.pending[key]
                state = engine.states[key]
                cancellations.append((value, (entry.ready, state.overtake_mode_energy,
                                               state.overtake_mode_deployments),
                                      entry.mode_active, state.overtake_mode_active_lap))
        return red

    monkeypatch.setattr(engine, "_leader_interval", interval)
    run()
    if control_kind == "red":
        assert engine.suspensions
    else:
        assert releases == [3]
    deployed = [row for row in starts if row["lap"] == 2 and row["active"]]
    assert deployed and all(row["after"] < row["before"] for row in deployed)
    assert cancellations
    assert all(before == after for before, after, _, _ in cancellations)
    assert all(not pending_active and not state_active
               for _, _, pending_active, state_active in cancellations)
    assert any(row["lap"] > 4 and row["active"] for row in starts)


def test_a_second_safety_car_requires_a_new_field_crossing(monkeypatch):
    engine, run, starts, decisions, releases = recovery_race(
        monkeypatch, service=250, deploy_laps=(1, 5), stop_laps=(2, 3),
    )
    run()
    assert releases == [2, 6]
    cleared = next(time for _, lap, time in engine.crossings if lap == 6)
    earlier_clear = next(time for _, lap, time in engine.crossings if lap == 2)
    assert any(driver == "C" and earlier_clear < time < cleared
               for driver, _, time in engine.crossings)
    first_tail_crossing = next(time for driver, _, time in engine.crossings
                              if driver == "C" and time > cleared)
    blocked = [row for row in starts if row["interval"] > 7
               and cleared <= row["time"] < first_tail_crossing]
    assert blocked and all(not row["allowed"] for row in blocked)
    assert any(row["time"] >= first_tail_crossing and row["allowed"] for row in decisions)


def test_an_interrupted_run_cannot_leave_a_recovery_gate_in_the_next_race(monkeypatch):
    engine, run, starts, _, _ = recovery_race(monkeypatch)
    actual_start = engine._start_lap

    def interrupted_start(*args, **kwargs):
        if engine.control_intervals > 2:
            raise RuntimeError("Controlled interruption")
        return actual_start(*args, **kwargs)

    monkeypatch.setattr(engine, "_start_lap", interrupted_start)
    with pytest.raises(RuntimeError, match="Controlled interruption"):
        run()
    engine.simulator.event_manager.clear_forced_events()
    monkeypatch.setattr(engine, "_start_lap", actual_start)
    run()
    assert any(row["lap"] == 2 and row["allowed"] for row in starts)
    assert any(row["lap"] == 2 and row["active"] for row in starts)


def test_finish_distance_guard_cannot_price_a_burst_before_field_recovery(monkeypatch):
    from test_chronological_finish_strategy import decision_snapshot

    engine, state, _ = decision_snapshot(monkeypatch)
    engine.control_intervals = 3
    engine.simulator.event_manager.sc_restart_lap_number = 3
    assert engine.simulator.event_manager.is_overtake_mode_allowed(4, engine.weather)
    state.overtake_mode_energy = 1
    traffic = StrategyTrafficSnapshot(.5, None, 0., (.5, None))
    physics = LapSimulator(np.random.default_rng(7))
    driver = state.driver.model_copy(deep=True)
    driver.current_tire_laps = 0
    first = physics.calculate_lap_time(
        driver, state.car, engine.track, state.current_tire, engine.weather, 1, 10,
        gap_to_car_ahead=.5, sample_variation=False,
    )
    deployed = physics.calculate_lap_time(
        driver, state.car, engine.track, state.current_tire, engine.weather, 1, 10,
        gap_to_car_ahead=.5, sample_variation=False, overtake_mode_active=True,
    )
    driver.current_tire_laps = 1
    second = physics.calculate_lap_time(
        driver, state.car, engine.track, state.current_tire, engine.weather, 2, 10,
        sample_variation=False,
    )
    assert first > deployed
    flag = first + second - (first - deployed) / 2
    monkeypatch.setattr(engine, "_projected_flag_time", lambda *a, **kw: flag)
    engine._overtake_restart_waiting.add("C")
    assert not engine._protect_elective_finish_distance(state, engine.track, 0, 0, traffic)
    engine._overtake_restart_waiting.clear()
    assert engine._protect_elective_finish_distance(state, engine.track, 0, 0, traffic)
