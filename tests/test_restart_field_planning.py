"""Suspension fits use one observed field before restart running or service."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.race import DriverStatus, RaceSimulator


@pytest.mark.parametrize("finite", [False, True])
@pytest.mark.parametrize("leader_paid", [False, True])
@pytest.mark.parametrize("follower_custom", [False, True])
def test_free_fits_finish_before_releasing_any_restart_lap(
    monkeypatch, finite, leader_paid, follower_custom,
):
    monkeypatch.setattr("f1sim.simulation.race_timing.RACING_TIME_LIMIT_SECONDS", 450.)
    simulator = RaceSimulator(np.random.default_rng(19))
    engine = ChronologicalRace(simulator, red_flag_pause_seconds=0.)
    control = simulator.event_manager
    control.set_forced_red_flag(2)
    monkeypatch.setattr(control, "_deploy_safety_measure", lambda *args, **kwargs: None)
    monkeypatch.setattr(control, "_check_mechanical_failure", lambda *args, **kwargs: None)
    monkeypatch.setattr(control, "_check_random_incident", lambda *args, **kwargs: None)
    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake",
                        lambda *args, **kwargs: (False, False))
    events, frozen, fitted, choices, leading_views = [], {}, {}, {}, []
    planning_track = engine._planning_track
    fit = engine._fit_red_flag_set
    begin_running = engine._begin_running
    choose = simulator._custom_plan_replacement_choice
    leading_context = engine._leading_finish_context

    def capture_leading_context(state, now, *, restart=False):
        context = leading_context(state, now, restart=restart)
        if restart:
            assert fitted.keys() == engine.states.keys()
            assert not engine.expected_box_releases
            leading_views.append((state.driver.id, context))
        return context

    def capture_horizon(state, now, *, restart=False):
        planning = planning_track(state, now, restart=restart)
        if restart:
            frozen[state.driver.id] = (planning, state.strategy_finish_context)
            events.append(("forecast", state.driver.id))
        return planning

    def capture_fit(state, planning, *args):
        assert planning is frozen[state.driver.id][0]
        assert state.strategy_finish_context == frozen[state.driver.id][1]
        # A paid service entered after the restart cannot affect a tyre choice
        # made while the whole field is still under suspension.
        assert not engine.expected_box_releases
        events.append(("fit", state.driver.id))
        fit(state, planning, *args)
        fitted[state.driver.id] = state.current_tire.compound

    def capture_choice(state, planning, weather, lap, **kwargs):
        choice = choose(state, planning, weather, lap, **kwargs)
        if kwargs.get("free_fit"):
            choices[state.driver.id] = choice
        return choice

    def capture_running(state, pending, start):
        if engine.suspensions and start == engine.suspensions[-1][1]:
            assert fitted.keys() == engine.states.keys()
            assert not engine.free_refits
            assert len(leading_views) == 1 and leading_views[0][0] == "A"
            assert state.strategy_leading_finish_context is (
                leading_views[0][1] if state.driver.id == "A" else None
            )
            events.append(("running", state.driver.id))
        begin_running(state, pending, start)

    monkeypatch.setattr(engine, "_planning_track", capture_horizon)
    monkeypatch.setattr(engine, "_fit_red_flag_set", capture_fit)
    monkeypatch.setattr(engine, "_begin_running", capture_running)
    monkeypatch.setattr(simulator, "_custom_plan_replacement_choice", capture_choice)
    monkeypatch.setattr(engine, "_leading_finish_context", capture_leading_context)
    drivers = [Driver(id=key, name=key, team_id=key, skill_rating=skill)
               for key, skill in (("A", .4), ("B", 1.))]
    cars = {key: Car(team_id=key, team_name=key, base_pace=pace)
            for key, pace in (("A", .4), ("B", 1.))}
    track = Track(id="T", name="Synthetic", country="Synthetic", total_laps=12,
                  base_lap_time=90., pit_lane_delta=20., safety_car_probability=0.,
                  overtake_difficulty=1.)
    records = {key: [dict(id="S", compound="soft", age=0),
                     dict(id="H", compound="hard", age=0)] for key in "AB"}
    plans = {"A": [dict(lap=3, compound="hard")] if leader_paid else []}
    if follower_custom:
        plans["B"] = [dict(lap=7, compound="hard")]
    results = engine.run(
        drivers, cars, track, Weather(change_probability=0.), list("AB"),
        starting_tires={key: TireCompound.SOFT for key in "AB"},
        tire_inventory=records if finite else None, pit_plans=plans,
    )

    assert set(events[:2]) == {("forecast", key) for key in "AB"}
    assert set(events[2:4]) == {("fit", key) for key in "AB"}
    assert all(kind == "running" for kind, _ in events[4:])
    assert frozen["A"][1] is not None
    assert frozen["B"][1] is None
    assert leading_views[0][1] is not None
    assert all(row.status == DriverStatus.FINISHED and row.race_time_limited for row in results)
    if finite and follower_custom and leader_paid:
        # Only keeping S available now can fulfill the later H request in the
        # frozen seven-lap follower horizon; fitting the sole H would skip it.
        assert frozen["B"][0].total_laps == 7
        assert choices["B"].set_id == "S"
        assert choices["B"].instructions == 1
    if leader_paid:
        leader = next(row for row in results if row.driver_id == "A")
        assert leader.pit_plan_history[0]["status"] == "executed"
        assert leader.pit_laps == [3]


@pytest.mark.parametrize("survivor", ["A", "B"])
def test_failed_free_fit_is_retired_before_releasing_survivors(monkeypatch, survivor):
    simulator = RaceSimulator(np.random.default_rng(23))
    engine = ChronologicalRace(simulator, red_flag_pause_seconds=0.)
    control = simulator.event_manager
    control.set_forced_red_flag(2)
    monkeypatch.setattr(control, "_deploy_safety_measure", lambda *args, **kwargs: None)
    monkeypatch.setattr(control, "_check_mechanical_failure", lambda *args, **kwargs: None)
    monkeypatch.setattr(control, "_check_random_incident", lambda *args, **kwargs: None)
    monkeypatch.setattr(simulator, "_should_pit", lambda *args, **kwargs: False)
    fit_ids, running_ids = [], []
    resume = engine._resume_if_collected
    fit = engine._fit_red_flag_set
    begin = engine._begin_running

    def wet_restart(now):
        active = {key for key, state in engine.states.items()
                  if state.status == DriverStatus.RACING}
        if active <= engine.red_waiting:
            engine.weather = Weather(track_wetness=.9, rain_intensity=.9, change_probability=0.)
        resume(now)

    def observe_fit(state, *args):
        fit_ids.append(state.driver.id)
        fit(state, *args)

    def observe_running(state, pending, now):
        if engine.suspensions and now == engine.suspensions[-1][1]:
            assert set(fit_ids) == set("AB")
            assert not engine.free_refits
            retired_id = next(key for key in "AB" if key != survivor)
            assert engine.states[retired_id].status == DriverStatus.DNF
            running_ids.append(state.driver.id)
        begin(state, pending, now)

    monkeypatch.setattr(engine, "_resume_if_collected", wet_restart)
    monkeypatch.setattr(engine, "_fit_red_flag_set", observe_fit)
    monkeypatch.setattr(engine, "_begin_running", observe_running)
    drivers = [Driver(id=key, name=key, team_id=key) for key in "AB"]
    cars = {key: Car(team_id=key, team_name=key) for key in "AB"}
    track = Track(id="T", name="Synthetic", country="Synthetic", total_laps=6,
                  base_lap_time=90., safety_car_probability=0.)
    records = {key: [dict(id="S", compound="soft", age=0)] for key in "AB"}
    records[survivor].append(dict(id="W", compound="wet", age=0))
    results = engine.run(drivers, cars, track, Weather(change_probability=0.), list("AB"),
                         starting_tires={key: TireCompound.SOFT for key in "AB"},
                         tire_inventory=records, pit_plans={key: [] for key in "AB"})
    assert running_ids == [survivor]
    assert engine.states[survivor].status == DriverStatus.FINISHED
    retired = next(row for row in results if row.driver_id != survivor)
    assert retired.status == DriverStatus.DNF
    assert retired.laps_completed == 2
    assert retired.pit_stops == 0
    assert retired.tire_inventory[0]["age"] == 2
