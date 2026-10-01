"""Committed fitting delays share one first-crossing clock across forecasts."""

from copy import deepcopy
from math import nextafter

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.chronological_race import ChronologicalRace, _PendingLap
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import DriverRaceState, RaceSimulator
from f1sim.simulation.race_timing import RaceFinishTimeline


def _snapshot(profile, *, fit=True, control="green"):
    simulator = RaceSimulator(np.random.default_rng(17), tire_warmup=profile)
    simulator.event_manager.vsc_active = control == "vsc"
    simulator.event_manager.safety_car_active = control == "sc"
    engine = ChronologicalRace(simulator)
    engine.track = Track(id="t", name="T", country="T", total_laps=90, base_lap_time=100)
    state = DriverRaceState(
        Driver(id="d", name="D", team_id="t"), Car(team_id="t", team_name="T"), 1,
        current_tire=TIRE_COMPOUNDS[TireCompound.HARD].model_copy(deep=True),
        fit_lap_pending=fit, laps_completed=70, total_time=7000,
    )
    engine.states = {"d": state}
    engine.order = []
    engine.running_paces = {"d": 100.0}
    engine.timeline = RaceFinishTimeline(90, ["d"])
    for lap in range(1, 71):
        engine.timeline.observe_crossing("d", lap, lap * 100.0, is_leader=True)
    engine.pending = {"d": _PendingLap(
        71, 7000, 999999, Weather(), False, state.current_tire, 0, 0,
        on_track=False, paid_stop=True, expected_exit=7099,
    )}
    return engine, state


@pytest.mark.parametrize("control", ["green", "vsc", "sc"])
@pytest.mark.parametrize("profile,fit,fee", [
    ({"hard": 2}, True, 2), ({"hard": 2}, False, 0),
    (None, True, 0), ({"hard": 0}, True, 0), ({"soft": 2}, True, 0),
])
def test_off_track_anchors_share_absolute_fit_delay_without_mutation(control, profile, fit, fee):
    engine, state = _snapshot(profile, fit=fit, control=control)
    pending = engine.pending["d"]
    before = deepcopy((state, pending, engine.running_paces,
                       engine.simulator.rng.bit_generator.state))
    modifier = engine.simulator.event_manager.get_lap_time_modifier()
    first_crossing = 7099 + 100 * modifier + fee
    first_update, pace, _ = engine._weather_projection_clock(7050)
    assert first_update == first_crossing
    assert pace == 100
    flag = engine._projected_flag_time(7050)
    assert (flag - first_crossing) % 100 == 0
    assert engine._projected_progress("d", 7099 + 50, 71, now=7050) == pytest.approx(
        50 / (100 * modifier + fee),
    )
    assert engine._pending_service_exit("d", pending, 7050) == 7099
    assert (state, pending, engine.running_paces,
            engine.simulator.rng.bit_generator.state) == before


def test_fit_crosses_deadline_and_preserves_terminal_exit_ties():
    engine, _ = _snapshot({"hard": 2})
    # The unpaid running component would cross at 7199; the known fit moves
    # that crossing past the 7200 deadline, so the following crossing flags.
    assert engine._weather_projection_clock(7050) == (7201, 100, 1)
    assert engine._projected_flag_time(7050) == 7301
    assert engine._projected_progress("d", 7201, 72, now=7050) == 1
    assert engine._projected_progress("d", 7301, 72, now=7050) == 1
    assert engine._projected_progress("d", nextafter(7301, float("inf")), 72, now=7050) is None


@pytest.mark.parametrize("control", ["green", "vsc"])
def test_native_service_fit_is_forecast_once_then_consumed(monkeypatch, control):
    simulator = RaceSimulator(np.random.default_rng(17), tire_warmup={"soft": .5, "hard": 1.75})
    manager = simulator.event_manager
    reset = manager.reset

    def reset_control():
        reset()
        manager.vsc_active = control == "vsc"

    monkeypatch.setattr(manager, "reset", reset_control)
    monkeypatch.setattr(manager, "process_lap", lambda *args, **kwargs: [])
    monkeypatch.setattr(manager, "_check_mechanical_failure", lambda *args, **kwargs: None)
    monkeypatch.setattr(manager, "_check_random_incident", lambda *args, **kwargs: None)
    monkeypatch.setattr(simulator, "_deploy_overtake_mode_if_eligible",
                        lambda *args, **kwargs: False)

    def stop(state, states, track, lap, *args, **kwargs):
        if lap == 3:
            state.dry_pit_proposal = (lap, TireCompound.HARD)
            return True
        return False

    monkeypatch.setattr(simulator, "_should_pit", stop)
    physics = simulator.lap_simulator.calculate_lap_time
    fuel_distances = []

    def mean_physics(*args, **kwargs):
        fuel_distances.append(args[6])
        return physics(*args, **{**kwargs, "sample_variation": False})

    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", mean_physics)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_pit_stop_time",
                        expected_stationary_time)
    monkeypatch.setattr(Weather, "evolve", lambda self, rng: self.project_surface())
    engine = ChronologicalRace(simulator)
    begin = engine._begin_running
    observations = []

    def observe_fit(state, pending, now):
        if pending.paid_stop:
            assert state.fit_lap_pending and not pending.on_track
            before = deepcopy((state, pending, simulator.rng.bit_generator.state))
            old_pace = engine.running_paces[state.driver.id]
            expected_exit = engine._pending_service_exit(state.driver.id, pending, now)
            anchor = max(now, expected_exit) + old_pace * manager.get_lap_time_modifier() + 1.75
            assert engine._weather_projection_clock(now)[0] == pytest.approx(anchor)
            assert (state, pending, simulator.rng.bit_generator.state) == before
        begin(state, pending, now)
        if pending.paid_stop:
            assert not state.fit_lap_pending
            assert pending.ready == pytest.approx(
                now + pending.running * pending.lap_time_modifier + 1.75,
            )
            assert engine._weather_projection_clock(now)[0] == pending.ready
            assert engine._pending_fit_cost(state.driver.id) == 0
            observations.append((now, pending.ready))

    monkeypatch.setattr(engine, "_begin_running", observe_fit)
    driver = Driver(id="d", name="D", team_id="t")
    car = Car(team_id="t", team_name="T")
    track = Track(id="t", name="T", country="T", total_laps=6, base_lap_time=100)
    result, = engine.run([driver], {"t": car}, track, Weather(), ["d"],
                         starting_tires={"d": TireCompound.SOFT})
    assert result.pit_laps == [3]
    assert len(observations) == 1
    assert all(distance == 6 for distance in fuel_distances)
    assert ("d", 3, observations[0][1]) in engine.crossings
