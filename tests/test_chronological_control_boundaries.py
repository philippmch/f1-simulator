"""Live control signals govern encounters without rewriting started lap clocks."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.events import EventType, RaceEvent
from f1sim.simulation.race import RaceSimulator


def setup(monkeypatch, signal=None, *, laps=3, paces=None, signal_lap=1,
          release=False):
    paces = paces or {"A": 90, "B": 110, "C": 100}
    sim = RaceSimulator(np.random.default_rng(5))
    engine = ChronologicalRace(sim)
    manager = sim.event_manager
    monkeypatch.setattr(manager, "_check_random_incident", lambda *a, **kw: None)
    monkeypatch.setattr(manager, "_check_mechanical_failure", lambda *a, **kw: None)
    monkeypatch.setattr(manager, "_deploy_safety_measure", lambda *a, **kw: None)
    monkeypatch.setattr(sim, "_should_pit", lambda *a, **kw: False)
    monkeypatch.setattr(Weather, "evolve", lambda self, rng: self.model_copy(deep=True))
    monkeypatch.setattr(sim.lap_simulator, "calculate_lap_time",
                        lambda driver, *a, **kw: paces[driver.id])
    calls, pre_flags, fits = [], [], []
    process = manager.process_lap

    def control(lap, *args, **kwargs):
        pre_flags.append(engine.timeline.chequered_time)
        events = process(lap, *args, **kwargs)
        if signal == "vsc" and lap == signal_lap:
            manager.vsc_active = True
            manager.vsc_laps_remaining = 1 if release else 10
            event = RaceEvent(EventType.VIRTUAL_SAFETY_CAR, lap)
            manager.events.append(event)
            events.append(event)
        if release and signal == "sc" and lap == signal_lap:
            manager.safety_car_laps_remaining = 1
        return events

    def passing(driver, car, defender, *args, **kwargs):
        calls.append((driver.id, defender.id, manager.current_lap,
                      manager.safety_car_active, manager.vsc_active, manager.red_flag_active,
                      engine.pending[driver.id].neutralized,
                      engine.pending[defender.id].neutralized))
        return True, False

    monkeypatch.setattr(manager, "process_lap", control)
    monkeypatch.setattr(sim.overtaking_model, "attempt_overtake", passing)
    actual_fit = sim._fit_tire

    def fit(state, compound):
        fits.append(state.driver.id)
        return actual_fit(state, compound)

    monkeypatch.setattr(sim, "_fit_tire", fit)
    if signal == "sc":
        manager.set_forced_safety_car(signal_lap)
    elif signal == "red":
        manager.set_forced_red_flag(signal_lap)
    drivers = [Driver(id=d, name=d, team_id=d) for d in paces]
    cars = {d: Car(team_id=d, team_name=d) for d in paces}
    track = Track(id="T", name="T", country="T", total_laps=laps, base_lap_time=90)

    def run():
        return engine.run(drivers, cars, track, Weather(track_wetness=.3, rain_intensity=.3),
                          list(paces), starting_tires={d: TireCompound.INTERMEDIATE for d in paces})

    return engine, run, calls, pre_flags, fits


@pytest.mark.parametrize("signal", ["sc", "vsc"])
@pytest.mark.parametrize("higher_lap", [False, True])
def test_live_signal_blocks_green_started_passes_and_blue_flags(monkeypatch, signal, higher_lap):
    engine, run, calls, _, _ = setup(
        monkeypatch, signal, signal_lap=2 if higher_lap else 1,
        paces={"A": 90, "B": 250, "C": 100} if higher_lap else None,
    )
    run()
    if higher_lap:
        # C completed lap one in green, then catches B's first lap after
        # A's second crossing has deployed control at t=180.
        b_cross = next(t for d, lap, t in engine.crossings if d == "B" and lap == 1)
        c_cross = next(t for d, lap, t in engine.crossings if d == "C" and lap == 2)
        assert b_cross == 250
        assert c_cross > b_cross
    else:
        assert engine.crossings[0] == ("A", 1, 90)
        assert engine.crossings[1] == ("B", 1, 110)
        assert engine.crossings[2][0] == "C"
        assert engine.crossings[2][2] > 110
    assert not any(any(call[3:]) for call in calls)


def test_green_control_allows_caught_predecessor_pass(monkeypatch):
    engine, run, calls, _, _ = setup(monkeypatch)
    run()
    assert engine.crossings[:2] == [("A", 1, 90), ("C", 1, 100)]
    assert any(call[:2] == ("C", "B") for call in calls)


def test_completed_pass_is_not_undone_by_later_signal(monkeypatch):
    engine, run, calls, _, _ = setup(monkeypatch, "sc", paces={"A": 90, "B": 110, "C": 80})
    run()
    assert engine.crossings[0] == ("C", 1, 80)
    assert [call[:2] for call in calls[:2]] == [("C", "B"), ("C", "A")]


@pytest.mark.parametrize("signal", ["sc", "vsc"])
def test_countdown_release_keeps_old_neutralized_laps_protected(monkeypatch, signal):
    engine, run, calls, _, _ = setup(monkeypatch, signal, laps=8, release=True)
    run()
    assert not any(any(call[3:]) for call in calls)
    assert any(call[2] >= 3 for call in calls)


@pytest.mark.parametrize("signal", ["sc", "vsc", "red"])
def test_final_interval_control_precedes_flag_without_restarting(monkeypatch, signal):
    engine, run, _, pre_flags, fits = setup(
        monkeypatch, signal, signal_lap=3, paces={"A": 90, "B": 110},
    )
    results = run()
    manager = engine.simulator.event_manager
    assert pre_flags == [None, None, None]
    assert engine.timeline.chequered_time == 270
    assert [result.total_time for result in results] == [270, 330]
    assert engine.green_streak == 0
    assert engine.has_two_green
    assert not engine.suspensions
    assert not engine.free_refits
    assert not fits
    assert not manager.red_flag_just_ended
    assert manager.red_flag_restart_lap_number is None
    if signal == "red":
        assert manager.red_flag_active
    types = {"sc": EventType.SAFETY_CAR, "vsc": EventType.VIRTUAL_SAFETY_CAR,
             "red": EventType.RED_FLAG}
    assert any(event.event_type == types[signal] and event.lap == 3 for event in manager.events)


@pytest.mark.parametrize("signal", ["sc", "vsc", "red"])
@pytest.mark.parametrize("laps", [2, 3])
def test_final_signal_protects_trailing_order_and_green_lap_points(monkeypatch, signal, laps):
    engine, run, calls, _, fits = setup(monkeypatch, signal, signal_lap=laps, laps=laps)

    def physics(driver, car, track, tire, weather, lap, total_laps, **kwargs):
        if lap == laps:
            return {"A": 90, "B": 150, "C": 90}[driver.id]
        return {"A": 90, "B": 100, "C": 110}[driver.id]

    monkeypatch.setattr(engine.simulator.lap_simulator, "calculate_lap_time", physics)
    results = run()
    # C catches B only after A has taken the flag under the new signal.
    # The finish boundary must neither release that restriction nor give
    # this neutralized final interval credit as a second green lap.
    assert [result.driver_id for result in results] == ["A", "B", "C"]
    assert results[0].total_time == 90 * laps
    assert results[1].total_time == 100 * (laps - 1) + 150
    assert results[2].total_time > results[1].total_time
    assert not any(call[2] == laps for call in calls)
    assert [result.points_awarded for result in results] == (
        [0, 0, 0] if laps == 2 else [25, 18, 15]
    )
    assert not fits and not engine.suspensions
    # Terminal control state from the first run cannot contaminate a reuse.
    assert run() == results
