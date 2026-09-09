"""A red flag closes pit exit and restarts the field on one absolute timeline."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.events import EventType, RaceEvent
from f1sim.simulation.race import DriverStatus, RaceSimulator


def setup(monkeypatch, *, pause=600, laps=3, red=(1,), paces=None, pit=False):
    paces = paces or {"A": 90, "B": 110}
    simulator = RaceSimulator(np.random.default_rng(7))
    engine = ChronologicalRace(simulator, red_flag_pause_seconds=pause)
    drivers = [Driver(id=key, name=key, team_id=key) for key in paces]
    cars = {key: Car(team_id=key, team_name=key) for key in paces}
    track = Track(id="t", name="T", country="T", total_laps=laps, base_lap_time=90)
    control = simulator.event_manager
    for lap in red:
        control.set_forced_red_flag(lap)
    monkeypatch.setattr(control, "_deploy_safety_measure", lambda *a: None)
    monkeypatch.setattr(control, "_check_mechanical_failure", lambda *a: None)
    monkeypatch.setattr(control, "_check_random_incident", lambda *a, **kw: None)
    monkeypatch.setattr(Weather, "evolve", lambda self, rng: self.model_copy(deep=True))
    monkeypatch.setattr(simulator, "_should_pit", lambda state, states, track, lap, *a, **k:
                        pit and state.driver.id == "B" and lap == 2)
    samples, services, fits, attempts = [], [], [], []

    def physics(driver, car, track, tire, weather, lap, total_laps, **kwargs):
        pending = engine.pending[driver.id]
        samples.append((driver.id, lap, pending.running_start, driver.current_tire_laps,
                        total_laps, pending.restart_boost, kwargs["overtake_mode_active"]))
        return paces[driver.id]

    def passing(*args, **kwargs):
        attempts.append(engine.regrouping)
        return True, False

    actual_fit = simulator._fit_tire

    def fit(state, compound):
        fits.append((state.driver.id, state.laps_completed))
        actual_fit(state, compound)

    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", physics)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_pit_stop_time",
                        lambda car: services.append(car.team_id) or 100)
    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake", passing)
    monkeypatch.setattr(simulator, "_fit_tire", fit)

    def run():
        return engine.run(drivers, cars, track, Weather(track_wetness=.3, rain_intensity=.3),
                          list(paces), starting_tires={key: TireCompound.INTERMEDIATE
                                                     for key in paces})

    return engine, run, samples, services, fits, attempts


@pytest.mark.parametrize("pause", [0, 600, 1800])
def test_field_waits_for_common_restart_without_rewinding_crossings(monkeypatch, pause):
    engine, run, samples, services, fits, _ = setup(monkeypatch, pause=pause)
    results = run()
    assert engine.crossings[:2] == [("A", 1, 90), ("B", 1, 110)]
    assert engine.suspensions == [(90, 110 + pause, ("A", "B"))]
    restart = [sample for sample in samples if sample[1] == 2]
    assert [sample[2] for sample in restart] == [110 + pause, 110 + pause]
    assert all(sample[3] == 0 and sample[5] and not sample[6] for sample in restart)
    assert fits == [("A", 1), ("B", 1)] and not services
    assert [row.total_time for row in results] == [290 + pause, 330 + pause]
    assert [row.fastest_lap for row in results] == [90, 110]
    assert engine.timeline.total_suspension_seconds == 20 + pause
    clocks = [time for _, _, time in engine.crossings]
    assert clocks == sorted(clocks)


def test_lapped_restart_keeps_completed_distance_and_saved_order(monkeypatch):
    engine, run, samples, _, _, _ = setup(
        monkeypatch, laps=6, red=(2,), paces={"A": 90, "B": 300},
    )
    run()
    assert engine.suspensions == [(180, 900, ("A", "B"))]
    restarted = [sample[:3] for sample in samples if sample[2] == 900]
    assert restarted == [("A", 3, 900), ("B", 2, 900)]
    assert engine.crossings[:3] == [("A", 1, 90), ("A", 2, 180), ("B", 1, 300)]


def test_paid_service_waits_at_closed_exit_and_runs_once_after_restart(monkeypatch):
    engine, run, samples, services, fits, _ = setup(
        monkeypatch, laps=4, red=(2,), paces={"A": 90, "B": 100}, pit=True,
    )
    results = run()
    assert engine.suspensions == [(180, 820, ("A", "B"))]
    assert engine.pit_exits == [("B", 2, 820)]
    paid_running = [sample for sample in samples if sample[:2] == ("B", 2)]
    assert len(paid_running) == 1 and paid_running[0][2] == 820
    assert paid_running[0][3] == 0 and paid_running[0][5] and not paid_running[0][6]
    assert services == ["B"]
    assert fits == [("B", 1), ("A", 2), ("B", 1)]  # Paid set, then two free sets.
    b = next(row for row in results if row.driver_id == "B")
    assert b.pit_stops == 1 and b.pit_laps == [2] and b.laps_completed == 3
    assert b.total_time == 1020


def test_collection_retirement_releases_survivor_without_extra_work(monkeypatch):
    engine, run, samples, _, fits, _ = setup(monkeypatch)

    def fail(driver, car, track, lap, weather):
        if driver.id == "B":
            driver.dnf = True
            driver.dnf_reason = "Controlled failure"
            return RaceEvent(EventType.MECHANICAL_FAILURE, lap, [driver.id])
        return None

    monkeypatch.setattr(engine.simulator.event_manager, "_check_mechanical_failure", fail)
    results = run()
    assert engine.suspensions == [(90, 710, ("A",))]
    assert fits == [("A", 1)]
    assert [sample[1] for sample in samples if sample[0] == "B"] == [1]
    assert results[1].status == DriverStatus.DNF and results[1].laps_completed == 0
    assert engine.timeline.states["B"].retirement_time == 110


def test_collection_forbids_passing_even_for_previously_green_laps(monkeypatch):
    engine, run, _, _, _, attempts = setup(monkeypatch, paces={"A": 90, "B": 110, "C": 100})
    run()
    assert next(time for driver, lap, time in engine.crossings if (driver, lap) == ("C", 1)) > 110
    assert not any(attempts)
    assert engine.suspensions[0][2] == ("A", "B", "C")


def test_restart_keeps_pass_completed_before_red_but_before_next_line_crossing(monkeypatch):
    engine, run, _, _, _, _ = setup(monkeypatch, paces={"A": 100, "B": 110, "C": 90})
    # C passes B at its provisional crossing, then waits behind A. A's red
    # flag must not restore stale last-crossing rank B/C over physical C/B.
    monkeypatch.setattr(engine.simulator.overtaking_model, "attempt_overtake",
                        lambda attacker, car, defender, *a, **k:
                        (attacker.id == "C" and defender.id == "B", False))
    run()
    assert engine.suspensions[0] == (100, 710, ("A", "C", "B"))


def test_suspension_extends_finish_deadline_but_not_original_fuel_schedule(monkeypatch):
    monkeypatch.setattr("f1sim.simulation.race_timing.RACING_TIME_LIMIT_SECONDS", 180)
    engine, run, samples, _, _, _ = setup(monkeypatch, laps=10)
    results = run()
    assert engine.timeline.time_limit_seconds == 800
    assert engine.timeline.final_lap == 3 and engine.timeline.chequered_time == 890
    assert results[0].race_time_limited and results[0].laps_completed == 3
    assert all(sample[4] == 10 for sample in samples)
    assert [sample[1] for sample in samples if sample[0] == "A"] == [1, 2, 3]


def test_repeated_suspensions_and_reuse_reset_barrier_and_duration(monkeypatch):
    engine, run, _, _, _, _ = setup(monkeypatch, laps=4, red=(1, 3))
    first = run()
    intervals = list(engine.suspensions)
    assert intervals == [(90, 710, ("A", "B")), (890, 1530, ("A", "B"))]
    assert engine.timeline.total_suspension_seconds == 1260
    assert run() == first
    assert engine.suspensions == intervals and not engine.red_waiting and not engine.regrouping


def test_red_flag_on_winners_final_crossing_does_not_restart_finished_race(monkeypatch):
    engine, run, _, _, fits, _ = setup(monkeypatch, red=(3,))
    results = run()
    assert not engine.suspensions and not fits
    assert engine.timeline.total_suspension_seconds == 0
    assert [row.total_time for row in results] == [270, 330]


@pytest.mark.parametrize("pause", [-1, True, float("inf"), float("nan"), "600"])
def test_invalid_pause_setting_is_rejected(pause):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        ChronologicalRace(RaceSimulator(), red_flag_pause_seconds=pause)
