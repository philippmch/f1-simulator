"""Chronological traffic uses physical progress and once-per-own-lap energy."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.chronological_race import ChronologicalRace, _PendingLap
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.race import DriverStatus, RaceSimulator


def setup(monkeypatch, paces=None, laps=4, weather=None):
    paces = paces or {"A": 90, "B": 90}
    drivers = [Driver(id=key, name=key, team_id=key) for key in paces]
    cars = {key: Car(team_id=key, team_name=key) for key in paces}
    track = Track(id="t", name="T", country="T", total_laps=laps, base_lap_time=90,
                  pit_lane_delta=1)
    simulator = RaceSimulator(np.random.default_rng(42))
    samples, attempts = [], []

    def running(driver, car, track, tire, weather, lap, total_laps, **kwargs):
        samples.append((driver.id, lap, kwargs.copy()))
        pace = paces[driver.id]
        return pace(lap) if callable(pace) else pace

    def passing(*args, **kwargs):
        attempts.append((args[0].id, kwargs.copy()))
        return True, False

    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", running)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_pit_stop_time", lambda car: 3)
    monkeypatch.setattr(simulator, "_should_pit", lambda *args, **kwargs: False)
    monkeypatch.setattr(simulator.event_manager, "process_lap", lambda *args, **kwargs: [])
    monkeypatch.setattr(simulator.event_manager, "_check_mechanical_failure", lambda *args: None)
    monkeypatch.setattr(simulator.event_manager, "_check_random_incident",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake", passing)
    monkeypatch.setattr(Weather, "evolve", lambda self, rng: self.model_copy(deep=True))
    engine = ChronologicalRace(simulator)
    args = drivers, cars, track, weather or Weather(), list(paces)
    return engine, args, samples, attempts


def run(engine, args):
    return engine.run(*args, starting_tires={key: TireCompound.SOFT for key in args[-1]})


def test_physical_gap_uses_progress_not_race_laps_or_crossing_clock(monkeypatch):
    engine, args, _, _ = setup(monkeypatch)
    run(engine, args)
    engine.order = ["A", "B"]
    for state in engine.states.values():
        state.status = DriverStatus.RACING
    engine.states["A"].laps_completed = 9
    engine.states["B"].laps_completed = 1
    engine.states["A"].total_time = 900
    engine.states["B"].total_time = 100
    engine.pending = {"A": _PendingLap(10, 200, 310, Weather(), False,
                       TIRE_COMPOUNDS[TireCompound.SOFT], 9, 110, running_start=200)}
    engine.running_paces = {"B": 90}
    assert engine._physical_gap_ahead("B", 201) == pytest.approx(90 / 110)
    assert engine._physical_gap_ahead("A", 201) is None
    engine.pending["A"].on_track = False
    assert engine._physical_gap_ahead("B", 201) is None
    engine.pending["A"].on_track = True
    engine.states["A"].status = DriverStatus.FINISHED
    assert engine._physical_gap_ahead("B", 201) is None
    assert engine._physical_gap_ahead("B", float("nan")) is None


def test_paid_lap_measures_traffic_at_exit_and_never_deploys_mode(monkeypatch):
    engine, args, samples, _ = setup(monkeypatch, {"A": 90, "B": 90.5}, laps=3)
    monkeypatch.setattr(engine.simulator, "_should_pit",
                        lambda state, states, track, lap, *args, **kwargs:
                        state.driver.id == "B" and lap == 2)
    run(engine, args)
    paid = next(options for driver, lap, options in samples if driver == "B" and lap == 2)
    assert engine.pit_exits == [("B", 2, 94.5)]
    assert paid["gap_to_car_ahead"] == pytest.approx(4.5 / 90 * 90.5)
    assert paid["overtake_mode_active"] is False
    assert sum(driver == "B" and lap == 2 for driver, lap, _ in samples) == 1


@pytest.mark.parametrize("first_gap,expected", [(0.5, True), (20, False)])
def test_passing_preserves_detection_snapshot_without_free_catch_deployment(
    monkeypatch, first_gap, expected,
):
    engine, args, samples, attempts = setup(
        monkeypatch, {"A": 90, "B": lambda lap: 90 + first_gap if lap == 1 else 60}, laps=3,
    )
    run(engine, args)
    start = next(options for driver, lap, options in samples if driver == "B" and lap == 2)
    assert start["overtake_mode_active"] is expected
    assert any(driver == "B" and options["overtake_mode_active"] is expected
               for driver, options in attempts)
    if not expected:
        assert not any(options["overtake_mode_active"]
                       for driver, options in attempts if driver == "B")


def test_restart_boost_and_mode_gate_use_global_interval_snapshot(monkeypatch):
    engine, args, samples, attempts = setup(
        monkeypatch, {"A": 90, "B": lambda lap: 90.5 if lap == 1 else 60}, laps=3,
    )
    control = engine.simulator.event_manager

    def events(interval, *args, **kwargs):
        if interval == 1:
            control.sc_restart_lap_number = 2
        return []

    monkeypatch.setattr(control, "process_lap", events)
    run(engine, args)
    start = next(options for driver, lap, options in samples if driver == "B" and lap == 2)
    assert not start["overtake_mode_active"]
    assert any(driver == "B" and options["restart_boost"] for driver, options in attempts)


@pytest.mark.parametrize("flag", ["safety_car_active", "vsc_active"])
def test_neutralization_disables_mode_and_passing(monkeypatch, flag):
    engine, args, samples, attempts = setup(monkeypatch, laps=3)
    control = engine.simulator.event_manager

    def events(interval, *args, **kwargs):
        setattr(control, flag, True)
        return []

    monkeypatch.setattr(control, "process_lap", events)
    run(engine, args)
    assert not any(options["overtake_mode_active"] for _, _, options in samples)
    assert not attempts


def test_wet_conditions_disable_mode(monkeypatch):
    engine, args, samples, _ = setup(monkeypatch, weather=Weather(track_wetness=0.3))
    run(engine, args)
    assert not any(options["overtake_mode_active"] for _, _, options in samples)


def test_energy_recharges_once_per_completed_own_lap_and_resets_on_reuse(monkeypatch):
    engine, args, _, _ = setup(monkeypatch, {"A": 90, "B": 110}, laps=10)
    recharges = []
    actual = engine.simulator._recharge_overtake_mode_energy

    def recharge(states, neutralized=False):
        recharges.extend(state.driver.id for state in states)
        actual(states, neutralized)

    monkeypatch.setattr(engine.simulator, "_recharge_overtake_mode_energy", recharge)
    run(engine, args)
    assert recharges.count("A") == 10 and recharges.count("B") == 9
    first = {key: (state.overtake_mode_energy, state.overtake_mode_deployments)
             for key, state in engine.states.items()}
    run(engine, args)
    assert {key: (state.overtake_mode_energy, state.overtake_mode_deployments)
            for key, state in engine.states.items()} == first
    assert all(0 <= state.overtake_mode_energy <= 1 for state in engine.states.values())


def test_actual_lap_model_applies_measured_dirty_air_and_mode_gain(monkeypatch):
    from f1sim.models.track import ActiveAeroZone

    engine, args, _, _ = setup(monkeypatch, laps=3)
    args[2].active_aero_zones = [ActiveAeroZone(zone_id=1, sector=1, time_gain=0.8)]
    physics = LapSimulator(np.random.default_rng(0))
    comparisons = []

    def running(*positional, **options):
        observed = physics.calculate_lap_time(*positional, **options, sample_variation=False)
        clean = physics.calculate_lap_time(*positional, **dict(options, gap_to_car_ahead=None,
                                              overtake_mode_active=False), sample_variation=False)
        no_mode = physics.calculate_lap_time(
            *positional, **dict(options, overtake_mode_active=False), sample_variation=False,
        )
        comparisons.append((positional[0].id, positional[5], options, observed, clean, no_mode))
        return observed

    monkeypatch.setattr(engine.simulator.lap_simulator, "calculate_lap_time", running)
    run(engine, args)
    first = next(row for row in comparisons if row[:2] == ("B", 1))
    assert first[2]["gap_to_car_ahead"] == 0
    assert first[3] - first[4] == pytest.approx(0.5)
    active = [row for row in comparisons if row[2]["overtake_mode_active"]]
    assert active
    for row in active:
        # Compare the actual mode-enabled call against the exact same traffic state.
        assert row[2]["gap_to_car_ahead"] <= args[2].overtake_mode_detection_gap
        assert row[3] < row[5]
