"""Executable chronological races use actual own-lap work, not result truncation."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.events import EventType, RaceEvent
from f1sim.simulation.race import DriverStatus, RaceSimulator


def fixture(monkeypatch, paces=None, laps=10):
    paces = paces or {"Fast": 90, "Slow": 110}
    drivers = [Driver(id=key, name=key, team_id="T") for key in paces]
    cars = {"T": Car(team_id="T", team_name="Team")}
    track = Track(id="t", name="T", country="T", total_laps=laps, base_lap_time=90,
                  pit_lane_delta=1)
    simulator = RaceSimulator(np.random.default_rng(2))
    calls, policy, exposure, control = [], [], [], []

    def running(driver, car, track, tire, weather, lap, total_laps, **kwargs):
        calls.append((driver.id, lap, total_laps, driver.current_tire_laps))
        return paces[driver.id]

    def should(state, states, track, lap, *args, **kwargs):
        policy.append((state.driver.id, lap))
        return False

    def mechanical(driver, car, track, lap, weather):
        exposure.append((driver.id, lap))
        return None

    def events(lap, *args, **kwargs):
        control.append(lap)
        return []

    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", running)
    monkeypatch.setattr(simulator, "_should_pit", should)
    monkeypatch.setattr(simulator.event_manager, "_check_mechanical_failure", mechanical)
    monkeypatch.setattr(simulator.event_manager, "_check_random_incident", lambda *args: None)
    monkeypatch.setattr(simulator.event_manager, "process_lap", events)
    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake", lambda *args, **kwargs:
                        (True, False))
    monkeypatch.setattr(Weather, "evolve", lambda self, rng: self.project_surface())
    engine = ChronologicalRace(simulator)
    args = drivers, cars, track, Weather(), list(paces)
    return engine, args, calls, policy, exposure, control


def run(engine, args):
    return engine.run(*args, starting_tires={key: TireCompound.INTERMEDIATE for key in args[-1]})


def test_lapped_finish_executes_actual_nine_laps_for_slower_car(monkeypatch):
    engine, args, calls, policy, exposure, control = fixture(monkeypatch)
    results = run(engine, args)
    assert [(r.driver_id, r.total_time, r.laps_completed) for r in results] == [
        ("Fast", 900, 10), ("Slow", 990, 9),
    ]
    assert sorted(calls) == sorted(
        (driver, lap, 10, lap - 1) for driver, lap, _ in engine.crossings
    )
    assert len(policy) == len(exposure) == 19
    assert control == list(range(1, 11))
    assert sorted(t for _, _, t in engine.crossings) == [t for _, _, t in engine.crossings]


def test_equal_clock_ties_flag_both_without_extra_lap(monkeypatch):
    engine, args, calls, _, _, _ = fixture(monkeypatch, {"Fast": 90, "Slow": 90})
    results = run(engine, args)
    assert [r.total_time for r in results] == [900, 900]
    assert [r.laps_completed for r in results] == [10, 10]
    assert len(calls) == 20


def test_failed_lapping_pass_cannot_sort_through_physical_predecessor(monkeypatch):
    engine, args, _, _, _, _ = fixture(monkeypatch)
    attempts = []

    def fail(*args, **kwargs):
        attempts.append((args[0].id, args[2].id))
        return False, False

    monkeypatch.setattr(engine.simulator.overtaking_model, "attempt_overtake", fail)
    results = run(engine, args)
    assert attempts
    assert results[0].total_time > 900
    assert any(driver == "Fast" and lap == 6 and time >= 550
               for driver, lap, time in engine.crossings)


@pytest.mark.parametrize("failure_lap", [8, 9])
def test_retirement_can_occur_before_or_after_winner_flag(monkeypatch, failure_lap):
    engine, args, calls, _, _, _ = fixture(monkeypatch)

    def mechanical(driver, car, track, lap, weather):
        if driver.id == "Slow" and lap == failure_lap:
            driver.dnf = True
            driver.dnf_reason = "Controlled failure"
            return RaceEvent(EventType.MECHANICAL_FAILURE, lap, [driver.id])
        return None

    monkeypatch.setattr(engine.simulator.event_manager, "_check_mechanical_failure", mechanical)
    results = run(engine, args)
    slow = next(r for r in results if r.driver_id == "Slow")
    assert slow.status == DriverStatus.DNF
    assert slow.laps_completed == failure_lap - 1
    assert slow.total_time == 110 * (failure_lap - 1)
    assert max(lap for driver, lap, _, _ in calls if driver == "Slow") == failure_lap
    assert engine.timeline.states["Slow"].retirement_time == failure_lap * 110


def test_persistent_team_queue_uses_staggered_actual_arrivals(monkeypatch):
    engine, args, _, _, _, _ = fixture(monkeypatch, {"Fast": 90, "Slow": 91}, laps=3)
    monkeypatch.setattr(engine.simulator, "_should_pit", lambda state, states, track, lap,
                        *args, **kwargs: lap == 2)
    services = []
    monkeypatch.setattr(engine.simulator.lap_simulator, "calculate_pit_stop_time",
                        lambda car: services.append(1) or 3)
    results = run(engine, args)
    assert engine.pit_exits == [("Fast", 2, 94), ("Slow", 2, 97)]
    assert len(services) == 2
    assert all(result.pit_stops == 1 and result.pit_laps == [2] for result in results)


def test_retired_longer_distance_car_ranks_above_shorter_finisher(monkeypatch):
    engine, args, _, _, _, _ = fixture(monkeypatch, {"Fast": 90, "Retired": 99, "Slow": 130})

    def failure(driver, car, track, lap, weather):
        if driver.id == "Retired" and lap == 10:
            driver.dnf = True
            driver.dnf_reason = "Controlled failure"
            return RaceEvent(EventType.MECHANICAL_FAILURE, lap, [driver.id])
        return None

    monkeypatch.setattr(engine.simulator.event_manager, "_check_mechanical_failure", failure)
    results = run(engine, args)
    assert [(r.driver_id, r.laps_completed, r.status) for r in results] == [
        ("Fast", 10, DriverStatus.FINISHED),
        ("Retired", 9, DriverStatus.DNF),
        ("Slow", 7, DriverStatus.FINISHED),
    ]


def test_own_random_spin_delays_crossing_without_extra_exposure(monkeypatch):
    engine, args, calls, _, _, _ = fixture(monkeypatch, {"Fast": 90}, laps=3)
    exposures = []

    def incident(drivers, track, weather, lap):
        exposures.append(lap)
        return (RaceEvent(EventType.SPIN, lap, [drivers[0].id], time_loss_seconds=4)
                if lap == 2 else None)

    monkeypatch.setattr(engine.simulator.event_manager, "_check_random_incident", incident)
    (result,) = run(engine, args)
    assert result.total_time == 274
    assert exposures == [1, 2, 3]
    assert len(calls) == 3
    assert engine.crossings == [("Fast", 1, 90), ("Fast", 2, 184), ("Fast", 3, 274)]


def test_random_retirement_does_not_credit_failed_lap_or_fastest(monkeypatch):
    engine, args, _, _, _, _ = fixture(monkeypatch, {"Fast": 90}, laps=3)

    def incident(drivers, track, weather, lap):
        if lap == 2:
            drivers[0].dnf = True
            drivers[0].dnf_reason = "Controlled crash"
            return RaceEvent(EventType.COLLISION, lap, [drivers[0].id])
        return None

    monkeypatch.setattr(engine.simulator.event_manager, "_check_random_incident", incident)
    (result,) = run(engine, args)
    assert result.laps_completed == 1 and result.total_time == 90
    assert result.fastest_lap == 90 and result.points_awarded == 0
    assert engine.timeline.winner_id is None


def test_no_weather_work_after_flag_and_no_lap_after_own_finish(monkeypatch):
    engine, args, calls, policy, _, _ = fixture(monkeypatch)
    evolves = []
    monkeypatch.setattr(Weather, "evolve", lambda self, rng: evolves.append(1) or self.model_copy())
    run(engine, args)
    assert len(evolves) == 9
    assert max(lap for driver, lap in policy if driver == "Fast") == 10
    assert max(lap for driver, lap, _, _ in calls if driver == "Slow") == 9


def test_shared_lap_physics_is_used_with_own_age_and_original_fuel(monkeypatch):
    from f1sim.simulation.lap import LapSimulator

    engine, args, _, _, _, _ = fixture(monkeypatch, {"Fast": 90}, laps=3)
    physics = LapSimulator(np.random.default_rng(5))
    monkeypatch.setattr(engine.simulator.lap_simulator, "calculate_lap_time",
                        lambda *args, **kwargs: physics.calculate_lap_time(
                            *args, **kwargs, sample_variation=False))
    driver, car, track, weather = args[0][0], args[1]["T"], args[2], args[3]
    from f1sim.models.tire import TIRE_COMPOUNDS

    expected = sum(physics.calculate_lap_time(
        driver.model_copy(update={"current_tire_laps": lap - 1}), car, track,
        TIRE_COMPOUNDS[TireCompound.INTERMEDIATE], weather, lap, 3, sample_variation=False,
    ) for lap in range(1, 4))
    (result,) = run(engine, args)
    assert result.total_time == pytest.approx(expected)


def test_dry_lapped_car_must_complete_two_compounds_before_own_flag(monkeypatch):
    engine, args, _, _, _, _ = fixture(monkeypatch, {"Fast": 10, "Slow": 110}, laps=10)
    args[2].pit_lane_delta = 300
    monkeypatch.setattr(engine.simulator, "_should_pit",
                        RaceSimulator._should_pit.__get__(engine.simulator))
    results = engine.run(*args, starting_tires={key: TireCompound.SOFT for key in args[-1]})
    assert all(len(set(result.strategy)) >= 2 for result in results)


def test_lapped_exact_ties_process_leading_distance_before_starting_another_lap(monkeypatch):
    engine, args, calls, _, _, _ = fixture(
        monkeypatch, {"Fast": 90, "Slow": 110, "Tied": 150, "Last": 200},
    )
    results = run(engine, args)
    assert [(r.driver_id, r.total_time, r.laps_completed) for r in results] == [
        ("Fast", 900, 10), ("Slow", 990, 9), ("Tied", 900, 6), ("Last", 1000, 5),
    ]
    assert max(lap for driver, lap, _, _ in calls if driver == "Tied") == 6
    assert [(driver, lap) for driver, lap, time in engine.crossings if time == 900] == [
        ("Fast", 10), ("Tied", 6),
    ]


def test_control_snapshots_apply_to_next_started_laps_without_clock_bunching(monkeypatch):
    from f1sim.simulation.events import EventManager

    engine, args, _, _, _, _ = fixture(monkeypatch, laps=3)
    control = engine.simulator.event_manager
    monkeypatch.setattr(control, "process_lap", EventManager.process_lap.__get__(control))
    monkeypatch.setattr(control, "_deploy_safety_measure", lambda *args: None)
    monkeypatch.setattr(control, "bunch_field", lambda *args: pytest.fail("clock rewind"))
    control.set_forced_safety_car(1)
    results = run(engine, args)
    assert [(r.driver_id, r.total_time) for r in results] == [("Fast", 342), ("Slow", 418)]
    assert len([event for event in control.events if event.event_type == EventType.SAFETY_CAR]) == 1
    crossing_times = [time for _, _, time in engine.crossings]
    assert crossing_times == sorted(crossing_times)


def test_red_flag_refits_each_continuing_car_once_without_rewriting_pending_lap(monkeypatch):
    from f1sim.simulation.events import EventManager

    engine, args, _, _, _, _ = fixture(monkeypatch, laps=3)
    control = engine.simulator.event_manager
    monkeypatch.setattr(control, "process_lap", EventManager.process_lap.__get__(control))
    monkeypatch.setattr(control, "_deploy_safety_measure", lambda *args: None)
    control.set_forced_red_flag(1)
    fits = []
    fit = engine.simulator._fit_tire

    def record(state, compound):
        fits.append((state.driver.id, state.laps_completed))
        fit(state, compound)

    monkeypatch.setattr(engine.simulator, "_fit_tire", record)
    results = run(engine, args)
    assert fits == [("Fast", 1), ("Slow", 1)]
    assert all(result.pit_stops == 0 and len(result.strategy) == 2 for result in results)
    assert all(result.total_time > 0 for result in results)


@pytest.mark.parametrize("neutralized,expected_horizon", [(False, 6), (True, 5)])
def test_own_horizon_forecast_uses_free_pace_current_modifier_and_absolute_pending_exit(
    monkeypatch, neutralized, expected_horizon,
):
    import copy

    from f1sim.simulation.chronological_race import _PendingLap
    from f1sim.simulation.race_timing import RaceFinishTimeline

    engine, args, _, _, _, _ = fixture(monkeypatch, laps=6)
    run(engine, args)
    engine.timeline = RaceFinishTimeline(6, engine.states)
    for driver, completed, clock in [("Fast", 2, 180), ("Slow", 1, 110)]:
        state = engine.states[driver]
        state.status = DriverStatus.RACING
        state.laps_completed = completed
        state.total_time = clock
        state.last_lap_time = 999  # Service/blocked time must not become projected pace.
    engine.running_paces = {"Fast": 90, "Slow": 110}
    engine.pending = {"Fast": _PendingLap(3, 180, 400, Weather(), False,
                       engine.states["Fast"].current_tire, 2, 90)}
    engine.simulator.event_manager.safety_car_active = neutralized
    before = copy.deepcopy((engine.pending, engine.running_paces,
                            engine.simulator.rng.bit_generator.state, vars(engine.timeline._clock)))
    planning = engine._planning_track(engine.states["Slow"], 200)
    assert planning.total_laps == expected_horizon
    assert (engine.pending, engine.running_paces,
            engine.simulator.rng.bit_generator.state, vars(engine.timeline._clock)) == before
    assert engine.track.total_laps == 6


def test_lapped_leadership_handoff_does_not_replay_control_intervals(monkeypatch):
    engine, args, _, _, _, control = fixture(monkeypatch, {"Fast": 10, "Slow": 100}, laps=12)

    def failure(driver, car, track, lap, weather):
        if driver.id == "Fast" and lap == 11:
            driver.dnf = True
            driver.dnf_reason = "Controlled failure"
            return RaceEvent(EventType.MECHANICAL_FAILURE, lap, [driver.id])
        return None

    monkeypatch.setattr(engine.simulator.event_manager, "_check_mechanical_failure", failure)
    results = run(engine, args)
    assert results[0].driver_id == "Slow" and results[0].laps_completed == 12
    assert engine.timeline.states["Fast"].completed_laps == 10
    assert control == list(range(1, len(control) + 1))
    assert len(control) > 12


def test_forced_global_red_flag_is_not_replayed_after_lapped_leader_handoff(monkeypatch):
    from f1sim.simulation.events import EventManager

    engine, args, _, _, _, _ = fixture(monkeypatch, {"Fast": 90, "Slow": 300})
    control = engine.simulator.event_manager
    intervals = []
    actual_process = EventManager.process_lap.__get__(control)

    def process(interval, *args, **kwargs):
        intervals.append(interval)
        return actual_process(interval, *args, **kwargs)

    def failure(driver, car, track, lap, weather):
        if driver.id == "Fast" and lap == 4:
            driver.dnf = True
            driver.dnf_reason = "Controlled failure"
            return RaceEvent(EventType.MECHANICAL_FAILURE, lap, [driver.id])
        return None

    monkeypatch.setattr(control, "process_lap", process)
    monkeypatch.setattr(control, "_deploy_safety_measure", lambda *args: None)
    monkeypatch.setattr(control, "_check_mechanical_failure", failure)
    control.set_forced_red_flag(2)
    results = run(engine, args)
    assert results[0].driver_id == "Slow" and results[0].laps_completed == 10
    assert intervals == list(range(1, 13))
    red_flags = [event for event in control.events if event.event_type == EventType.RED_FLAG]
    assert len(red_flags) == 1 and red_flags[0].lap == 2
    assert control.red_flag_restart_lap_number == 3
    assert control.current_lap == 12
