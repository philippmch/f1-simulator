"""Green-lap points follow completed work and actual control exposure."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, Track, Weather
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.events import EventType, RaceEvent
from f1sim.simulation.race import RaceSimulator


def quiet(monkeypatch, simulator):
    for name in ("_check_random_incident", "_check_mechanical_failure",
                 "_check_red_flag_conditions"):
        monkeypatch.setattr(simulator.event_manager, name, lambda *a, **kw: None)


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("control", ["safety_car", "vsc"])
@pytest.mark.parametrize("announcement", [1, 2])
@pytest.mark.parametrize("source", ["scheduled", "during_lap"])
def test_points_require_two_completed_green_laps(
    monkeypatch, engine, control, announcement, source,
):
    schedule = ([dict(lap=announcement, control=control, duration_laps=6)]
                if source == "scheduled" else None)
    simulator = RaceSimulator(np.random.default_rng(91), control_schedule=schedule)
    quiet(monkeypatch, simulator)
    manager = simulator.event_manager
    if source == "during_lap":
        def deploy(lap, *args, **kwargs):
            if lap != announcement:
                return None
            if control == "safety_car":
                manager.safety_car_active = True
                manager.safety_car_laps_remaining = 6
                kind = EventType.SAFETY_CAR
            else:
                manager.vsc_active = True
                manager.vsc_laps_remaining = 6
                kind = EventType.VIRTUAL_SAFETY_CAR
            return RaceEvent(kind, lap, duration_laps=6)

        monkeypatch.setattr(manager, "_deploy_safety_measure", deploy)
    samples = []
    actual_physics = simulator.lap_simulator.calculate_lap_time

    def physics(*args, **kwargs):
        if kwargs.get("sample_variation", True):
            samples.append(manager.safety_car_active or manager.vsc_active)
        return actual_physics(*args, **kwargs)

    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", physics)
    laps = announcement + 6
    track = Track(id="T", name="Test", country="Synthetic", total_laps=laps, base_lap_time=90)
    run = simulator.simulate_race if engine == "standard" else ChronologicalRace(simulator).run
    (result,) = run([Driver(id="A", name="A", team_id="T")],
                   {"T": Car(team_id="T", team_name="T")}, track,
                   Weather(change_probability=0), ["A"], starting_tires={"A": "medium"},
                   pit_plans={"A": [{"lap": 3, "compound": "hard"}]})
    assert samples == [False] * announcement + [True] * 6
    assert result.laps_completed == laps and result.classified
    assert result.pit_laps == [3]
    assert result.points_awarded == (25 if source == "scheduled" and announcement == 2 else 0)
    assert all(event.announced_after_crossing == (source == "scheduled")
               for event in manager.events if event.event_type in (
                   EventType.SAFETY_CAR, EventType.VIRTUAL_SAFETY_CAR))


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("control", ["safety_car", "vsc"])
@pytest.mark.parametrize("timed", [False, True])
def test_scheduled_final_crossing_keeps_completed_green_pair(monkeypatch, engine, control, timed):
    if timed:
        monkeypatch.setattr("f1sim.simulation.race_timing.RACING_TIME_LIMIT_SECONDS", 80.)
    simulator = RaceSimulator(np.random.default_rng(91), control_schedule=[
        dict(lap=2, control=control, duration_laps=6),
    ])
    quiet(monkeypatch, simulator)
    track = Track(id="T", name="Test", country="Synthetic", total_laps=10 if timed else 2,
                  base_lap_time=90)
    run = simulator.simulate_race if engine == "standard" else ChronologicalRace(simulator).run
    (result,) = run([Driver(id="A", name="A", team_id="T")],
                   {"T": Car(team_id="T", team_name="T")}, track,
                   Weather(change_probability=0), ["A"], starting_tires={"A": "medium"},
                   pit_plans={"A": [{"lap": 2, "compound": "hard"}]})
    assert result.laps_completed == 2 and result.classified
    assert result.race_time_limited is timed
    assert result.points_awarded == (6 if timed else 25)
    manager = simulator.event_manager
    assert manager.safety_car_active if control == "safety_car" else manager.vsc_active
    assert manager.get_control_schedule_history() == [
        dict(lap=2, control=control, duration_laps=6,
             status="applied", reason="scheduled_announcement"),
    ]


@pytest.mark.parametrize("control", ["safety_car", "vsc"])
@pytest.mark.parametrize("exposure", ["service", "running"])
def test_lapped_successors_keep_control_exposure(monkeypatch, control, exposure):
    simulator = RaceSimulator(np.random.default_rng(7), control_schedule=[
        dict(lap=1, control=control, duration_laps=1),
        dict(lap=4, control=control, duration_laps=6),
    ])
    quiet(monkeypatch, simulator)
    engine = ChronologicalRace(simulator)
    if exposure == "service":
        # C pays for a compulsory stop while A clears the control, then
        # inherits the lead after A retires. Its track-entry snapshot is green.
        paces = {"A": 90., "C": 100.}
        retire_laps = {"A": 3}
        plans = {"A": [], "C": [{"lap": 2, "compound": "hard"}]}
        target = ("C", 2)
    else:
        # A retires before the slow first laps finish. C then clears the
        # control and retires before B finishes its already sampled green lap.
        paces = {"A": 90., "C": 300., "B": 450.}
        retire_laps = {"A": 2, "C": 2}
        plans = {"A": [], "C": [], "B": [{"lap": 2, "compound": "hard"}]}
        target = ("B", 1)
    samples, crossings, windows = [], [], []
    active_start = None

    def physics(driver, car, track, tire, weather, lap, total_laps, **kwargs):
        manager = simulator.event_manager
        samples.append((driver.id, lap, engine.pending[driver.id].running_start,
                        manager.safety_car_active or manager.vsc_active))
        return 90. if exposure == "running" and driver.id == "C" and lap == 2 else paces[driver.id]

    def incident(drivers, track, weather, lap, **kwargs):
        if exposure == "service" and drivers and drivers[0].id == "C" and lap == 1:
            # A compulsory repair cannot be vetoed to retain finish distance.
            return RaceEvent(EventType.PUNCTURE, lap, ["C"], forces_pit_stop=True)
        return None

    def mechanical(driver, car, track, lap, weather):
        if lap == retire_laps.get(driver.id):
            driver.dnf = True
            driver.dnf_reason = "scripted retirement"
            return RaceEvent(EventType.MECHANICAL_FAILURE, lap, [driver.id])
        return None

    actual_interval = engine._leader_interval

    def interval(pending):
        nonlocal active_start
        identifier = next(key for key, value in engine.pending.items() if value is pending)
        result = actual_interval(pending)
        active = simulator.event_manager.safety_car_active or simulator.event_manager.vsc_active
        if active and active_start is None:
            active_start = pending.ready
        elif not active and active_start is not None:
            windows.append((active_start, pending.ready))
            active_start = None
        crossings.append((identifier, pending.lap, pending.start, pending.ready,
                          engine.green_streak))
        return result

    monkeypatch.setattr(engine, "_leader_interval", interval)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", physics)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_pit_stop_time", lambda car: 140.)
    monkeypatch.setattr(simulator.event_manager, "_check_mechanical_failure", mechanical)
    monkeypatch.setattr(simulator.event_manager, "_check_random_incident", incident)
    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake",
                        lambda *a, **kw: (False, False))
    drivers = [Driver(id=key, name=key, team_id=key) for key in paces]
    results = engine.run(drivers, {key: Car(team_id=key, team_name=key) for key in paces},
                         Track(id="T", name="Test", country="Synthetic", total_laps=4,
                               base_lap_time=90), Weather(change_probability=0), list(paces),
                         starting_tires={key: "medium" for key in paces},
                         pit_plans=plans)
    if active_start is not None:
        windows.append((active_start, float("inf")))
    # Independent interval-overlap accounting includes pit service. A signal
    # exactly at this crossing has no overlap with its completed running.
    streak = 0
    eligible_pair = False
    for _, _, start, end, actual_streak in crossings:
        exposed = any(start < clear and end > signal for signal, clear in windows)
        streak = 0 if exposed else streak + 1
        eligible_pair |= streak >= 2
        assert actual_streak == streak
    assert not eligible_pair and not engine.has_two_green
    assert any((identifier, lap) == target for identifier, lap, *_ in crossings)
    # The snapshot is green at track entry; the complete own lap still
    # encountered the earlier procedure, before the lapped lead handoff.
    assert next(active for identifier, lap, _, active in samples
                if (identifier, lap) == target) is False
    assert results[0].classified
    assert all(result.points_awarded == 0 for result in results)
