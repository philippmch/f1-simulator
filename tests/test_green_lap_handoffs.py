"""Leading intervals cannot count the same green lap distance twice."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, Track, Weather
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.events import EventType, RaceEvent
from f1sim.simulation.race import RaceSimulator


@pytest.mark.parametrize("control", ["safety_car", "vsc"])
@pytest.mark.parametrize("announcement", [2, 3])
@pytest.mark.parametrize("retirement", [2, 3])
@pytest.mark.parametrize("timed", [False, True])
def test_green_pair_requires_distinct_consecutive_leading_laps(
    monkeypatch, control, announcement, retirement, timed,
):
    if timed:
        monkeypatch.setattr("f1sim.simulation.race_timing.RACING_TIME_LIMIT_SECONDS", 750.)
    simulator = RaceSimulator(np.random.default_rng(7), control_schedule=[
        dict(lap=announcement, control=control, duration_laps=6),
    ])
    manager = simulator.event_manager
    monkeypatch.setattr(manager, "_check_random_incident", lambda *a, **kw: None)
    monkeypatch.setattr(manager, "_check_red_flag_conditions", lambda *a, **kw: None)

    def mechanical(driver, car, track, lap, weather):
        if driver.id == "A" and lap == retirement:
            driver.dnf = True
            driver.dnf_reason = "scripted retirement"
            return RaceEvent(EventType.MECHANICAL_FAILURE, lap, ["A"])
        return None

    monkeypatch.setattr(manager, "_check_mechanical_failure", mechanical)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time",
                        lambda driver, *a, **kw: 90. if driver.id == "A" else 300.)
    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake",
                        lambda *a, **kw: (False, False))
    engine = ChronologicalRace(simulator)
    intervals = []
    original = engine._leader_interval

    def observe(pending):
        driver_id = next(key for key, value in engine.pending.items() if value is pending)
        result = original(pending)
        intervals.append((driver_id, pending.lap, pending.start, pending.ready,
                          engine.green_streak, engine.has_two_green))
        return result

    monkeypatch.setattr(engine, "_leader_interval", observe)
    track = Track(id="T", name="Test", country="Synthetic", total_laps=20 if timed else 5,
                  base_lap_time=90)
    results = engine.run([Driver(id=key, name=key, team_id=key) for key in "AB"],
                         {key: Car(team_id=key, team_name=key) for key in "AB"}, track,
                         Weather(track_wetness=.3, rain_intensity=.3, change_probability=0),
                         ["A", "B"], starting_tires={key: "intermediate" for key in "AB"},
                         pit_plans={key: [] for key in "AB"})
    winner = results[0]
    assert winner.driver_id == "B" and winner.classified
    assert winner.laps_completed == (4 if timed else 5)
    assert winner.race_time_limited is timed
    if retirement == 2:
        # Both cars' first green laps start at zero. These overlapping lap-one
        # completions cannot establish a consecutive pair, despite two control ticks.
        assert intervals[:2] == [
            ("A", 1, 0., 90., 1, False),
            ("B", 1, 0., 300., 1, False),
        ]
    else:
        # A really completes laps one and two green before retiring. A lapped
        # successor cannot erase that already earned eligibility.
        assert intervals[:2] == [
            ("A", 1, 0., 90., 1, False),
            ("A", 2, 90., 180., 2, True),
        ]
    eligible = retirement == 3 or announcement == 3
    assert engine.has_two_green is eligible
    assert winner.points_awarded == ((6 if timed else 25) if eligible else 0)
    assert manager.get_control_schedule_history() == [
        dict(lap=announcement, control=control, duration_laps=6,
             status="applied", reason="scheduled_announcement"),
    ]


@pytest.mark.parametrize("control", ["safety_car", "vsc"])
def test_successive_green_distances_can_be_completed_by_different_leaders(monkeypatch, control):
    simulator = RaceSimulator(np.random.default_rng(7), control_schedule=[
        dict(lap=2, control=control, duration_laps=6),
    ])
    manager = simulator.event_manager
    monkeypatch.setattr(manager, "_check_random_incident", lambda *a, **kw: None)
    monkeypatch.setattr(manager, "_check_red_flag_conditions", lambda *a, **kw: None)

    def mechanical(driver, car, track, lap, weather):
        if driver.id == "A" and lap == 2:
            driver.dnf = True
            return RaceEvent(EventType.MECHANICAL_FAILURE, lap, ["A"])
        return None

    monkeypatch.setattr(manager, "_check_mechanical_failure", mechanical)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time",
                        lambda driver, *a, **kw: 90. if driver.id == "A" else 100.)
    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake",
                        lambda *a, **kw: (False, False))
    engine = ChronologicalRace(simulator)
    crossings = []
    original = engine._leader_interval

    def observe(pending):
        output = original(pending)
        crossings.append((pending.lap, engine.green_streak, engine.has_two_green))
        return output

    monkeypatch.setattr(engine, "_leader_interval", observe)
    results = engine.run([Driver(id=key, name=key, team_id=key) for key in "AB"],
                         {key: Car(team_id=key, team_name=key) for key in "AB"},
                         Track(id="T", name="T", country="T", total_laps=5, base_lap_time=90),
                         Weather(track_wetness=.3, rain_intensity=.3, change_probability=0),
                         ["A", "B"], starting_tires={key: "intermediate" for key in "AB"},
                         pit_plans={key: [] for key in "AB"})
    assert engine.has_two_green
    assert crossings[:2] == [(1, 1, False), (2, 2, True)]
    assert results[0].driver_id == "B" and results[0].points_awarded == 25
    assert results[0].race_points_context.has_two_green_laps
    assert manager.get_control_schedule_history()[0]["status"] == "applied"
