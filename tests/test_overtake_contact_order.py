"""Native engines preserve failed-contact order through completed crossings."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.events import EventType, RaceEvent
from f1sim.simulation.race import RaceSimulator


def run_contact_race(monkeypatch, engine_name, laps):
    simulator = RaceSimulator(np.random.default_rng(8))
    drivers = [Driver(id=key, name=key, team_id=key) for key in ("A", "B")]
    cars = {key: Car(team_id=key, team_name=key) for key in ("A", "B")}
    track = Track(id="contact", name="Contact", country="Test", total_laps=laps,
                  base_lap_time=90, pit_lane_delta=1)
    monkeypatch.setattr(simulator, "_should_pit", lambda *args, **kwargs: False)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time",
                        lambda driver, *args, **kwargs: 90 if driver.id == "A" else 89)
    monkeypatch.setattr(simulator.overtaking_model, "should_attempt_overtake",
                        lambda *args, **kwargs: True)
    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake",
                        lambda attacker, *args, **kwargs: (False, attacker.id == "B"))
    monkeypatch.setattr(simulator.event_manager, "_check_mechanical_failure",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(simulator.event_manager, "_check_random_incident",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(simulator.event_manager, "process_lap", lambda *args, **kwargs: [])
    snapshots = []
    reconcile = simulator._reconcile_racing_times

    def record_crossing(states):
        reconcile(states)
        snapshots.append({state.driver.id: (state.total_time, state.last_lap_time)
                          for state in states})

    monkeypatch.setattr(simulator, "_reconcile_racing_times", record_crossing)
    run_kwargs = {"starting_tires": {key: TireCompound.SOFT for key in ("A", "B")}}
    args = (drivers, cars, track, Weather(change_probability=0), ["A", "B"])
    if engine_name == "standard":
        results = simulator.simulate_race(*args, **run_kwargs)
    else:
        engine = ChronologicalRace(simulator)
        results = engine.run(*args, **run_kwargs)
        clocks = {"A": 0, "B": 0}
        for lap in range(1, laps + 1):
            current = {key: time for key, completed, time in engine.crossings if completed == lap}
            snapshots.append({key: (time, time - clocks[key]) for key, time in current.items()})
            clocks.update(current)
    return results, simulator.event_manager.events, snapshots, drivers


def test_native_restart_does_not_count_far_slow_follower(monkeypatch):
    simulator = RaceSimulator(np.random.default_rng(42))
    drivers = [Driver(id=key, name=key, team_id=key) for key in ("A", "B")]
    cars = {key: Car(team_id=key, team_name=key) for key in ("A", "B")}
    track = Track(id="restart", name="Restart", country="Test", total_laps=3,
                  base_lap_time=90)
    monkeypatch.setattr(simulator, "_should_pit", lambda *args, **kwargs: False)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time",
                        lambda driver, *args, **kwargs: 90 if driver.id == "A" else 93)
    monkeypatch.setattr(simulator.event_manager, "_check_mechanical_failure",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(simulator.event_manager, "_check_random_incident",
                        lambda *args, **kwargs: None)

    def control(lap, **kwargs):
        if lap == 1:
            event = RaceEvent(EventType.SAFETY_CAR, lap, duration_laps=1)
            simulator.event_manager.events.append(event)
            simulator.event_manager.safety_car_active = True
            return [event]
        if lap == 2:
            simulator.event_manager.safety_car_active = False
            simulator.event_manager.sc_restart_lap_number = 3
        return []

    monkeypatch.setattr(simulator.event_manager, "process_lap", control)
    native_process = simulator._process_overtakes
    encounters = []

    def process(states, *args, **kwargs):
        queue = sorted(states, key=lambda state: state.position)
        encounters.append((kwargs["lap"], kwargs["restart_lap"],
                           queue[1].total_time - queue[0].total_time))
        return native_process(states, *args, **kwargs)

    monkeypatch.setattr(simulator, "_process_overtakes", process)
    results = simulator.simulate_race(
        drivers, cars, track, Weather(change_probability=0), ["A", "B"],
        starting_tires={key: TireCompound.SOFT for key in ("A", "B")},
    )
    assert [(lap, restart) for lap, restart, gap in encounters] == [(1, False), (3, True)]
    assert encounters[-1][2] > 2
    assert [row.driver_id for row in results] == ["A", "B"]
    assert all((row.overtake_attempts, row.overtake_successes, row.overtake_contacts)
               == (0, 0, 0) for row in results)
    assert [event.event_type for event in simulator.event_manager.events] == [EventType.SAFETY_CAR]


@pytest.mark.parametrize("laps", [1, 3])
def test_failed_contact_faster_attacker_stays_blocked_in_both_native_engines(monkeypatch, laps):
    runs = [run_contact_race(monkeypatch, engine, laps)
            for engine in ("standard", "chronological")]
    for results, events, snapshots, drivers in runs:
        assert [row.driver_id for row in results] == ["A", "B"]
        assert len(events) == laps
        assert all(event.event_type == EventType.COLLISION for event in events)
        assert events[0].applied_time_losses == pytest.approx(
            {"B": 1.6539445532111214, "A": 1.9809152650068882}
        )
        assert len(snapshots) == laps
        previous = {"A": 0, "B": 0}
        for event, snapshot in zip(events, snapshots):
            assert snapshot["B"][0] >= snapshot["A"][0]
            for key, pace in (("A", 90), ("B", 89)):
                clock, running = snapshot[key]
                assert clock > previous[key]
                assert running == pytest.approx(clock - previous[key])
                sampled = pace + event.applied_time_losses[key]
                if key == "A":
                    assert running == pytest.approx(sampled)
                else:
                    # Waiting behind the defender is charged once in addition
                    # to the event's actual loss, never as a second collision.
                    assert clock == pytest.approx(max(previous[key] + sampled,
                                                      snapshot["A"][0]), abs=0.01)
                previous[key] = clock
        for row in results:
            assert row.laps_completed == laps
            assert row.total_time == pytest.approx(snapshots[-1][row.driver_id][0])
            assert row.fastest_lap == pytest.approx(
                min(snapshot[row.driver_id][1] for snapshot in snapshots)
            )
            assert (row.overtake_attempts, row.overtake_successes, row.overtake_contacts) == (
                (laps, 0, laps) if row.driver_id == "B" else (0, 0, 0)
            )
        assert [driver.current_tire_laps for driver in drivers] == [laps, laps]
    assert [event.applied_time_losses for event in runs[0][1]] == [
        event.applied_time_losses for event in runs[1][1]
    ]
