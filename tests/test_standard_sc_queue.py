"""Standard-engine SC queues conserve completed clocks, distance and pit service."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.events import EventType, RaceEvent
from f1sim.simulation.race import DriverStatus, RaceSimulator


def run_controlled(
    monkeypatch, *, paces=(90, 100), first_laps=None, laps=8,
    mode="sc", deploy=2, duration=4, pits=(), retirement=None, penalty=None,
):
    names = "ABCD"[:len(paces)]
    drivers = [Driver(id=name, name=name, team_id=name) for name in names]
    cars = {name: Car(team_id=name, team_name=name) for name in names}
    track = Track(id="t", name="T", country="T", total_laps=laps,
                  base_lap_time=90, pit_lane_delta=20)
    simulator = RaceSimulator(np.random.default_rng(1))
    control = simulator.event_manager
    calls, crossings, decisions, battles = [], {}, [], []

    def pace(driver, lap_number, total_laps, **kwargs):
        calls.append((driver.id, lap_number, total_laps,
                      kwargs["active_aero_enabled"], kwargs["overtake_mode_active"]))
        values = first_laps if lap_number == 1 and first_laps is not None else paces
        return values[names.index(driver.id)]

    def should_pit(state, states, planning_track, lap, *args, **kwargs):
        decisions.append((state.driver.id, lap, state.total_time, planning_track.total_laps))
        return (state.driver.id, lap) in pits

    def mechanical(driver, car, track, lap, weather):
        if (driver.id, lap) == retirement:
            driver.dnf = True
            driver.dnf_reason = "controlled failure"
            return RaceEvent(EventType.MECHANICAL_FAILURE, lap, [driver.id])
        return None

    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", pace)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_pit_stop_time", lambda *a: 3)
    monkeypatch.setattr(simulator, "_should_pit", should_pit)
    monkeypatch.setattr(simulator, "_process_overtakes",
                        lambda *a, **kw: battles.append(kw["lap"]) or 0)
    monkeypatch.setattr(control, "_check_mechanical_failure", mechanical)
    monkeypatch.setattr(control, "_check_random_incident", lambda *a, **kw: None)
    monkeypatch.setattr(control, "_deploy_safety_measure", lambda *a, **kw: None)
    monkeypatch.setattr(Weather, "evolve", lambda self, rng: self.project_surface())
    original_events = control.process_lap

    def events(lap, **kwargs):
        result = original_events(lap=lap, **kwargs)
        if penalty is not None and lap == penalty[1]:
            result.append(RaceEvent(EventType.SPIN, lap, [penalty[0]],
                                    time_loss_seconds=penalty[2]))
        if mode and lap == deploy:
            if mode == "sc":
                control.safety_car_active = True
                control.safety_car_laps_remaining = duration
                kind = EventType.SAFETY_CAR
            else:
                control.vsc_active = True
                control.vsc_laps_remaining = duration
                kind = EventType.VIRTUAL_SAFETY_CAR
            result.append(RaceEvent(kind, lap, duration_laps=duration))
        return result

    original_update = simulator._update_positions

    def observe(states):
        original_update(states)
        for state in states:
            if state.status == DriverStatus.RACING:
                crossings[state.driver.id, state.laps_completed + 1] = (
                    state.total_time, state.last_lap_time, state.position, state.tire_laps,
                )

    monkeypatch.setattr(control, "process_lap", events)
    monkeypatch.setattr(simulator, "_update_positions", observe)
    monkeypatch.setattr(control, "bunch_field", lambda *a: pytest.fail("SC rewrote clocks"))
    results = simulator.simulate_race(
        drivers, cars, track, Weather(), list(names),
        starting_tires={name: TireCompound.INTERMEDIATE for name in names},
    )
    # These are accounting invariants across every case, including pit laps,
    # blocked cars and retirements; each completed lap must actually be paid.
    for name in names:
        previous = 0
        for (driver_id, _), (time, lap_time, _, _) in crossings.items():
            if driver_id == name:
                assert time > previous
                assert time - previous == pytest.approx(lap_time)
                previous = time
        assert next(row for row in results if row.driver_id == name).total_time == previous
    assert all(call[2] == laps for call in calls)
    return results, crossings, decisions, calls, battles


def test_deployment_cannot_rewind_a_car_past_its_previous_crossing(monkeypatch):
    _, crossings, *_ = run_controlled(monkeypatch, paces=(90, 200), laps=3)
    assert crossings["B", 1][:2] == (200, 200)
    assert crossings["B", 2][:2] == (400, 200)
    assert crossings["B", 3][:2] == (600, 200)  # Too slow to catch the queue.


def test_queue_pace_propagates_through_every_follower(monkeypatch):
    _, crossings, *_ = run_controlled(
        monkeypatch, paces=(90, 90, 90), first_laps=(90, 100, 110), deploy=1, laps=4,
    )
    for lap in (2, 3, 4):
        leader = 90 + (lap - 1) * 126
        assert [crossings[name, lap][0] for name in "ABC"] == pytest.approx(
            [leader, leader + 1, leader + 2],
        )


def test_large_gap_closes_over_multiple_real_laps(monkeypatch):
    _, crossings, *_ = run_controlled(
        monkeypatch, paces=(90, 90), first_laps=(90, 170), deploy=1, laps=5,
    )
    assert [crossings["B", lap][1] for lap in range(2, 6)] == pytest.approx([90, 90, 119, 126])
    assert [crossings["B", lap][0] - crossings["A", lap][0]
            for lap in range(2, 6)] == pytest.approx([44, 8, 1, 1])


def test_slow_car_blocks_the_following_queue_without_automatic_passes(monkeypatch):
    _, crossings, *_ = run_controlled(monkeypatch, paces=(90, 200, 90), deploy=1, laps=3)
    assert [crossings[name, 3][0] for name in "ABC"] == pytest.approx([342, 600, 600])
    assert [crossings[name, 3][2] for name in "ABC"] == [1, 2, 3]


def test_vsc_preserves_its_individual_pace_without_sc_catchup(monkeypatch):
    _, crossings, *_ = run_controlled(monkeypatch, mode="vsc")
    assert crossings["A", 3][0] == 288
    assert crossings["B", 3][0] == 320


def test_pit_loss_is_paid_once_before_bounded_catchup(monkeypatch):
    results, crossings, decisions, *_ = run_controlled(monkeypatch, pits=(("B", 3),))
    assert crossings["B", 2][0] == 200
    assert crossings["B", 3][:2] == pytest.approx((314, 114))
    assert crossings["B", 4][0] == pytest.approx(433)
    result = next(row for row in results if row.driver_id == "B")
    assert result.pit_laps == [3]
    assert result.pit_stop_details[0]["service_time"] == 3
    assert next(time for name, lap, time, _ in decisions if name == "B" and lap == 4) == 314


def test_pitting_leader_can_lose_position_without_an_on_track_pass(monkeypatch):
    _, crossings, *_ = run_controlled(monkeypatch, paces=(90, 91), pits=(("A", 3),))
    assert crossings["B", 3][2] == 1
    assert crossings["A", 3][2] == 2
    assert crossings["A", 3][0] - crossings["B", 3][0] == pytest.approx(1)


def test_pitter_cannot_pass_a_slow_blocker_when_crossing_clocks_tie(monkeypatch):
    _, crossings, *_ = run_controlled(
        monkeypatch, paces=(90, 90, 200), first_laps=(90, 90, 90),
        deploy=1, laps=3, pits=(("A", 2),),
    )
    # A rejoins behind C. Its free pace would be faster, but it has to stay
    # behind C through both crossings, even though it began the stop in P1.
    for lap, time in ((2, 290), (3, 490)):
        assert crossings["A", lap][0] == crossings["C", lap][0] == time
        assert [crossings[name, lap][2] for name in "BCA"] == [1, 2, 3]


def test_ending_sc_lap_keeps_catchup_and_mode_restrictions(monkeypatch):
    _, crossings, _, calls, battles = run_controlled(monkeypatch, duration=1, laps=4)
    assert crossings["B", 3][0] - crossings["A", 3][0] == pytest.approx(1)
    assert crossings["B", 4][1] == 100
    assert all(not aero and not mode for _, lap, _, aero, mode in calls if lap == 3)
    assert all(aero and not mode for _, lap, _, aero, mode in calls if lap == 4)
    assert battles == [1, 2, 4]


def test_final_lap_deployment_preserves_actual_finishing_gap(monkeypatch):
    results, *_ = run_controlled(monkeypatch, laps=2)
    assert [row.total_time for row in results] == [180, 200]
    assert results[1].gap_to_leader == 20


def test_incident_position_loss_survives_deployment_and_later_catchup(monkeypatch):
    _, crossings, *_ = run_controlled(
        monkeypatch, paces=(90, 92, 94), penalty=("B", 2, 30), laps=4,
    )
    assert [crossings[name, 2][0] for name in "ABC"] == [180, 214, 188]
    assert [crossings[name, 2][2] for name in "ABC"] == [1, 3, 2]
    assert [crossings[name, 4][2] for name in "ABC"] == [1, 3, 2]


def test_retirement_retains_uncompressed_completed_distance_and_time(monkeypatch):
    results, crossings, *_ = run_controlled(
        monkeypatch, paces=(90, 200), retirement=("B", 3), laps=4,
    )
    retired = next(row for row in results if row.driver_id == "B")
    assert retired.status == DriverStatus.DNF
    assert retired.laps_completed == 2
    assert retired.total_time == crossings["B", 2][0] == 400


def test_new_leader_keeps_conserved_time_for_timed_finish_and_planning(monkeypatch):
    monkeypatch.setattr("f1sim.simulation.race_timing.RACING_TIME_LIMIT_SECONDS", 500)
    results, _, decisions, *_ = run_controlled(
        monkeypatch, paces=(90, 200), retirement=("A", 3), laps=10,
    )
    assert results[0].driver_id == "B"
    assert results[0].laps_completed == 4
    assert results[0].race_time_limited
    assert next(horizon for name, lap, _, horizon in decisions if name == "B" and lap == 4) == 4
