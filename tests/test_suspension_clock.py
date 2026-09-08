"""Suspensions extend the wall-clock deadline without replaying race distance."""

from copy import deepcopy

import pytest

from f1sim.simulation.race_timing import RaceFinishClock, RaceFinishTimeline


def test_suspension_extends_deadline_at_exact_boundary():
    clock = RaceFinishClock(100)
    clock.observe_leader_crossing(1, 100)
    clock.begin_suspension(200)
    clock.end_suspension(800)
    assert clock.total_suspension_seconds == 600
    assert clock.time_limit_seconds == 7800
    assert clock.observe_leader_crossing(2, 7799) == 100
    assert clock.observe_leader_crossing(3, 7800) == 4
    assert clock.observe_leader_crossing(4, 7900) == 4
    assert clock.winner_time == 7900


def test_multiple_suspensions_accumulate_but_extension_is_capped():
    clock = RaceFinishClock(100)
    for start, end in [(100, 1900), (2000, 4400)]:
        clock.begin_suspension(start)
        clock.end_suspension(end)
    assert clock.total_suspension_seconds == 4200
    assert clock.time_limit_seconds == 10800
    assert clock.observe_leader_crossing(1, 10799) == 100
    assert clock.observe_leader_crossing(2, 10800) == 3


def test_announced_distance_stays_latched_across_suspension():
    clock = RaceFinishClock(100)
    assert clock.observe_leader_crossing(1, 7200) == 2
    clock.begin_suspension(7200)
    clock.end_suspension(7300)
    assert clock.observe_leader_crossing(2, 7350) == 2
    assert clock.winner_time == 7350


def test_announced_handoff_after_suspension_does_not_restart_countdown():
    clock = RaceFinishClock(100)
    clock.observe_leader_crossing(1, 7200)
    clock.begin_suspension(7200)
    clock.end_suspension(7300)
    clock.observe_leader_crossing(1, 7350, allow_leadership_reset=True)
    assert clock.winner_time == 7350
    assert clock.final_lap == 2 and clock.completed_laps == 1


@pytest.mark.parametrize("invalid", [-1, float("inf"), float("nan"), True, "100"])
@pytest.mark.parametrize("method", ["begin_suspension", "end_suspension"])
def test_invalid_suspension_times_are_atomic(invalid, method):
    clock = RaceFinishClock(100)
    if method == "end_suspension":
        clock.begin_suspension(0)
    before = vars(clock).copy()
    with pytest.raises(ValueError):
        getattr(clock, method)(invalid)
    assert vars(clock) == before


def test_overlap_missing_begin_and_replayed_observations_are_atomic():
    clock = RaceFinishClock(100)
    clock.observe_leader_crossing(1, 100)
    before = vars(clock).copy()
    for method, time in [("begin_suspension", 99), ("end_suspension", 101)]:
        with pytest.raises(ValueError):
            getattr(clock, method)(time)
        assert vars(clock) == before
    clock.begin_suspension(200)
    before = vars(clock).copy()
    for method, time in [("begin_suspension", 201), ("end_suspension", 199)]:
        with pytest.raises(ValueError):
            getattr(clock, method)(time)
        assert vars(clock) == before
    with pytest.raises(ValueError, match="during suspension"):
        clock.observe_leader_crossing(2, 201)
    assert vars(clock) == before
    clock.end_suspension(300)
    before = vars(clock).copy()
    with pytest.raises(ValueError, match="chronological"):
        clock.observe_leader_crossing(2, 299)
    assert vars(clock) == before


def test_flag_prevents_new_suspension():
    clock = RaceFinishClock(1)
    clock.observe_leader_crossing(1, 90)
    before = vars(clock).copy()
    with pytest.raises(ValueError, match="chequered"):
        clock.begin_suspension(90)
    assert vars(clock) == before


def test_timeline_collects_followers_and_retirements_before_restart():
    timeline = RaceFinishTimeline(100, ["leader", "follower", "retiree"])
    timeline.observe_crossing("leader", 1, 90, is_leader=True)
    timeline.begin_suspension(90)
    timeline.observe_crossing("follower", 1, 100)
    timeline.retire("retiree", 105)
    timeline.end_suspension(120)
    assert timeline.total_suspension_seconds == 30
    assert timeline.time_limit_seconds == 7230
    assert timeline.states["follower"].last_crossing_time == 100
    assert timeline.states["retiree"].retirement_time == 105
    timeline.observe_crossing("leader", 2, 210, is_leader=True)


def test_timeline_validation_preserves_both_ledgers():
    timeline = RaceFinishTimeline(100, ["leader", "follower"])
    timeline.observe_crossing("leader", 1, 90, is_leader=True)
    timeline.observe_crossing("follower", 1, 100)
    before = deepcopy(vars(timeline))
    with pytest.raises(ValueError, match="chronological"):
        timeline.begin_suspension(99)
    assert vars(timeline._clock) == vars(before.pop("_clock"))
    assert {key: value for key, value in vars(timeline).items() if key != "_clock"} == before
    timeline.begin_suspension(110)
    with pytest.raises(ValueError, match="during suspension"):
        timeline.observe_crossing("leader", 2, 200, is_leader=True)
    # Rejected leading crossing must not prevent an earlier valid restart.
    timeline.end_suspension(150)
    assert timeline.total_suspension_seconds == 40
    with pytest.raises(ValueError, match="chronological"):
        timeline.retire("follower", 149)
    timeline.observe_crossing("leader", 2, 240, is_leader=True)


def test_restart_cannot_precede_collected_follower():
    timeline = RaceFinishTimeline(100, ["leader", "follower"])
    timeline.observe_crossing("leader", 1, 90, is_leader=True)
    timeline.begin_suspension(90)
    timeline.observe_crossing("follower", 1, 100)
    with pytest.raises(ValueError, match="chronological"):
        timeline.end_suspension(99)
    assert timeline.total_suspension_seconds == 0
    timeline.end_suspension(100)
    assert timeline.total_suspension_seconds == 10


def test_infinite_test_deadline_remains_supported(monkeypatch):
    monkeypatch.setattr("f1sim.simulation.race_timing.RACING_TIME_LIMIT_SECONDS", float("inf"))
    clock = RaceFinishClock(100)
    clock.begin_suspension(0)
    clock.end_suspension(100)
    assert clock.time_limit_seconds == float("inf")
    assert clock.observe_leader_crossing(1, 100000) == 100
