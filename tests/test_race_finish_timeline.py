"""Chronological flag semantics without truncating running after the winner."""

import heapq

import pytest

from f1sim.simulation.race_timing import RaceFinishClock, RaceFinishTimeline


def snapshot(timeline):
    return timeline.final_lap, timeline.chequered_time, timeline.winner_id, dict(timeline.states)


def test_lapped_car_finishes_at_own_next_crossing():
    timeline = RaceFinishTimeline(10, ["Fast", "Slow"])
    queue = [(90, "Fast", 1), (110, "Slow", 1)]
    while queue:
        time, driver, lap = heapq.heappop(queue)
        timeline.observe_crossing(driver, lap, time, is_leader=driver == "Fast")
        if timeline.can_start_next_lap(driver):
            heapq.heappush(queue, (time + (90 if driver == "Fast" else 110), driver, lap + 1))
    assert timeline.winner_id == "Fast" and timeline.chequered_time == 900
    assert timeline.states["Fast"].finish_time == 900
    assert timeline.states["Slow"].finish_time == 990
    assert timeline.states["Slow"].completed_laps == 9
    assert not timeline.can_start_next_lap("Fast")
    assert not timeline.can_start_next_lap("Slow")


def test_exact_ties_follow_explicit_leader_first_order():
    timeline = RaceFinishTimeline(2, ["A", "B"])
    for lap in (1, 2):
        timeline.observe_crossing("A", lap, lap * 90, is_leader=True)
        timeline.observe_crossing("B", lap, lap * 90)
    assert [state.finish_time for state in timeline.states.values()] == [180, 180]
    assert timeline.winner_id == "A"


def test_timed_boundary_and_active_car_can_retire_after_winner():
    timeline = RaceFinishTimeline(10, ["A", "B", "C"])
    timeline.retire("C", 1)
    for lap in range(1, 6):
        timeline.observe_crossing("A", lap, lap * 1800, is_leader=True)
    assert timeline.final_lap == 5 and timeline.chequered_time == 9000
    assert timeline.can_start_next_lap("B")  # May still incur events before its crossing.
    timeline.retire("B", 9050)
    assert timeline.states["B"].retired
    assert timeline.states["B"].finish_time is None
    assert timeline.states["B"].completed_laps == 0
    assert not timeline.can_start_next_lap("B")


def test_retirements_do_not_create_winner():
    timeline = RaceFinishTimeline(10, ["A", "B"])
    timeline.observe_crossing("A", 1, 90, is_leader=True)
    timeline.retire("A", 95)
    timeline.retire("B", 100)
    assert timeline.winner_id is None and timeline.chequered_time is None
    assert timeline.states["A"].completed_laps == 1
    assert all(state.finish_time is None for state in timeline.states.values())


@pytest.mark.parametrize("lap,time", [(1, 91), (3, 91), (2, 89), (2, float("nan")),
                                      (2, float("inf")), (2, -1), (True, 91)])
def test_invalid_crossings_are_atomic(lap, time):
    timeline = RaceFinishTimeline(10, ["A"])
    timeline.observe_crossing("A", 1, 90, is_leader=True)
    before = snapshot(timeline)
    with pytest.raises(ValueError):
        timeline.observe_crossing("A", lap, time, is_leader=True)
    assert snapshot(timeline) == before


def test_failed_leader_validation_does_not_commit_driver_crossing():
    timeline = RaceFinishTimeline(3, ["A", "B"])
    timeline.observe_crossing("A", 1, 90, is_leader=True)
    before = snapshot(timeline)
    with pytest.raises(ValueError):
        timeline.observe_crossing("B", 1, 95, is_leader=True)
    assert snapshot(timeline) == before
    timeline.observe_crossing("B", 1, 95)
    timeline.observe_crossing("B", 2, 180, is_leader=True)  # Leader identity may change.
    assert timeline.states["B"].completed_laps == 2


@pytest.mark.parametrize("retired", [False, True])
def test_terminal_driver_rejects_crossings_and_retirements_atomically(retired):
    timeline = RaceFinishTimeline(1, ["A"])
    if retired:
        timeline.retire("A", 90)
    else:
        timeline.observe_crossing("A", 1, 90, is_leader=True)
    before = snapshot(timeline)
    with pytest.raises(ValueError):
        timeline.observe_crossing("A", 2, 180)
    with pytest.raises(ValueError):
        timeline.retire("A", 180)
    assert snapshot(timeline) == before


@pytest.mark.parametrize("scheduled,pace,finish", [(10, 1800, 5), (10, 1799, 6), (4, 1800, 4)])
def test_leader_clock_distance_and_time_caps(scheduled, pace, finish):
    clock = RaceFinishClock(scheduled)
    for lap in range(1, finish + 1):
        clock.observe_leader_crossing(lap, lap * pace)
    assert clock.final_lap == finish
    assert clock.winner_time == finish * pace
    before = vars(clock).copy()
    with pytest.raises(ValueError):
        clock.observe_leader_crossing(finish + 1, (finish + 1) * pace)
    assert vars(clock) == before


def test_state_view_cannot_be_mutated():
    timeline = RaceFinishTimeline(1, ["A"])
    with pytest.raises(TypeError):
        timeline.states["A"] = None


@pytest.mark.parametrize("time", [89, -1, float("nan"), float("inf")])
def test_invalid_retirement_is_atomic(time):
    timeline = RaceFinishTimeline(3, ["A", "B"])
    timeline.observe_crossing("A", 1, 90, is_leader=True)
    before = snapshot(timeline)
    with pytest.raises(ValueError):
        timeline.retire("B", time)
    assert snapshot(timeline) == before


@pytest.mark.parametrize("lap,time", [(2, 90), (1, float("nan")), (1, float("inf")), (1, -1)])
def test_invalid_first_leader_crossing_is_atomic(lap, time):
    clock = RaceFinishClock(10)
    before = vars(clock).copy()
    with pytest.raises(ValueError):
        clock.observe_leader_crossing(lap, time)
    assert vars(clock) == before


@pytest.mark.parametrize("scheduled", [1, 2])
def test_missing_leader_on_first_crossing_is_atomic(scheduled):
    timeline = RaceFinishTimeline(scheduled, ["A"])
    before = snapshot(timeline)
    with pytest.raises(ValueError, match="identify the leader"):
        timeline.observe_crossing("A", 1, 90)
    assert snapshot(timeline) == before


def test_follower_first_at_exact_finish_time_is_rejected_atomically():
    timeline = RaceFinishTimeline(2, ["A", "B"])
    timeline.observe_crossing("A", 1, 90, is_leader=True)
    timeline.observe_crossing("B", 1, 90)
    before = snapshot(timeline)
    with pytest.raises(ValueError, match="identify the leader"):
        timeline.observe_crossing("B", 2, 180)
    assert snapshot(timeline) == before
    timeline.observe_crossing("A", 2, 180, is_leader=True)
    timeline.observe_crossing("B", 2, 180)
    assert timeline.states["B"].finish_time == 180


def test_lapped_successor_can_reach_retired_leaders_previous_lap():
    timeline = RaceFinishTimeline(4, ["A", "B"])
    for time, driver, lap in [(90, "A", 1), (110, "B", 1), (180, "A", 2),
                              (220, "B", 2), (270, "A", 3)]:
        timeline.observe_crossing(driver, lap, time, is_leader=driver == "A")
    timeline.retire("A", 275)
    timeline.observe_crossing("B", 3, 330, is_leader=True)
    timeline.observe_crossing("B", 4, 440, is_leader=True)
    assert timeline.winner_id == "B" and timeline.chequered_time == 440
    assert timeline.states["A"].retired and timeline.states["A"].completed_laps == 3


@pytest.mark.parametrize("successor_lap", [2, 3])
def test_same_or_lower_distance_successor_crossing_announces_time_limit(successor_lap):
    timeline = RaceFinishTimeline(10, ["A", "B"])
    observations = [(90, "A", 1), (110, "B", 1), (180, "A", 2), (270, "A", 3)]
    if successor_lap == 3:
        observations.append((220, "B", 2))
    for time, driver, lap in sorted(observations):
        timeline.observe_crossing(driver, lap, time, is_leader=driver == "A")
    timeline.retire("A", 275)
    timeline.observe_crossing("B", successor_lap, 7500, is_leader=True)
    assert timeline.final_lap == successor_lap + 1
    assert timeline.chequered_time is None
    timeline.observe_crossing("B", successor_lap + 1, 7600, is_leader=True)
    assert timeline.winner_id == "B" and timeline.chequered_time == 7600


def test_retirement_reset_cannot_promote_driver_trailing_another_active_car():
    timeline = RaceFinishTimeline(10, ["A", "B", "C"])
    for lap in range(1, 4):
        timeline.observe_crossing("A", lap, lap * 90, is_leader=True)
        timeline.observe_crossing("C", lap, lap * 90 + 1)
    timeline.retire("A", 275)
    before = snapshot(timeline)
    with pytest.raises(ValueError, match="trail another active"):
        timeline.observe_crossing("B", 1, 300, is_leader=True)
    assert snapshot(timeline) == before


def test_clock_reset_requires_explicit_opt_in_and_cannot_skip_distance():
    clock = RaceFinishClock(10)
    for lap in range(1, 4):
        clock.observe_leader_crossing(lap, lap * 90)
    before = vars(clock).copy()
    for lap, reset in [(2, False), (5, True)]:
        with pytest.raises(ValueError):
            clock.observe_leader_crossing(lap, 7500, allow_leadership_reset=reset)
        assert vars(clock) == before
    assert clock.observe_leader_crossing(2, 7500, allow_leadership_reset=True) == 3
    assert clock.observe_leader_crossing(3, 7600) == 3
    assert clock.winner_time == 7600


@pytest.mark.parametrize("successor_lap", [2, 4])
@pytest.mark.parametrize("scheduled", [5, 10])
def test_post_announcement_regressing_handoff_is_explicitly_unsupported_and_atomic(
    successor_lap, scheduled,
):
    timeline = RaceFinishTimeline(scheduled, ["A", "B"])
    observations = [(1800 * lap, "A", lap) for lap in range(1, 5)]
    observations += [(1800 * lap + 10, "B", lap) for lap in range(1, successor_lap)]
    for time, driver, lap in sorted(observations):
        timeline.observe_crossing(driver, lap, time, is_leader=driver == "A")
    assert timeline.final_lap == 5
    timeline.retire("A", 7250)
    before = snapshot(timeline)
    clock_before = vars(timeline._clock).copy()
    leader_before = timeline._leader_id
    with pytest.raises(NotImplementedError, match="after a two-hour"):
        timeline.observe_crossing("B", successor_lap, 7300, is_leader=True)
    assert snapshot(timeline) == before
    assert vars(timeline._clock) == clock_before
    assert timeline._leader_id == leader_before
    assert timeline.can_start_next_lap("B")
