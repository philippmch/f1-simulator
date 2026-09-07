"""Queue gaps close through positive running time, never retroactive clock edits."""

import pytest

from f1sim.simulation.neutralization import safety_car_running_time


def test_large_gap_closes_over_laps_without_crossing_free_pace_bound():
    # Cars start the same numbered lap 80 seconds apart. The leader travels
    # at 126 s/lap; the follower can catch up at no faster than 90 s/lap.
    leader_clock, follower_clock = 0.0, 80.0
    observed_gaps, durations = [], []
    for _ in range(4):
        gap = follower_clock - leader_clock
        running = safety_car_running_time(90, 126, gap)
        previous = follower_clock
        leader_clock += safety_car_running_time(90, 126, None)
        follower_clock += running
        assert follower_clock > previous
        durations.append(running)
        observed_gaps.append(follower_clock - leader_clock)
    assert durations == [90, 90, 119, 126]
    assert observed_gaps == [44, 8, 1, 1]


def test_catchup_does_not_spread_an_already_compact_queue():
    assert safety_car_running_time(90, 126, 0.5) == 126
    assert safety_car_running_time(90, 126, 1) == 126
    assert safety_car_running_time(90, 126, 30) == 97


def test_slower_follower_joins_common_safety_car_pace():
    # Although its free lap takes 100 seconds, this car follows the same
    # 126-second SC pace as the 90-second leader once their queue has formed.
    follower_crossing = 20 + safety_car_running_time(100, 126, 20)
    assert follower_crossing - 126 == 1
    next_crossing = follower_crossing + safety_car_running_time(100, 126, 1)
    assert next_crossing - 2 * 126 == 1


@pytest.mark.parametrize("free,nominal,gap", [
    (0, 126, 1), (90, 80, 1), (90, float("inf"), 1),
    (90, 126, float("nan")), (90, 126, -1),
])
def test_invalid_clock_inputs_are_rejected(free, nominal, gap):
    with pytest.raises(ValueError):
        safety_car_running_time(free, nominal, gap)
