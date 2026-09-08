"""The strategy forecast predicts a flag without committing a finish crossing."""

import math

import pytest

from f1sim.simulation.race_timing import RaceFinishClock, forecast_final_lap


@pytest.mark.parametrize(('completed', 'clock', 'pace', 'deadline', 'modifier', 'expected'), [
    (40, 4000, 100, 7200, 1, 73),
    (40, 4000, 100, 7201, 1, 74),
    (40, 4000, 100, 7199, 1, 73),
    (40, 7200, 100, 7200, 1, 41),
    (40, 7300, 100, 7200, 1, 41),
    (40, 7100, 100, 7200, 1, 42),
    (40, 7050, 100, 7200, 2, 42),
    (40, 4000, 100, 7200, 2, 72),
    (40, 4000, 100, 10800, 1, 90),
    (89, 8900, 100, 7200, 1, 90),
])
def test_forecast_deadline_and_current_control(
    completed, clock, pace, deadline, modifier, expected,
):
    assert forecast_final_lap(90, completed, clock, pace, deadline, modifier) == expected


@pytest.mark.parametrize('pace', [None, 0, -1, math.nan, math.inf, True])
def test_without_valid_observed_pace_keep_scheduled_distance(pace):
    assert forecast_final_lap(90, 40, 4000, pace, 7200) == 90


@pytest.mark.parametrize('deadline', [math.inf, math.nan])
def test_unbounded_or_unknown_deadline_keeps_schedule(deadline):
    assert forecast_final_lap(90, 40, 4000, 100, deadline) == 90


def test_strategy_forecast_does_not_announce_finish():
    clock = RaceFinishClock(90)
    clock.observe_leader_crossing(1, 100)
    assert forecast_final_lap(90, 1, 100, 100, clock.time_limit_seconds) == 73
    assert clock.final_lap == 90
    assert not clock.time_limit_announced
    assert clock.winner_time is None
