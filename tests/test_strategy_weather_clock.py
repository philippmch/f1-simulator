"""Pure event-timeline tests for strategy weather update clocks."""

from dataclasses import FrozenInstanceError
from math import isclose

import pytest

from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock


def _clock(**overrides) -> StrategyWeatherClock:
    values = {
        "lap_start_offsets": (0.0, 170.0, 340.0, 510.0, 680.0),
        "first_update_after": 10.0,
        "update_interval": 90.0,
        "max_updates": 8,
        "current_stop_delay": 24.750946,
        "future_stop_delay": 24.750946,
    }
    values.update(overrides)
    return StrategyWeatherClock(**values)


def _timeline_count(clock: StrategyWeatherClock, elapsed: float) -> int:
    events = [
        clock.first_update_after + index * clock.update_interval
        for index in range(clock.max_updates)
    ]
    return min(clock.max_updates, sum(event <= elapsed + 1.0e-12 for event in events))


def test_stay_first_stop_future_stops_and_queue_are_counted_on_one_timeline() -> None:
    clock = _clock(current_stop_delay=30.0, future_stop_delay=40.0)

    # Staying reaches the first two external events at own offset 170.
    assert clock.updates(1, 0) == _timeline_count(clock, 170.0)
    # A first stop consumes its actual delay, then two later stops use the
    # future delay.  This is physical elapsed time, independent of traffic cost.
    assert clock.updates(1, 1, stopped_first=True) == _timeline_count(clock, 200.0)
    assert clock.updates(1, 3, stopped_first=True) == _timeline_count(clock, 280.0)


def test_decision_snapshot_does_not_count_equal_time_background_update() -> None:
    clock = _clock(first_update_after=0.0)

    assert clock.updates(0, 0) == 0
    assert clock.updates(0, 1) == 1


@pytest.mark.parametrize(
    ("elapsed", "expected"),
    [
        (9.999, 0),
        (10.0, 1),
        (99.999, 1),
        (100.0, 2),
        (190.0, 3),
        (1000.0, 8),
    ],
)
def test_equal_boundaries_skipped_events_and_cap(elapsed: float, expected: int) -> None:
    clock = _clock(lap_start_offsets=(0.0, elapsed))
    assert clock.updates(1, 0) == expected


def test_repeated_queries_are_cumulative_and_do_not_mutate_clock() -> None:
    clock = _clock()
    original = clock

    assert clock.updates(2, 0) == 4
    assert clock.updates(2, 0) == 4
    assert clock == original
    assert hash(clock) == hash(original)
    with pytest.raises(FrozenInstanceError):
        clock.max_updates = 3


def test_horizon_must_match_lap_start_offsets() -> None:
    clock = _clock()

    clock.validate_horizon(5)
    with pytest.raises(ValueError, match="horizon"):
        clock.validate_horizon(4)
    with pytest.raises(ValueError, match="horizon"):
        clock.validate_horizon(True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lap_start_offsets": ()},
        {"lap_start_offsets": (1.0, 2.0)},
        {"lap_start_offsets": (0.0, 2.0, 1.0)},
        {"lap_start_offsets": (0.0, float("nan"))},
        {"first_update_after": float("inf")},
        {"update_interval": 0.0},
        {"max_updates": -1},
        {"max_updates": True},
        {"current_stop_delay": -1.0},
        {"future_stop_delay": True},
    ],
)
def test_constructor_rejects_invalid_clock_context(kwargs) -> None:
    with pytest.raises(ValueError):
        _clock(**kwargs)


@pytest.mark.parametrize(
    "args",
    [
        (True, 0, False),
        (-1.0, 0, False),
        (5, 0, False),
        (0, True, False),
        (0, -1, False),
        (0, 0, True),
        (0, 0, 1),
    ],
)
def test_updates_rejects_invalid_inputs(args) -> None:
    with pytest.raises(ValueError):
        _clock().updates(*args)


def test_clock_reproduces_external_leader_event_timeline() -> None:
    # Relative to a decision at t=170, an external leader updates at
    # t=180,270,360.  Own starts are 0,170,340,... from that decision.
    clock = _clock(first_update_after=10.0, update_interval=90.0, max_updates=3)

    assert [clock.updates(offset, 0) for offset in range(3)] == [0, 2, 3]
    assert isclose(clock.current_stop_delay, 24.750946)
