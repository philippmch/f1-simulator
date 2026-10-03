"""Pure event-timeline tests for strategy weather update clocks."""

from dataclasses import FrozenInstanceError, replace
from math import isclose

import numpy as np
import pytest

from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock, _nonnegative_integer


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


def test_nonnegative_integer_accepts_large_builtin_int_and_rejects_negative():
    value = int("1000000")

    assert _nonnegative_integer(value, "value") == 1000000
    with pytest.raises(ValueError, match="nonnegative integer"):
        _nonnegative_integer(-1, "value")


@pytest.mark.parametrize("value", [True, 1.0, np.int64(-1)])
def test_nonnegative_integer_rejects_invalid_or_negative_integral_inputs(value):
    with pytest.raises(ValueError, match="nonnegative integer"):
        _nonnegative_integer(value, "value")


@pytest.mark.parametrize("value", [np.int64(3), np.uint64(3)])
def test_nonnegative_integer_accepts_numpy_integral_fallback(value):
    assert _nonnegative_integer(value, "value") == 3


@pytest.mark.parametrize("first_running", [(80., 110.), (200., 150.)])
@pytest.mark.parametrize("stopped", [False, True])
def test_observed_running_moves_later_entries_without_changing_physical_delays(
    first_running, stopped,
):
    clock = _clock(current_running_times=first_running,
                   current_stop_delay=30., future_stop_delay=40.)
    for offset in range(1, len(clock.lap_start_offsets)):
        for later_stops in range(3):
            paid = later_stops + stopped
            for fit in (0., 15.):
                elapsed = first_running[int(stopped)] + (offset - 1) * 170.
                elapsed += later_stops * 40. + (30. if stopped else 0.) + fit
                assert clock.updates(offset, paid, stopped, fit_delay=fit) == (
                    _timeline_count(clock, elapsed))
    assert clock.current_stop_delay == 30.
    assert clock.future_stop_delay == 40.
    assert clock.lap_start_offsets == (0., 170., 340., 510., 680.)


def test_running_override_does_not_move_current_entry_or_mutate_its_clock():
    clock = _clock(current_running_times=(80., 110.))
    original = replace(clock)
    baseline = replace(clock, current_running_times=None)
    assert clock.updates(0, 0) == baseline.updates(0, 0)
    assert clock.updates(0, 1, True, fit_delay=15.) == baseline.updates(0, 1, True)
    assert clock.updates(1, 0) != baseline.updates(1, 0)
    assert clock == original and hash(clock) == hash(original)
    assert clock != baseline and hash(clock) != hash(baseline)
    with pytest.raises(FrozenInstanceError):
        clock.current_running_times = (90., 90.)


@pytest.mark.parametrize("value", [[], [80., 110.], (), (80.,), (80., 110., 90.),
                                   (0., 80.), (80., -1.), (True, 80.),
                                   (80., float("nan")), (float("inf"), 80.)])
def test_running_override_rejects_invalid_observations(value):
    with pytest.raises(ValueError, match="current_running_times"):
        _clock(current_running_times=value)


def test_one_lap_clock_ignores_future_running_and_large_offsets_do_not_overflow_early():
    clock = _clock(lap_start_offsets=(0.,), current_running_times=(80., 110.))
    assert clock.updates(0, 0) == 0
    assert clock.updates(0, 1, True) == 1
    clock = _clock(lap_start_offsets=(0., 1.e308), first_update_after=1.e308,
                   update_interval=1.e308, current_running_times=(1.e308, 1.e308))
    assert clock.updates(1, 0) == 1
