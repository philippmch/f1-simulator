"""Chronological finish deadline with bounded suspension-time extensions."""

from dataclasses import dataclass, replace
from math import ceil, isfinite
from numbers import Integral, Real
from types import MappingProxyType
from typing import Iterable, Mapping

RACING_TIME_LIMIT_SECONDS = 7200.0


def forecast_final_lap(
    scheduled_final_lap: int,
    completed_lap: int,
    crossing_time: float,
    running_pace: float | None,
    time_limit_seconds: float,
    current_lap_time_modifier: float = 1.0,
) -> int:
    """Estimate strategy distance without announcing or changing the finish.

    The upcoming lap uses current race control; subsequent laps assume green.
    Observed running pace excludes pit service and other elapsed-time losses.
    Without a usable observation, retain the scheduled strategy horizon.
    """
    values = (crossing_time, running_pace, time_limit_seconds, current_lap_time_modifier)
    if any(isinstance(value, bool) or not isinstance(value, Real)
           or not isfinite(value) for value in values):
        return scheduled_final_lap
    if running_pace <= 0 or current_lap_time_modifier <= 0 or crossing_time < 0:
        return scheduled_final_lap
    if crossing_time >= time_limit_seconds:
        return min(scheduled_final_lap, completed_lap + 1)
    next_crossing = crossing_time + running_pace * current_lap_time_modifier
    additional = max(0, ceil((time_limit_seconds - next_crossing) / running_pace))
    return min(scheduled_final_lap, completed_lap + 2 + additional)


def announced_final_lap(current_final_lap: int, completed_lap: int, leader_time: float) -> int:
    """After two hours expire, finish the following lap or the scheduled last lap."""
    if leader_time >= RACING_TIME_LIMIT_SECONDS:
        return min(current_final_lap, completed_lap + 1)
    return current_final_lap


def _valid_lap(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _valid_time(value: float, previous: float | None) -> None:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not isfinite(value) or value < 0):
        raise ValueError("crossing time must be finite and nonnegative")
    if previous is not None and value < previous:
        raise ValueError("observations must be in chronological order")


class RaceFinishClock:
    """Leader-only distance/deadline state, independent of driver identity.

    Leading crossings must be observed consecutively from lap one. Equal clock
    values are allowed; the caller controls exact-time observation ordering.
    """

    def __init__(self, scheduled_laps: int):
        _valid_lap(scheduled_laps, "scheduled_laps")
        self.scheduled_laps = scheduled_laps
        self.final_lap = scheduled_laps
        self.completed_laps = 0
        self.last_crossing_time: float | None = None
        self.winner_time: float | None = None
        self._time_limit_announced = False
        self._last_observation_time: float | None = None
        self._suspension_start: float | None = None
        self._total_suspension_seconds = 0.0

    @property
    def total_suspension_seconds(self) -> float:
        """Duration of completed suspensions, including collection and pause."""
        return self._total_suspension_seconds

    @property
    def time_limit_seconds(self) -> float:
        return RACING_TIME_LIMIT_SECONDS + min(self.total_suspension_seconds, 3600.0)

    @property
    def time_limit_announced(self) -> bool:
        """Whether a timed final-crossing announcement has been made."""
        return self._time_limit_announced

    def begin_suspension(self, time: float) -> None:
        _valid_time(time, self._last_observation_time)
        if self.winner_time is not None:
            raise ValueError("cannot suspend after the chequered flag")
        if self._suspension_start is not None:
            raise ValueError("a suspension is already open")
        self._suspension_start = time
        self._last_observation_time = time

    def end_suspension(self, time: float) -> None:
        _valid_time(time, self._last_observation_time)
        if self._suspension_start is None:
            raise ValueError("no suspension is open")
        self._total_suspension_seconds += time - self._suspension_start
        self._suspension_start = None
        self._last_observation_time = time

    def observe_leader_crossing(
        self, completed_lap: int, time: float, *, allow_leadership_reset: bool = False,
    ) -> int:
        """Commit a crossing; opt-in reset permits a lapped successor after retirement.

        The caller must verify the previous leader retired before opting in.
        A reset can lower the active leading distance, but cannot skip ahead.
        """
        _valid_lap(completed_lap, "completed_lap")
        _valid_time(time, self._last_observation_time)
        if self._suspension_start is not None:
            raise ValueError("leader crossings are forbidden during suspension")
        if self.winner_time is not None:
            raise ValueError("the leader has already received the chequered flag")
        if not isinstance(allow_leadership_reset, bool):
            raise ValueError("allow_leadership_reset must be a boolean")
        valid_reset = (allow_leadership_reset and self.completed_laps > 0
                       and completed_lap <= self.completed_laps + 1)
        if completed_lap != self.completed_laps + 1 and not valid_reset:
            raise ValueError("leader laps must be consecutive without duplicates")
        finish_pending = self._time_limit_announced
        final_lap = self.final_lap
        if not finish_pending and time >= self.time_limit_seconds:
            final_lap = min(final_lap, completed_lap + 1)
        self.final_lap = final_lap
        self._time_limit_announced |= (time >= self.time_limit_seconds
                                       and completed_lap < final_lap)
        self.completed_laps = completed_lap
        self.last_crossing_time = time
        self._last_observation_time = time
        # The timed signal belongs to the next leading crossing, not the old
        # leader's personal lap number. A retired leader's lapped successor
        # must not restart the countdown or drive extra laps to reach it.
        if finish_pending or completed_lap == final_lap:
            self.winner_time = time
        return final_lap


@dataclass(frozen=True)
class DriverFinishState:
    completed_laps: int = 0
    last_crossing_time: float | None = None
    finish_time: float | None = None
    retired: bool = False
    retirement_time: float | None = None


class RaceFinishTimeline:
    """Pure chronological crossing ledger, including running after the winner.

    The caller identifies the authoritative leader and submits observations in
    chronological order. At identical timestamps it must submit the leader's
    crossing first, then other crossings in its deterministic physical order.
    An unfinished car may still pit, suffer incidents, or retire after the
    winner's flag; it finishes only at its own next crossing. This controller
    neither schedules those events nor truncates their running time.
    """

    def __init__(self, scheduled_laps: int, driver_ids: Iterable[str]):
        self._clock = RaceFinishClock(scheduled_laps)
        ids = list(driver_ids)
        if len(ids) != len(set(ids)):
            raise ValueError("driver IDs must be unique")
        self._states = {driver_id: DriverFinishState() for driver_id in ids}
        self._last_observation_time: float | None = None
        self._leader_id: str | None = None
        self.winner_id: str | None = None

    @property
    def final_lap(self) -> int:
        return self._clock.final_lap

    @property
    def chequered_time(self) -> float | None:
        return self._clock.winner_time

    @property
    def total_suspension_seconds(self) -> float:
        return self._clock.total_suspension_seconds

    @property
    def time_limit_seconds(self) -> float:
        return self._clock.time_limit_seconds

    @property
    def time_limit_announced(self) -> bool:
        return self._clock.time_limit_announced

    def begin_suspension(self, time: float) -> None:
        _valid_time(time, self._last_observation_time)
        self._clock.begin_suspension(time)
        self._last_observation_time = time

    def end_suspension(self, time: float) -> None:
        _valid_time(time, self._last_observation_time)
        self._clock.end_suspension(time)
        self._last_observation_time = time

    @property
    def states(self) -> Mapping[str, DriverFinishState]:
        return MappingProxyType(self._states)

    def _active_state(self, driver_id: str) -> DriverFinishState:
        if driver_id not in self._states:
            raise ValueError(f"unknown driver ID: {driver_id}")
        state = self._states[driver_id]
        if state.retired or state.finish_time is not None:
            raise ValueError(f"driver is already finished or retired: {driver_id}")
        return state

    def can_start_next_lap(self, driver_id: str) -> bool:
        if driver_id not in self._states:
            raise ValueError(f"unknown driver ID: {driver_id}")
        state = self._states[driver_id]
        return not state.retired and state.finish_time is None

    def observe_crossing(
        self, driver_id: str, completed_lap: int, time: float, *, is_leader: bool = False,
    ) -> DriverFinishState:
        state = self._active_state(driver_id)
        _valid_lap(completed_lap, "completed_lap")
        _valid_time(time, self._last_observation_time)
        if completed_lap != state.completed_laps + 1:
            raise ValueError("driver laps must be consecutive without duplicates")
        if completed_lap > self._clock.scheduled_laps:
            raise ValueError("driver distance exceeds the scheduled race distance")
        if not isinstance(is_leader, bool):
            raise ValueError("is_leader must be a boolean")
        active_distance = max(
            active.completed_laps for active in self._states.values()
            if not active.retired and active.finish_time is None
        )
        if self.chequered_time is None:
            if not is_leader and completed_lap > active_distance:
                raise ValueError("a crossing leading the active race must identify the leader")
            if is_leader and completed_lap <= active_distance:
                raise ValueError(
                    "a leader cannot trail another active car's completed distance or tie it"
                )
        leadership_reset = (self._leader_id is not None
                            and self._states[self._leader_id].retired)
        # The clock validates before mutation. No validation can fail after
        # this call, so invalid observations leave both ledgers untouched.
        if is_leader:
            self._clock.observe_leader_crossing(
                completed_lap, time, allow_leadership_reset=leadership_reset,
            )
            self._leader_id = driver_id
            if self.chequered_time is not None:
                self.winner_id = driver_id
        finished = self.chequered_time is not None
        updated = replace(state, completed_laps=completed_lap, last_crossing_time=time,
                          finish_time=time if finished else None)
        self._states[driver_id] = updated
        self._last_observation_time = time
        return updated

    def retire(self, driver_id: str, time: float) -> DriverFinishState:
        """Retire at an event timestamp without fabricating a completed crossing."""
        state = self._active_state(driver_id)
        _valid_time(time, self._last_observation_time)
        updated = replace(state, retired=True, retirement_time=time)
        self._states[driver_id] = updated
        self._last_observation_time = time
        return updated
