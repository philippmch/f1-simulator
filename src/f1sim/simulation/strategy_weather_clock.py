"""Pure weather-update clocks for strategy projections.

The clock represents updates observed from an external leading car.  It does
not advance weather itself; callers use :meth:`StrategyWeatherClock.updates`
to obtain a cumulative count and apply that many surface projections.  A
driver whose decision snapshot is already current has ``offset == 0`` and no
paid stops, which intentionally returns zero even when an equal-time
background update is queued.
"""

from dataclasses import dataclass
from math import floor, isfinite
from numbers import Integral, Real


def _finite_nonnegative(value, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not isfinite(value) or value < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _positive_finite(value, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _nonnegative_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


@dataclass(frozen=True, slots=True)
class StrategyWeatherClock:
    """Immutable cumulative update clock anchored to an external leader.

    ``lap_start_offsets`` contains the projected own-lap start offsets from a
    decision snapshot; ``updates`` receives an index into this tuple.
    ``first_update_after`` and ``update_interval`` define the external leader's
    update events.  A paid stop consumes its physical
    duration on this clock: the first stop uses ``current_stop_delay`` and
    later stops use ``future_stop_delay``.  Traffic pricing adjustments belong
    outside this object and must not be folded into those physical delays.
    """

    lap_start_offsets: tuple[float, ...]
    first_update_after: float
    update_interval: float
    max_updates: int
    current_stop_delay: float
    future_stop_delay: float

    def __post_init__(self) -> None:
        if not isinstance(self.lap_start_offsets, tuple) or not self.lap_start_offsets:
            raise ValueError("lap_start_offsets must be a nonempty tuple")
        offsets = tuple(
            _finite_nonnegative(value, "lap_start_offsets values")
            for value in self.lap_start_offsets
        )
        if offsets[0] != 0.0:
            raise ValueError("lap_start_offsets must start at zero")
        if any(previous > current for previous, current in zip(offsets, offsets[1:])):
            raise ValueError("lap_start_offsets must be nondecreasing")
        object.__setattr__(self, "lap_start_offsets", offsets)
        object.__setattr__(
            self, "first_update_after", _finite_nonnegative(
                self.first_update_after, "first_update_after"
            )
        )
        object.__setattr__(
            self, "update_interval", _positive_finite(
                self.update_interval, "update_interval"
            )
        )
        object.__setattr__(self, "max_updates", _nonnegative_integer(
            self.max_updates, "max_updates"
        ))
        object.__setattr__(
            self, "current_stop_delay", _finite_nonnegative(
                self.current_stop_delay, "current_stop_delay"
            )
        )
        object.__setattr__(
            self, "future_stop_delay", _finite_nonnegative(
                self.future_stop_delay, "future_stop_delay"
            )
        )

    def validate_horizon(self, horizon: int) -> None:
        """Require a caller's planning horizon to match the stored starts."""

        horizon = _nonnegative_integer(horizon, "horizon")
        if horizon != len(self.lap_start_offsets):
            raise ValueError("horizon must match lap_start_offsets")

    def updates(self, offset: int, paid_stops: int, stopped_first: bool = False) -> int:
        """Return external leading updates at or before a projected own start.

        ``offset`` indexes ``lap_start_offsets``.  The event time is that
        nominal start plus ``paid_stops * future_stop_delay`` with the first
        stop correction ``current_stop_delay - future_stop_delay`` when
        ``stopped_first`` is true.  Counts are cumulative, capped, and use a
        small boundary tolerance so mathematically equal events remain equal
        after ordinary floating-point arithmetic.
        """

        offset = _nonnegative_integer(offset, "offset")
        if offset >= len(self.lap_start_offsets):
            raise ValueError("offset must be within lap_start_offsets")
        paid_stops = _nonnegative_integer(paid_stops, "paid_stops")
        if not isinstance(stopped_first, bool):
            raise ValueError("stopped_first must be a boolean")
        if stopped_first and paid_stops == 0:
            raise ValueError("stopped_first requires at least one paid stop")
        if offset == 0 and paid_stops == 0:
            return 0
        if self.max_updates == 0:
            return 0

        try:
            elapsed = self.lap_start_offsets[offset] + paid_stops * self.future_stop_delay
        except OverflowError:
            return self.max_updates
        if stopped_first:
            elapsed += self.current_stop_delay - self.future_stop_delay
        if not isfinite(elapsed):
            return self.max_updates
        if elapsed < self.first_update_after:
            return 0
        ratio = (elapsed - self.first_update_after) / self.update_interval
        if not isfinite(ratio):
            return self.max_updates
        count = floor(ratio + 1.0e-12) + 1
        return max(0, min(self.max_updates, count))
