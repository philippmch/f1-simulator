"""Frozen first-lap queue observations for full-safety-car strategy costs."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real

from f1sim.models._native import register_forecast_helpers
from f1sim.simulation.neutralization import safety_car_running_time


def _finite(value, name, *, positive=False, nonnegative=False):
    if (isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value)
            or (positive and value <= 0) or (nonnegative and value < 0)):
        raise ValueError(f"{name} must be finite with a valid sign")


@dataclass(frozen=True, slots=True)
class SafetyCarBranch:
    """One candidate's expected track entry relative to its observed queue.

    A missing queue pace means this candidate anchors the queue. Standard
    running uses the predecessor's crossing relative to entry; chronological
    running uses its fractional progress at entry. Neither contains service
    samples or a future rival decision. Fitting fees belong after running.
    """

    queue_pace: float | None
    traffic_gap: float | None = None
    ahead_crossing: float | None = None
    ahead_progress: float | None = None
    blocked_until: float | None = None

    def __post_init__(self):
        if self.queue_pace is not None:
            _finite(self.queue_pace, "queue_pace", positive=True)
        if self.traffic_gap is not None:
            _finite(self.traffic_gap, "traffic_gap", nonnegative=True)
        if self.ahead_crossing is not None:
            _finite(self.ahead_crossing, "ahead_crossing")
        if self.blocked_until is not None:
            _finite(self.blocked_until, "blocked_until")
        if self.ahead_progress is not None:
            _finite(self.ahead_progress, "ahead_progress", nonnegative=True)
            if self.ahead_progress > 1:
                raise ValueError("ahead_progress must not exceed one lap")
        if self.ahead_crossing is not None and self.ahead_progress is not None:
            raise ValueError("use either a crossing or fractional progress")

    def running_time(self, free_running, modifier):
        if self.queue_pace is None:
            return free_running * modifier
        nominal = max(free_running, self.queue_pace)
        gap = (max(0., nominal - self.ahead_crossing)
               if self.ahead_crossing is not None else
               nominal * self.ahead_progress if self.ahead_progress is not None else None)
        value = safety_car_running_time(free_running, nominal, gap)
        return max(value, self.ahead_crossing) if self.ahead_crossing is not None else value


@dataclass(frozen=True, slots=True)
class StrategySafetyCarSnapshot:
    """Retained and paid-stop alternatives from the same decision snapshot."""

    retained: SafetyCarBranch
    stopped: SafetyCarBranch

    def __post_init__(self):
        if not all(isinstance(branch, SafetyCarBranch) for branch in (self.retained, self.stopped)):
            raise ValueError("safety-car alternatives must be SafetyCarBranch observations")

    @property
    def traffic_gaps(self):
        return self.retained.traffic_gap, self.stopped.traffic_gap

    def for_paid_fit(self):
        """Rank replacements after a stop has already been committed."""
        return StrategySafetyCarSnapshot(self.stopped, self.stopped)


def current_running_time(free_running, modifier, safety_car=None, *, stopped=False):
    """Apply control only to running, preserving the legacy uniform forecast."""
    if safety_car is None:
        return free_running * modifier
    branch = safety_car.stopped if stopped else safety_car.retained
    return branch.running_time(free_running, modifier)


def current_fitted_time(running, fitting_cost, safety_car=None, *, stopped=False):
    """Add fitting sensitivity, then enforce the observed SC predecessor.

    Fitting remains unscaled and may itself absorb blocked running. A car
    cannot cross a still-neutralized predecessor whose fitting delay is known.
    """
    value = running + fitting_cost
    if safety_car is None:
        return value
    branch = safety_car.stopped if stopped else safety_car.retained
    return max(value, branch.blocked_until) if branch.blocked_until is not None else value


register_forecast_helpers(globals(), ("safety_car_running_time",))
register_forecast_helpers(vars(SafetyCarBranch), ("running_time",))
