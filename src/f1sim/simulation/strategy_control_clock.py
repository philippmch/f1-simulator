"""Observed field clocks for a bounded SC/VSC prefix of strategy planning."""

from copy import copy
from dataclasses import dataclass, replace
from math import inf, isclose, isfinite

from f1sim.models._native import register_forecast_helpers
from f1sim.simulation.chronological_finish import (
    ChronologicalFinishContext,
    ObservedChronologicalField,
)
from f1sim.simulation.finish_strategy import SafetyCarFinishField
from f1sim.simulation.neutralization import safety_car_running_times
from f1sim.simulation.strategy_neutralization import _finite


@dataclass(frozen=True, slots=True)
class StandardControlContext:
    """Frozen first-lap entries in the standard engine's shared lap order."""

    current_lap: int
    entry_time: float
    field: SafetyCarFinishField
    modifier: float
    safety_car: bool
    control_intervals: int
    stop_delay: float
    pre_stop_order: tuple[str | None, ...] = ()
    committed_pitters: tuple[str, ...] = ()

    def __post_init__(self):
        if (type(self.current_lap) is not int or self.current_lap < 1
                or type(self.field) is not SafetyCarFinishField
                or type(self.safety_car) is not bool
                or type(self.control_intervals) is not int or self.control_intervals < 1):
            raise ValueError("invalid standard control context")
        _finite(self.entry_time, "entry_time", nonnegative=True)
        _finite(self.modifier, "modifier", positive=True)
        _finite(self.stop_delay, "stop_delay", nonnegative=True)
        if not isfinite(self.entry_time + self.stop_delay):
            raise ValueError("invalid expected stop exit")
        identifiers = {None, *(row.identifier for row in self.field.retained.rows
                               if row is not None)}
        if (type(self.pre_stop_order) is not tuple
                or self.pre_stop_order and (len(self.pre_stop_order) != len(identifiers)
                                           or set(self.pre_stop_order) != identifiers)
                or type(self.committed_pitters) is not tuple
                or len(set(self.committed_pitters)) != len(self.committed_pitters)
                or any(key is None or key not in identifiers for key in self.committed_pitters)):
            raise ValueError("invalid committed standard pit order")


class ObservedStandardField:
    """Independent cumulative entries through known lockstep control laps.

    Rival free pace is held. A later paid stop rejoins the physical queue by
    expected elapsed clock. SC catch-up recovers eligible gaps on each lap;
    fitting fees and no-passing barriers follow running. VSC uses its uniform
    running modifier and the engine's completed-clock pit rejoin order.
    """

    def __init__(self, context):
        self.context = context
        self.rows = list(context.field.retained.rows)
        self.now = context.entry_time
        self.lap = context.current_lap
        self.intervals_left = context.control_intervals
        self.safety_car = context.safety_car
        self.updates = 0
        self.entered = False
        self.pre_stop_order = context.pre_stop_order or tuple(
            None if row is None else row.identifier for row in self.rows)
        self.pitting = context.committed_pitters

    def fork(self):
        branch = copy(self)
        branch.rows = self.rows.copy()
        return branch

    @property
    def controlled(self):
        return self.intervals_left > 0

    @property
    def running_modifier(self):
        return self.context.modifier if self.controlled else 1.

    @property
    def projection_required(self):
        return self.controlled

    @property
    def finished(self):
        return False

    def enter(self, stop_delay=None):
        if self.entered:
            raise ValueError("candidate already entered this projected lap")
        if stop_delay is not None:
            _finite(stop_delay, "stop_delay", nonnegative=True)
            ready = self.now + stop_delay
            if not isfinite(ready):
                raise ValueError("invalid expected stop exit")
            if (self.lap == self.context.current_lap
                    and isclose(stop_delay, self.context.stop_delay, rel_tol=0., abs_tol=1.e-9)):
                self.rows = list(self.context.field.stopped.rows)
            else:
                previous = self.rows.index(None)
                self.rows.remove(None)
                # Staying cars retain their relative physical order. At equal
                # entry clocks the candidate retains its previous tie order.
                insertion = next((index for index, row in enumerate(self.rows)
                                  if (ready, previous) < (row.entry_time,
                                                         index + (index >= previous))),
                                 len(self.rows))
                self.rows.insert(insertion, None)
            self.now = ready
            self.pitting = (*self.pitting, None)
        self.entered = True
        return self.now

    def gap_ahead(self, reference):
        if not self.entered:
            raise ValueError("candidate must enter before observing its predecessor")
        index = self.rows.index(None)
        return None if index == 0 else max(0., self.now - self.rows[index - 1].entry_time)

    def cross(self, free_running, fitting_cost=0.):
        if not self.entered:
            raise ValueError("candidate must enter before running")
        _finite(free_running, "free_running", positive=True)
        _finite(fitting_cost, "fitting_cost", nonnegative=True)
        observations = [(None, self.now, free_running) if row is None else
                        (row.identifier, row.entry_time, row.free_running) for row in self.rows]
        running = (safety_car_running_times(observations, self.running_modifier)
                   if self.controlled and self.safety_car else
                   {key: free * self.running_modifier for key, _, free in observations})
        crossings = {key: entry + running[key] + (fitting_cost if row is None else row.fitting_cost)
                     for row, (key, entry, _) in zip(self.rows, observations)}
        if not self.safety_car and self.pitting:
            # The standard VSC engine resolves paid rejoin order using the
            # completed lap clocks. SC instead keeps its pit-exit queue.
            positions = {key: index for index, key in enumerate(self.pre_stop_order)}
            pitting = sorted(self.pitting, key=lambda key: (crossings[key], positions[key]))
            staying = [key for key in self.pre_stop_order if key not in self.pitting]
            ordered, index = [], 0
            for key in staying:
                while index < len(pitting) and (crossings[pitting[index]],
                                                positions[pitting[index]]) < (
                                                    crossings[key], positions[key]):
                    ordered.append(pitting[index])
                    index += 1
                ordered.append(key)
            ordered.extend(pitting[index:])
            rows_by_id = {None if row is None else row.identifier: row for row in self.rows}
            self.rows = [rows_by_id[key] for key in ordered]
        previous, rows = None, []
        for row in self.rows:
            key = None if row is None else row.identifier
            entry = self.now if row is None else row.entry_time
            crossing = crossings[key]
            if self.controlled and previous is not None:
                crossing = max(crossing, previous)
            if not isfinite(crossing) or crossing <= entry:
                raise ValueError("invalid projected running crossing")
            previous = crossing
            if row is None:
                self.now = crossing
                rows.append(None)
            else:
                rows.append(replace(row, entry_time=crossing, fitting_cost=0.))
        self.rows = rows
        self.pre_stop_order = tuple(None if row is None else row.identifier for row in rows)
        self.pitting = ()
        self.intervals_left = max(0, self.intervals_left - 1)
        self.lap += 1
        self.updates += 1
        self.entered = False


def _field_observation(field):
    if type(field) is StandardControlContext:
        return field
    values = dict(vars(field))
    timeline = values.pop("timeline")
    if timeline is None:
        return values, None
    ledger = dict(vars(timeline))
    clock = ledger.pop("_clock")
    return values, ledger, dict(vars(clock))


@dataclass(frozen=True, slots=True, eq=False)
class StrategyControlContext:
    """A dry decision's frozen field, expected current stop and paid-fit phase."""

    field: StandardControlContext | ChronologicalFinishContext
    now: float
    current_stop_delay: float
    paid_fit: bool = False

    __hash__ = None

    def __eq__(self, other):
        if type(other) is not StrategyControlContext:
            return NotImplemented
        # A copied private timeline has a different object identity. Compare
        # its actual observations so state snapshots still detect mutation.
        return (self.now == other.now and self.current_stop_delay == other.current_stop_delay
                and self.paid_fit == other.paid_fit
                and _field_observation(self.field) == _field_observation(other.field))

    def __post_init__(self):
        if (type(self.field) not in (StandardControlContext, ChronologicalFinishContext)
                or type(self.paid_fit) is not bool):
            raise ValueError("invalid strategy control context")
        _finite(self.now, "now", nonnegative=True)
        _finite(self.current_stop_delay, "current_stop_delay", nonnegative=True)
        if not isfinite(self.now + self.current_stop_delay):
            raise ValueError("invalid expected stop exit")

    def new_field(self):
        if type(self.field) is StandardControlContext:
            return ObservedStandardField(self.field)
        return ObservedChronologicalField(self.field, self.now)

    def for_paid_fit(self):
        return replace(self, paid_fit=True)


@dataclass(frozen=True, slots=True)
class ProjectedControlCost:
    """Legal finishes precede retirements, then distance and elapsed time rank."""

    laps: int
    seconds: float
    finished: bool = True

    @property
    def cost(self):
        """A retirement never supplies a finite completion cost."""
        return self.seconds if self.finished else inf

    @property
    def rank(self):
        return self.finished and self.laps >= 0, self.laps, -self.seconds

    def prepend_lap(self, seconds):
        """Keep an accepted incoming crossing when its suffix cannot finish."""
        if self.laps < 0:
            return self
        return ProjectedControlCost(self.laps + 1, seconds + self.seconds, self.finished)


def observed_control_key(field):
    """Keep physical event priority, dropping obsolete scheduler counters."""
    if type(field) is ObservedStandardField:
        # A lone lockstep car has no external clock whose phase can change
        # running, fuel or a future pit discount.
        return (field.lap, field.intervals_left,
                field.now if len(field.rows) > 1 else None, tuple(field.rows),
                field.pre_stop_order, field.pitting)
    queue = tuple((time, distance, kind, identifier)
                  for time, distance, _, kind, identifier, generation in sorted(field.queue)
                  if identifier in field.pending
                  and field.pending[identifier].generation == generation)
    pending = tuple((key, tuple(value for name, value in vars(row).items()
                               if name != "generation")) for key, row in field.pending.items())
    return (field.now, field.intervals_left, field.updates, tuple(field.order),
            tuple(vars(field.timeline._clock).items()), tuple(field.timeline.states.items()),
            pending, queue, tuple(field.free_paces.items()))


register_forecast_helpers(globals(), (
    "StandardControlContext", "ObservedStandardField", "StrategyControlContext",
    "ObservedChronologicalField", "safety_car_running_times", "_finite",
    "_field_observation", "ProjectedControlCost", "observed_control_key",
))
register_forecast_helpers(vars(ObservedStandardField), (
    "__init__", "fork", "controlled", "running_modifier", "projection_required", "finished",
    "enter", "gap_ahead", "cross",
))
register_forecast_helpers(vars(StrategyControlContext), ("new_field", "for_paid_fit", "__eq__"))
register_forecast_helpers(vars(ProjectedControlCost), ("cost", "rank", "prepend_lap"))
