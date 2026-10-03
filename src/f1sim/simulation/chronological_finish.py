"""Conditional finish distance through a frozen chronological SC/VSC field.

Already committed running and expected unfinished service remain observable.
Observed control duration advances at leading crossings; pending no-passing
restrictions survive until those laps cross. Later green passage is free,
rivals hold observed free pace, and the candidate retains its mean tyre pace
or uses the absolute lap floor after an optimistic paid outlap. No future
policy, atmosphere, incident, service or passing draw is consumed.
"""

import heapq
from copy import copy, deepcopy
from dataclasses import dataclass, replace
from math import isfinite

from f1sim.cancellation import cancellation_checkpoint
from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models._native import register_forecast_helpers, register_forecast_values
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.finish_strategy import (
    FinishProtectionResult,
    _copy_driver,
    _copy_tire,
    _invalid_result,
    _valid_nonnegative,
    _valid_positive,
    replacement_options,
)
from f1sim.simulation.lap import LapSimulator, minimum_lap_time
from f1sim.simulation.neutralization import safety_car_running_time
from f1sim.simulation.race_timing import DriverFinishState, RaceFinishClock, RaceFinishTimeline
from f1sim.simulation.warmup import tire_warmup_seconds, validate_tire_warmup
from f1sim.simulation.weather_schedule import WeatherForecastContext


@dataclass(frozen=True, slots=True)
class ChronologicalFinishCar:
    """One rival's committed running or expected service and recurring pace."""

    identifier: str
    completed_laps: int
    free_running: float
    ready: float
    running_start: float | None
    neutralized: bool
    event_order: int
    fitting_cost: float = 0.

    def __post_init__(self):
        if (not isinstance(self.identifier, str) or not self.identifier
                or type(self.completed_laps) is not int or self.completed_laps < 0
                or _valid_positive(self.free_running) is None
                or _valid_nonnegative(self.ready) is None
                or type(self.neutralized) is not bool
                or type(self.event_order) is not int or self.event_order < 0
                or _valid_nonnegative(self.fitting_cost) is None
                or (self.running_start is not None
                    and (_valid_nonnegative(self.running_start) is None
                         or self.ready <= self.running_start))):
            raise ValueError("invalid chronological finish rival")


@dataclass(frozen=True)
class ChronologicalFinishContext:
    """Independent ledger snapshot and circular on-track order at a decision.

    A rival with no running start is awaiting an expected pit exit at ``ready``.
    Every path owns a deep copy of this ledger, so the supplied snapshot is
    never advanced. The candidate has just crossed and has no pending lap.
    """

    identifier: str
    timeline: RaceFinishTimeline
    order: tuple[str, ...]
    rivals: tuple[ChronologicalFinishCar, ...]
    own_pace: float
    modifier: float
    safety_car: bool
    forecast_context: WeatherForecastContext | None = None
    control_intervals: int = 1


@dataclass
class _ProjectedLap:
    lap: int
    ready: float
    running_start: float | None
    free_running: float
    neutralized: bool
    fitting_cost: float = 0.
    generation: int = 0


def _validate_context(context, now, scheduled_laps):
    if (type(context) is not ChronologicalFinishContext
            or not isinstance(context.identifier, str) or not context.identifier
            or type(context.timeline) is not RaceFinishTimeline
            or type(context.order) is not tuple or type(context.rivals) is not tuple
            or any(type(row) is not ChronologicalFinishCar for row in context.rivals)
            or _valid_positive(context.own_pace) is None
            or _valid_positive(context.modifier) is None
            or type(context.safety_car) is not bool
            or type(context.control_intervals) is not int or context.control_intervals < 0
            or (context.forecast_context is not None
                and type(context.forecast_context) is not WeatherForecastContext)):
        return False
    identifiers = [row.identifier for row in context.rivals]
    if (context.identifier in identifiers or len(set(identifiers)) != len(identifiers)
            or any(not isinstance(key, str) for key in context.order)
            or len(set(context.order)) != len(context.order)
            or set(context.order) != {context.identifier, *(
                row.identifier for row in context.rivals if row.running_start is not None)}
            or len({row.event_order for row in context.rivals}) != len(context.rivals)):
        return False
    ledger = context.timeline.states
    active = {key for key, row in ledger.items() if not row.retired and row.finish_time is None}
    last_observation = context.timeline._last_observation_time
    if (active != {context.identifier, *identifiers}
            or context.timeline._clock.scheduled_laps != scheduled_laps
            or context.timeline._clock._suspension_start is not None
            or last_observation is not None and last_observation > now
            or any(row.last_crossing_time is not None and row.last_crossing_time > now
                   for row in ledger.values())
            or any(row.completed_laps != ledger[row.identifier].completed_laps
                   or row.ready < now
                   or row.running_start is not None and row.running_start > now
                   for row in context.rivals)):
        return False
    return ledger[context.identifier].completed_laps < scheduled_laps


@dataclass(frozen=True, slots=True)
class ObservedFieldCrossing:
    """One private crossing, including the timed flag and leading update."""

    identifier: str
    time: float
    leading: bool
    flag_time: float | None


_FINISH_FIELDS = frozenset(DriverFinishState.__dataclass_fields__)
_TIMELINE_FIELDS = frozenset((
    "_clock", "_states", "_last_observation_time", "_leader_id", "winner_id",
))
_CLOCK_FIELDS = frozenset((
    "scheduled_laps", "final_lap", "completed_laps", "last_crossing_time", "winner_time",
    "_time_limit_announced", "_last_observation_time", "_suspension_start",
    "_total_suspension_seconds",
))
_SCALAR_TYPES = (int, float, str, bool, type(None))


def _copy_timeline(timeline):
    # Native driver observations are frozen records of scalar values. Share
    # those values while copying every mutable ledger/clock container. Any
    # non-native records or additional attributes retain ordinary deep copying.
    immutable = {id(row): row for row in timeline.states.values()
                 if type(row) is DriverFinishState and vars(row).keys() == _FINISH_FIELDS
                 and all(type(value) in _SCALAR_TYPES for value in vars(row).values())}
    clock = timeline._clock
    if (type(timeline) is RaceFinishTimeline and type(clock) is RaceFinishClock
            and vars(timeline).keys() == _TIMELINE_FIELDS
            and vars(clock).keys() == _CLOCK_FIELDS
            and len(immutable) == len(timeline.states)
            and all(type(value) in _SCALAR_TYPES for value in vars(clock).values())
            and all(type(value) in _SCALAR_TYPES for name, value in vars(timeline).items()
                    if name not in ("_clock", "_states"))):
        branch = copy(timeline)
        branch._clock = copy(clock)
        branch._states = dict(timeline.states)
        return branch
    return deepcopy(timeline, immutable)


class ObservedChronologicalField:
    """Branchable observed field at the candidate's own lap boundary.

    Rivals keep their observed free pace after committed running or expected
    service. The caller supplies candidate mean pace at each actual track
    entry. Leading crossings consume known control intervals; neutralized
    pending laps retain their no-passing restriction after control ends.
    Fitting penalties follow running, so later SC laps can recover that gap.
    Every branch owns its ledger, pending events and physical order.
    """

    def __init__(self, context, now):
        scheduled = context.timeline._clock.scheduled_laps
        if (_valid_nonnegative(now) is None
                or not _validate_context(context, now, scheduled)):
            raise ValueError("invalid chronological field context")
        self.identifier = context.identifier
        self.timeline = _copy_timeline(context.timeline)
        self.active_distance = max(row.completed_laps for row in self.timeline.states.values()
                                   if not row.retired and row.finish_time is None)
        self.order = list(context.order)
        self.modifier = context.modifier
        self.safety_car = context.safety_car
        self.intervals_left = context.control_intervals
        self.now = now
        self.updates = 0
        self.events = []
        self.free_paces = {context.identifier: context.own_pace,
                           **{row.identifier: row.free_running for row in context.rivals}}
        self.pending = {row.identifier: _ProjectedLap(
            row.completed_laps + 1, row.ready, row.running_start, row.free_running,
            row.neutralized, row.fitting_cost,
        ) for row in context.rivals}
        self.serial = max((row.event_order for row in context.rivals), default=-1) + 1
        self.queue = [(row.ready, -row.completed_laps - 1, row.event_order,
                       "cross" if row.running_start is not None else "exit", row.identifier, 0)
                      for row in context.rivals]
        heapq.heapify(self.queue)
        self.entered = False

    def fork(self):
        """Copy only mutable projection state; no engine or model is retained."""
        branch = copy(self)
        branch.timeline = _copy_timeline(self.timeline)
        branch.order = self.order.copy()
        branch.free_paces = self.free_paces.copy()
        branch.pending = {key: replace(row) for key, row in self.pending.items()}
        branch.queue = self.queue.copy()
        branch.events = self.events.copy()
        return branch

    @property
    def controlled(self):
        return self.intervals_left > 0

    @property
    def running_modifier(self):
        return self.modifier if self.controlled else 1.

    @property
    def projection_required(self):
        return self.controlled or any(row.neutralized for row in self.pending.values())

    @property
    def finished(self):
        return self.timeline.states[self.identifier].finish_time is not None

    def _enqueue(self, identifier, kind):
        row = self.pending[identifier]
        heapq.heappush(self.queue, (row.ready, -row.lap, self.serial, kind,
                                   identifier, row.generation))
        self.serial += 1

    def _gap_ahead(self, identifier, entry, reference):
        index = self.order.index(identifier)
        if index == 0:
            return None
        ahead = self.pending[self.order[index - 1]]
        duration = ahead.ready - ahead.running_start
        if duration <= 0 or ahead.running_start > entry:
            raise ValueError("invalid chronological predecessor")
        return min(1., (entry - ahead.running_start) / duration) * reference

    def gap_ahead(self, reference):
        if not self.entered:
            raise ValueError("candidate must enter before observing its predecessor")
        return self._gap_ahead(self.identifier, self.now, reference)

    def _begin(self, identifier, entry, free_running, fitting_cost=0.):
        row = self.pending[identifier]
        row.free_running = free_running
        row.neutralized = self.controlled
        row.running_start = entry
        running = free_running * self.running_modifier
        self.free_paces[identifier] = free_running
        if self.controlled and self.safety_car:
            ledger = self.timeline.states
            leader = min(self.order, key=lambda key: -ledger[key].completed_laps)
            if identifier != leader:
                nominal = max(free_running, self.pending[leader].free_running * self.modifier)
                running = safety_car_running_time(
                    free_running, nominal, self._gap_ahead(identifier, entry, nominal))
        row.ready = entry + running + fitting_cost
        row.fitting_cost = 0.
        if not isfinite(row.ready) or row.ready <= entry:
            raise ValueError("invalid projected running crossing")
        self._enqueue(identifier, "cross")

    def _advance(self, target):
        while self.queue:
            cancellation_checkpoint()
            time, _, _, kind, identifier, generation = heapq.heappop(self.queue)
            row = self.pending.get(identifier)
            if row is None or row.generation != generation:
                continue
            if kind == "exit":
                self.order.append(identifier)
                if identifier == self.identifier:
                    self.now = time
                    return
                self._begin(identifier, time, self.free_paces[identifier], row.fitting_cost)
                continue
            while self.order[0] != identifier:
                index = self.order.index(identifier)
                ahead = self.pending[self.order[index - 1]]
                if not (self.controlled or row.neutralized or ahead.neutralized):
                    self.order[index - 1], self.order[index] = identifier, self.order[index - 1]
                    continue
                ready = max(row.ready, ahead.ready + 1.e-9)
                if not isfinite(ready) or ready <= time:
                    raise ValueError("invalid projected no-passing crossing")
                row.ready, row.generation = ready, row.generation + 1
                self._enqueue(identifier, "cross")
                break
            else:
                leading = (self.timeline.chequered_time is None
                           and row.lap > self.active_distance)
                if leading:
                    # This held field has no future retirement. Before the
                    # flag, only a leading crossing can increase its distance.
                    self.active_distance = row.lap
                    self.intervals_left = max(0, self.intervals_left - 1)
                crossing = self.timeline.observe_crossing(identifier, row.lap, time,
                                                         is_leader=leading)
                event = ObservedFieldCrossing(identifier, time, leading,
                                             self.timeline.chequered_time)
                self.events.append(event)
                if leading and self.timeline.chequered_time is None:
                    self.updates += 1
                del self.pending[identifier]
                self.order.remove(identifier)
                if crossing.finish_time is None:
                    self.order.append(identifier)
                if identifier == target:
                    self.now = time
                    return event
                if crossing.finish_time is None:
                    self.pending[identifier] = _ProjectedLap(row.lap + 1, time, None,
                                                             row.free_running, False)
                    self._begin(identifier, time, row.free_running)
        raise ValueError("candidate has no projected event")

    def enter(self, stop_delay=None):
        """Advance expected service, then expose conditions at actual entry.

        A zero-delay paid stop still removes and rejoins the candidate when
        ``stop_delay`` is explicitly supplied. ``None`` retains its position.
        No fitting penalty is consumed before track entry.
        """
        if self.entered or self.finished:
            raise ValueError("candidate cannot enter this projected lap")
        stopped = stop_delay is not None
        if stopped and _valid_nonnegative(stop_delay) is None:
            raise ValueError("invalid expected stop delay")
        ready = self.now + (stop_delay if stopped else 0.)
        if not isfinite(ready):
            raise ValueError("invalid expected stop exit")
        lap = self.timeline.states[self.identifier].completed_laps + 1
        self.pending[self.identifier] = _ProjectedLap(lap, ready, None, 0., True)
        if stopped:
            self.order.remove(self.identifier)
            self._enqueue(self.identifier, "exit")
            self._advance(self.identifier)
        self.entered = True
        return self.now

    def cross(self, free_running, fitting_cost=0.):
        """Run and resolve the candidate's next crossing on this branch."""
        if not self.entered:
            raise ValueError("candidate must enter before running")
        if (_valid_positive(free_running) is None
                or _valid_nonnegative(fitting_cost) is None):
            raise ValueError("invalid candidate running pace or fitting cost")
        self._begin(self.identifier, self.now, free_running, fitting_cost)
        crossing = self._advance(self.identifier)
        self.entered = False
        return crossing


def evaluate_chronological_finish_protection(
    driver, car, track, current_tire, tire_age, weather, now, context, *,
    expected_lane_loss=0., expected_service_time=0., expected_queue_delay=0.,
    replacements=None, tire_warmup=None, current_fit_pending=False, lap_simulator=None,
    current_overtake_mode_active=False,
    _clock_observer=None,
):
    """Compare complete field clocks for retention and an optimistic stop.

    The first paid outlap uses conditions at its expected entry, including
    leading weather/control changes during service. Subsequent stop-side laps
    use the shared absolute floor. Mean retained laps remain unsuitable when
    their tyre becomes critical. Physical neutralized crossing barriers apply
    to both branches and never modify the supplied ledger, models or clocks.
    """
    try:
        valid_context = (type(track.total_laps) is int and track.total_laps > 0
                         and _validate_context(context, now, track.total_laps))
    except (TypeError, ValueError, OverflowError, AttributeError, KeyError):
        valid_context = False
    if (_valid_nonnegative(now) is None or type(tire_age) is not int or tire_age < 0
            or type(current_fit_pending) is not bool
            or type(current_overtake_mode_active) is not bool or not valid_context):
        return _invalid_result("invalid chronological finish context")
    delays = (expected_lane_loss, expected_service_time, expected_queue_delay)
    if any(_valid_nonnegative(value) is None for value in delays):
        return _invalid_result("invalid expected stop delay")
    stop_entry = now + expected_lane_loss + expected_service_time + expected_queue_delay
    if not isfinite(stop_entry):
        return _invalid_result("invalid expected stop exit")
    try:
        tire_warmup = validate_tire_warmup(tire_warmup)
    except ValueError:
        return _invalid_result("invalid tire warmup profile")
    physics = lap_simulator or LapSimulator()
    current_lap = context.timeline.states[context.identifier].completed_laps + 1

    def path(tire, age, *, stopped):
        field = ObservedChronologicalField(context, now)
        surface = weather.model_copy(deep=True)
        projection_driver = _copy_driver(driver)
        projection_car = car.model_copy(deep=True)
        projection_track = track.model_copy(deep=True)
        tire = _copy_tire(tire)
        updates, observed = 0, 0

        def observe():
            nonlocal observed
            if _clock_observer is not None and not stopped:
                for event in field.events[observed:]:
                    _clock_observer(event.identifier, event.time, event.leading, event.flag_time)
            observed = len(field.events)

        while not field.finished:
            cancellation_checkpoint()
            lap = field.timeline.states[field.identifier].completed_laps + 1
            first = lap == current_lap
            field.enter(stop_entry - now if stopped and first else None)
            observe()
            while updates < field.updates:
                forecast = context.forecast_context
                surface = (surface.project_surface() if forecast is None else
                           forecast.advanced(updates).project_next(surface))
                updates += 1
            if stopped and not first:
                free = minimum_lap_time(projection_track)
            else:
                if surface.tire_mismatch(tire.compound) == "critical":
                    return None
                projection_driver.current_tire_laps = age + lap - current_lap
                # A paid outlap is priced in clean air for its upper bound;
                # retained traffic is observable at each simulated entry.
                reference = field.free_paces[field.identifier] * field.running_modifier
                gap = None if stopped else field.gap_ahead(reference)
                free = physics.calculate_lap_time(
                    projection_driver, projection_car, projection_track, tire, surface,
                    lap, track.total_laps, gap_to_car_ahead=gap,
                    active_aero_enabled=not field.controlled,
                    overtake_mode_active=(current_overtake_mode_active and not stopped and first),
                    sample_variation=False,
                )
            fit_cost = (tire_warmup_seconds(tire_warmup, tire.compound)
                        if first and (stopped or current_fit_pending) else 0.)
            field.cross(free, fit_cost)
            observe()
        return field.timeline.states[field.identifier].completed_laps - current_lap + 1, field.now

    def project(tire, age, *, stopped):
        try:
            return path(tire, age, stopped=stopped)
        except (TypeError, ValueError, OverflowError, AttributeError, KeyError):
            return None

    best = None
    for option in replacement_options(replacements):
        cancellation_checkpoint()
        result = project(TIRE_COMPOUNDS[option.compound], option.age, stopped=True)
        if result is not None and (best is None or (result[0], -result[1]) > (best[0], -best[1])):
            best = (*result, option.compound)
    horizon = track.total_laps - current_lap + 1
    if best is not None and best[0] == horizon:
        return FinishProtectionResult(None, best[0], None, best[1], best[2], False, True,
                                      False, "optimistic stop reaches scheduled cap")
    retained = project(current_tire, tire_age, stopped=False)
    veto = retained is not None and best is not None and retained[0] > best[0]
    reason = ("retained forecast infeasible" if retained is None else
              "stop forecast infeasible" if best is None else
              "retained distance exceeds optimistic stop bound" if veto else None)
    return FinishProtectionResult(
        None if retained is None else retained[0], None if best is None else best[0],
        None if retained is None else retained[1], None if best is None else best[1],
        None if best is None else best[2], retained is not None, best is not None, veto,
        reason,
    )


@dataclass(frozen=True, slots=True)
class ObservedChronologicalClock:
    """Held-pace field crossings and weather events from a private ledger."""

    identifier: str
    flag_time: float
    own_crossings: tuple[float, ...]
    leading_updates: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _ObservedRunningPace:
    """One held observation, without creating a new class for every forecast."""

    pace: float

    def calculate_lap_time(self, *args, **kwargs):
        return self.pace


def project_observed_chronological_clock(context, now, *, own_fitting_cost=0.):
    """Project the observed field without future policies or any physics draw.

    The finish-distance projector owns the queue, service, control and ledger
    laws. Reusing it here keeps strategy distance and weather on those same
    crossings. A constant pace adapter holds only the candidate's latest free
    running observation; rivals already hold theirs in that projector.
    """
    if _valid_nonnegative(own_fitting_cost) is None:
        return None
    try:
        scheduled = context.timeline._clock.scheduled_laps
        identifier = context.identifier
    except AttributeError:
        return None
    try:
        valid = (type(scheduled) is int and scheduled > 0
                 and _valid_nonnegative(now) is not None
                 and _validate_context(context, now, scheduled))
    except (TypeError, ValueError, OverflowError, AttributeError, KeyError):
        valid = False
    if not valid:
        return None
    own, updates, flags = [], [], []

    def observe(key, time, leading, flag):
        if key == identifier:
            own.append(time)
        if leading and flag is None:
            updates.append(time)
        if flag is not None:
            flags.append(flag)

    result = evaluate_chronological_finish_protection(
        Driver(id=identifier, name=identifier, team_id="clock"),
        Car(team_id="clock", team_name="clock"),
        Track(id="clock", name="clock", country="clock", total_laps=scheduled,
              base_lap_time=context.own_pace),
        TIRE_COMPOUNDS[TireCompound.MEDIUM],
        0, Weather(), now, replace(context, forecast_context=None), replacements=(),
        lap_simulator=_ObservedRunningPace(context.own_pace),
        current_fit_pending=own_fitting_cost > 0., tire_warmup={"medium": own_fitting_cost},
        _clock_observer=observe,
    )
    if not result.retained_feasible or not own or not flags:
        return None
    return ObservedChronologicalClock(identifier, flags[0], tuple(own), tuple(updates))


register_forecast_helpers(globals(), (
    "ChronologicalFinishCar", "ChronologicalFinishContext",
    "evaluate_chronological_finish_protection",
    "_validate_context", "_copy_driver", "_copy_tire", "_valid_positive", "_valid_nonnegative",
    "replacement_options", "safety_car_running_time", "minimum_lap_time",
    "project_observed_chronological_clock", "ObservedChronologicalClock", "_ObservedRunningPace",
    "ObservedChronologicalField", "ObservedFieldCrossing", "_copy_timeline", "DriverFinishState",
    "RaceFinishTimeline", "RaceFinishClock",
))
register_forecast_values(globals(), (
    "_FINISH_FIELDS", "_TIMELINE_FIELDS", "_CLOCK_FIELDS", "_SCALAR_TYPES",
))
register_forecast_helpers(vars(_ObservedRunningPace), ("calculate_lap_time",))
register_forecast_helpers(vars(ObservedChronologicalField), (
    "__init__", "fork", "controlled", "running_modifier", "projection_required", "finished",
    "_enqueue", "_gap_ahead", "gap_ahead", "_begin", "_advance", "enter", "cross",
))
