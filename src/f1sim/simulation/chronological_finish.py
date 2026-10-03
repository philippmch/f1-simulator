"""Conditional finish distance through a frozen chronological SC/VSC field.

Already committed running and expected unfinished service remain observable.
Observed control duration advances at leading crossings; pending no-passing
restrictions survive until those laps cross. Later green passage is free,
rivals hold observed free pace, and the candidate retains its mean tyre pace
or uses the absolute lap floor after an optimistic paid outlap. No future
policy, atmosphere, incident, service or passing draw is consumed.
"""

import heapq
from copy import deepcopy
from dataclasses import dataclass
from itertools import count
from math import isfinite

from f1sim.cancellation import cancellation_checkpoint
from f1sim.models._native import register_forecast_helpers
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
from f1sim.simulation.race_timing import RaceFinishTimeline
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


def evaluate_chronological_finish_protection(
    driver, car, track, current_tire, tire_age, weather, now, context, *,
    expected_lane_loss=0., expected_service_time=0., expected_queue_delay=0.,
    replacements=None, tire_warmup=None, current_fit_pending=False, lap_simulator=None,
    current_overtake_mode_active=False,
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
        candidate = context.identifier
        timeline = deepcopy(context.timeline)
        order = list(context.order)
        surface = weather.model_copy(deep=True)
        projection_driver = _copy_driver(driver)
        projection_car = car.model_copy(deep=True)
        projection_track = track.model_copy(deep=True)
        tire = _copy_tire(tire)
        free_paces = {context.identifier: context.own_pace,
                      **{row.identifier: row.free_running for row in context.rivals}}
        pending = {row.identifier: _ProjectedLap(
            row.completed_laps + 1, row.ready, row.running_start, row.free_running,
            row.neutralized, row.fitting_cost,
        ) for row in context.rivals}
        serial = count(max((row.event_order for row in context.rivals), default=-1) + 1)
        queue = []
        intervals_left, updates = context.control_intervals, 0
        controlled = intervals_left > 0
        projected_age = age

        def enqueue(identifier, kind):
            row = pending[identifier]
            heapq.heappush(queue, (row.ready, -row.lap, next(serial), kind,
                                   identifier, row.generation))

        for row in context.rivals:
            heapq.heappush(queue, (row.ready, -row.completed_laps - 1, row.event_order,
                                   "cross" if row.running_start is not None else "exit",
                                   row.identifier, 0))

        def gap_ahead(identifier, entry, reference):
            index = order.index(identifier)
            if index == 0:
                return None
            ahead = pending[order[index - 1]]
            duration = ahead.ready - ahead.running_start
            if duration <= 0 or ahead.running_start > entry:
                raise ValueError("invalid chronological predecessor")
            return min(1., (entry - ahead.running_start) / duration) * reference

        def begin(identifier, entry):
            row = pending[identifier]
            modifier = context.modifier if controlled else 1.
            if identifier == candidate:
                if stopped and row.lap > current_lap:
                    free = minimum_lap_time(projection_track)
                else:
                    if surface.tire_mismatch(tire.compound) == "critical":
                        return False
                    projection_driver.current_tire_laps = projected_age
                    # A paid outlap is priced in clean air for its upper bound;
                    # retained traffic is observable at each simulated entry.
                    reference = free_paces[candidate] * modifier
                    gap = None if stopped else gap_ahead(identifier, entry, reference)
                    free = physics.calculate_lap_time(
                        projection_driver, projection_car, projection_track, tire, surface,
                        row.lap, track.total_laps, gap_to_car_ahead=gap,
                        active_aero_enabled=not controlled,
                        overtake_mode_active=(current_overtake_mode_active and not stopped
                                              and row.lap == current_lap),
                        sample_variation=False,
                    )
                if _valid_positive(free) is None:
                    return False
                free_paces[candidate] = free
            else:
                free = free_paces[identifier]
            running = free * modifier
            row.free_running = free
            row.neutralized = controlled
            row.running_start = entry
            if controlled and context.safety_car:
                leader = min(order, key=lambda key: -timeline.states[key].completed_laps)
                if identifier != leader:
                    anchor = pending[leader].free_running
                    nominal = max(free, anchor * modifier)
                    running = safety_car_running_time(
                        free, nominal, gap_ahead(identifier, entry, nominal))
            row.ready = entry + running + row.fitting_cost
            row.fitting_cost = 0.
            if not isfinite(row.ready) or row.ready <= entry:
                return False
            enqueue(identifier, "cross")
            return True

        fit_cost = (tire_warmup_seconds(tire_warmup, tire.compound)
                    if stopped or current_fit_pending else 0.)
        pending[candidate] = _ProjectedLap(current_lap, stop_entry if stopped else now,
                                            None, 0., True, fit_cost)
        if stopped:
            order.remove(candidate)
            enqueue(candidate, "exit")
        elif not begin(candidate, now):
            return None
        while queue:
            cancellation_checkpoint()
            time, _, _, kind, identifier, generation = heapq.heappop(queue)
            row = pending.get(identifier)
            if row is None or row.generation != generation:
                continue
            if kind == "exit":
                order.append(identifier)
                if not begin(identifier, time):
                    return None
                continue
            while order[0] != identifier:
                index = order.index(identifier)
                ahead = pending[order[index - 1]]
                if not (controlled or row.neutralized or ahead.neutralized):
                    order[index - 1], order[index] = identifier, order[index - 1]
                    continue
                ready = max(row.ready, ahead.ready + 1.e-9)
                if not isfinite(ready) or ready <= time:
                    return None
                row.ready, row.generation = ready, row.generation + 1
                enqueue(identifier, "cross")
                break
            else:
                active_distance = max(value.completed_laps for value in timeline.states.values()
                                      if not value.retired and value.finish_time is None)
                leading = timeline.chequered_time is None and row.lap > active_distance
                if leading:
                    intervals_left = max(0, intervals_left - 1)
                    controlled = intervals_left > 0
                crossing = timeline.observe_crossing(identifier, row.lap, time, is_leader=leading)
                del pending[identifier]
                order.remove(identifier)
                if identifier == candidate:
                    if crossing.finish_time is not None:
                        return row.lap - current_lap + 1, time
                    projected_age += 1
                if leading and timeline.chequered_time is None:
                    forecast = context.forecast_context
                    surface = (surface.project_surface() if forecast is None else
                               forecast.advanced(updates).project_next(surface))
                    updates += 1
                if crossing.finish_time is None:
                    order.append(identifier)
                    pending[identifier] = _ProjectedLap(row.lap + 1, time, None,
                                                        row.free_running, False)
                    if not begin(identifier, time):
                        return None
        return None

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


register_forecast_helpers(globals(), (
    "ChronologicalFinishCar", "ChronologicalFinishContext",
    "evaluate_chronological_finish_protection",
    "_validate_context", "_copy_driver", "_copy_tire", "_valid_positive", "_valid_nonnegative",
    "replacement_options", "safety_car_running_time", "minimum_lap_time",
))
