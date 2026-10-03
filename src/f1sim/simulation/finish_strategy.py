"""Bounded finish-distance checks for elective pit stops.

The ordinary strategy planner prices a stop over its estimated lap horizon.
These helpers separately compare the distance achieved by two deterministic
continuations before committing it:

* staying on the fitted set, using the observed gap only for the first lap;
* stopping now, using an optimistic replacement and clean air thereafter.

Followers use an external projected flag. Leading candidates instead supply
their own crossings to a conditional timed clock, alongside frozen observed
rival forecasts. Each path can therefore change the leading announcement and
flag time. Standard execution shares the final lap; chronological execution
finishes each car at its own crossing after the flag.

After a mean outlap the stop runs at the native lap model's absolute floor,
without further service, weather or traffic constraints. It is an optimistic
distance bound under that model and the frozen rival streams. A veto requires
a feasible retained path with strictly more laps. No strategy policy, physical
fit, reservation or future random draw is made, and no global optimum is claimed.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from math import isfinite
from numbers import Integral, Real

from f1sim.cancellation import cancellation_checkpoint
from f1sim.models import Car, Driver, Tire, TireCompound, Track, Weather
from f1sim.models._native import register_forecast_helpers
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.lap import LapSimulator, minimum_lap_time
from f1sim.simulation.neutralization import safety_car_running_times
from f1sim.simulation.warmup import tire_warmup_seconds, validate_tire_warmup
from f1sim.simulation.weather_schedule import WeatherForecastContext


@dataclass(frozen=True)
class RivalFinishForecast:
    """A frozen rival crossing stream, excluding future elective services."""

    completed_laps: int
    next_crossing_time: float
    running_pace: float
    identifier: str | None = None
    fitting_cost: float = 0.


@dataclass(frozen=True, slots=True)
class SafetyCarFinishCar:
    """One rival's mean first running lap after its expected pit entry."""

    identifier: str
    entry_time: float
    free_running: float
    fitting_cost: float = 0.

    def __post_init__(self):
        if (not isinstance(self.identifier, str) or not self.identifier
                or _valid_nonnegative(self.entry_time) is None
                or _valid_positive(self.free_running) is None
                or _valid_nonnegative(self.fitting_cost) is None):
            raise ValueError("invalid safety-car finish rival")


@dataclass(frozen=True, slots=True)
class SafetyCarFinishBranch:
    """Frozen pit-exit order; ``None`` marks the candidate's own position."""

    rows: tuple[SafetyCarFinishCar | None, ...]

    def __post_init__(self):
        if (type(self.rows) is not tuple or sum(row is None for row in self.rows) != 1
                or any(row is not None and type(row) is not SafetyCarFinishCar
                       for row in self.rows)):
            raise ValueError("invalid safety-car finish order")
        identifiers = [row.identifier for row in self.rows if row is not None]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("duplicate safety-car finish rival")

    def gap_at(self, entry_time):
        index = self.rows.index(None)
        return None if index == 0 else max(0., entry_time - self.rows[index - 1].entry_time)

    def project(self, entry_time, free_running, fitting_cost, modifier):
        """Return candidate and rival crossings after running and fit barriers."""
        observations = [(None, entry_time, free_running) if row is None else
                        (row.identifier, row.entry_time, row.free_running) for row in self.rows]
        running = safety_car_running_times(observations, modifier)
        crossings, previous = {}, None
        for row, (identifier, entry, _) in zip(self.rows, observations):
            cancellation_checkpoint()
            fee = fitting_cost if row is None else row.fitting_cost
            crossing = entry + running[identifier] + fee
            crossing = max(crossing, previous) if previous is not None else crossing
            if not isfinite(crossing):
                raise ValueError("invalid safety-car finish crossing")
            crossings[identifier], previous = crossing, crossing
        return crossings.pop(None), crossings


@dataclass(frozen=True, slots=True)
class SafetyCarFinishField:
    """Independent first-lap field observations for retention and paid service."""

    retained: SafetyCarFinishBranch
    stopped: SafetyCarFinishBranch

    def __post_init__(self):
        if (type(self.retained) is not SafetyCarFinishBranch
                or type(self.stopped) is not SafetyCarFinishBranch):
            raise ValueError("invalid safety-car finish field")
        def identifiers(branch):
            return {row.identifier for row in branch.rows if row is not None}
        if identifiers(self.retained) != identifiers(self.stopped):
            raise ValueError("safety-car finish branches have different rivals")


@dataclass(frozen=True)
class LeadingFinishContext:
    """Observed timed signal and rival crossings for a leading candidate.

    The candidate supplies its own counterfactual crossings. Each rival keeps
    its observed recurring pace after the first expected crossing. ``lockstep``
    retains the standard engine's shared distance and one-update-per-lap weather.
    An optional safety-car field resolves separate first-lap queues for staying
    out and stopping, before those recurring rival streams begin.
    """

    time_limit_seconds: float
    announced: bool = False
    rivals: tuple[RivalFinishForecast, ...] = ()
    forecast_context: WeatherForecastContext | None = None
    lockstep: bool = False
    safety_car_field: SafetyCarFinishField | None = None


class _LeadingProjection:
    """Advance only leading crossings established by this path's history."""

    def __init__(self, context, current_lap, maximum_lap):
        self.context = context
        self.next_lap = current_lap
        self.maximum_lap = maximum_lap
        self.announced = context.announced
        self.flag_time = None
        self.flag_lap = None
        self.updates = 0
        self.crossings = {}
        for lap in range(current_lap, maximum_lap + 1):
            cancellation_checkpoint()
            self.crossings[lap] = min((
                rival.next_crossing_time
                + (lap - rival.completed_laps - 1) * rival.running_pace
                for rival in context.rivals
            ), default=float("inf"))

    def observe_until(self, time, *, candidate_lap=None):
        if candidate_lap is not None:
            self.crossings[candidate_lap] = min(self.crossings[candidate_lap], time)
        while self.flag_time is None and self.next_lap <= self.maximum_lap:
            crossing = self.crossings[self.next_lap]
            if crossing > time:
                break
            if self.announced or self.next_lap == self.maximum_lap:
                self.flag_time, self.flag_lap = crossing, self.next_lap
                break
            self.announced = crossing >= self.context.time_limit_seconds
            self.updates += 1
            self.next_lap += 1


@dataclass(frozen=True)
class ReplacementOption:
    """A possible replacement tyre for a finish-distance upper bound.

    ``identifier`` is informational.  Finite inventories can pass immutable
    records through :func:`replacement_options`; the helper never mutates
    those records or the inventory that owns them.
    """

    compound: TireCompound
    age: int = 0
    identifier: str | None = None


@dataclass(frozen=True)
class FinishProtectionResult:
    """Pure result of a retained-versus-stop finish-distance comparison."""

    retained_laps: int | None
    stop_laps: int | None
    retained_crossing_time: float | None
    stop_crossing_time: float | None
    stop_compound: TireCompound | None
    retained_feasible: bool
    stop_feasible: bool
    veto: bool
    reason: str | None = None

def _invalid_result(reason: str) -> FinishProtectionResult:
    return FinishProtectionResult(
        retained_laps=None,
        stop_laps=None,
        retained_crossing_time=None,
        stop_crossing_time=None,
        stop_compound=None,
        retained_feasible=False,
        stop_feasible=False,
        veto=False,
        reason=reason,
    )


def replacement_options(
    values: Iterable[ReplacementOption] | None,
) -> tuple[ReplacementOption, ...]:
    """Normalize immutable replacement records for a forecast.

    ``None`` intentionally means all configured fresh compounds.  Passing an
    iterable means a finite inventory view; no caller-owned object is changed.
    """
    if values is None:
        return tuple(ReplacementOption(compound) for compound in TIRE_COMPOUNDS)
    result: list[ReplacementOption] = []
    for option in values:
        if isinstance(option, ReplacementOption) and option.age >= 0:
            result.append(option)
    return tuple(result)


def _copy_driver(driver: Driver) -> Driver:
    """Clone a driver without relying on or mutating live race state."""
    return driver.model_copy(deep=True)


def _copy_tire(tire: Tire) -> Tire:
    return tire.model_copy(deep=True)


def _valid_nonnegative(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not isfinite(value) or value < 0:
        return None
    return value


def _valid_positive(value) -> float | None:
    value = _valid_nonnegative(value)
    return value if value is not None and value > 0 else None


def _surface_copy(surface) -> Weather | None:
    if surface is None:
        return None
    return surface.model_copy(deep=True)


def evaluate_finish_protection(
    driver: Driver,
    car: Car,
    track: Track,
    current_tire: Tire,
    tire_age: int,
    current_lap: int,
    weather: Weather,
    now: float,
    projected_flag_time: float | None,
    max_scheduled_lap: int,
    *,
    lap_simulator: LapSimulator | None = None,
    physical_total_laps: int | None = None,
    expected_lane_loss: float = 0.0,
    expected_service_time: float = 0.0,
    expected_queue_delay: float = 0.0,
    current_lap_time_modifier: float = 1.0,
    observed_gap: float | None = None,
    current_overtake_mode_active: bool = False,
    replacements: Iterable[ReplacementOption] | None = None,
    projected_surface_at: Callable[[float], Weather] | None = None,
    tire_warmup=None,
    current_fit_pending: bool = False,
    leading_finish_context: LeadingFinishContext | None = None,
    active_aero_enabled: bool = True,
) -> FinishProtectionResult:
    """Compare retained and optimistic-stop distances through the finish.

    ``projected_surface_at`` is a pure callback over absolute track-entry
    times.  It lets the chronological engine use its frozen leading weather
    clock, including an expected pit exit, without evolving live weather.
    When omitted, the supplied snapshot is held constant for direct callers.

    The returned lap counts include ``current_lap`` and stop's outlap.  A path
    ends at the first crossing at or after ``projected_flag_time`` or at
    ``max_scheduled_lap``. Equal completed distances preserve the native
    decision, including when a terminal crossing equals the projected flag.

    A leading candidate can instead supply ``leading_finish_context``. The
    timed announcement then follows the first leading crossing at expiry and
    the flag follows the next leading crossing, for each candidate path. Rival
    streams can supply those crossings while the candidate is in service.

    The first retained lap and mean outlap use the supplied current control
    modifier and Active Aero state. Service and fitting costs remain outside
    the running modifier; subsequent laps retain the green forecast. Callers
    can supply complete standard safety-car fields to resolve each branch's
    first-lap running and fitting barriers. Other neutralized fields whose
    constraints change the conditional crossing streams must be excluded.
    """
    if not isinstance(current_lap, Integral) or isinstance(current_lap, bool):
        return _invalid_result("invalid current lap")
    if not isinstance(max_scheduled_lap, Integral) or isinstance(max_scheduled_lap, bool):
        return _invalid_result("invalid scheduled lap")
    current_lap = int(current_lap)
    max_scheduled_lap = int(max_scheduled_lap)
    if current_lap < 1 or max_scheduled_lap < current_lap:
        return _invalid_result("invalid finish horizon")
    if not isinstance(tire_age, Integral) or isinstance(tire_age, bool) or tire_age < 0:
        return _invalid_result("invalid tire age")
    if _valid_nonnegative(now) is None:
        return _invalid_result("invalid current time")
    flag = _valid_nonnegative(projected_flag_time)
    if flag is None and leading_finish_context is None:
        return _invalid_result("missing projected flag")
    if leading_finish_context is not None:
        context = leading_finish_context
        if (type(context) is not LeadingFinishContext
                or _valid_nonnegative(context.time_limit_seconds) is None
                or type(context.announced) is not bool or type(context.lockstep) is not bool
                or type(context.rivals) is not tuple
                or (context.safety_car_field is not None
                    and (type(context.safety_car_field) is not SafetyCarFinishField
                         or not context.lockstep))
                or (context.forecast_context is not None
                    and type(context.forecast_context) is not WeatherForecastContext)):
            return _invalid_result("invalid leading finish context")
        for rival in context.rivals:
            if (type(rival) is not RivalFinishForecast
                    or type(rival.completed_laps) is not int
                    or not 0 <= rival.completed_laps < current_lap
                    or _valid_nonnegative(rival.next_crossing_time) is None
                    or rival.next_crossing_time < now
                    or _valid_nonnegative(rival.fitting_cost) is None
                    or _valid_positive(rival.running_pace) is None):
                return _invalid_result("invalid rival finish forecast")
        if context.safety_car_field is not None and (
                any(not isinstance(rival.identifier, str) for rival in context.rivals)
                or {row.identifier for row in context.safety_car_field.retained.rows
                 if row is not None} != {rival.identifier for rival in context.rivals}
                or len({rival.identifier for rival in context.rivals}) != len(context.rivals)):
            return _invalid_result("invalid safety-car finish rivals")
    modifier = _valid_positive(current_lap_time_modifier)
    if modifier is None:
        return _invalid_result("invalid current lap modifier")
    if type(active_aero_enabled) is not bool:
        return _invalid_result("invalid active aero state")
    lane = _valid_nonnegative(expected_lane_loss)
    service = _valid_nonnegative(expected_service_time)
    queue = _valid_nonnegative(expected_queue_delay)
    if lane is None or service is None or queue is None:
        return _invalid_result("invalid expected stop delay")
    physical = track.total_laps if physical_total_laps is None else physical_total_laps
    if (not isinstance(physical, Integral) or isinstance(physical, bool)
            or physical < max_scheduled_lap or physical < current_lap):
        return _invalid_result("invalid physical distance")
    if not isinstance(weather, Weather):
        return _invalid_result("invalid current weather")

    try:
        tire_warmup = validate_tire_warmup(tire_warmup)
    except ValueError:
        return _invalid_result("invalid tire warmup profile")
    if type(current_fit_pending) is not bool:
        return _invalid_result("invalid pending tire fit")
    physics = lap_simulator or LapSimulator()
    base_surface = _surface_copy(weather)
    if base_surface is None:
        return _invalid_result("invalid current weather")

    def surface_at(absolute_time: float) -> Weather | None:
        if projected_surface_at is None:
            return _surface_copy(base_surface)
        try:
            return _surface_copy(projected_surface_at(absolute_time))
        except (TypeError, ValueError, AttributeError):
            return None

    def path(
        tire: Tire,
        age: int,
        entry_time: float,
        *,
        first_gap: float | None,
        future_green_floor: bool,
        fitted: bool = False,
    ) -> tuple[int, float] | None:
        """Return (distance, terminal crossing) for one deterministic path."""
        projection_driver = _copy_driver(driver)
        projection_car = car.model_copy(deep=True)
        projection_track = track.model_copy(deep=True)
        tire = _copy_tire(tire)
        entry = entry_time
        projected_age = int(age)
        leading = (_LeadingProjection(leading_finish_context, current_lap, max_scheduled_lap)
                   if leading_finish_context is not None else None)
        field = (leading_finish_context.safety_car_field
                 if leading_finish_context is not None else None)
        branch = (field.stopped if future_green_floor else field.retained) if field else None
        surface = _surface_copy(base_surface)
        surface_updates = 0
        for lap_number in range(current_lap, max_scheduled_lap + 1):
            cancellation_checkpoint()
            if future_green_floor and lap_number > current_lap:
                # This is deliberately an optimistic bound: after the actual
                # outlap, future stop-side laps use the shared absolute green
                # floor directly.  They do not rerun tyre/weather physics or
                # demand that the original replacement remain suitable.
                lap_time = minimum_lap_time(track)
            else:
                if leading is None:
                    surface = surface_at(entry)
                else:
                    if branch is None or lap_number > current_lap:
                        leading.observe_until(entry)
                    updates = (lap_number - current_lap if leading.context.lockstep
                               else leading.updates)
                    try:
                        while surface_updates < updates:
                            cancellation_checkpoint()
                            forecast = leading.context.forecast_context
                            surface = (surface.project_surface() if forecast is None else
                                       forecast.advanced(surface_updates).project_next(surface))
                            surface_updates += 1
                        surface = _surface_copy(surface)
                    except (TypeError, ValueError, AttributeError, OverflowError):
                        return None
                if surface is None or surface.tire_mismatch(tire.compound) == "critical":
                    return None
                projection_driver.current_tire_laps = projected_age
                try:
                    lap_time = physics.calculate_lap_time(
                        projection_driver,
                        projection_car,
                        projection_track,
                        tire,
                        surface,
                        lap_number,
                        int(physical),
                        gap_to_car_ahead=(branch.gap_at(entry) if branch is not None
                                         and not future_green_floor and lap_number == current_lap
                                         else first_gap if lap_number == current_lap else None),
                        active_aero_enabled=(active_aero_enabled
                                             if lap_number == current_lap else True),
                        overtake_mode_active=(current_overtake_mode_active
                                              and not future_green_floor
                                              and lap_number == current_lap),
                        sample_variation=False,
                    )
                except (TypeError, ValueError, OverflowError, AttributeError, KeyError):
                    # Identity-keyed extensions may have no observation for a
                    # copied driver. An unavailable counterfactual cannot veto.
                    return None
                if (isinstance(lap_time, bool) or not isinstance(lap_time, Real)
                        or not isfinite(float(lap_time)) or float(lap_time) <= 0):
                    return None
                lap_time = float(lap_time)
            if lap_number == current_lap:
                fitting = (tire_warmup_seconds(tire_warmup, tire.compound)
                           if tire_warmup and (fitted or current_fit_pending) else 0.)
                if branch is not None:
                    try:
                        first, rivals = branch.project(entry, lap_time, fitting, modifier)
                    except (TypeError, ValueError, OverflowError):
                        return None
                    lap_time = first - entry
                    context = replace(leading_finish_context, rivals=tuple(
                        replace(rival, next_crossing_time=rivals[rival.identifier])
                        for rival in leading_finish_context.rivals))
                    leading = _LeadingProjection(context, current_lap, max_scheduled_lap)
                else:
                    lap_time *= modifier
                    lap_time += fitting
            crossing = entry + lap_time
            if not isfinite(crossing) or crossing < entry:
                return None
            distance = lap_number - current_lap + 1
            if leading is not None:
                leading.observe_until(crossing, candidate_lap=lap_number)
                finished = (leading.flag_lap is not None and lap_number >= leading.flag_lap
                            if leading.context.lockstep else
                            leading.flag_time is not None and crossing >= leading.flag_time)
            else:
                finished = crossing >= flag
            if finished or lap_number == max_scheduled_lap:
                return distance, crossing
            entry = crossing
            projected_age += 1
        return None

    stop_entry = float(now) + lane + service + queue
    if not isfinite(stop_entry):
        return _invalid_result("invalid expected stop exit")
    stop_candidates = replacement_options(replacements)
    best: tuple[int, float, TireCompound] | None = None
    # The options are evaluated independently.  No set is fitted, reserved or
    # aged, and the same immutable replacement can be considered by callers
    # that intentionally pass duplicate records.
    for option in stop_candidates:
        cancellation_checkpoint()
        tire = TIRE_COMPOUNDS[option.compound].model_copy(deep=True)
        candidate = path(
            tire,
            option.age,
            stop_entry,
            first_gap=None,
            future_green_floor=True,
            fitted=True,
        )
        if candidate is None:
            continue
        distance, crossing = candidate
        # Prefer the greatest distance, then earliest terminal crossing.
        if best is None or (distance, -crossing) > (best[0], -best[1]):
            best = (distance, crossing, option.compound)

    horizon = max_scheduled_lap - current_lap + 1
    if best is not None and best[0] >= horizon:
        # Once an optimistic stop reaches the physical scheduled cap, no
        # retained path can complete more distance.  Avoid the extra physics
        # walk and leave the native timing decision untouched.
        return FinishProtectionResult(
            retained_laps=None,
            stop_laps=best[0],
            retained_crossing_time=None,
            stop_crossing_time=best[1],
            stop_compound=best[2],
            retained_feasible=False,  # Not evaluated: it cannot beat the cap.
            stop_feasible=True,
            veto=False,
            reason="optimistic stop reaches scheduled cap",
        )

    retained = path(
        _copy_tire(current_tire), int(tire_age), float(now),
        first_gap=observed_gap, future_green_floor=False,
    )
    if retained is None:
        return FinishProtectionResult(
            retained_laps=None,
            stop_laps=best[0] if best else None,
            retained_crossing_time=None,
            stop_crossing_time=best[1] if best else None,
            stop_compound=best[2] if best else None,
            retained_feasible=False,
            stop_feasible=best is not None,
            veto=False,
            reason="retained forecast infeasible",
        )
    if best is None:
        return FinishProtectionResult(
            retained_laps=retained[0],
            stop_laps=None,
            retained_crossing_time=retained[1],
            stop_crossing_time=None,
            stop_compound=None,
            retained_feasible=True,
            stop_feasible=False,
            veto=False,
            reason="stop forecast infeasible",
        )
    veto = retained[0] > best[0]
    return FinishProtectionResult(
        retained_laps=retained[0],
        stop_laps=best[0],
        retained_crossing_time=retained[1],
        stop_crossing_time=best[1],
        stop_compound=best[2],
        retained_feasible=True,
        stop_feasible=True,
        veto=veto,
        reason="retained distance exceeds optimistic stop bound" if veto else None,
    )


register_forecast_helpers(globals(), (
    "LeadingFinishContext", "RivalFinishForecast", "_LeadingProjection", "replacement_options",
    "evaluate_finish_protection", "_copy_driver", "_copy_tire", "_valid_nonnegative",
    "_valid_positive", "_surface_copy", "safety_car_running_times",
))
register_forecast_helpers(vars(_LeadingProjection), ("__init__", "observe_until"))
register_forecast_helpers(vars(SafetyCarFinishBranch), ("gap_at", "project"))
