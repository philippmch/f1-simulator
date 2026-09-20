"""Bounded finish-distance checks for chronological elective pit stops.

The chronological engine's ordinary strategy planner answers a local question:
whether a stop pays for itself under its configured policy horizon.  The
helpers in this module answer a narrower safety question before that stop is
committed.  They compare two direct, deterministic distance forecasts under
the same projected leading flag:

* staying on the fitted set, using the observed gap only for the first lap;
* stopping now, using an optimistic replacement and clean air thereafter.

The stop forecast is intentionally an upper bound on what a stop can achieve.
It does not run a strategy policy, reserve an inventory set, sample service or
traffic, or claim a global optimum.  A veto is therefore made only when the
feasible retained forecast completes strictly more laps than that optimistic
bound.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real

from f1sim.models import Car, Driver, Tire, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.lap import LapSimulator, minimum_lap_time


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
    projected_flag_time: float,
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
    if flag is None:
        return _invalid_result("missing projected flag")
    modifier = _valid_positive(current_lap_time_modifier)
    if modifier is None:
        return _invalid_result("invalid current lap modifier")
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

    projection_driver = _copy_driver(driver)

    def path(
        tire: Tire,
        age: int,
        entry_time: float,
        *,
        first_gap: float | None,
        future_green_floor: bool,
    ) -> tuple[int, float] | None:
        """Return (distance, terminal crossing) for one deterministic path."""
        entry = entry_time
        projected_age = int(age)
        for lap_number in range(current_lap, max_scheduled_lap + 1):
            if future_green_floor and lap_number > current_lap:
                # This is deliberately an optimistic bound: after the actual
                # outlap, future stop-side laps use the shared absolute green
                # floor directly.  They do not rerun tyre/weather physics or
                # demand that the original replacement remain suitable.
                lap_time = minimum_lap_time(track)
            else:
                surface = surface_at(entry)
                if surface is None or surface.tire_mismatch(tire.compound) == "critical":
                    return None
                projection_driver.current_tire_laps = projected_age
                try:
                    lap_time = physics.calculate_lap_time(
                        projection_driver,
                        car,
                        track,
                        tire,
                        surface,
                        lap_number,
                        int(physical),
                        gap_to_car_ahead=(first_gap if lap_number == current_lap else None),
                        active_aero_enabled=True,
                        overtake_mode_active=(current_overtake_mode_active
                                              and not future_green_floor
                                              and lap_number == current_lap),
                        sample_variation=False,
                    )
                except (TypeError, ValueError, OverflowError, AttributeError):
                    return None
                if (isinstance(lap_time, bool) or not isinstance(lap_time, Real)
                        or not isfinite(float(lap_time)) or float(lap_time) <= 0):
                    return None
                lap_time = float(lap_time)
            if lap_number == current_lap:
                lap_time *= modifier
            crossing = entry + lap_time
            if not isfinite(crossing) or crossing < entry:
                return None
            distance = lap_number - current_lap + 1
            if crossing >= flag or lap_number == max_scheduled_lap:
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
        tire = TIRE_COMPOUNDS[option.compound].model_copy(deep=True)
        candidate = path(
            tire,
            option.age,
            stop_entry,
            first_gap=None,
            future_green_floor=True,
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
