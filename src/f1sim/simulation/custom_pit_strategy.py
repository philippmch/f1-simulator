"""Replacement forecasts that retain a driver's remaining custom pit schedule.

Only compulsory choices are optimized. Requested services keep their lap and
compound, including a deliberately slow request or an unavailable request that
execution would skip. The caller supplies the current estimated finish horizon;
fuel always uses the original scheduled distance.
"""

from dataclasses import dataclass
from functools import lru_cache
from math import inf, isfinite

import numpy as np

from f1sim.cancellation import cancellation_checkpoint
from f1sim.models._native import forecast_decision, native_physics, register_forecast_helpers
from f1sim.models.tire import TIRE_COMPOUNDS, TireCompound
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.strategy_lap import isolated_strategy_lap
from f1sim.simulation.strategy_neutralization import current_fitted_time, current_running_time
from f1sim.simulation.strategy_traffic import normalize_current_traffic_gaps
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.surface_projection import normalize_weather_intervals, projected_surfaces
from f1sim.simulation.tire_inventory import tire_slot_usable
from f1sim.simulation.warmup import tire_warmup_seconds, validate_tire_warmup
from f1sim.simulation.weather_schedule import ScheduledWeatherIntervals, project_next_surface


@dataclass(frozen=True)
class CustomPitFinishContext:
    """Observed leader clock, used without announcing the live race's finish."""

    now: float
    time_limit_seconds: float
    announced: bool = False


@dataclass(frozen=True)
class CustomPitChoice:
    cost: float
    compound: TireCompound | None = None
    set_id: str | None = None
    laps: int = 0
    instructions: int = 0
    partial_time: float = inf


@forecast_decision
def choose_custom_pit_replacement(
    driver, car, track, weather, current_tire, tire_age, current_lap, pit_plan, *,
    pit_plan_index=0, inventory=None, used_compounds=(), free_fit=False,
    current_fit_pending=False, pit_lane_factor=1., current_lap_time_modifier=1.,
    active_aero_enabled=True, physical_total_laps=None, weather_intervals=None,
    weather_clock=None, additional_current_stop_cost=0., tire_warmup=None,
    current_traffic_gaps=None, forecast_context=None, finish_context=None,
    safety_car=None,
):
    """Price a committed paid replacement or a free restart fit without mutation.

    No elective automatic stops are inserted, even for an empty plan. Future
    weather repairs and the final compound correction are still compulsory.
    Finite requests select the least worn available physical set, with pool
    order breaking ties, exactly as execution does. A free retained set keeps
    its wear and pending fitting penalty; an unrun fit earns no compound credit.

    Costs use deterministic physics and expected service. Observed stay/rejoin
    gaps enter the first running lap before clipping and control scaling; later
    laps assume clean air and green. Only a paid service on the current lap
    receives its expected queue cost, including a request after a free fit. External
    weather updates include physical stop delays and previously run warmup fees.
    An observed leader clock can shorten each branch at its following crossing
    after expiry. Completed distance is ranked first, then fulfilled requests,
    then time. A cheap choice cannot win solely by making a later safe request
    unavailable when an equally long continuation can honor it. Other cars retain
    the caller's estimated own-lap horizon. No live finish signal is changed.
    Legal finishes take priority. If every continuation retires, rank accepted
    distance and time at the last crossing, without credit for fulfilled requests.
    Failed completion costs remain infinite; ``partial_time`` retains the finite
    retirement trace. No choice is returned when no candidate can complete a lap.
    """
    horizon = track.total_laps - current_lap + 1
    if horizon < 1:
        return CustomPitChoice(inf)
    if pit_plan is None:
        raise ValueError("custom replacement forecast requires a pit plan")
    intervals = normalize_weather_intervals(
        horizon, weather_intervals, weather=weather, forecast_context=forecast_context,
    )
    if isinstance(intervals, ScheduledWeatherIntervals):
        forecast_context = intervals.context
    if weather_clock is not None:
        if not isinstance(weather_clock, StrategyWeatherClock):
            raise ValueError("weather_clock must be a StrategyWeatherClock")
        weather_clock.validate_horizon(horizon)
    physical = track.total_laps if physical_total_laps is None else physical_total_laps
    gaps = normalize_current_traffic_gaps(current_traffic_gaps)
    if safety_car is not None:
        gaps = safety_car.traffic_gaps
    warmup = validate_tire_warmup(tire_warmup)
    driver = driver.model_copy(deep=True)
    car = car.model_copy(deep=True)
    simulator = LapSimulator(np.random.default_rng(0))
    prepared = simulator.prepare_deterministic_lap_time(driver, car, track, physical)
    interchangeable = prepared is not None and native_physics(driver, car, track, weather)
    current_stop = (track.pit_lane_delta * pit_lane_factor
                    + expected_stationary_time(car) + additional_current_stop_cost)
    green_stop = track.pit_lane_delta + expected_stationary_time(car)
    requests = {item["lap"]: TireCompound(item["compound"])
                for item in pit_plan[pit_plan_index:]
                if current_lap <= item["lap"] <= track.total_laps}
    if not free_fit:
        # This committed service resolves any overridden request on this lap.
        requests.pop(current_lap, None)
    finite = inventory is not None
    if finite:
        records = inventory.snapshot(tire_age)
        compounds = tuple(TireCompound(item["compound"]) for item in records)
        identifiers = tuple(item["id"] for item in records)
        ages = tuple(item["age"] for item in records)
        expiries = tuple(-1 if item.get("remaining_laps") is None else
                         item["age"] + item["remaining_laps"] for item in records)
        unavailable = frozenset(index for index, item in enumerate(records)
                                if item["unavailable"])
        current = identifiers.index(inventory.current_set_id)
    else:
        compounds = tuple(TireCompound)
        identifiers = (None,) * len(compounds)
        ages = (tire_age,)
        unavailable = frozenset()
        current = compounds.index(current_tire.compound)
    bits = {compound: 1 << index if index < 3 else 8
            for index, compound in enumerate(TireCompound)}
    initial_used = 0
    for compound in used_compounds:
        initial_used |= bits[TireCompound(compound)]
    surface_path = [projected_surfaces(weather, 1)[0]]
    # Completion status precedes distance, request credit and crossing time.
    # A retired suffix accepts no further laps and incurs no further service.
    retired = (1, 0, 0, 0.)
    unknown = (inf, inf, inf, inf)

    def add_stint(laps, seconds, tail, instructions=0):
        return (tail[0], tail[1] - laps,
                0 if tail[0] else tail[2] - instructions, tail[3] + seconds)

    def finish(laps, seconds, used, instructions=0):
        return (0, -laps, -instructions, seconds) if legal(used) else retired

    def after_crossing(elapsed, seconds, announced):
        if finish_context is None:
            return 0., False
        elapsed += seconds
        return elapsed, announced or elapsed >= finish_context.time_limit_seconds

    def surface(offset, paid, fit_delay, stopped_first):
        update = (weather_clock.updates(offset, paid, stopped_first, fit_delay=fit_delay)
                  if weather_clock is not None else
                  intervals[offset] if intervals is not None else offset)
        while len(surface_path) <= update:
            cancellation_checkpoint()
            surface_path.append(project_next_surface(
                surface_path[-1], forecast_context, len(surface_path) - 1,
            ))
        return update, surface_path[update]

    def legal(used):
        return physical <= 1 or bool(used & 8) or (used & 7).bit_count() >= 2

    def age_at(ages, selected):
        return ages[selected] if finite else ages[0]

    def increment(ages, selected):
        if finite:
            return ages[:selected] + (ages[selected] + 1,) + ages[selected + 1:]
        return (ages[0] + 1,)

    def replacements(current, entry, ages):
        return tuple(index for index, compound in enumerate(compounds)
                     if index not in unavailable and (not finite or index != current)
                     and (not finite or tire_slot_usable(ages[index], expiries[index]))
                     and entry.tire_mismatch(compound) != "critical")

    @lru_cache(maxsize=4096)
    def running(offset, selected, age, update, stopped_first):
        tire = TIRE_COMPOUNDS[compounds[selected]]
        aero = active_aero_enabled if offset == 0 else True
        gap = gaps[int(stopped_first)] if offset == 0 and gaps is not None else None
        if prepared is None:
            value = isolated_strategy_lap(
                simulator, driver, car, track, tire, surface_path[update], current_lap + offset,
                physical, tire_age=age, active_aero_enabled=aero,
                gap_to_car_ahead=gap,
            )
        else:
            value = prepared(tire, surface_path[update], current_lap + offset, age,
                             active_aero_enabled=aero, gap_to_car_ahead=gap)
        return (current_running_time(value, current_lap_time_modifier, safety_car,
                                     stopped=stopped_first) if offset == 0 else value)

    def run(offset, selected, ages, used, paid, fit_delay, pending, stopped_first):
        if finite and not tire_slot_usable(ages[selected], expiries[selected]):
            return None
        update, after = surface(offset, paid, fit_delay, stopped_first)
        if after.tire_mismatch(compounds[selected]) == "critical":
            return None
        fee = tire_warmup_seconds(warmup, compounds[selected]) if pending else 0.
        seconds = running(offset, selected, age_at(ages, selected), update,
                          stopped_first if offset == 0 else False)
        seconds = (current_fitted_time(seconds, fee, safety_car, stopped=stopped_first)
                   if offset == 0 else seconds + fee)
        return (seconds,
                increment(ages, selected), used | bits[compounds[selected]], fit_delay + fee)

    def paid_fit(offset, selected, ages, used, paid, fit_delay, stopped_first, elapsed, announced,
                 requested=False):
        fitted_ages = ages if finite else (0,)
        stopped_first = stopped_first or offset == 0
        outcome = run(offset, selected, fitted_ages, used, paid + 1, fit_delay, True,
                      stopped_first)
        if outcome is None:
            return retired
        cost, next_ages, next_used, next_delay = outcome
        stop = current_stop if offset == 0 else green_stop
        seconds = stop + cost
        if announced:
            return finish(1, seconds, next_used, int(requested))
        next_elapsed, next_announced = after_crossing(elapsed, seconds, announced)
        return add_stint(1, seconds, continuation(
            offset + 1, selected, next_ages, next_used, paid + 1, next_delay,
            False, stopped_first, next_elapsed, next_announced,
        ), int(requested))

    @lru_cache(maxsize=4096)
    def continuation(offset, current, ages, used, paid, fit_delay, pending, stopped_first,
                     elapsed, announced):
        cost = 0.
        completed = 0
        # Advance deterministic retained stints iteratively. Recursion occurs
        # only at services, so a long scheduled distance cannot exhaust Python's
        # call stack merely by keeping the same set.
        while offset < horizon:
            cancellation_checkpoint()
            lap = current_lap + offset
            _, entry = surface(offset, paid, fit_delay, stopped_first)
            final = lap >= max(2, track.total_laps) or announced
            compulsory = (current in unavailable
                          or finite and not tire_slot_usable(ages[current], expiries[current])
                          or entry.tire_mismatch(compounds[current]) == "critical"
                          or (final and not legal(used | bits[compounds[current]])))
            options = None
            honors_request = False
            if lap in requests:
                requested = [index for index in replacements(current, entry, ages)
                             if compounds[index] == requests[lap]
                             and (not final or legal(used | bits[compounds[index]]))]
                if requested:
                    options = (min(requested, key=lambda index: (age_at(ages, index), index)),)
                    honors_request = True
            if options is None and compulsory:
                options = tuple(index for index in replacements(current, entry, ages)
                                if not final or legal(used | bits[compounds[index]]))
            if options is not None:
                if finite and interchangeable:
                    # Requests select one physical ID already. Compulsory
                    # alternatives with equal compound, wear and expiry have
                    # identical suffix costs; retain their first input identity
                    # here without removing any remaining copies from the pool.
                    unique = {}
                    for index in options:
                        unique.setdefault((compounds[index], ages[index], expiries[index]), index)
                    options = tuple(unique.values())
                best = min((paid_fit(offset, index, ages, used, paid, fit_delay,
                                     stopped_first, elapsed, announced, honors_request)
                            for index in options), default=retired)
                return add_stint(completed, cost, best)
            outcome = run(offset, current, ages, used, paid, fit_delay, pending, stopped_first)
            if outcome is None:
                return add_stint(completed, cost, retired)
            value, ages, used, fit_delay = outcome
            cost += value
            completed += 1
            if announced:
                return finish(completed, cost, used)
            elapsed, announced = after_crossing(elapsed, value, announced)
            pending = False
            offset += 1
        return finish(completed, cost, used)

    # Eligibility is the observed commitment surface; the delayed pit-exit
    # surface is used for running pace and feasibility, not eligibility.
    candidates = list(replacements(current, surface_path[0], ages))
    if free_fit and finite and current not in unavailable \
            and tire_slot_usable(ages[current], expiries[current]) \
            and surface_path[0].tire_mismatch(compounds[current]) != "critical":
        candidates.insert(0, current)
    choice = CustomPitChoice(inf)
    best = unknown
    elapsed = 0. if finish_context is None else finish_context.now
    announced = False if finish_context is None else finish_context.announced
    try:
        for index in candidates:
            cancellation_checkpoint()
            if free_fit:
                fitted = not finite or index != current
                score = continuation(0, index, ages if finite else (0,), initial_used,
                                     0, 0., fitted or current_fit_pending, False,
                                     elapsed, announced)
            else:
                if ((current_lap >= max(2, track.total_laps) or announced)
                        and not legal(initial_used | bits[compounds[index]])):
                    continue
                score = paid_fit(0, index, ages, initial_used, 0, 0., False, elapsed, announced)
            if isfinite(score[3]) and score[1] < 0 and score < best:
                best = score
                choice = CustomPitChoice(score[3] if not score[0] else inf,
                                         compounds[index], identifiers[index],
                                         -score[1], -score[2],
                                         score[3] if score[0] else inf)
        return choice
    finally:
        continuation.cache_clear()
        running.cache_clear()


register_forecast_helpers(globals(), ("choose_custom_pit_replacement", "project_next_surface",
                                      "normalize_current_traffic_gaps"))
register_forecast_helpers(globals(), ("current_running_time", "current_fitted_time"))
register_forecast_helpers(globals(), ("tire_slot_usable",))
register_forecast_helpers(globals(), ("isolated_strategy_lap", "native_physics"))
