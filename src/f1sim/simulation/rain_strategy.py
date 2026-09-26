"""Paid rain stints and compound transitions under deterministic surface projection."""

import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from math import inf, isfinite
from numbers import Integral, Real
from threading import RLock

import numpy as np

from f1sim.cancellation import cancellation_checkpoint
from f1sim.models import Car, Driver, Tire, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.strategy_traffic import normalize_current_traffic_gaps
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.surface_projection import (
    normalize_weather_intervals,
    projected_surfaces,
    suffix_weather_intervals,
)
from f1sim.simulation.warmup import tire_warmup_seconds, validate_tire_warmup


@dataclass(frozen=True)
class RainStopDecision:
    pit_now_cost: float
    wait_cost: float

    def should_pit(self, tolerance: float = 0.0) -> bool:
        _nonnegative(tolerance, "tolerance")
        return self.pit_now_cost + tolerance < self.wait_cost


def _nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _validate_weather_clock(weather_clock, horizon):
    if weather_clock is not None:
        if not isinstance(weather_clock, StrategyWeatherClock):
            raise ValueError("weather_clock must be a StrategyWeatherClock")
        weather_clock.validate_horizon(horizon)


def _clock_rain_stop(
    driver, car, track, weather, current_tire, tire_age, current_lap, remaining_stops,
    *, pit_lane_factor, additional_current_stop_cost, current_lap_time_modifier,
    active_aero_enabled, physical_total_laps, current_traffic_gaps, weather_clock,
    tire_warmup, current_fit_pending,
):
    """Compare complete same-compound stints on the paid-stop weather clock."""
    gaps = normalize_current_traffic_gaps(current_traffic_gaps)
    horizon = track.total_laps - current_lap + 1
    driver = driver.model_copy(deep=True)
    driver.reset_race_state()
    driver.id = driver.name = driver.team_id = "projection"
    clean = car.model_copy(update={"team_id": "projection", "team_name": "projection"})
    models = tuple(model.model_dump_json() for model in (driver, clean, track))
    retained_json = current_tire.model_dump_json()
    fresh = TIRE_COMPOUNDS[current_tire.compound]
    fresh_json = fresh.model_dump_json()
    service = expected_stationary_time(clean)
    green_stop = track.pit_lane_delta + service
    warmup = tire_warmup
    if warmup and type(weather_clock) is not StrategyWeatherClock:
        raise ValueError("tire_warmup requires the native StrategyWeatherClock")
    projected = [weather]
    native_clock = type(weather_clock) is StrategyWeatherClock
    absolute_updates = {}

    def schedule(paid, stopped_first, fit_delay=0.0):
        key = (paid, stopped_first, fit_delay)
        values = absolute_updates.get(key)
        if values is None:
            values = tuple(
                (weather_clock.updates(index, paid, stopped_first, fit_delay=fit_delay)
                 if fit_delay else weather_clock.updates(index, paid, stopped_first))
                for index in range(horizon)
            )
            absolute_updates[key] = values
        return values

    @lru_cache(maxsize=None)
    def row(offset, paid, stopped_first, retained=False, fit_delay=0.0,
            first_fit_fee=0.0):
        if native_clock:
            clock_updates = schedule(paid, stopped_first, fit_delay)
            first = clock_updates[offset]
        else:
            first = weather_clock.updates(offset, paid, stopped_first)
        while len(projected) <= first:
            projected.append(projected[-1].project_surface())
        if native_clock:
            intervals = tuple(
                ((weather_clock.updates(index, paid, stopped_first,
                                        fit_delay=fit_delay + first_fit_fee)
                  if fit_delay + first_fit_fee and index > offset
                  else clock_updates[index]) - first)
                for index in range(offset, horizon)
            )
        else:
            intervals = tuple(
                weather_clock.updates(index, paid, stopped_first) - first
                for index in range(offset, horizon)
            )
        costs = _running_row(
            models, projected[first].model_dump_json(),
            retained_json if retained else fresh_json,
            tire_age if retained else 0, current_lap + offset,
            physical_total_laps, intervals,
        )
        if first_fit_fee and costs:
            costs = (costs[0] + first_fit_fee, *costs[1:])
        return costs

    cache = {}

    def future(initial):
        """Every edge ends a stint, retaining every allowed stop schedule."""
        if initial in cache:
            return cache[initial]
        frames = [[initial, None, 0, inf]]
        while frames:
            cancellation_checkpoint()
            state, actions, index, best = frames[-1]
            if actions is None:
                offset, paid, stopped_first, fit_delay, first_fit_fee = state
                costs = row(offset, paid, stopped_first, fit_delay=fit_delay,
                            first_fit_fee=first_fit_fee)
                actions = []
                prefix = 0.0
                if paid < remaining_stops:
                    for next_stop in range(offset + 1, horizon):
                        prefix += costs[next_stop - offset - 1]
                        next_fee = tire_warmup_seconds(warmup, fresh.compound) if warmup else 0.0
                        actions.append(((next_stop, paid + 1, stopped_first,
                                         fit_delay + first_fit_fee, next_fee),
                                        prefix + green_stop))
                frames[-1][1:] = [actions, 0, sum(costs)]
                continue
            if index < len(actions):
                child, cost = actions[index]
                frames[-1][2] += 1
                if child in cache:
                    frames[-1][3] = min(frames[-1][3], cost + cache[child])
                else:
                    frames.append([child, None, 0, inf])
                continue
            cache[state] = best
            frames.pop()
            if frames:
                parent = frames[-1]
                _child, cost = parent[1][parent[2] - 1]
                parent[3] = min(parent[3], cost + best)
        return cache[initial]

    simulator = LapSimulator(np.random.default_rng(0))

    def first_running(tire, age, paid, stopped_first, gap, fit_fee=0.0):
        first = (schedule(paid, stopped_first)[0] if native_clock
                 else weather_clock.updates(0, paid, stopped_first))
        while len(projected) <= first:
            projected.append(projected[-1].project_surface())
        driver.current_tire_laps = age
        value = simulator.calculate_lap_time(
            driver, clean, track, tire, projected[first], current_lap,
            physical_total_laps, active_aero_enabled=active_aero_enabled,
            sample_variation=False, gap_to_car_ahead=gap,
        ) * current_lap_time_modifier
        return value + fit_fee

    current_fee = (tire_warmup_seconds(warmup, current_tire.compound)
                   if current_fit_pending and warmup else 0.0)
    fresh_fee = tire_warmup_seconds(warmup, fresh.compound) if warmup else 0.0
    old = row(0, 0, False, True, first_fit_fee=current_fee)
    first_old = first_running(current_tire, tire_age, 0, False,
                              gaps[0] if gaps is not None else None, current_fee)
    wait = first_old + sum(old[1:])
    prefix = first_old
    if remaining_stops:
        for next_stop in range(1, horizon):
            cancellation_checkpoint()
            if next_stop > 1:
                prefix += old[next_stop - 1]
            wait = min(wait, prefix + green_stop + future((
                next_stop, 1, False, current_fee, fresh_fee,
            )))
    pit = inf
    if remaining_stops:
        fresh_row = row(0, 1, True, first_fit_fee=fresh_fee)
        first_fresh = first_running(fresh, 0, 1, True,
                                    gaps[1] if gaps is not None else None, fresh_fee)
        pit = (track.pit_lane_delta * pit_lane_factor + service
               + additional_current_stop_cost + future((0, 1, True, 0.0, fresh_fee))
               + first_fresh - fresh_row[0])
    return RainStopDecision(pit, wait)


def _clock_rain_transition(
    driver, car, track, weather, current_tire, tire_age, current_lap, remaining_stops,
    *, pit_lane_factor, additional_current_stop_cost, current_lap_time_modifier,
    active_aero_enabled, physical_total_laps, remaining_dry_stops,
    remaining_damp_stops, used_mask, current_traffic_gaps, weather_clock,
    tire_warmup, current_fit_pending,
):
    """Evaluate rain/slick transitions while paid stops move an external clock."""
    gaps = normalize_current_traffic_gaps(current_traffic_gaps)
    horizon = track.total_laps - current_lap + 1
    simulator = LapSimulator(np.random.default_rng(0))
    driver = driver.model_copy(deep=True)
    driver.reset_race_state()
    clean = car.model_copy(deep=True)
    service = expected_stationary_time(clean)
    warmup = tire_warmup
    if warmup and type(weather_clock) is not StrategyWeatherClock:
        raise ValueError("tire_warmup requires the native StrategyWeatherClock")
    bits = {compound: (1 << index if index < 3 else 8)
            for index, compound in enumerate(TireCompound)}

    projected = [weather]
    branch_surfaces = {}
    surface_json = {id(weather): weather.model_dump_json()}

    def updates(offset, paid_stops, stopped_first, fit_delay=0.0):
        if fit_delay:
            return weather_clock.updates(offset, paid_stops, stopped_first,
                                        fit_delay=fit_delay)
        return weather_clock.updates(offset, paid_stops, stopped_first)

    def branch_surface(offset, paid_stops, stopped_first, fit_delay=0.0):
        branch = (offset, paid_stops, stopped_first, fit_delay)
        if branch in branch_surfaces:
            return branch_surfaces[branch]
        update_index = updates(offset, paid_stops, stopped_first, fit_delay)
        while len(projected) <= update_index:
            value = projected[-1].project_surface()
            projected.append(value)
            surface_json[id(value)] = value.model_dump_json()
        value = projected[update_index]
        branch_surfaces[branch] = value
        return value

    @lru_cache(maxsize=None)
    def running(offset, tire_key, compound, age, gap_kind, surface_json):
        tire = current_tire if tire_key == "retained" else TIRE_COMPOUNDS[compound]
        driver.current_tire_laps = age
        branch_surface = Weather.model_validate_json(surface_json)
        value = simulator.calculate_lap_time(
            driver, clean, track, tire, branch_surface,
            current_lap + offset, physical_total_laps,
            active_aero_enabled=(active_aero_enabled if offset == 0 else True),
            sample_variation=False,
            gap_to_car_ahead=(
                gaps[gap_kind] if gaps is not None and gap_kind in (0, 1) else None
            ),
        )
        return value * current_lap_time_modifier if offset == 0 else value

    def run(offset, tire_key, compound, age, branch_surface, gap_kind=-1, fitted=False):
        if isinstance(compound, TireCompound):
            compound = compound.value
        elif isinstance(compound, Tire):
            compound = compound.compound.value
        value = running(offset, tire_key, compound, age, gap_kind,
                        surface_json[id(branch_surface)])
        if fitted and warmup:
            value += tire_warmup_seconds(warmup, compound)
        return value

    def legal(mask):
        return bool(mask & 8) or (mask & 7).bit_count() >= 2

    def reduced(value):
        return None if value is None else max(0, value - 1)

    solve_cache = {}

    def solve(initial):
        """Evaluate the transition DAG with an explicit stack."""
        if initial in solve_cache:
            return solve_cache[initial]
        frames = [[initial, None, 0, inf]]
        while frames:
            cancellation_checkpoint()
            state, actions, index, best = frames[-1]
            if state in solve_cache:
                frames.pop()
                continue
            (offset, tire_key, compound, age, left, dry, damp, used,
             paid_stops, stopped_first, fit_delay) = state
            if actions is None:
                if offset == horizon:
                    # Keep the terminal value in the frame so the common
                    # completion path can add its incoming edge.
                    frames[-1][1:] = [[], 0, 0.0 if legal(used) else inf]
                    continue
                before = branch_surface(offset, paid_stops, stopped_first, fit_delay)
                current = TireCompound(compound)
                critical = before.tire_mismatch(current) == "critical"
                actions = []
                if not critical:
                    actions.append((
                        (offset + 1, tire_key, compound, age + 1, left, dry, damp,
                         used | bits[current], paid_stops, stopped_first, fit_delay),
                        run(offset, tire_key, current, age, before),
                    ))
                rain = before.fresh_rain_compound()
                candidates = ((rain,) if rain is not None else (
                    TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD,
                ))
                limit = dry if before.track_wetness < .08 and before.rain_intensity < .15 else damp
                allowed = (critical or (left > 0 and (
                    current in (TireCompound.INTERMEDIATE, TireCompound.WET)
                    or before.track_wetness > .3 or limit is None or limit > 0
                )))
                compliant = legal(used)
                if allowed or not compliant:
                    for candidate in candidates:
                        cancellation_checkpoint()
                        if before.tire_mismatch(candidate) == "critical":
                            continue
                        if not allowed and (compliant or used & bits[candidate]):
                            continue
                        after_paid = paid_stops + 1
                        after = branch_surface(offset, after_paid, stopped_first, fit_delay)
                        fit_cost = tire_warmup_seconds(warmup, candidate) if warmup else 0.0
                        actions.append((
                            (offset + 1, candidate.value, candidate.value, 1,
                             max(0, left - 1), reduced(dry), reduced(damp),
                             used | bits[candidate], after_paid, stopped_first,
                             fit_delay + fit_cost),
                            track.pit_lane_delta + service
                            + run(offset, candidate.value, candidate, 0, after,
                                  fitted=bool(warmup)),
                        ))
                frames[-1][1:] = [actions, 0, inf]
                continue
            if index < len(actions):
                child, edge = actions[index]
                frames[-1][2] += 1
                if child in solve_cache:
                    frames[-1][3] = min(frames[-1][3], edge + solve_cache[child])
                else:
                    frames.append([child, None, 0, inf])
                continue
            solve_cache[state] = best
            frames.pop()
            if frames:
                parent = frames[-1]
                child, edge = parent[1][parent[2] - 1]
                parent[3] = min(parent[3], edge + solve_cache[state])
        return solve_cache[initial]

    first = branch_surface(0, 0, False, 0.0)
    wait = inf
    if first.tire_mismatch(current_tire.compound) != "critical":
        current_fee = (tire_warmup_seconds(warmup, current_tire.compound)
                       if current_fit_pending and warmup else 0.0)
        wait = run(0, "retained", current_tire.compound, tire_age, first, 0,
                   fitted=bool(current_fee))
        wait += solve((1, "retained", current_tire.compound.value, tire_age + 1,
                       remaining_stops, remaining_dry_stops, remaining_damp_stops,
                       used_mask | bits[current_tire.compound], 0, False, current_fee))

    pit = inf
    selected = None
    rain = first.fresh_rain_compound()
    candidates = ((rain,) if rain is not None else (
        TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD,
    ))
    critical = first.tire_mismatch(current_tire.compound) == "critical"
    limit = remaining_dry_stops if first.track_wetness < .08 and first.rain_intensity < .15 \
        else remaining_damp_stops
    allowed = critical or (remaining_stops > 0 and (
        current_tire.compound in (TireCompound.INTERMEDIATE, TireCompound.WET)
        or first.track_wetness > .3 or limit is None or limit > 0
    ))
    compliant = legal(used_mask)
    if allowed or not compliant:
        for candidate in candidates:
            if first.tire_mismatch(candidate) == "critical":
                continue
            if not allowed and (compliant or used_mask & bits[candidate]):
                continue
            after = branch_surface(0, 1, True, 0.0)
            fit_cost = tire_warmup_seconds(warmup, candidate) if warmup else 0.0
            cost = (track.pit_lane_delta * pit_lane_factor + service
                    + additional_current_stop_cost
                    + run(0, candidate.value, candidate, 0, after, 1,
                          fitted=bool(warmup)))
            cost += solve((1, candidate.value, candidate.value, 1,
                           max(0, remaining_stops - 1),
                           reduced(remaining_dry_stops), reduced(remaining_damp_stops),
                           used_mask | bits[candidate], 1, True, fit_cost))
            if cost < pit:
                pit, selected = cost, candidate
    return RainTransitionDecision(pit, wait, selected)


def _clock_same_rain_stint(weather, current_tire, weather_clock):
    """Return whether a transition forecast has no legal compound crossover.

    The transition planner has a much larger state space because every
    projected surface can introduce a slick or alternate rain candidate.  If
    the external clock can only reach surfaces that continue to recommend the
    fitted rain compound, and that compound never becomes critical, its
    transition state graph is exactly the same-compound stop graph.  Keep this
    check bounded by the clock's capped update count; a future crossover still
    uses the full transition solver.
    """
    if current_tire.compound not in (TireCompound.INTERMEDIATE, TireCompound.WET):
        return False
    surface = weather
    for _ in range(weather_clock.max_updates + 1):
        cancellation_checkpoint()
        if (surface.fresh_rain_compound() != current_tire.compound
                or surface.tire_mismatch(current_tire.compound) == "critical"):
            return False
        surface = surface.project_surface()
    return True


@lru_cache(maxsize=4096)
def _running_row(models, weather_json, tire_json, age, lap, physical, intervals=None):
    """Green stint beginning here; no stop budget or current-control key."""
    driver = Driver.model_validate_json(models[0])
    car = Car.model_construct(**json.loads(models[1]))
    track = Track.model_validate_json(models[2])
    surface = Weather.model_validate_json(weather_json)
    tire = Tire.model_validate_json(tire_json)
    simulator = LapSimulator(np.random.default_rng(0))
    row = []
    for offset, surface in enumerate(projected_surfaces(
        surface, track.total_laps - lap + 1, intervals,
    )):
        cancellation_checkpoint()
        driver.current_tire_laps = age + offset
        row.append(simulator.calculate_lap_time(
            driver, car, track, tire, surface, lap + offset, physical,
            sample_variation=False,
        ))
    return tuple(row)


@lru_cache(maxsize=4096)
def _surfaces(weather_json, horizon, intervals=None):
    return tuple(surface.model_dump_json() for surface in projected_surfaces(
        Weather.model_validate_json(weather_json), horizon, intervals,
    ))


@lru_cache(maxsize=8192)
def _fresh_future(models, weather_json, fresh_json, lap, budget, physical, intervals=None,
                  warmup_profile=()):
    """Best green cost after a fresh set is fitted; its service is excluded."""
    row = _running_row(models, weather_json, fresh_json, 0, lap, physical, intervals)
    compound = Tire.model_validate_json(fresh_json).compound.value
    fit_cost = dict(warmup_profile).get(compound, 0.0)
    total = sum(row) + fit_cost
    if budget == 0:
        return total
    car = Car.model_construct(**json.loads(models[1]))
    track = Track.model_validate_json(models[2])
    stop = track.pit_lane_delta + expected_stationary_time(car)
    surfaces = _surfaces(weather_json, len(row), intervals)
    # The first running lap is charged even when this fitted set is replaced
    # later in the forecast, so carry its fee into every stop branch.
    stint = fit_cost
    for offset in range(1, len(row)):
        stint += row[offset - 1]
        total = min(total, stint + stop + _fresh_future(
            models, surfaces[offset], fresh_json, lap + offset, budget - 1, physical,
            suffix_weather_intervals(intervals, offset),
            warmup_profile,
        ))
    return total


@lru_cache(maxsize=256)
def _plan(snapshots, tire_age, current_lap, budget, lane, queue, modifier, aero, physical,
          intervals=None, gaps=None, warmup_profile=(), current_fit_pending=False):
    models = snapshots[:3]
    weather_json, retained_json, fresh_json = snapshots[3:]
    car = Car.model_construct(**json.loads(models[1]))
    track = Track.model_validate_json(models[2])
    row = _running_row(models, weather_json, retained_json, tire_age, current_lap,
                       physical, intervals)
    fresh_row = _running_row(models, weather_json, fresh_json, 0, current_lap, physical, intervals)
    surfaces = _surfaces(weather_json, len(row), intervals)
    service = expected_stationary_time(car)
    retained_compound = Tire.model_validate_json(retained_json).compound.value
    current_fee = (dict(warmup_profile).get(retained_compound, 0.0)
                   if current_fit_pending else 0.0)
    wait = sum(row) + current_fee
    if budget:
        # If we wait before refitting, the pending current-set cost belongs
        # to the prefix of every such branch as well as the straight-through
        # forecast above.
        stint = current_fee
        for offset in range(1, len(row)):
            stint += row[offset - 1]
            wait = min(wait, stint + track.pit_lane_delta + service + _fresh_future(
                models, surfaces[offset], fresh_json, current_lap + offset, budget - 1, physical,
                suffix_weather_intervals(intervals, offset),
                warmup_profile,
            ))
    pit = inf if budget == 0 else (
        track.pit_lane_delta * lane + service + queue + _fresh_future(
            models, weather_json, fresh_json, current_lap, budget - 1, physical, intervals,
            warmup_profile,
        )
    )
    first_old, first_fresh = row[0], fresh_row[0]
    if not aero or gaps is not None:
        driver = Driver.model_validate_json(models[0])
        simulator = LapSimulator(np.random.default_rng(0))
        weather = Weather.model_validate_json(weather_json)
        def first(tire_json, age, gap):
            driver.current_tire_laps = age
            return simulator.calculate_lap_time(
                driver, car, track, Tire.model_validate_json(tire_json), weather,
                current_lap, physical, active_aero_enabled=aero, sample_variation=False,
                gap_to_car_ahead=gap,
            )
        first_old = first(retained_json, tire_age, gaps[0] if gaps else None)
        first_fresh = first(fresh_json, 0, gaps[1] if gaps else None)
    return RainStopDecision(pit + first_fresh * modifier - fresh_row[0],
                            wait + first_old * modifier - row[0])


def plan_rain_stop(
    driver: Driver, car: Car, track: Track, weather: Weather, current_tire: Tire,
    tire_age: int, current_lap: int, remaining_stops: int, *, pit_lane_factor: float = 1.0,
    additional_current_stop_cost: float = 0.0, current_lap_time_modifier: float = 1.0,
    active_aero_enabled: bool = True, physical_total_laps: int | None = None,
    weather_intervals: tuple[int, ...] | None = None,
    current_traffic_gaps: tuple[float | None, float | None] | None = None,
    weather_clock: StrategyWeatherClock | None = None,
    tire_warmup=None,
    current_fit_pending: bool = False,
) -> RainStopDecision:
    """Compare stopping now with driving at least one lap before any stop.

    Every refit pays service and lane loss and consumes the bounded stop budget.
    Current neutralization and queue costs apply once; future laps assume green
    clean air. Rainfall stays fixed while surface wetness evolves. The caller
    must ensure the same rain compound remains appropriate over the horizon.
    Fuel follows physical_total_laps even when track bounds a shorter plan.
    weather_intervals optionally supplies cumulative surface-update counts for
    each remaining own lap, starting at zero; None uses one update per lap.
    """
    for name, value, minimum in (("tire_age", tire_age, 0), ("current_lap", current_lap, 1),
                                 ("remaining_stops", remaining_stops, 0)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if current_lap > track.total_laps:
        raise ValueError("current_lap must not exceed the planning distance")
    physical = track.total_laps if physical_total_laps is None else physical_total_laps
    if (isinstance(physical, bool) or not isinstance(physical, Integral)
            or physical < track.total_laps):
        raise ValueError("physical_total_laps must be an integer >= planning distance")
    if current_tire.compound not in (TireCompound.INTERMEDIATE, TireCompound.WET):
        raise ValueError("current_tire must be a rain compound")
    if (isinstance(additional_current_stop_cost, bool)
            or not isinstance(additional_current_stop_cost, Real)
            or not isfinite(additional_current_stop_cost)):
        raise ValueError("additional_current_stop_cost must be finite")
    for name, value in (("pit_lane_factor", pit_lane_factor),
                        ("current_lap_time_modifier", current_lap_time_modifier)):
        _nonnegative(value, name)
    if current_lap_time_modifier == 0:
        raise ValueError("current_lap_time_modifier must be positive")
    if not isinstance(active_aero_enabled, bool):
        raise ValueError("active_aero_enabled must be boolean")
    tire_warmup = validate_tire_warmup(tire_warmup)
    if type(current_fit_pending) is not bool:
        raise ValueError("current_fit_pending must be boolean")
    gaps = normalize_current_traffic_gaps(current_traffic_gaps)
    intervals = normalize_weather_intervals(
        track.total_laps - current_lap + 1, weather_intervals, weather=weather,
    )
    horizon = track.total_laps - current_lap + 1
    _validate_weather_clock(weather_clock, horizon)
    if weather_clock is not None:
        return _clock_rain_stop(
            driver, car, track, weather, current_tire, int(tire_age), int(current_lap),
            min(int(remaining_stops), horizon), pit_lane_factor=float(pit_lane_factor),
            additional_current_stop_cost=float(additional_current_stop_cost),
            current_lap_time_modifier=float(current_lap_time_modifier),
            active_aero_enabled=active_aero_enabled, physical_total_laps=int(physical),
            current_traffic_gaps=current_traffic_gaps, weather_clock=weather_clock,
            tire_warmup=tire_warmup, current_fit_pending=current_fit_pending,
        )
    clean = driver.model_copy(deep=True)
    clean.reset_race_state()
    # Lap physics reads performance attributes, never names or identifiers.
    # Keep every performance field, including consistency used to compute std.
    clean.id = clean.name = clean.team_id = "projection"
    clean_car = car.model_copy(update={"team_id": "projection", "team_name": "projection"})
    snapshots = tuple(model.model_dump_json() for model in (
        clean, clean_car, track, weather, current_tire, TIRE_COMPOUNDS[current_tire.compound],
    ))
    return _plan(snapshots, int(tire_age), int(current_lap),
                 min(int(remaining_stops), track.total_laps - current_lap + 1),
                 float(pit_lane_factor), float(additional_current_stop_cost),
                 float(current_lap_time_modifier), active_aero_enabled, int(physical),
                 intervals, gaps, tuple(sorted(tire_warmup.items())),
                 current_fit_pending)


@dataclass(frozen=True)
class RainTransitionDecision(RainStopDecision):
    compound: TireCompound | None


# Completed immutable suffix costs are shared across later decisions. Accesses
# are synchronized; computing duplicate misses concurrently is harmless.
_TRANSITION_SUFFIX_LIMIT = 8192
_transition_suffixes = OrderedDict()
_transition_suffix_lock = RLock()


def _reset_transition_cache_after_fork():
    # Another parent thread may own the inherited lock. Replace it without
    # acquiring it; the child's cache starts independently of parent updates.
    global _transition_suffix_lock, _transition_suffixes
    _transition_suffix_lock = RLock()
    _transition_suffixes = OrderedDict()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_transition_cache_after_fork)


def _remember_transition(key, value):
    with _transition_suffix_lock:
        _transition_suffixes[key] = value
        _transition_suffixes.move_to_end(key)
        while len(_transition_suffixes) > _TRANSITION_SUFFIX_LIMIT:
            _transition_suffixes.popitem(last=False)


def _transition_stop_eligibility_row(surfaces, critical, compound, left, dry, damp):
    """Return exact stop eligibility for one compound and remaining budget state."""
    rain_compound = compound in (TireCompound.INTERMEDIATE, TireCompound.WET)
    row = []
    for offset, surface in enumerate(surfaces):
        if critical[offset]:
            row.append(True)
        elif left <= 0:
            row.append(False)
        elif rain_compound or surface.track_wetness > .3:
            row.append(True)
        else:
            limit = (dry if surface.track_wetness < .08 and surface.rain_intensity < .15
                     else damp)
            row.append(limit is None or limit > 0)
    return tuple(row)


@lru_cache(maxsize=256)
def _transition_plan(snapshots, tire_age, current_lap, budget, lane, queue,
                     modifier, aero, physical, dry_budget, damp_budget, intervals=None, gaps=None,
                     used_mask=8, warmup_profile=(), current_fit_pending=False):
    models = snapshots[:3]
    weather_json, retained_json, tires_json = snapshots[3:]
    track = Track.model_validate_json(models[2])
    car = Car.model_construct(**json.loads(models[1]))
    fresh = {TireCompound(key): Tire.model_validate(value).model_dump_json()
             for key, value in json.loads(tires_json).items()}
    retained = Tire.model_validate_json(retained_json)
    horizon = track.total_laps - current_lap + 1
    surface_json = _surfaces(weather_json, horizon, intervals)
    surfaces = [Weather.model_validate_json(value) for value in surface_json]
    cadence_suffixes = tuple(suffix_weather_intervals(intervals, offset, surface)
                             for offset, surface in enumerate(surfaces))
    service = expected_stationary_time(car)
    stop_cost = track.pit_lane_delta + service
    candidates = []
    critical = {}
    for compound in TireCompound:
        critical[compound] = [s.tire_mismatch(compound) == "critical" for s in surfaces]
    for surface in surfaces:
        rain = surface.fresh_rain_compound()
        candidates.append((rain,) if rain is not None else (
            TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD,
        ))

    def reduced(limit):
        return None if limit is None else max(0, limit - 1)

    bits = {compound: (1 << index if index < 3 else 8)
            for index, compound in enumerate(TireCompound)}

    def legal(mask):
        return bool(mask & 8) or (mask & 7).bit_count() >= 2

    eligibility_rows = {}

    def stop_eligibility(compound, left, dry, damp):
        key = (compound, left, dry, damp)
        row = eligibility_rows.get(key)
        if row is None:
            row = _transition_stop_eligibility_row(
                surfaces, critical[compound], compound, left, dry, damp,
            )
            eligibility_rows[key] = row
        return row

    def cache_key(state):
        offset, compound, left, dry, damp, mask = state
        return (models, surface_json[offset], tires_json, warmup_profile, current_lap + offset,
                compound, left, dry, damp, mask, physical, cadence_suffixes[offset])

    solved = {}
    def stint(start, compound, age, tire_json, left, dry, damp, mask, fitted=False):
        row = _running_row(models, surface_json[start], tire_json, age,
                           current_lap + start, physical,
                           cadence_suffixes[start])
        total, best_cost = 0.0, inf
        # This set must run its fitting/current lap before any future stop.
        # A critical set cannot run that lap and earns no compound-use credit.
        if critical[compound][start]:
            return best_cost
        total += row[0]
        if fitted and warmup_profile:
            total += dict(warmup_profile).get(compound.value, 0.0)
        mask |= bits[compound]
        compliant = legal(mask)
        next_left, next_dry, next_damp = max(0, left - 1), reduced(dry), reduced(damp)
        # Keeping this same set cannot change either its used-compound mask or
        # the remaining stop allowances. Only the surface eligibility varies.
        allowed_by_offset = stop_eligibility(compound, left, dry, damp)
        for offset in range(start + 1, horizon):
            allowed = allowed_by_offset[offset]
            for candidate in candidates[offset]:
                if (not critical[candidate][offset] and (
                    allowed or (not compliant and not mask & bits[candidate])
                )):
                    value = yield (
                        offset, candidate, next_left, next_dry, next_damp, mask,
                    )
                    best_cost = min(best_cost, total + stop_cost + value)
            if critical[compound][offset]:
                return best_cost
            total += row[offset - start]
        return min(best_cost, total if compliant else inf)

    def evaluate(frame, initial=None):
        # Suspend each stint at its unresolved edge. Rescanning a parent's
        # earlier laps after each child would repeat both physics-row lookups
        # and every previously visited edge. Offsets strictly increase, so this
        # explicit DFS stack needs no recursion or in-progress cache entries.
        pending = [(initial, frame)]
        value = None
        while pending:
            cancellation_checkpoint()
            state, frame = pending[-1]
            try:
                child = frame.send(value)
            except StopIteration as finished:
                value = finished.value
                if state is not None:
                    solved[state] = value
                    _remember_transition(cache_key(state), value)
                pending.pop()
                continue
            if child in solved:
                value = solved[child]
                continue
            key = cache_key(child)
            with _transition_suffix_lock:
                value = _transition_suffixes.get(key)
                if value is not None:
                    _transition_suffixes.move_to_end(key)
            if value is not None:
                solved[child] = value
                continue
            offset, compound, left, dry, damp, mask = child
            pending.append((child, stint(
                offset, compound, 0, fresh[compound], left, dry, damp, mask,
                fitted=True,
            )))
        return value

    wait = evaluate(stint(0, retained.compound, tire_age, retained_json,
                          budget, dry_budget, damp_budget, used_mask,
                          fitted=current_fit_pending))

    def first(tire_json, age, gap):
        row = _running_row(models, weather_json, tire_json, age, current_lap, physical, intervals)
        if aero and gap is None:
            return row[0] * modifier - row[0]
        driver = Driver.model_validate_json(models[0])
        driver.current_tire_laps = age
        simulator = LapSimulator(np.random.default_rng(0))
        actual = simulator.calculate_lap_time(
            driver, car, track, Tire.model_validate_json(tire_json), surfaces[0],
            current_lap, physical, active_aero_enabled=aero, sample_variation=False,
            gap_to_car_ahead=gap,
        )
        return actual * modifier - row[0]

    wait += first(retained_json, tire_age, gaps[0] if gaps else None)
    pit, compound = inf, None
    current_stop_allowed = stop_eligibility(
        retained.compound, budget, dry_budget, damp_budget,
    )[0]
    if current_stop_allowed or not legal(used_mask):
        for candidate in candidates[0]:
            if critical[candidate][0] or (
                not current_stop_allowed
                and (legal(used_mask) or used_mask & bits[candidate])
            ):
                continue
            state = (0, candidate, max(0, budget - 1),
                     reduced(dry_budget), reduced(damp_budget), used_mask)
            def root():
                return (yield state)

            value = evaluate(root())
            cost = (track.pit_lane_delta * lane + service + queue + value
                    + first(fresh[candidate], 0, gaps[1] if gaps else None))
            if cost < pit:
                pit, compound = cost, candidate
    return RainTransitionDecision(pit, wait, compound)


def plan_rain_transition(
    driver: Driver, car: Car, track: Track, weather: Weather, current_tire: Tire,
    tire_age: int, current_lap: int, remaining_stops: int, *, pit_lane_factor: float = 1.0,
    additional_current_stop_cost: float = 0.0, current_lap_time_modifier: float = 1.0,
    active_aero_enabled: bool = True, physical_total_laps: int | None = None,
    weather_intervals: tuple[int, ...] | None = None,
    remaining_dry_stops: int | None = None, remaining_damp_stops: int | None = None,
    used_compounds: set[TireCompound] | None = None,
    current_traffic_gaps: tuple[float | None, float | None] | None = None,
    weather_clock: StrategyWeatherClock | None = None,
    tire_warmup=None,
    current_fit_pending: bool = False,
) -> RainTransitionDecision:
    """Plan bounded paid stops across rain/slick transitions under fixed rainfall.

    weather_intervals optionally supplies cumulative surface-update counts for
    each remaining own lap, starting at zero; None uses one update per lap.

    A retained set may run while noncritical. Fresh fits follow the surface's
    rain-compound recommendation, or consider all slicks when it recommends
    none. Critical replacements and required compound corrections may exceed
    the elective stop budget. Explicit used_compounds records actual race use;
    prior tyre wear gives no credit. Two slicks or actual rain-tyre use are
    required at the finish. Slick callers must provide it; omitting it for
    retained rain tyres preserves the legacy wet exemption.
    Every fit runs its fitting lap; future costs assume green clean air.
    Optional slick-state allowances count all paid fits from this decision,
    including earlier rain fits; they never reset on a compound transition.
    """
    for name, value, minimum in (("tire_age", tire_age, 0), ("current_lap", current_lap, 1),
                                 ("remaining_stops", remaining_stops, 0)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if current_lap > track.total_laps:
        raise ValueError("current_lap must not exceed the planning distance")
    physical = track.total_laps if physical_total_laps is None else physical_total_laps
    if (isinstance(physical, bool) or not isinstance(physical, Integral)
            or physical < track.total_laps):
        raise ValueError("physical_total_laps must be an integer >= planning distance")
    if (isinstance(additional_current_stop_cost, bool)
            or not isinstance(additional_current_stop_cost, Real)
            or not isfinite(additional_current_stop_cost)):
        raise ValueError("additional_current_stop_cost must be finite")
    for name, value in (("pit_lane_factor", pit_lane_factor),
                        ("current_lap_time_modifier", current_lap_time_modifier)):
        _nonnegative(value, name)
    if current_lap_time_modifier == 0:
        raise ValueError("current_lap_time_modifier must be positive")
    if not isinstance(active_aero_enabled, bool):
        raise ValueError("active_aero_enabled must be boolean")
    tire_warmup = validate_tire_warmup(tire_warmup)
    if type(current_fit_pending) is not bool:
        raise ValueError("current_fit_pending must be boolean")
    for name, value in (("remaining_dry_stops", remaining_dry_stops),
                        ("remaining_damp_stops", remaining_damp_stops)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, Integral) or value < 0
        ):
            raise ValueError(f"{name} must be a nonnegative integer or None")
    # None preserves the historical rain-only caller's assumed wet exemption.
    if used_compounds is None and current_tire.compound in {
        TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD,
    }:
        raise ValueError("used_compounds is required for a retained slick compound")
    mask = 8 if used_compounds is None else 0
    if used_compounds is not None:
        if not isinstance(used_compounds, (set, frozenset, list, tuple)):
            raise ValueError("used_compounds must contain valid tyre compounds")
        for compound in used_compounds:
            try:
                compound = TireCompound(compound)
            except (ValueError, TypeError) as exc:
                raise ValueError("used_compounds must contain valid tyre compounds") from exc
            mask |= {TireCompound.SOFT: 1, TireCompound.MEDIUM: 2,
                     TireCompound.HARD: 4}.get(compound, 8)
    gaps = normalize_current_traffic_gaps(current_traffic_gaps)
    intervals = normalize_weather_intervals(
        track.total_laps - current_lap + 1, weather_intervals, weather=weather,
    )
    horizon = track.total_laps - current_lap + 1
    _validate_weather_clock(weather_clock, horizon)
    if weather_clock is not None:
        # An unrecorded rain fit can still allow a compound-rule correction
        # beyond the elective budget. The same-compound solver has no mask,
        # so leave that case with the general transition solver.
        compliant = bool(mask & 8) or (mask & 7).bit_count() >= 2
        if ((remaining_stops > 0 or compliant)
                and _clock_same_rain_stint(weather, current_tire, weather_clock)):
            same_compound = _clock_rain_stop(
                driver, car, track, weather, current_tire, int(tire_age), int(current_lap),
                min(int(remaining_stops), horizon),
                pit_lane_factor=float(pit_lane_factor),
                additional_current_stop_cost=float(additional_current_stop_cost),
                current_lap_time_modifier=float(current_lap_time_modifier),
                active_aero_enabled=active_aero_enabled, physical_total_laps=int(physical),
                current_traffic_gaps=current_traffic_gaps, weather_clock=weather_clock,
                tire_warmup=tire_warmup, current_fit_pending=current_fit_pending,
            )
            return RainTransitionDecision(
                same_compound.pit_now_cost,
                same_compound.wait_cost,
                (current_tire.compound
                 if same_compound.pit_now_cost < inf else None),
            )
        return _clock_rain_transition(
            driver, car, track, weather, current_tire, int(tire_age), int(current_lap),
            min(int(remaining_stops), horizon), pit_lane_factor=float(pit_lane_factor),
            additional_current_stop_cost=float(additional_current_stop_cost),
            current_lap_time_modifier=float(current_lap_time_modifier),
            active_aero_enabled=active_aero_enabled, physical_total_laps=int(physical),
            remaining_dry_stops=(None if remaining_dry_stops is None
                                 else int(remaining_dry_stops)),
            remaining_damp_stops=(None if remaining_damp_stops is None
                                  else int(remaining_damp_stops)),
            used_mask=mask, current_traffic_gaps=current_traffic_gaps,
            weather_clock=weather_clock,
            tire_warmup=tire_warmup, current_fit_pending=current_fit_pending,
        )
    clean = driver.model_copy(deep=True)
    clean.reset_race_state()
    clean.id = clean.name = clean.team_id = "projection"
    clean_car = car.model_copy(update={"team_id": "projection", "team_name": "projection"})
    snapshots = tuple(model.model_dump_json() for model in (
        clean, clean_car, track, weather, current_tire,
    )) + (json.dumps({compound.value: tire.model_dump(mode="json")
                     for compound, tire in TIRE_COMPOUNDS.items()}, sort_keys=True),)
    return _transition_plan(
        snapshots, int(tire_age), int(current_lap),
        min(int(remaining_stops), track.total_laps - current_lap + 1),
        float(pit_lane_factor), float(additional_current_stop_cost),
        float(current_lap_time_modifier), active_aero_enabled, int(physical),
        None if remaining_dry_stops is None else int(remaining_dry_stops),
        None if remaining_damp_stops is None else int(remaining_damp_stops), intervals, gaps, mask,
        tuple(sorted(tire_warmup.items())), current_fit_pending,
    )
