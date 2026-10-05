"""Paid rain stints and compound transitions under deterministic surface projection."""

import json
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from math import inf, isfinite
from numbers import Integral, Real
from threading import RLock

import numpy as np

from f1sim.cancellation import cancellation_checkpoint
from f1sim.models import Car, Driver, Tire, TireCompound, Track, Weather
from f1sim.models._native import (
    forecast_decision,
    forecast_dump,
    forecast_json,
    native_forecast_cache,
    native_physics,
    register_forecast_helpers,
    restore_model,
    shared_forecast_available,
)
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation import lap as lap_physics  # noqa: F401 - extension compatibility
from f1sim.simulation.controlled_weather_strategy import (
    plan_controlled_weather,
    usable_weather_control,
)
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.strategy_lap import isolated_strategy_lap, strategy_projection_models
from f1sim.simulation.strategy_neutralization import current_fitted_time, current_running_time
from f1sim.simulation.strategy_traffic import normalize_current_traffic_gaps
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.surface_projection import (
    normalize_weather_intervals,
    projected_surfaces,
    suffix_weather_intervals,
)
from f1sim.simulation.warmup import tire_warmup_seconds, validate_tire_warmup
from f1sim.simulation.weather_schedule import (
    paid_compound_candidates,
    project_next_surface,
)

# Share only scalar native green costs, never models or prepared evaluators.
_GREEN_LAP_LIMIT = 65_536
_green_laps = OrderedDict()
_green_lap_lock = RLock()
# Exact future weather graphs and strategy costs contain immutable values only.
_CLOCK_NODE_LIMIT = 65_536
_REFIT_COST_LIMIT = 65_536


class _ForecastCostCache:
    """Keep reusable refit branches from being displaced by retained tails.

    Both pools share the existing total bound. A pool can use spare capacity;
    when full, preserve one eighth for refits and evict within each pool in
    least-recently-used order. Only immutable keys and scalar costs are stored.
    Callers hold the shared forecast lock.
    """

    def __init__(self):
        self.refits = OrderedDict()
        self.tails = OrderedDict()

    def _pool(self, key):
        return self.refits if key[0] == "refit" else self.tails

    def get(self, key, default=None):
        return self._pool(key).get(key, default)

    def __setitem__(self, key, value):
        self._pool(key)[key] = value

    def move_to_end(self, key):
        self._pool(key).move_to_end(key)

    def __len__(self):
        return len(self.refits) + len(self.tails)

    def popitem(self, last=False):
        pool = (self.refits if len(self.refits) > _REFIT_COST_LIMIT // 8 or not self.tails
                else self.tails)
        return pool.popitem(last=last)


_clock_nodes = OrderedDict()
_clock_node_sequence = 0
_refit_costs = _ForecastCostCache()
_refit_lock = RLock()
def _native_green_cache_available(simulator):
    return (shared_forecast_available()
            and simulator._native_deterministic_evaluator_available())


def _native_green_model_available(model):
    from f1sim.models._native import native_model

    return native_model(model)


def _reset_green_cache_after_fork():
    global _green_laps, _green_lap_lock, _clock_nodes
    global _refit_costs, _refit_lock
    _green_laps = OrderedDict()
    _green_lap_lock = RLock()
    _clock_nodes = OrderedDict()
    # Inherited decision-local IDs can remain alive after a fork. Never reuse
    # their numbers when rebuilding the caches in the child.
    _refit_costs = _ForecastCostCache()
    _refit_lock = RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_green_cache_after_fork)


def _shared_green_lap(key, evaluate, tire, surface, lap, age):
    with _green_lap_lock:
        value = _green_laps.get(key)
        if value is not None:
            _green_laps.move_to_end(key)
            return value
    value = evaluate(tire, surface, lap, age)
    with _green_lap_lock:
        _green_laps[key] = value
        _green_laps.move_to_end(key)
        while len(_green_laps) > _GREEN_LAP_LIMIT:
            _green_laps.popitem(last=False)
    return value


@dataclass(frozen=True)
class RainStopDecision:
    pit_now_cost: float
    wait_cost: float
    pit_now_laps: int | None = field(default=None, kw_only=True)
    wait_laps: int | None = field(default=None, kw_only=True)

    def should_pit(self, tolerance: float = 0.0) -> bool:
        _nonnegative(tolerance, "tolerance")
        if (self.pit_now_laps is not None and self.wait_laps is not None
                and self.pit_now_laps != self.wait_laps):
            return self.pit_now_laps > self.wait_laps
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
    tire_warmup, current_fit_pending, forecast_context=None, safety_car=None,
):
    """Compare complete same-compound stints on the paid-stop weather clock."""
    gaps = normalize_current_traffic_gaps(current_traffic_gaps)
    if safety_car is not None:
        gaps = safety_car.traffic_gaps
    horizon = track.total_laps - current_lap + 1
    driver, clean = strategy_projection_models(driver, car, track, weather, current_tire)
    models = tuple(forecast_json(model) for model in (driver, clean, track))
    retained_json = forecast_json(current_tire)
    fresh = TIRE_COMPOUNDS[current_tire.compound]
    fresh_json = forecast_json(fresh)
    service = expected_stationary_time(clean)
    green_stop = track.pit_lane_delta + service
    warmup = tire_warmup
    if warmup and type(weather_clock) is not StrategyWeatherClock:
        raise ValueError("tire_warmup requires the native StrategyWeatherClock")
    projected = [weather]
    native_clock = type(weather_clock) is StrategyWeatherClock
    simulator = LapSimulator(np.random.default_rng(0))
    prepared = (simulator.prepare_deterministic_lap_time(
        driver, clean, track, physical_total_laps,
    ) if native_clock and type(weather) is Weather and type(current_tire) is Tire
                and type(fresh) is Tire else None)
    shared_green = (prepared is not None and _native_green_cache_available(simulator)
                    and all(_native_green_model_available(model)
                            for model in (driver, clean, track, current_tire, fresh, weather)))
    if shared_green:
        # Reuse existing exact snapshots; string hashes are cached by Python.
        # Each projected surface is serialized once, never on a cache lookup.
        package = models + (physical_total_laps,)
        tire_values = (fresh_json, retained_json)
        surface_values = [forecast_json(weather)]
    absolute_updates = {}

    @lru_cache(maxsize=None)
    def running(retained, lap, age, update_index):
        """Only immutable costs escape this decision's shared surface path.

        The two complete tyre models and driver/car/track/fuel package are
        fixed for this call; retained distinguishes their full parameters.
        Absolute update counts preserve paid-stop and fitting-delay cadence.
        """
        while len(projected) <= update_index:
            projected.append(project_next_surface(
                projected[-1], forecast_context, len(projected) - 1))
        if shared_green:
            while len(surface_values) <= update_index:
                value = projected[len(surface_values)]
                surface_values.append(forecast_json(value)
                                      if _native_green_model_available(value) else None)
        tire = current_tire if retained else fresh
        surface = projected[update_index]
        if shared_green and surface_values[update_index] is not None:
            key = (package, tire_values[retained], surface_values[update_index], lap, age)
            return _shared_green_lap(key, prepared, tire, surface, lap, age)
        return prepared(tire, surface, lap, age)

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
        if prepared is not None:
            delayed = (schedule(paid, stopped_first, fit_delay + first_fit_fee)
                       if first_fit_fee else clock_updates)
            values = []
            for index in range(offset, horizon):
                cancellation_checkpoint()
                values.append(running(
                    retained, current_lap + index,
                    (tire_age if retained else 0) + index - offset,
                    first if index == offset else delayed[index],
                ))
            costs = tuple(values)
            if first_fit_fee and costs:
                costs = (costs[0] + first_fit_fee, *costs[1:])
            return costs
        while len(projected) <= first:
            projected.append(project_next_surface(
                projected[-1], forecast_context, len(projected) - 1))
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
            models, forecast_json(projected[first]),
            retained_json if retained else fresh_json,
            tire_age if retained else 0, current_lap + offset,
            physical_total_laps, normalize_weather_intervals(
                len(intervals), intervals, forecast_context=(
                    forecast_context.advanced(first) if forecast_context is not None else None),
            ), True,
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

    def first_running(tire, age, paid, stopped_first, gap, fit_fee=0.0):
        first = (schedule(paid, stopped_first)[0] if native_clock
                 else weather_clock.updates(0, paid, stopped_first))
        while len(projected) <= first:
            projected.append(project_next_surface(
                projected[-1], forecast_context, len(projected) - 1))
        driver.current_tire_laps = age
        value = simulator.calculate_lap_time(
            driver, clean, track, tire, projected[first], current_lap,
            physical_total_laps, active_aero_enabled=active_aero_enabled,
            sample_variation=False, gap_to_car_ahead=gap,
        )
        value = current_running_time(value, current_lap_time_modifier, safety_car,
                                     stopped=stopped_first)
        return current_fitted_time(value, fit_fee, safety_car, stopped=stopped_first)

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


def _shared_clock_node(key):
    """Intern complete immutable suffix signatures, with nonrecycled IDs."""
    global _clock_node_sequence
    with _refit_lock:
        identity = _clock_nodes.get(key)
        if identity is not None:
            _clock_nodes.move_to_end(key)
            return identity
        _clock_node_sequence += 1
        identity = _clock_node_sequence
        _clock_nodes[key] = identity
        while len(_clock_nodes) > _CLOCK_NODE_LIMIT:
            _clock_nodes.popitem(last=False)
        return identity


def _shared_refit_cost(key):
    with _refit_lock:
        value = _refit_costs.get(key)
        if value is not None:
            _refit_costs.move_to_end(key)
        return value


def _store_refit_cost(key, value):
    with _refit_lock:
        _refit_costs[key] = value
        _refit_costs.move_to_end(key)
        while len(_refit_costs) > _REFIT_COST_LIMIT:
            _refit_costs.popitem(last=False)


def _equivalent_clock_branches(weather_clock, horizon, *, surface_key=None, current_lap=1,
                               update_counts=None):
    """Quotient complete native future clock paths, without fitting delays.

    A node describes the surface before and after a paid fit, and both next
    nodes. Equal signatures therefore give identical update counts for every
    remaining fit/retain sequence. All paid counts reachable with at most one
    fit per lap are included, even after the elective allowance is exhausted.
    Representatives only live in this decision; current-lap costs stay outside.
    Optional shared IDs describe full surface values and both child graphs,
    allowing equal future paths from different raw clocks to share scalar costs.
    """
    rows = [None] * horizon
    updates = weather_clock.updates if update_counts is None else update_counts
    intern = lru_cache(maxsize=None)(_shared_clock_node)
    shared_ids = [None] * horizon if surface_key is not None else None
    next_ids = None
    next_shared = None
    for offset in range(horizon - 1, 0, -1):
        representatives = {}
        ids = {}
        row = {}
        shared_row = {}
        for paid in range(offset + 1):
            for first in (False, True) if paid else (False,):
                cancellation_checkpoint()
                signature = (
                    updates(offset, paid, first),
                    updates(offset, paid + 1, first),
                    next_ids[paid, first] if next_ids is not None else 0,
                    next_ids[paid + 1, first] if next_ids is not None else 0,
                )
                if signature not in representatives:
                    representatives[signature] = ((paid, first), len(representatives))
                representative, identity = representatives[signature]
                row[paid, first] = representative
                ids[paid, first] = identity
                if surface_key is not None:
                    shared_row[paid, first] = intern((
                        current_lap + offset, surface_key(signature[0]), surface_key(signature[1]),
                        next_shared[paid, first] if next_shared is not None else 0,
                        next_shared[paid + 1, first] if next_shared is not None else 0,
                    ))
        rows[offset] = row
        next_ids = ids
        if shared_ids is not None:
            shared_ids[offset] = shared_row
            next_shared = shared_row
    return rows, shared_ids


def _budget_clock_branches(weather_clock, horizon, remaining_stops, surface_key, current_lap,
                           *, update_counts=None):
    """Exact suffixes when every compound stays noncritical throughout.

    With no compulsory replacements, a legal history permits only its remaining
    elective fits. An incomplete history after a running lap needs at most one
    extra fit: an unused slick or rain set completes the rule. Keep both bounds
    until the actual use mask selects one, including a paid opening fit when
    the caller supplied no prior race-use credit.
    """
    maximum_paid = max(remaining_stops + 1, 2)
    updates = weather_clock.updates if update_counts is None else update_counts
    intern = lru_cache(maxsize=None)(_shared_clock_node)
    rows = [None] * horizon
    next_row = None
    for offset in range(horizon - 1, 0, -1):
        row = {}
        for paid in range(min(offset, maximum_paid) + 1):
            left = max(0, remaining_stops - paid)
            for first in (False, True) if paid else (False,):
                before = surface_key(updates(offset, paid, first))
                for room in (left, left + 1) if paid < maximum_paid else (left,):
                    cancellation_checkpoint()
                    after = (surface_key(updates(offset, paid + 1, first))
                             if room else None)
                    keep = next_row[paid, first, room] if next_row is not None else 0
                    fit = (next_row[paid + 1, first, room - 1]
                           if room and next_row is not None else 0)
                    row[paid, first, room] = intern((
                        current_lap + offset, before, after, keep, fit,
                    ))
        rows[offset] = row
        next_row = row
    return rows


def _clock_rain_transition(
    driver, car, track, weather, current_tire, tire_age, current_lap, remaining_stops,
    *, pit_lane_factor, additional_current_stop_cost, current_lap_time_modifier,
    active_aero_enabled, physical_total_laps, remaining_dry_stops,
    remaining_damp_stops, used_mask, current_traffic_gaps, weather_clock,
    tire_warmup, current_fit_pending, forecast_context=None, safety_car=None,
):
    """Evaluate rain/slick transitions while paid stops move an external clock."""
    gaps = normalize_current_traffic_gaps(current_traffic_gaps)
    if safety_car is not None:
        gaps = safety_car.traffic_gaps
    horizon = track.total_laps - current_lap + 1
    simulator = LapSimulator(np.random.default_rng(0))
    driver = driver.model_copy(deep=True)
    isolated = not native_physics(driver, car, track, weather, current_tire)
    if not isolated:
        driver.reset_race_state()
    clean = car.model_copy(deep=True)
    service = expected_stationary_time(clean)
    warmup = tire_warmup
    if warmup and type(weather_clock) is not StrategyWeatherClock:
        raise ValueError("tire_warmup requires the native StrategyWeatherClock")
    bits = {compound: (1 << index if index < 3 else 8)
            for index, compound in enumerate(TireCompound)}
    compound_values = {compound: compound.value for compound in TireCompound}
    compounds_by_value = {value: compound for compound, value in compound_values.items()}

    projected = [weather]
    branch_surfaces = {}
    observed_surfaces = {id(weather): weather}
    native_safety = (type(weather_clock) is StrategyWeatherClock
                     and shared_forecast_available()
                     and all(_native_green_model_available(model)
                             for model in (driver, clean, track, weather, current_tire)))
    safety_tables = {}
    prepared = (simulator.prepare_deterministic_lap_time(
        driver, clean, track, physical_total_laps,
    ) if native_safety else None)
    shared_green = prepared is not None and _native_green_cache_available(simulator)
    if shared_green:
        projection_driver = driver.model_copy(update={
            "id": "projection", "name": "projection", "team_id": "projection",
        })
        projection_car = clean.model_copy(update={
            "team_id": "projection", "team_name": "projection",
        })
        package = tuple(forecast_json(model) for model in (
            projection_driver, projection_car, track,
        )) + (physical_total_laps,)
        tire_values = {compound.value: forecast_json(tire)
                       for compound, tire in TIRE_COMPOUNDS.items()}
        tire_values["retained"] = forecast_json(current_tire)
        surface_values = {}

    def surface_safety(surface):
        key = id(surface)
        if key not in safety_tables:
            candidates = paid_compound_candidates(surface, forecast_context)
            safe = frozenset(candidates)
            safety_tables[key] = (candidates, frozenset(TireCompound) - safe)
        return safety_tables[key]

    def candidates_for(surface):
        return (surface_safety(surface)[0] if native_safety else
                paid_compound_candidates(surface, forecast_context))

    def critical_on(surface, compound):
        return (compound in surface_safety(surface)[1] if native_safety else
                surface.tire_mismatch(compound) == "critical")

    def used_after_running(mask, compound):
        completed = mask | bits[compound]
        # Once the compound rule is satisfied, neither slick identities nor
        # how it was satisfied can affect any later action. Use one internal
        # completed-rule marker; actual race-use history stays with the caller.
        # Extensions retain their masks and original hook call counts.
        return (8 if native_safety and (
            completed & 8 or (completed & 7).bit_count() >= 2
        ) else completed)

    # Native clocks are immutable and their original method is registered with
    # the extension guard. Share validated counts between graph construction
    # and running branches; custom clocks and fitting delays keep dispatching.
    native_updates = (lru_cache(maxsize=None)(weather_clock.updates)
                      if native_safety and not warmup else None)

    def updates(offset, paid_stops, stopped_first, fit_delay=0.0):
        if fit_delay:
            return weather_clock.updates(offset, paid_stops, stopped_first,
                                        fit_delay=fit_delay)
        if native_updates is not None:
            return native_updates(offset, paid_stops, stopped_first)
        return weather_clock.updates(offset, paid_stops, stopped_first)

    def branch_surface(offset, paid_stops, stopped_first, fit_delay=0.0):
        branch = (offset, paid_stops, stopped_first, fit_delay)
        if branch in branch_surfaces:
            return branch_surfaces[branch]
        update_index = updates(offset, paid_stops, stopped_first, fit_delay)
        while len(projected) <= update_index:
            value = project_next_surface(projected[-1], forecast_context, len(projected) - 1)
            projected.append(value)
            observed_surfaces[id(value)] = value
        value = projected[update_index]
        branch_surfaces[branch] = value
        return value

    @lru_cache(maxsize=None)
    def running(offset, tire_key, compound, age, gap_kind, surface_id):
        tire = current_tire if tire_key == "retained" else TIRE_COMPOUNDS[compound]
        branch_surface = observed_surfaces[surface_id]
        if offset and prepared is not None:
            if shared_green:
                if surface_id not in surface_values:
                    surface_values[surface_id] = forecast_json(branch_surface)
                key = (package, tire_values[tire_key], surface_values[surface_id],
                       current_lap + offset, age)
                return _shared_green_lap(key, prepared, tire, branch_surface,
                                         current_lap + offset, age)
            return prepared(tire, branch_surface, current_lap + offset, age)
        options = dict(active_aero_enabled=active_aero_enabled if offset == 0 else True,
                       gap_to_car_ahead=gaps[gap_kind]
                       if gaps is not None and gap_kind in (0, 1) else None)
        if isolated:
            value = isolated_strategy_lap(
                simulator, driver, clean, track, tire, branch_surface,
                current_lap + offset, physical_total_laps, tire_age=age, **options)
        else:
            driver.current_tire_laps = age
            value = simulator.calculate_lap_time(
                driver, clean, track, tire, branch_surface, current_lap + offset,
                physical_total_laps, sample_variation=False, **options)
        return (current_running_time(value, current_lap_time_modifier, safety_car,
                                     stopped=gap_kind == 1) if offset == 0 else value)

    def run(offset, tire_key, compound, age, branch_surface, gap_kind=-1, fitted=False):
        if isinstance(compound, TireCompound):
            compound = compound.value
        elif isinstance(compound, Tire):
            compound = compound.compound.value
        value = running(offset, tire_key, compound, age, gap_kind,
                        id(branch_surface))
        fee = tire_warmup_seconds(warmup, compound) if fitted and warmup else 0.
        return (current_fitted_time(value, fee, safety_car, stopped=gap_kind == 1)
                if offset == 0 else value + fee)

    def legal(mask):
        return bool(mask & 8) or (mask & 7).bit_count() >= 2

    def reduced(value):
        return None if value is None else max(0, value - 1)

    solve_cache = {}

    # A cached recursive call consumes interpreter capacity as well as one
    # Python frame. Keep a conservative allowance and account for callers,
    # including an application that deliberately lowers its recursion limit.
    frame = sys._getframe()
    depth = 0
    while frame is not None:
        depth += 1
        frame = frame.f_back
    recursive = native_safety and 6 * horizon + depth + 64 < sys.getrecursionlimit()

    def projected_surface_key(update_index):
        while len(projected) <= update_index:
            value = project_next_surface(projected[-1], forecast_context, len(projected) - 1)
            projected.append(value)
            observed_surfaces[id(value)] = value
        value = projected[update_index]
        identity = id(value)
        if identity not in surface_values:
            surface_values[identity] = forecast_json(value)
        return surface_values[identity]

    clock_branches = shared_clock_ids = None
    budget_clock = False
    if recursive and not warmup:
        if shared_green and horizon > 1:
            maximum_paid = min(horizon, max(remaining_stops + 1, 2))
            last_update = max(updates(horizon - 1, maximum_paid, first)
                              for first in (False, True))
            projected_surface_key(last_update)
            # This sufficient proof includes skipped updates too. A later
            # critical scheduled step keeps the full compulsory-fit graph.
            budget_clock = all(len(surface_safety(surface)[0]) == len(TireCompound)
                               for surface in projected)
        if budget_clock:
            shared_clock_ids = _budget_clock_branches(
                weather_clock, horizon, remaining_stops, projected_surface_key, current_lap,
                update_counts=native_updates,
            )
        else:
            clock_branches, shared_clock_ids = _equivalent_clock_branches(
                weather_clock, horizon,
                surface_key=projected_surface_key if shared_green else None,
                current_lap=current_lap,
                update_counts=native_updates,
            )
    if shared_clock_ids is not None:
        refit_package = (package, tuple(tire_values[compound_values[value]]
                                       for value in TireCompound))

    def clock_state(state):
        if clock_branches is None:
            return state
        offset = state[0]
        # No further weather query or paid action follows a terminal state.
        paid, first = ((0, False) if offset == horizon else
                       clock_branches[offset][state[8], state[9]])
        return (*state[:8], paid, first, state[10])

    def shared_clock(offset, paid, first, left, used):
        if budget_clock:
            room = left + (not legal(used))
            return shared_clock_ids[offset][paid, first, room]
        return shared_clock_ids[offset][paid, first]

    @lru_cache(maxsize=None)
    def recursive_refit(offset, left, dry, damp, used, paid_stops, stopped_first,
                        fit_delay, allowed):
        """Fresh choices share costs across retained compounds and tyre ages.

        Eligibility has already accounted for the retained set. Once that
        verdict is fixed, every fresh-fit edge and child is independent of the
        removed set. Keep all budgets, use history and clock dimensions.
        """
        cancellation_checkpoint()
        shared_key = None
        if shared_clock_ids is not None:
            shared_key = ("refit", refit_package,
                          shared_clock(offset, paid_stops, stopped_first, left, used),
                          left, dry, damp, used, allowed)
            cached = _shared_refit_cost(shared_key)
            if cached is not None:
                return cached
        before = branch_surface(offset, paid_stops, stopped_first, fit_delay)
        compliant = legal(used)
        best = inf
        for candidate in candidates_for(before):
            if not allowed and (compliant or used & bits[candidate]):
                continue
            candidate_value = compound_values[candidate]
            after_paid = paid_stops + 1
            after = branch_surface(offset, after_paid, stopped_first, fit_delay)
            fit_cost = tire_warmup_seconds(warmup, candidate) if warmup else 0.0
            child = (offset + 1, candidate_value, candidate_value, 1,
                     max(0, left - 1), reduced(dry), reduced(damp),
                     used_after_running(used, candidate), after_paid, stopped_first,
                     fit_delay + fit_cost)
            edge = track.pit_lane_delta + service + run(
                offset, candidate_value, candidate_value, 0, after, fitted=bool(warmup),
            )
            best = min(best, edge + recursive_solve(clock_state(child)))
        if shared_key is not None:
            _store_refit_cost(shared_key, best)
        return best

    @lru_cache(maxsize=None)
    def recursive_solve(state):
        cancellation_checkpoint()
        (offset, tire_key, compound, age, left, dry, damp, used,
         paid_stops, stopped_first, fit_delay) = state
        if offset == horizon:
            return 0.0 if legal(used) else inf
        shared_key = None
        if shared_clock_ids is not None:
            # Full tyre parameters and age distinguish retained and fresh sets.
            # The complete suffix graph prices every future retain/refit path;
            # current traffic/control costs and any already-paid fit stay out.
            shared_key = ("tail", refit_package, tire_values[tire_key], age,
                          shared_clock(offset, paid_stops, stopped_first, left, used),
                          left, dry, damp, used)
            cached = _shared_refit_cost(shared_key)
            if cached is not None:
                return cached
        before = branch_surface(offset, paid_stops, stopped_first, fit_delay)
        current = compounds_by_value[compound]
        if left == 0 and legal(used):
            # A safe set must be retained after elective fits run out. Fold
            # this forced stint in one frame instead of publishing every
            # interior age to the bounded shared cache. A later critical
            # surface still admits all compulsory replacements, even beyond
            # the budget; their paid clock and fitting fees remain unchanged.
            edges = []
            while offset < horizon:
                cancellation_checkpoint()
                before = branch_surface(offset, paid_stops, stopped_first, fit_delay)
                if critical_on(before, current):
                    best = recursive_refit(
                        offset, left, dry, damp, used, paid_stops, stopped_first,
                        fit_delay, True,
                    )
                    break
                edges.append(run(offset, tire_key, compound, age, before))
                offset += 1
                age += 1
                used = used_after_running(used, current)
            else:
                best = 0.0
            # Preserve the original right-associative addition order exactly.
            for edge in reversed(edges):
                best = edge + best
            if shared_key is not None:
                _store_refit_cost(shared_key, best)
            return best
        critical = critical_on(before, current)
        best = inf
        if not critical:
            edge = run(offset, tire_key, compound, age, before)
            child = (offset + 1, tire_key, compound, age + 1, left, dry, damp,
                     used_after_running(used, current), paid_stops, stopped_first, fit_delay)
            best = min(best, edge + recursive_solve(clock_state(child)))
        limit = dry if before.track_wetness < .08 and before.rain_intensity < .15 else damp
        allowed = (critical or (left > 0 and (
            current in (TireCompound.INTERMEDIATE, TireCompound.WET)
            or before.track_wetness > .3 or limit is None or limit > 0
        )))
        compliant = legal(used)
        if allowed or not compliant:
            best = min(best, recursive_refit(
                offset, left, dry, damp, used, paid_stops, stopped_first, fit_delay, allowed,
            ))
        if shared_key is not None:
            _store_refit_cost(shared_key, best)
        return best

    def solve(initial):
        """Evaluate the transition DAG with an explicit stack."""
        if recursive:
            return recursive_solve(clock_state(initial))
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
                current = compounds_by_value[compound]
                critical = critical_on(before, current)
                actions = []
                if not critical:
                    actions.append((
                        (offset + 1, tire_key, compound, age + 1, left, dry, damp,
                         used_after_running(used, current), paid_stops, stopped_first, fit_delay),
                        run(offset, tire_key, compound, age, before),
                    ))
                candidates = candidates_for(before)
                limit = dry if before.track_wetness < .08 and before.rain_intensity < .15 else damp
                allowed = (critical or (left > 0 and (
                    current in (TireCompound.INTERMEDIATE, TireCompound.WET)
                    or before.track_wetness > .3 or limit is None or limit > 0
                )))
                compliant = legal(used)
                if allowed or not compliant:
                    for candidate in candidates:
                        cancellation_checkpoint()
                        if not native_safety and critical_on(before, candidate):
                            continue
                        if not allowed and (compliant or used & bits[candidate]):
                            continue
                        candidate_value = compound_values[candidate]
                        after_paid = paid_stops + 1
                        after = branch_surface(offset, after_paid, stopped_first, fit_delay)
                        fit_cost = tire_warmup_seconds(warmup, candidate) if warmup else 0.0
                        actions.append((
                            (offset + 1, candidate_value, candidate_value, 1,
                             max(0, left - 1), reduced(dry), reduced(damp),
                             used_after_running(used, candidate), after_paid, stopped_first,
                             fit_delay + fit_cost),
                            track.pit_lane_delta + service
                            + run(offset, candidate_value, candidate_value, 0, after,
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

    try:
        first = branch_surface(0, 0, False, 0.0)
        wait = inf
        if not critical_on(first, current_tire.compound):
            current_fee = (tire_warmup_seconds(warmup, current_tire.compound)
                           if current_fit_pending and warmup else 0.0)
            wait = run(0, "retained", current_tire.compound, tire_age, first, 0,
                       fitted=bool(current_fee))
            wait += solve((1, "retained", current_tire.compound.value, tire_age + 1,
                           remaining_stops, remaining_dry_stops, remaining_damp_stops,
                           used_after_running(used_mask, current_tire.compound),
                           0, False, current_fee))

        pit = inf
        selected = None
        candidates = candidates_for(first)
        critical = critical_on(first, current_tire.compound)
        limit = remaining_dry_stops if first.track_wetness < .08 and first.rain_intensity < .15 \
            else remaining_damp_stops
        allowed = critical or (remaining_stops > 0 and (
            current_tire.compound in (TireCompound.INTERMEDIATE, TireCompound.WET)
            or first.track_wetness > .3 or limit is None or limit > 0
        ))
        compliant = legal(used_mask)
        if allowed or not compliant:
            for candidate in candidates:
                if not native_safety and critical_on(first, candidate):
                    continue
                if not allowed and (compliant or used_mask & bits[candidate]):
                    continue
                candidate_value = compound_values[candidate]
                after = branch_surface(0, 1, True, 0.0)
                fit_cost = tire_warmup_seconds(warmup, candidate) if warmup else 0.0
                cost = (track.pit_lane_delta * pit_lane_factor + service
                        + additional_current_stop_cost
                        + run(0, candidate_value, candidate_value, 0, after, 1,
                              fitted=bool(warmup)))
                cost += solve((1, candidate_value, candidate_value, 1,
                               max(0, remaining_stops - 1),
                               reduced(remaining_dry_stops), reduced(remaining_damp_stops),
                               used_after_running(used_mask, candidate), 1, True, fit_cost))
                if cost < pit:
                    pit, selected = cost, candidate
        return RainTransitionDecision(pit, wait, selected)
    finally:
        # Break closure cycles promptly, including on cancellation.
        recursive_solve.cache_clear()
        recursive_refit.cache_clear()
        recursive_solve = None
        recursive_refit = None


@native_forecast_cache(maxsize=4096)
def _running_row(models, weather_json, tire_json, age, lap, physical, intervals=None,
                 observed_surface=False):
    """Green stint beginning here; no stop budget or current-control key."""
    driver = restore_model(Driver, models[0])
    car = restore_model(Car, models[1])
    track = restore_model(Track, models[2])
    surface = restore_model(Weather, weather_json)
    tire = restore_model(Tire, tire_json)
    simulator = LapSimulator(np.random.default_rng(0))
    prepared = simulator.prepare_deterministic_lap_time(driver, car, track, physical)
    row = []
    horizon = track.total_laps - lap + 1
    if observed_surface:
        projected = []
        previous = 0
        for interval in (range(horizon) if intervals is None else intervals):
            for update in range(previous, interval):
                surface = project_next_surface(
                    surface, getattr(intervals, "context", None), update)
            projected.append(surface)
            previous = interval
    else:
        projected = projected_surfaces(surface, horizon, intervals)
    for offset, surface in enumerate(projected):
        cancellation_checkpoint()
        if prepared is not None:
            value = prepared(tire, surface, lap + offset, age + offset)
        else:
            value = isolated_strategy_lap(
                simulator, driver, car, track, tire, surface, lap + offset, physical,
                tire_age=age + offset)
        row.append(value)
    return tuple(row)


@native_forecast_cache(maxsize=4096)
def _surfaces(weather_json, horizon, intervals=None):
    return tuple(forecast_json(surface) for surface in projected_surfaces(
        restore_model(Weather, weather_json), horizon, intervals,
    ))


@native_forecast_cache(maxsize=8192)
def _fresh_future(models, weather_json, fresh_json, lap, budget, physical, intervals=None,
                  warmup_profile=()):
    """Best green cost after a fresh set is fitted; its service is excluded."""
    row = _running_row(models, weather_json, fresh_json, 0, lap, physical, intervals)
    compound = restore_model(Tire, fresh_json).compound.value
    fit_cost = dict(warmup_profile).get(compound, 0.0)
    total = sum(row) + fit_cost
    if budget == 0:
        return total
    car = restore_model(Car, models[1])
    track = restore_model(Track, models[2])
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


@native_forecast_cache(maxsize=256)
def _plan(snapshots, tire_age, current_lap, budget, lane, queue, modifier, aero, physical,
          intervals=None, gaps=None, warmup_profile=(), current_fit_pending=False,
          safety_car=None):
    models = snapshots[:3]
    weather_json, retained_json, fresh_json = snapshots[3:]
    car = restore_model(Car, models[1])
    track = restore_model(Track, models[2])
    row = _running_row(models, weather_json, retained_json, tire_age, current_lap,
                       physical, intervals)
    fresh_row = _running_row(models, weather_json, fresh_json, 0, current_lap, physical, intervals)
    surfaces = _surfaces(weather_json, len(row), intervals)
    service = expected_stationary_time(car)
    retained_compound = restore_model(Tire, retained_json).compound.value
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
        driver = restore_model(Driver, models[0])
        simulator = LapSimulator(np.random.default_rng(0))
        weather = restore_model(Weather, weather_json)
        def first(tire_json, age, gap):
            driver.current_tire_laps = age
            return simulator.calculate_lap_time(
                driver, car, track, restore_model(Tire, tire_json), weather,
                current_lap, physical, active_aero_enabled=aero, sample_variation=False,
                gap_to_car_ahead=gap,
            )
        first_old = first(retained_json, tire_age, gaps[0] if gaps else None)
        first_fresh = first(fresh_json, 0, gaps[1] if gaps else None)
    if safety_car is None:
        return RainStopDecision(pit + first_fresh * modifier - fresh_row[0],
                                wait + first_old * modifier - row[0])
    fresh_fee = dict(warmup_profile).get(restore_model(Tire, fresh_json).compound.value, 0.)
    paid_first = current_running_time(first_fresh, modifier, safety_car, stopped=True)
    old_first = current_running_time(first_old, modifier, safety_car)
    return RainStopDecision(
        pit + current_fitted_time(paid_first, fresh_fee, safety_car, stopped=True)
        - fresh_row[0] - fresh_fee,
        wait + current_fitted_time(old_first, current_fee, safety_car) - row[0] - current_fee,
    )


@forecast_decision
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
    forecast_context=None,
    safety_car=None,
    control_context=None,
) -> RainStopDecision:
    """Compare stopping now with driving at least one lap before any stop.

    Every refit pays service and lane loss and consumes the bounded stop budget.
    A usable control_context prices known neutralized entries and later stops;
    otherwise current neutralization and queue costs apply once. Green suffixes
    retain the existing clean-air policy. Rainfall stays fixed while surface
    wetness evolves. The caller must ensure the same rain compound remains
    appropriate over the horizon.
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
    if safety_car is not None:
        gaps = safety_car.traffic_gaps
    intervals = normalize_weather_intervals(
        track.total_laps - current_lap + 1, weather_intervals, weather=weather,
        forecast_context=forecast_context,
    )
    forecast_context = getattr(intervals, "context", forecast_context)
    horizon = track.total_laps - current_lap + 1
    _validate_weather_clock(weather_clock, horizon)
    if usable_weather_control(control_context, weather_clock):
        result = plan_controlled_weather(
            driver, car, track, weather, current_tire, tire_age, current_lap,
            min(int(remaining_stops), horizon), control_context=control_context,
            physical_total_laps=int(physical), tire_warmup=tire_warmup,
            current_fit_pending=current_fit_pending, forecast_context=forecast_context,
            require_compound_rule=False, same_compound=True)
        return RainStopDecision(result.pit.seconds, result.wait.seconds,
                                pit_now_laps=result.pit.laps, wait_laps=result.wait.laps)
    if weather_clock is not None:
        return _clock_rain_stop(
            driver, car, track, weather, current_tire, int(tire_age), int(current_lap),
            min(int(remaining_stops), horizon), pit_lane_factor=float(pit_lane_factor),
            additional_current_stop_cost=float(additional_current_stop_cost),
            current_lap_time_modifier=float(current_lap_time_modifier),
            active_aero_enabled=active_aero_enabled, physical_total_laps=int(physical),
            current_traffic_gaps=current_traffic_gaps, weather_clock=weather_clock,
            tire_warmup=tire_warmup, current_fit_pending=current_fit_pending,
            forecast_context=forecast_context,
            safety_car=safety_car,
        )
    clean, clean_car = strategy_projection_models(driver, car, track, weather, current_tire)
    snapshots = tuple(forecast_json(model) for model in (
        clean, clean_car, track, weather, current_tire, TIRE_COMPOUNDS[current_tire.compound],
    ))
    return _plan(snapshots, int(tire_age), int(current_lap),
                 min(int(remaining_stops), track.total_laps - current_lap + 1),
                 float(pit_lane_factor), float(additional_current_stop_cost),
                 float(current_lap_time_modifier), active_aero_enabled, int(physical),
                 intervals, gaps, tuple(sorted(tire_warmup.items())),
                 current_fit_pending, safety_car)


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


@native_forecast_cache(maxsize=256)
def _transition_plan(snapshots, tire_age, current_lap, budget, lane, queue,
                     modifier, aero, physical, dry_budget, damp_budget, intervals=None, gaps=None,
                     used_mask=8, warmup_profile=(), current_fit_pending=False, safety_car=None):
    models = snapshots[:3]
    weather_json, retained_json, tires_json = snapshots[3:]
    track = restore_model(Track, models[2])
    car = restore_model(Car, models[1])
    fresh = {TireCompound(key): forecast_json(restore_model(Tire, value))
             for key, value in json.loads(tires_json).items()}
    retained = restore_model(Tire, retained_json)
    horizon = track.total_laps - current_lap + 1
    surface_json = _surfaces(weather_json, horizon, intervals)
    surfaces = [restore_model(Weather, value) for value in surface_json]
    cadence_suffixes = tuple(suffix_weather_intervals(intervals, offset, surface)
                             for offset, surface in enumerate(surfaces))
    service = expected_stationary_time(car)
    stop_cost = track.pit_lane_delta + service
    candidates = []
    critical = {}
    for compound in TireCompound:
        critical[compound] = [s.tire_mismatch(compound) == "critical" for s in surfaces]
    for surface in surfaces:
        candidates.append(paid_compound_candidates(surface, getattr(intervals, "context", None)))

    def reduced(limit):
        return None if limit is None else max(0, limit - 1)

    bits = {compound: (1 << index if index < 3 else 8)
            for index, compound in enumerate(TireCompound)}

    def legal(mask):
        return bool(mask & 8) or (mask & 7).bit_count() >= 2

    native_masks = shared_forecast_available()

    def equivalent_mask(mask, compliant):
        # After actual rain-tyre use or two different slicks, no later choice
        # depends on which compounds satisfied the rule. Share that completed
        # credit without changing stop budgets, stint costs or candidate order.
        # Behavioral extensions retain their original evaluation call counts.
        return 8 if native_masks and compliant else mask

    used_mask = equivalent_mask(used_mask, legal(used_mask))
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
        mask = equivalent_mask(mask, compliant)
        next_left, next_dry, next_damp = max(0, left - 1), reduced(dry), reduced(damp)
        # Keeping this same set cannot change either its used-compound mask or
        # the remaining stop allowances. Only the surface eligibility varies.
        allowed_by_offset = stop_eligibility(compound, left, dry, damp)
        for offset in range(start + 1, horizon):
            if native_masks:
                # Resolved edges no longer suspend the generator, so retain
                # cancellation checks while scanning each future lap.
                cancellation_checkpoint()
            allowed = allowed_by_offset[offset]
            for candidate in candidates[offset]:
                if (not critical[candidate][offset] and (
                    allowed or (not compliant and not mask & bits[candidate])
                )):
                    child = (
                        offset, candidate, next_left, next_dry, next_damp, mask,
                    )
                    # A completed local suffix has exactly the value that the
                    # outer stack would send back. Preserve candidate order
                    # and arithmetic while avoiding one suspension per hit.
                    value = solved.get(child) if native_masks else None
                    if value is None:
                        value = yield child
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
                    if shared_forecast_available():
                        _remember_transition(cache_key(state), value)
                pending.pop()
                continue
            if child in solved:
                value = solved[child]
                continue
            key = cache_key(child)
            with _transition_suffix_lock:
                value = (_transition_suffixes.get(key)
                         if shared_forecast_available() else None)
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

    def first(tire_json, age, gap, *, stopped=False):
        row = _running_row(models, weather_json, tire_json, age, current_lap, physical, intervals)
        fee = (dict(warmup_profile).get(restore_model(Tire, tire_json).compound.value, 0.)
               if stopped or current_fit_pending else 0.)
        if aero and gap is None:
            if safety_car is None:
                return row[0] * modifier - row[0]
            value = current_running_time(row[0], modifier, safety_car, stopped=stopped)
            return current_fitted_time(value, fee, safety_car, stopped=stopped) - row[0] - fee
        driver = restore_model(Driver, models[0])
        driver.current_tire_laps = age
        simulator = LapSimulator(np.random.default_rng(0))
        actual = simulator.calculate_lap_time(
            driver, car, track, restore_model(Tire, tire_json), surfaces[0],
            current_lap, physical, active_aero_enabled=aero, sample_variation=False,
            gap_to_car_ahead=gap,
        )
        if safety_car is None:
            return actual * modifier - row[0]
        value = current_running_time(actual, modifier, safety_car, stopped=stopped)
        return current_fitted_time(value, fee, safety_car, stopped=stopped) - row[0] - fee

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
                    + first(fresh[candidate], 0, gaps[1] if gaps else None, stopped=True))
            if cost < pit:
                pit, compound = cost, candidate
    return RainTransitionDecision(pit, wait, compound)


@forecast_decision
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
    forecast_context=None,
    safety_car=None,
    control_context=None,
) -> RainTransitionDecision:
    """Plan bounded paid stops on a deterministic rainfall/surface projection.

    weather_intervals optionally supplies cumulative surface-update counts for
    each remaining own lap, starting at zero; None uses one update per lap.

    A retained set may run while noncritical. Every currently noncritical
    fresh compound is priced, including suboptimal alternatives, on either
    fixed rainfall or a prescribed scenario. Eligibility uses the observed
    commitment surface; an external clock's projected rejoin surface prices
    running after the stop. Critical replacements and compound corrections may exceed
    the elective stop budget. Explicit used_compounds records actual race use;
    prior tyre wear gives no credit. Two slicks or actual rain-tyre use are
    required at the finish. Slick callers must provide it; omitting it for
    retained rain tyres preserves the legacy wet exemption.
    Every fit runs its fitting lap. A usable control_context prices known
    neutralized entries and stops before resuming the existing green policy.
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
    if safety_car is not None:
        gaps = safety_car.traffic_gaps
    intervals = normalize_weather_intervals(
        track.total_laps - current_lap + 1, weather_intervals, weather=weather,
        forecast_context=forecast_context,
    )
    forecast_context = getattr(intervals, "context", forecast_context)
    horizon = track.total_laps - current_lap + 1
    _validate_weather_clock(weather_clock, horizon)
    if usable_weather_control(control_context, weather_clock):
        result = plan_controlled_weather(
            driver, car, track, weather, current_tire, tire_age, current_lap,
            min(int(remaining_stops), horizon), control_context=control_context,
            physical_total_laps=int(physical), tire_warmup=tire_warmup,
            current_fit_pending=current_fit_pending, forecast_context=forecast_context,
            used_compounds={TireCompound.WET} if used_compounds is None else used_compounds,
            remaining_dry_stops=remaining_dry_stops, remaining_damp_stops=remaining_damp_stops)
        return RainTransitionDecision(result.pit.seconds, result.wait.seconds, result.compound,
                                      pit_now_laps=result.pit.laps, wait_laps=result.wait.laps)
    if weather_clock is not None:
        # Alternative safe compounds can improve a steady-rain stint too;
        # only the explicit plan_rain_stop API keeps same-compound semantics.
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
            forecast_context=forecast_context,
            safety_car=safety_car,
        )
    clean, clean_car = strategy_projection_models(driver, car, track, weather, current_tire)
    snapshots = tuple(forecast_json(model) for model in (
        clean, clean_car, track, weather, current_tire,
    )) + (json.dumps({compound.value: forecast_dump(tire)
                     for compound, tire in TIRE_COMPOUNDS.items()}, sort_keys=True),)
    return _transition_plan(
        snapshots, int(tire_age), int(current_lap),
        min(int(remaining_stops), track.total_laps - current_lap + 1),
        float(pit_lane_factor), float(additional_current_stop_cost),
        float(current_lap_time_modifier), active_aero_enabled, int(physical),
        None if remaining_dry_stops is None else int(remaining_dry_stops),
        None if remaining_damp_stops is None else int(remaining_damp_stops), intervals, gaps, mask,
        tuple(sorted(tire_warmup.items())), current_fit_pending, safety_car,
    )


register_forecast_helpers(globals(), ('_running_row', '_fresh_future', '_surfaces'))

register_forecast_helpers(globals(), (
    "project_next_surface", "paid_compound_candidates",
    "_native_green_model_available", "_equivalent_clock_branches",
    "_shared_clock_node", "_shared_refit_cost", "_store_refit_cost",
    "_budget_clock_branches",
    "strategy_projection_models", "isolated_strategy_lap", "native_physics",
    "plan_controlled_weather", "usable_weather_control",
))
register_forecast_helpers(vars(StrategyWeatherClock), ("updates", "validate_horizon"))
register_forecast_helpers(globals(), ("current_running_time", "current_fitted_time"))
