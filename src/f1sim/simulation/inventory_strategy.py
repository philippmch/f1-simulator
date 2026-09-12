"""Exact remaining-race planning over reusable physical tyre sets."""

from dataclasses import dataclass
from functools import lru_cache
from math import inf, isfinite, nextafter
from numbers import Real

import numpy as np

from f1sim.models.tire import TIRE_COMPOUNDS, TireCompound
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.strategy_traffic import normalize_current_traffic_gaps
from f1sim.simulation.surface_projection import normalize_weather_intervals, projected_surfaces


@dataclass(frozen=True)
class InventoryDecision:
    pit_now_cost: float
    wait_cost: float
    set_id: str | None
    compound: TireCompound | None

    def should_pit(self, timing_bias=0.0):
        return self.pit_now_cost < self.wait_cost + max(-.1, min(.1, timing_bias))


def plan_inventory_strategy(
    driver, car, track, weather, inventory, current_lap, *, tire_age=0,
    remaining_stops=3, remaining_dry_stops=None, remaining_damp_stops=None,
    used_compounds=(), pit_lane_factor=1., additional_current_stop_cost=0.,
    current_lap_time_modifier=1., active_aero_enabled=True, physical_total_laps=None,
    weather_intervals=None, current_traffic_gaps=None, force_stop=False, free_fit=False,
    require_compound_rule=True,
):
    """Minimize deterministic total time without inventing or freshening sets.

    The anonymous future pool retains compound, age and multiplicity. Only
    interchangeable IDs are merged. Local memoization is confined to this
    call, so every model, surface and cadence is intrinsically in its context.
    """
    for name, value in (("current_lap", current_lap), ("tire_age", tire_age),
                        ("remaining_stops", remaining_stops),
                        ("remaining_dry_stops", remaining_dry_stops),
                        ("remaining_damp_stops", remaining_damp_stops)):
        if value is None and name in ("remaining_dry_stops", "remaining_damp_stops"):
            continue
        if type(value) is not int or value < (1 if name == "current_lap" else 0):
            raise ValueError(f"{name} must be a nonnegative integer")
    if current_lap > track.total_laps:
        raise ValueError("current_lap exceeds planning distance")
    physical = track.total_laps if physical_total_laps is None else physical_total_laps
    if type(physical) is not int or physical < track.total_laps:
        raise ValueError("physical_total_laps must cover planning distance")
    for name, value in (("pit_lane_factor", pit_lane_factor),
                        ("additional_current_stop_cost", additional_current_stop_cost),
                        ("current_lap_time_modifier", current_lap_time_modifier)):
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
            raise ValueError(f"{name} must be finite")
        if name != "additional_current_stop_cost" and value < 0:
            raise ValueError(f"{name} must be nonnegative")
    if current_lap_time_modifier == 0:
        raise ValueError("current_lap_time_modifier must be positive")
    for value in (active_aero_enabled, force_stop, free_fit, require_compound_rule):
        if type(value) is not bool:
            raise ValueError("inventory strategy flags must be booleans")
    horizon = track.total_laps - current_lap + 1
    intervals = normalize_weather_intervals(horizon, weather_intervals, weather=weather)
    surfaces = tuple(projected_surfaces(weather, horizon, intervals))
    gaps = normalize_current_traffic_gaps(current_traffic_gaps)
    driver = driver.model_copy(deep=True)
    simulator = LapSimulator(np.random.default_rng(0))
    bits = {compound.value: 1 << index if index < 3 else 8
            for index, compound in enumerate(TireCompound)}
    mask = 0
    for compound in used_compounds:
        mask |= bits[TireCompound(compound).value]
    green_stop = track.pit_lane_delta + expected_stationary_time(car)
    current_stop = (track.pit_lane_delta * pit_lane_factor + expected_stationary_time(car)
                    + additional_current_stop_cost)

    def legal(used):
        return not require_compound_rule or bool(used & 8) or (used & 7).bit_count() >= 2

    critical = {c.value: tuple(s.tire_mismatch(c) == "critical" for s in surfaces)
                for c in TireCompound}

    def allowed(offset, compound, left, dry, damp, used, candidate):
        limit = dry if (surfaces[offset].track_wetness < .08
                        and surfaces[offset].rain_intensity < .15) else damp
        return (critical[compound][offset]
                or (left > 0 and (compound in ("wet", "intermediate")
                    or surfaces[offset].track_wetness > .3 or limit is None or limit > 0))
                or (not legal(used) and not used & bits[candidate]))

    def reduced(value):
        return None if value is None else max(0, value - 1)

    @lru_cache(maxsize=None)
    def running(offset, compound, age, first_kind=None):
        first = first_kind is not None
        driver.current_tire_laps = age
        value = simulator.calculate_lap_time(
            driver, car, track, TIRE_COMPOUNDS[TireCompound(compound)], surfaces[offset],
            current_lap + offset, physical, sample_variation=False,
            active_aero_enabled=active_aero_enabled if first else True,
            gap_to_car_ahead=gaps[first_kind] if first and gaps is not None else None,
        )
        return value * current_lap_time_modifier if first else value

    # Relax physical inventory conservation and all stop charges: at each
    # future lap independently choose any age that any undamaged initial set
    # could have reached. This is an optimistic lower bound, never a schedule.
    initial_ages = tuple((item.compound.value,
                          tire_age if item.id == inventory.current_set_id else item.age)
                         for item in inventory.sets.values()
                         if item.id not in inventory.unavailable_ids)
    @lru_cache(maxsize=1)
    def lower_bounds():
        # Plans without eligible future replacements need no relaxation table.
        lower = [0.] * (horizon + 1)
        for offset in range(horizon - 1, 0, -1):
            minimum = min((running(offset, compound, age + elapsed)
                           for compound, age in initial_ages if not critical[compound][offset]
                           for elapsed in range(offset + 1)), default=inf)
            lower[offset] = nextafter(minimum + lower[offset + 1], -inf)
        return tuple(lower)

    def exchange(pool, index, current):
        return tuple(sorted(pool[:index] + pool[index + 1:] + (current,)))

    solved = {}

    def frame(state):
        offset, compound, age, pool, left, dry, damp, used = state
        if offset == horizon:
            return 0.0 if legal(used) else inf
        if left == 0 and legal(used) and not any(critical[compound][offset:]):
            total = 0.
            for number in range(horizon - 1, offset - 1, -1):
                total = running(number, compound, age + number - offset) + total
            return total
        best = inf
        if not critical[compound][offset]:
            cost = running(offset, compound, age)
            child = (offset + 1, compound, age + 1, pool, left, dry, damp,
                     used | bits[compound])
            best = cost + (yield child)
        previous = None
        for index, candidate in enumerate(pool):
            if candidate == previous:
                continue
            previous = candidate
            target, target_age = candidate
            if critical[target][offset] or not allowed(
                offset, compound, left, dry, damp, used, target,
            ):
                continue
            child = (offset + 1, target, target_age + 1,
                     exchange(pool, index, (compound, age)), max(0, left - 1),
                     reduced(dry), reduced(damp), used | bits[target])
            cost = green_stop + running(offset, target, target_age)
            if nextafter(cost + lower_bounds()[offset + 1], -inf) < best:
                best = min(best, cost + (yield child))
        return best

    def solve(initial):
        if initial in solved:
            return solved[initial]
        stack = [(initial, frame(initial))]
        value = None
        while stack:
            state, generator = stack[-1]
            try:
                child = generator.send(value)
            except StopIteration as result:
                value = result.value
                solved[state] = value
                stack.pop()
                continue
            if child in solved:
                value = solved[child]
            else:
                stack.append((child, frame(child)))
                value = None
        return value

    current_id = inventory.current_set_id
    current = inventory.sets.get(current_id)
    usable_current = current is not None and current_id not in inventory.unavailable_ids
    stock = tuple(inventory.replacements())
    pool = tuple(sorted((item.compound.value, item.age) for item in stock))

    def initial_cost(item, age, available, charge, consume, kind):
        compound = item.compound.value
        if critical[compound][0]:
            return inf
        state = (1, compound, age + 1, available,
                 max(0, remaining_stops - consume),
                 reduced(remaining_dry_stops) if consume else remaining_dry_stops,
                 reduced(remaining_damp_stops) if consume else remaining_damp_stops,
                 mask | bits[compound])
        return charge + running(0, compound, age, kind) + solve(state)

    wait = inf
    if usable_current and (free_fit or not force_stop):
        wait = initial_cost(current, tire_age, pool, 0., 0, 0)
    best, selected = inf, None
    choices = ((current,) if free_fit and usable_current else ()) + stock
    for item in choices:
        if item.id == current_id:
            cost = wait
        else:
            if not free_fit and not force_stop and usable_current and not allowed(
                0, current.compound.value, remaining_stops, remaining_dry_stops,
                remaining_damp_stops, mask, item.compound.value,
            ):
                continue
            candidate = (item.compound.value, item.age)
            index = pool.index(candidate)
            available = pool[:index] + pool[index + 1:]
            if usable_current:
                available = tuple(sorted(available + ((current.compound.value, tire_age),)))
            cost = initial_cost(item, item.age, available, 0. if free_fit else current_stop,
                                0 if free_fit else 1, 0 if free_fit else 1)
        if cost < best:
            best, selected = cost, item
    return InventoryDecision(best, wait, selected.id if selected else None,
                             selected.compound if selected else None)
