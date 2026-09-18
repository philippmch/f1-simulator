"""Exact remaining-race planning over reusable physical tyre sets."""

from dataclasses import dataclass
from functools import lru_cache
from heapq import nsmallest
from math import inf, isfinite, nextafter
from numbers import Real

import numpy as np

from f1sim.models.tire import TIRE_COMPOUNDS, TireCompound
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.strategy_traffic import normalize_current_traffic_gaps
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.surface_projection import normalize_weather_intervals, projected_surfaces


@dataclass(frozen=True)
class InventoryDecision:
    pit_now_cost: float
    wait_cost: float
    set_id: str | None
    compound: TireCompound | None

    def should_pit(self, timing_bias=0.0):
        return self.pit_now_cost < self.wait_cost + max(-.1, min(.1, timing_bias))


def _validate_weather_clock(weather_clock, horizon):
    if weather_clock is not None:
        if not isinstance(weather_clock, StrategyWeatherClock):
            raise ValueError("weather_clock must be a StrategyWeatherClock")
        weather_clock.validate_horizon(horizon)


def _clock_inventory_strategy(
    driver, car, track, weather, inventory, current_lap, *, tire_age,
    remaining_stops, remaining_dry_stops, remaining_damp_stops, used_mask,
    pit_lane_factor, additional_current_stop_cost, current_lap_time_modifier,
    active_aero_enabled, physical_total_laps, current_traffic_gaps, force_stop,
    free_fit, require_compound_rule, weather_clock,
):
    """Exact finite-pool search whose branch surfaces include paid-stop delay."""
    horizon = track.total_laps - current_lap + 1
    driver = driver.model_copy(deep=True)
    simulator = LapSimulator(np.random.default_rng(0))
    service = expected_stationary_time(car)
    green_stop = track.pit_lane_delta + service
    current_stop = track.pit_lane_delta * pit_lane_factor + service \
        + additional_current_stop_cost
    gaps = normalize_current_traffic_gaps(current_traffic_gaps)
    bits = {compound.value: 1 << index if index < 3 else 8
            for index, compound in enumerate(TireCompound)}

    @lru_cache(maxsize=None)
    def updates(offset, paid_stops, stopped_first):
        return weather_clock.updates(offset, paid_stops, stopped_first)

    @lru_cache(maxsize=None)
    def projected_surface(updates):
        value = weather
        for _ in range(updates):
            value = value.project_surface()
        return value

    @lru_cache(maxsize=None)
    def surface(offset, paid_stops, stopped_first):
        return projected_surface(updates(offset, paid_stops, stopped_first))

    @lru_cache(maxsize=None)
    def running(offset, compound, age, first_kind, updates):
        driver.current_tire_laps = age
        value = simulator.calculate_lap_time(
            driver, car, track, TIRE_COMPOUNDS[TireCompound(compound)],
            projected_surface(updates), current_lap + offset,
            physical_total_laps, sample_variation=False,
            active_aero_enabled=(active_aero_enabled if offset == 0 else True),
            gap_to_car_ahead=(gaps[first_kind] if first_kind is not None and gaps is not None
                              else None),
        )
        return value * current_lap_time_modifier if offset == 0 else value

    def run(offset, compound, age, updates, first_kind=None):
        return running(offset, compound.value if isinstance(compound, TireCompound)
                       else compound, age, first_kind, updates)

    def legal(used):
        return (not require_compound_rule or bool(used & 8)
                or (used & 7).bit_count() >= 2)

    def canonical_used(used):
        if not require_compound_rule:
            return 0
        if used & 8:
            return 8
        slick = used & 7
        return 7 if slick.bit_count() >= 2 else slick

    def canonical_clock_state(offset, paid_stops, stopped_first):
        if offset >= horizon:
            return horizon + 1, True
        if updates(offset, paid_stops, stopped_first) == weather_clock.max_updates:
            # The update cap makes all later weather surfaces independent of
            # both the exact paid count and which first-stop delay was used.
            return horizon + 1, True
        return paid_stops, stopped_first

    def allowed(offset, compound, left, dry, damp, used, candidate, before):
        critical = before.tire_mismatch(TireCompound(compound)) == "critical"
        limit = dry if before.track_wetness < .08 and before.rain_intensity < .15 else damp
        return (critical or (left > 0 and (
            TireCompound(compound) in (TireCompound.INTERMEDIATE, TireCompound.WET)
            or before.track_wetness > .3 or limit is None or limit > 0
        )) or (not legal(used) and not used & bits[TireCompound(candidate).value]))

    def reduced(value):
        return None if value is None else max(0, value - 1)

    @lru_cache(maxsize=None)
    def retained_tail(offset, compound, age, paid_stops, stopped_first):
        """Cost of the compulsory final stint at one fixed clock position.

        Once no paid stops remain, a legal current compound can be retained
        exactly when it is valid for every remaining surface.  The paid-stop
        count is then fixed, so this sum uses the same delayed surface at each
        lap and preserves both physical wear and the external cadence.
        """
        total = 0.0
        for index in range(offset, horizon):
            before = surface(index, paid_stops, stopped_first)
            if before.tire_mismatch(TireCompound(compound)) == "critical":
                return inf
            total += run(
                index, compound, age + index - offset,
                updates(index, paid_stops, stopped_first),
            )
        return total

    @lru_cache(maxsize=None)
    def retainable(offset, compound, paid_stops, stopped_first):
        fitted = TireCompound(compound)
        return all(
            surface(index, paid_stops, stopped_first).tire_mismatch(fitted) != "critical"
            for index in range(offset, horizon)
        )

    def make_actions(state):
        offset, compound, age, pool, left, dry, damp, used, paid_stops, stopped_first = state
        before = surface(offset, paid_stops, stopped_first)
        current = TireCompound(compound)
        actions = []
        if before.tire_mismatch(current) != "critical":
            next_paid, next_stopped = canonical_clock_state(
                offset + 1, paid_stops, stopped_first,
            )
            actions.append((
                (offset + 1, current.value, age + 1, pool, left, dry, damp,
                 canonical_used(used | bits[current.value]),
                 next_paid, next_stopped),
                run(offset, current, age,
                    updates(offset, paid_stops, stopped_first)),
            ))
        previous = None
        for index, candidate in enumerate(pool):
            if candidate == previous:
                continue
            previous = candidate
            target, target_age = candidate
            if before.tire_mismatch(TireCompound(target)) == "critical":
                continue
            if not allowed(offset, compound, left, dry, damp, used, target,
                           before):
                continue
            after_paid = paid_stops + 1
            after_updates = updates(
                offset, after_paid, stopped_first,
            )
            exchanged = tuple(sorted(pool[:index] + pool[index + 1:]
                                     + ((compound, age),)))
            next_paid, next_stopped = canonical_clock_state(
                offset + 1, after_paid, stopped_first,
            )
            actions.append((
                (offset + 1, target, target_age + 1, exchanged,
                 max(0, left - 1), reduced(dry), reduced(damp),
                 canonical_used(used | bits[target]),
                 next_paid, next_stopped),
                green_stop + run(offset, target, target_age, after_updates),
            ))
        actions.sort(key=lambda action: action[1] + completion_bound(action[0]))
        return actions

    solve_cache = {}

    def solve(initial):
        """Evaluate the finite-pool strategy DAG without recursion."""
        if initial in solve_cache:
            return solve_cache[initial]
        frames = [[initial, None, 0, inf]]
        while frames:
            state, actions, index, best = frames[-1]
            if state in solve_cache:
                frames.pop()
                continue
            (offset, compound, age, pool, left, dry, damp, used,
             paid_stops, stopped_first) = state
            if actions is None:
                if offset >= horizon:
                    # Keep the terminal value in the frame so the common
                    # completion path can add its incoming edge.
                    frames[-1][1:] = [[], 0, 0.0 if legal(used) else inf]
                    continue
                if (left == 0 and legal(used)
                        and retainable(offset, compound, paid_stops, stopped_first)):
                    frames[-1][1:] = [[], 0, retained_tail(
                        offset, compound, age, paid_stops, stopped_first)]
                    continue
                actions = make_actions(state)
                frames[-1][1:] = [actions, 0, inf]
                continue
            if index < len(actions):
                child, edge = actions[index]
                frames[-1][2] += 1
                if child in solve_cache:
                    frames[-1][3] = min(frames[-1][3], edge + solve_cache[child])
                elif nextafter(edge + completion_bound(child), -inf) < frames[-1][3]:
                    frames.append([child, None, 0, inf])
                continue
            solve_cache[state] = best
            frames.pop()
            if frames:
                parent = frames[-1]
                child, edge = parent[1][parent[2] - 1]
                parent[3] = min(parent[3], edge + solve_cache[state])
        return solve_cache[initial]

    current_id = inventory.current_set_id
    current = inventory.sets.get(current_id)
    usable_current = current is not None and current_id not in inventory.unavailable_ids
    stock = tuple(inventory.replacements())
    pool = tuple(sorted((item.compound.value, item.age) for item in stock))

    # A branch's paid-stop count changes its future weather surfaces.  Build a
    # relaxation over every clock surface that can occur in this horizon; the
    # minimum at each lap is therefore no greater than the branch's actual
    # surface cost.  The conserved-wear helper then also relaxes set order and
    # stopping, while retaining the exact one-use-per-physical-set constraint.
    # Stops beyond the elective budget must repair legality or a critical
    # mismatch. At most two new slick compounds complete the former. After
    # the initial forced entry, every mismatch repair requires a previously
    # eligible compound to become critical on the advancing weather path.
    # Counting all such transitions (even for unavailable sets) is an upper
    # bound, including transitions skipped over while servicing a stop.
    critical_changes = 0
    previous = {compound: weather.tire_mismatch(compound) == "critical"
                for compound in TireCompound}
    for update in range(1, weather_clock.max_updates + 1):
        projected = projected_surface(update)
        for compound in TireCompound:
            critical = projected.tire_mismatch(compound) == "critical"
            critical_changes += critical and not previous[compound]
            previous[compound] = critical
    rule_stops = 0 if legal(used_mask) else 2 - (used_mask & 7).bit_count()
    max_paid_stops = remaining_stops + rule_stops + 1 + critical_changes
    clock_surfaces = {}
    for offset in range(horizon):
        clock_surfaces[offset] = tuple({
            updates(offset, paid_stops, stopped_first)
            # A strategy can pay at most once per started lap, including the
            # current one.  ``stopped_first=False`` additionally means that
            # the first paid stop was on a later lap, so it cannot have as
            # many stops as a branch whose first stop was on lap zero.  Keep
            # only these reachable clock states in the relaxation.
            for paid_stops in range(min(offset + 1, max_paid_stops) + 1)
            for stopped_first in (
                (False,) if paid_stops == 0 else
                ((False, True) if paid_stops <= offset else (True,))
            )
        })

    @lru_cache(maxsize=None)
    def lower_running(offset, compound, age):
        return min(
            run(offset, compound, age, updates)
            for updates in clock_surfaces[offset]
        )

    initial_ages = tuple((item.compound.value,
                          tire_age if item.id == inventory.current_set_id else item.age)
                         for item in inventory.sets.values()
                         if item.id not in inventory.unavailable_ids)
    lower_critical = {compound.value: (False,) * horizon for compound in TireCompound}

    @lru_cache(maxsize=1)
    def lower_bounds():
        value = _conserved_wear_lower_bounds(
            horizon, initial_ages, lower_critical, lower_running,
        )
        return value

    completion_rows = {}

    def completion_bound(state):
        """Relax everything after the first stop, retaining its unavoidable cost.

        Before its first future stop, a completion must run this physical tyre
        at its actual age on the unchanged paid-stop timeline. At the stop we
        charge one green pit entry, then use the optimistic conserved-wear
        bound for all remaining running. We also allow retaining to the end
        when legal. Ignoring replacement availability and all subsequent stop
        costs only lowers this bound; no age-monotonicity assumption is needed.
        """
        offset, compound, age, _pool, _left, _dry, _damp, used, paid, stopped = state
        compliant = legal(used)
        if offset >= horizon:
            return 0.0 if compliant else inf
        base_age = age - offset
        key = compound, base_age, paid, stopped, compliant
        if key not in completion_rows:
            completion_rows[key] = [horizon, [None] * horizon + [0.0 if compliant else inf]]
        first, row = completion_rows[key]
        for index in range(first - 1, offset - 1, -1):
            best = green_stop + lower_bounds()[index]
            if surface(index, paid, stopped).tire_mismatch(TireCompound(compound)) != "critical":
                stay = run(index, compound, base_age + index, updates(index, paid, stopped))
                best = min(best, stay + row[index + 1])
            row[index] = nextafter(best, -inf)
        completion_rows[key][0] = min(first, offset)
        return max(lower_bounds()[offset], row[offset])

    def initial_cost(item, available, charge, consume, first_kind, paid_stops,
                     stopped_first, age_override=None):
        compound = item.compound.value
        age = item.age if age_override is None else age_override
        before = surface(0, paid_stops, stopped_first)
        if before.tire_mismatch(item.compound) == "critical":
            return inf
        after_updates = updates(0, paid_stops, stopped_first)
        if consume:
            after_updates = updates(0, paid_stops + 1, True)
        next_stopped = stopped_first or bool(consume)
        next_paid, next_stopped = canonical_clock_state(
            1, paid_stops + consume, next_stopped,
        )
        state = (1, compound, age + 1, available,
                 max(0, remaining_stops - consume),
                 reduced(remaining_dry_stops) if consume else remaining_dry_stops,
                 reduced(remaining_damp_stops) if consume else remaining_damp_stops,
                 canonical_used(used_mask | bits[compound]), next_paid, next_stopped)
        return (charge + run(
                    0, compound, age,
                    after_updates,
                    first_kind,
                )
                + solve(state))

    wait = inf
    if usable_current and (free_fit or not force_stop):
        wait = initial_cost(current, pool, 0.0, 0, 0, 0, False, tire_age)

    best, selected = inf, None
    choices = ((current,) if free_fit and usable_current else ()) + stock
    for item in choices:
        if item.id == current_id:
            cost = wait
        else:
            if not free_fit and not force_stop and usable_current and not allowed(
                0, current.compound.value, remaining_stops, remaining_dry_stops,
                remaining_damp_stops, used_mask, item.compound.value,
                surface(0, 0, False),
            ):
                continue
            candidate = (item.compound.value, item.age)
            index = pool.index(candidate)
            available = pool[:index] + pool[index + 1:]
            if usable_current:
                available = tuple(sorted(available + ((current.compound.value, tire_age),)))
            consume = 0 if free_fit else 1
            charge = 0.0 if free_fit else current_stop
            # A free fit still starts in the same occupied lane as the
            # retained branch.  It consumes no pit time, but it must keep
            # the observed traffic-gap semantics for its first lap.
            cost = initial_cost(item, tuple(available), charge, consume,
                                1 if not free_fit else 0, 0, False)
        if cost < best:
            best, selected = cost, item
    return InventoryDecision(best, wait, selected.id if selected else None,
                             selected.compound if selected else None)


def _conserved_wear_lower_bounds(horizon, initial_ages, critical, running):
    """Bound future running cost while retaining unique physical-set/age uses.

    Each lap's cheapest reachable cost is its baseline. A potential use gets
    its cheapest excess above baseline over all eligible future laps. Taking
    the cheapest distinct uses relaxes their order, earlier consumption and
    all stop costs, but cannot reuse the same physical set at the same age.
    Duplicate sets retain separate uses; identical running costs can be cached
    by the caller. No assumption about age monotonicity is needed.
    """
    lower = [0.] * (horizon + 1)
    residuals = [inf] * (len(initial_ages) * horizon)
    baseline = 0.
    for offset in range(horizon - 1, 0, -1):
        eligible = [(index * horizon + elapsed, running(offset, compound, age + elapsed))
                    for index, (compound, age) in enumerate(initial_ages)
                    if not critical[compound][offset]
                    for elapsed in range(offset + 1)]
        minimum = min((cost for _, cost in eligible), default=inf)
        if minimum == inf or baseline == inf:
            baseline = lower[offset] = inf
            continue
        baseline = nextafter(minimum + baseline, -inf)
        for slot, cost in eligible:
            residuals[slot] = min(residuals[slot], nextafter(cost - minimum, -inf))
        extra = 0.
        for value in nsmallest(horizon - offset, residuals):
            extra = nextafter(extra + value, -inf)
        lower[offset] = max(baseline, nextafter(baseline + extra, -inf))
    return tuple(lower)


def plan_inventory_strategy(
    driver, car, track, weather, inventory, current_lap, *, tire_age=0,
    remaining_stops=3, remaining_dry_stops=None, remaining_damp_stops=None,
    used_compounds=(), pit_lane_factor=1., additional_current_stop_cost=0.,
    current_lap_time_modifier=1., active_aero_enabled=True, physical_total_laps=None,
    weather_intervals=None, current_traffic_gaps=None, force_stop=False, free_fit=False,
    require_compound_rule=True, weather_clock=None,
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
    _validate_weather_clock(weather_clock, horizon)
    gaps = normalize_current_traffic_gaps(current_traffic_gaps)
    driver = driver.model_copy(deep=True)
    simulator = LapSimulator(np.random.default_rng(0))
    bits = {compound.value: 1 << index if index < 3 else 8
            for index, compound in enumerate(TireCompound)}
    mask = 0
    for compound in used_compounds:
        mask |= bits[TireCompound(compound).value]
    if weather_clock is not None:
        return _clock_inventory_strategy(
            driver, car, track, weather, inventory, current_lap,
            tire_age=tire_age, remaining_stops=remaining_stops,
            remaining_dry_stops=remaining_dry_stops,
            remaining_damp_stops=remaining_damp_stops, used_mask=mask,
            pit_lane_factor=float(pit_lane_factor),
            additional_current_stop_cost=float(additional_current_stop_cost),
            current_lap_time_modifier=float(current_lap_time_modifier),
            active_aero_enabled=active_aero_enabled,
            physical_total_laps=physical, current_traffic_gaps=current_traffic_gaps,
            force_stop=force_stop, free_fit=free_fit,
            require_compound_rule=require_compound_rule, weather_clock=weather_clock,
        )
    surfaces = tuple(projected_surfaces(weather, horizon, intervals))
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

    # A physical set can run each completed age only once, even when removed
    # and refitted. Keep multiplicity when constructing those potential uses.
    initial_ages = tuple((item.compound.value,
                          tire_age if item.id == inventory.current_set_id else item.age)
                         for item in inventory.sets.values()
                         if item.id not in inventory.unavailable_ids)
    @lru_cache(maxsize=1)
    def lower_bounds():
        # Plans without eligible future replacements need no relaxation table.
        return _conserved_wear_lower_bounds(horizon, initial_ages, critical, running)

    def exchange(pool, index, current):
        return tuple(sorted(pool[:index] + pool[index + 1:] + (current,)))

    solved = {}

    @lru_cache(maxsize=None)
    def retained_tail(offset, compound, age):
        # Different exhausted pools can leave the same compulsory final
        # stint. Its cost depends only on this call's lap, compound and age.
        # Keep the original reverse summation order for identical rounding.
        total = 0.
        for number in range(horizon - 1, offset - 1, -1):
            total = running(number, compound, age + number - offset) + total
        return total

    def frame(state):
        offset, compound, age, pool, left, dry, damp, used = state
        if offset == horizon:
            return 0.0 if legal(used) else inf
        if left == 0 and legal(used) and not any(critical[compound][offset:]):
            return retained_tail(offset, compound, age)
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
