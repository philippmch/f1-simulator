"""Exact remaining-race planning over reusable physical tyre sets."""

from dataclasses import dataclass
from functools import lru_cache
from heapq import nsmallest
from math import inf, isfinite, nextafter, ulp
from numbers import Real

import numpy as np

from f1sim.cancellation import cancellation_checkpoint
from f1sim.models._native import forecast_json, native_physics, register_forecast_helpers
from f1sim.models.tire import TIRE_COMPOUNDS, TireCompound
from f1sim.simulation.controlled_weather_strategy import (
    _green_weather_clock_key,
    plan_controlled_weather,
    usable_weather_control,
)
from f1sim.simulation.lap import LapSimulator, minimum_lap_time
from f1sim.simulation.pit_strategy import SLICKS, _floor_tables, expected_stationary_time
from f1sim.simulation.strategy_control_clock import (
    ObservedStandardField,
    ProjectedControlCost,
    StrategyControlContext,
    observed_control_key,
)
from f1sim.simulation.strategy_lap import (
    control_lap_memo,
    control_relaxation_memo,
    control_wear_bound,
    isolated_strategy_lap,
    memoized_control_lap,
)
from f1sim.simulation.strategy_neutralization import current_fitted_time, current_running_time
from f1sim.simulation.strategy_traffic import normalize_current_traffic_gaps
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.surface_projection import normalize_weather_intervals, projected_surfaces
from f1sim.simulation.tire_inventory import (
    exchange_tire_slots,
    tire_set_slot,
    tire_slot_usable,
)
from f1sim.simulation.warmup import tire_warmup_seconds, validate_tire_warmup
from f1sim.simulation.weather_schedule import project_next_surface


@dataclass(frozen=True)
class InventoryDecision:
    pit_now_cost: float
    wait_cost: float
    set_id: str | None
    compound: TireCompound | None
    pit_now_laps: int | None = None
    wait_laps: int | None = None
    pit_now_partial_time: float = inf
    wait_partial_time: float = inf

    @classmethod
    def from_continuations(cls, pit, wait, set_id, compound):
        return cls(pit.cost, wait.cost, set_id, compound, pit.laps, wait.laps,
                   pit.seconds if not pit.finished else inf,
                   wait.seconds if not wait.finished else inf)

    def continuation(self, stopped):
        """Recover the full ranking when a controlled prefix delegates to green."""
        cost = self.pit_now_cost if stopped else self.wait_cost
        laps = self.pit_now_laps if stopped else self.wait_laps
        elapsed = self.pit_now_partial_time if stopped else self.wait_partial_time
        return ProjectedControlCost(-1 if laps is None else laps,
                                    cost if isfinite(cost) else elapsed, isfinite(cost))

    def should_pit(self, timing_bias=0.0):
        if isfinite(self.pit_now_cost) != isfinite(self.wait_cost):
            return isfinite(self.pit_now_cost)
        if (self.pit_now_laps is not None and self.wait_laps is not None
                and self.pit_now_laps != self.wait_laps):
            return self.pit_now_laps > self.wait_laps
        pit = self.pit_now_cost if isfinite(self.pit_now_cost) else self.pit_now_partial_time
        wait = self.wait_cost if isfinite(self.wait_cost) else self.wait_partial_time
        return pit < wait + max(-.1, min(.1, timing_bias))


def _validate_weather_clock(weather_clock, horizon):
    if weather_clock is not None:
        if not isinstance(weather_clock, StrategyWeatherClock):
            raise ValueError("weather_clock must be a StrategyWeatherClock")
        weather_clock.validate_horizon(horizon)


def _expired_inventory_state(state):
    """Forget an exhausted active slot, preserving all usable future stock.

    Once its allowance is exhausted, the active set cannot run or return to
    the pool. Its compound and wear no longer affect any future action; used
    compounds, stop budgets and clocks remain separate parts of the state.
    Native searches can therefore share the suffix across different last
    exhausted sets. Keep a valid, explicitly expired slot for existing checks.
    """
    if not tire_slot_usable(state[2], state[-1]):
        return (state[0], "soft", 0, *state[3:-1], 0)
    return state


def _dead_inventory_compounds(critical):
    """Identify stock that has no eligible entry anywhere in a future suffix."""
    horizon = len(next(iter(critical.values())))
    rows = [frozenset()] * (horizon + 1)
    dead = set(critical)
    for offset in range(horizon - 1, -1, -1):
        cancellation_checkpoint()
        dead = {name for name in dead if critical[name][offset]}
        rows[offset] = frozenset(dead)
    return tuple(rows)


def _canonical_inventory_state(state, horizon, require_rule, dead=()):
    """Share native suffixes only when the removed facts affect no future action."""
    offset, used = state[0], state[7]
    if offset >= horizon:
        # At the boundary only compound compliance survives. Incoming running,
        # service and fitting charges already belong to the parent edge.
        compliant = not require_rule or bool(used & 8) or (used & 7).bit_count() >= 2
        used = 7 if require_rule and compliant else 0
        clock = (horizon + 1, True, 0.) if len(state) == 12 else ()
        return (offset, "soft", 0, (), 0, 0, 0, used, *clock, 0)
    credit = (0 if not require_rule else 8 if used & 8 else
              7 if (used & 7).bit_count() >= 2 else used & 7)
    if not dead and credit == used:
        return _expired_inventory_state(state)
    _, compound, age, pool, left, dry, damp, _, *tail = state
    if dead:
        pool = tuple(slot for slot in pool if slot[0] not in dead)
        if compound in dead:
            # This tyre can neither run nor be fitted again. An expired
            # placeholder preserves the compulsory-entry checks without
            # returning that impossible set to the replacement stock.
            compound, age, tail[-1] = "soft", 0, 0
    if (compound, age, pool, credit, tuple(tail)) != (
            state[1], state[2], state[3], state[7], state[8:]):
        state = (offset, compound, age, pool, left, dry, damp, credit, *tail)
    return _expired_inventory_state(state)


def _bounded_inventory_suffix(initial, solved, excluded, actions_for, terminal):
    """Carry an incumbent through a native DAG without caching truncated values.

    An excluded branch only proves a lower bound on a legal finish. Its partial
    distance is unknown, so pruning starts only after a finish has been found.
    Exact results and excluded-cost bounds have separate decision-local caches.
    """
    if initial in solved:
        return solved[initial]
    retired = ProjectedControlCost(0, 0., False)
    # state, actions, next action, best, incoming cutoff, omitted lower bound,
    # whether any action was omitted. Each parent keeps its incoming edge.
    frames = [[initial, None, 0, retired, inf, inf, False]]
    while frames:
        cancellation_checkpoint()
        state, actions, index, best, cutoff, ignored, omitted = frames[-1]
        if actions is None:
            value = terminal(state)
            if value is not None:
                frames[-1][1:4] = [[], 0, value]
            else:
                frames[-1][1] = actions_for(state)
            continue
        if index < len(actions):
            child, edge, bound = actions[index]
            frames[-1][2] += 1
            if child in solved:
                option = solved[child].prepend_lap(edge)
                if option.rank > best.rank:
                    frames[-1][3] = option
                continue
            limit = min(cutoff, best.seconds if best.finished else inf)
            lower = nextafter(edge + max(bound, excluded.get(child, -inf)), -inf)
            if isfinite(limit) and lower >= limit:
                frames[-1][5] = min(ignored, lower)
                frames[-1][6] = True
                continue
            child_cutoff = nextafter(limit - edge, inf) if isfinite(limit) else inf
            frames.append([child, None, 0, retired, child_cutoff, inf, False])
            continue
        exact = not omitted or best.finished and best.seconds <= ignored
        if exact:
            solved[state] = best
        else:
            # This is a bound, not a fabricated continuation. A later visit
            # with a looser cutoff must still search for the exact result.
            excluded[state] = max(excluded.get(state, -inf), ignored)
        frames.pop()
        if frames:
            parent = frames[-1]
            _child, edge, _bound = parent[1][parent[2] - 1]
            option = best.prepend_lap(edge)
            if option.rank > parent[3].rank:
                parent[3] = option
            if not exact:
                parent[5] = min(parent[5], nextafter(edge + ignored, -inf))
                parent[6] = True
    # The root has no incoming cutoff: every omitted branch is bounded by
    # a completion found within this root, so its final value is exact.
    return solved[initial]


def _fresh_inventory_completion_bound(horizon, compounds, running, eligible,
                                      advance, canonical, green_stop, *, solved=None,
                                      fitted_running=None, fitted_advance=None, max_work=None):
    """Relax physical stock to unlimited fresh sets, retaining paid weather time.

    Native wear cannot improve a set over a fresh copy. Allowing every future
    replacement fresh stock, ignoring stop limits and compound compliance,
    therefore lowers completion cost. Every fitted stint still ages, pays its
    entry. Optional fitting delays may relax later weather observations; the
    retained physical set keeps its exact timeline. This is never an executable plan.
    A work limit may return zero for unfinished queries when all costs are
    nonnegative. Only fully solved service costs enter the shared table.
    """
    work = 0
    if solved is None:
        solved = {}

    def service(offset, clock):
        nonlocal work
        if offset >= horizon:
            return 0.
        initial = offset, canonical(offset, clock)
        if initial in solved:
            return solved[initial]
        # Every dependency advances the own-lap offset. Resume suspended
        # stint evaluations explicitly so long clocks do not grow either the
        # Python or native call stack through recursive cache wrappers.
        frames = [(initial, service_frame(*initial))]
        value = None
        while frames:
            cancellation_checkpoint()
            if max_work is not None:
                if work >= max_work:
                    return 0.
                work += 1
            key, frame = frames[-1]
            try:
                child = frame.send(value)
            except StopIteration as result:
                value = solved[key] = result.value
                frames.pop()
                continue
            if child in solved:
                value = solved[child]
            else:
                frames.append((child, service_frame(*child)))
                value = None
        return solved[initial]

    def service_frame(offset, clock):
        best = inf
        after = advance(clock)
        fitted_clock = after if fitted_advance is None else fitted_advance(after)
        for compound in compounds:
            if not eligible(offset, compound, clock):
                continue
            total = nextafter(green_stop, -inf)
            for number in range(offset, horizon):
                cancellation_checkpoint()
                # Entry eligibility precedes service; subsequent retained
                # entries see its paid delay, just as in the physical search.
                active = after if number == offset else fitted_clock
                if number > offset and not eligible(number, compound, active):
                    break
                evaluate = (fitted_running if number == offset and fitted_running is not None
                            else running)
                total = nextafter(total + evaluate(number, compound, number - offset, active), -inf)
                suffix = (0. if number + 1 == horizon else
                          (yield (number + 1, canonical(number + 1, fitted_clock))))
                best = min(best, nextafter(total + suffix, -inf))
        return best

    @lru_cache(maxsize=None)
    def bound(offset, compound, age, clock, expiry):
        best = service(offset, clock)
        if max_work is not None and best == 0.:
            return 0.
        total = 0.
        for number in range(offset, horizon):
            cancellation_checkpoint()
            current_age = age + number - offset
            if not tire_slot_usable(current_age, expiry) or not eligible(number, compound, clock):
                break
            total = nextafter(total + running(number, compound, current_age, clock), -inf)
            best = min(best, nextafter(total + service(number + 1, clock), -inf))
        return best

    return bound


def _fitting_inventory_completion_bound(horizon, compounds, running, updates, critical,
                                        canonical, warmup, green_stop, *, solved=None):
    """Price fresh service while bounding every future compound-specific delay.

    Retained clocks have one exact fitting delay. Fresh clocks carry its lower
    and upper endpoints. Each new fit charges its compound's actual fee, then
    widens later entries by the smallest/largest fee across all fresh compounds.
    This relaxation shares no raw future fitting histories or physical stock.
    """
    fit_floor = min(tire_warmup_seconds(warmup, compound) for compound in compounds)
    fit_ceiling = max(warmup.values(), default=0.)

    @lru_cache(maxsize=None)
    def service_surfaces(offset, clock):
        paid, stopped, low, *upper = clock
        high = upper[0] if upper else low
        return range(updates(offset, paid, stopped, low),
                     updates(offset, paid, stopped, high) + 1)

    def service_running(offset, compound, age, clock):
        return min(running(offset, compound, age, update)
                   for update in service_surfaces(offset, clock))

    def service_clock(offset, clock):
        paid, stopped, low, *upper = clock
        value = canonical(offset, paid, stopped, low)
        if value[0] >= horizon + 1:
            return (*value, 0.)
        return paid, stopped, low, upper[0] if upper else low

    return _fresh_inventory_completion_bound(
        horizon, compounds, service_running,
        lambda offset, compound, clock: any(not critical(update, compound)
                                            for update in service_surfaces(offset, clock)),
        lambda clock: (clock[0] + 1, *clock[1:]), service_clock, green_stop,
        solved=solved, max_work=(4 * horizon * horizon if fit_floor != fit_ceiling else None),
        fitted_running=lambda offset, compound, age, clock:
            service_running(offset, compound, age, clock) + tire_warmup_seconds(warmup, compound),
        fitted_advance=lambda clock:
            (*clock[:2], clock[2] + fit_floor, clock[3] + fit_ceiling))


def _inventory_clock_surfaces(horizon, clock, warmup, max_paid_stops, *, native=False):
    """Bound reachable entry surfaces without constraining any physical plan.

    Each previous own lap can contribute at most one fitting cost. Repeated
    addition of the largest configured fee bounds every actual cumulative fee,
    including a pending or free initial fit. The current entry sees no new fee.
    Preserve the full surface envelope for custom clocks and larger horizons.
    """
    rows = {}
    bounded_fit = bool(warmup) and native and horizon <= 100
    max_fit_delay = 0.
    max_fit_cost = max(warmup.values(), default=0.)
    for offset in range(horizon):
        cancellation_checkpoint()
        if warmup and not bounded_fit:
            rows[offset] = tuple(range(clock.max_updates + 1))
            continue
        observations = set()
        for paid in range(min(offset + 1, max_paid_stops) + 1):
            for stopped in ((False,) if paid == 0 else
                            ((False, True) if paid <= offset else (True,))):
                first = clock.updates(offset, paid, stopped)
                last = (clock.updates(offset, paid, stopped, fit_delay=max_fit_delay)
                        if bounded_fit else first)
                # Every integer event count between the endpoint delays is a
                # superset of all compound-specific fitting histories.
                observations.update(range(first, last + 1))
        rows[offset] = tuple(observations)
        max_fit_delay += max_fit_cost
    return rows


def _clock_inventory_strategy(
    driver, car, track, weather, inventory, current_lap, *, tire_age,
    remaining_stops, remaining_dry_stops, remaining_damp_stops, used_mask,
    pit_lane_factor, additional_current_stop_cost, current_lap_time_modifier,
    active_aero_enabled, physical_total_laps, current_traffic_gaps, force_stop,
    free_fit, require_compound_rule, weather_clock, tire_warmup,
    current_fit_pending, forecast_context=None, safety_car=None,
):
    """Exact finite-pool search whose branch surfaces include paid-stop delay."""
    horizon = track.total_laps - current_lap + 1
    driver = driver.model_copy(deep=True)
    simulator = LapSimulator(np.random.default_rng(0))
    prepared_lap_time = simulator.prepare_deterministic_lap_time(
        driver, car, track, physical_total_laps,
    )
    native = prepared_lap_time is not None and native_physics(driver, car, track, weather)
    native_clock = native and type(weather_clock) is StrategyWeatherClock
    constant_surface = (native and type(weather_clock) is StrategyWeatherClock
                        and forecast_context is None
                        and weather.track_wetness == weather.rain_intensity)
    absorbing_update = None
    dead_stock = None
    shared_laps = (control_lap_memo(driver, car, track, physical_total_laps)
                   if native else None)
    service = expected_stationary_time(car)
    green_stop = track.pit_lane_delta + service
    current_stop = track.pit_lane_delta * pit_lane_factor + service \
        + additional_current_stop_cost
    gaps = normalize_current_traffic_gaps(current_traffic_gaps)
    if safety_car is not None:
        gaps = safety_car.traffic_gaps
    bits = {compound.value: 1 << index if index < 3 else 8
            for index, compound in enumerate(TireCompound)}

    warmup = tire_warmup
    invalid = ProjectedControlCost(-1, inf)
    retired = ProjectedControlCost(0, 0., False)

    @lru_cache(maxsize=None)
    def updates(offset, paid_stops, stopped_first, fit_delay=0.0):
        # canonical_clock_state uses this reserved count as a saturated-clock
        # sentinel. Do not recompute elapsed time from it: the sentinel has
        # already recorded that every remaining entry sees the update cap,
        # including when a large fitting fee caused that saturation.
        if paid_stops >= horizon + 1:
            return weather_clock.max_updates
        if fit_delay:
            return weather_clock.updates(offset, paid_stops, stopped_first,
                                        fit_delay=fit_delay)
        return weather_clock.updates(offset, paid_stops, stopped_first)

    surface_path = [weather]

    def projected_surface(updates):
        while len(surface_path) <= updates:
            surface_path.append(project_next_surface(
                surface_path[-1], forecast_context, len(surface_path) - 1))
        return surface_path[updates]

    @lru_cache(maxsize=None)
    def surface(offset, paid_stops, stopped_first, fit_delay=0.0):
        return projected_surface(updates(offset, paid_stops, stopped_first, fit_delay))

    @lru_cache(maxsize=None)
    def critical_at(update_index, compound):
        # Many branches share a projected surface; it is fixed within this forecast.
        return (
            projected_surface(update_index).tire_mismatch(TireCompound(compound))
            == "critical"
        )

    @lru_cache(maxsize=None)
    def running(offset, compound, age, first_kind, updates):
        tire = TIRE_COMPOUNDS[TireCompound(compound)]
        surface = projected_surface(updates)
        lap_number = current_lap + offset
        aero_enabled = active_aero_enabled if offset == 0 else True
        gap = (gaps[first_kind]
               if first_kind is not None and gaps is not None else None)
        if shared_laps is not None:
            value = memoized_control_lap(shared_laps, prepared_lap_time, tire, surface,
                                         lap_number, age, gap, aero_enabled)
        elif prepared_lap_time is None:
            value = isolated_strategy_lap(
                simulator, driver, car, track, tire, surface, lap_number,
                physical_total_laps, tire_age=age,
                active_aero_enabled=aero_enabled, gap_to_car_ahead=gap,
            )
        else:
            value = prepared_lap_time(
                tire, surface, lap_number, age, gap, aero_enabled,
            )
        return (current_running_time(value, current_lap_time_modifier, safety_car,
                                     stopped=first_kind == 1) if offset == 0 else value)

    def run(offset, compound, age, updates, first_kind=None, fitted=False):
        name = compound.value if isinstance(compound, TireCompound) else compound
        value = running(offset, name, age, first_kind, updates)
        fee = tire_warmup_seconds(warmup, name) if fitted and warmup else 0.
        return (current_fitted_time(value, fee, safety_car, stopped=first_kind == 1)
                if offset == 0 else value + fee)

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

    def canonical_clock_state(offset, paid_stops, stopped_first, fit_delay):
        if constant_surface:
            # Native surface projection is stationary at rainfall equilibrium.
            # Stop and fitting delays still enter costs, but cannot change any
            # future surface, eligibility or pace in this green suffix.
            return horizon + 1, True, 0.0
        if offset >= horizon:
            return horizon + 1, True, 0.0
        observed = updates(offset, paid_stops, stopped_first, fit_delay)
        if (observed == weather_clock.max_updates
                or absorbing_update is not None and observed >= absorbing_update):
            # The update cap makes all later weather surfaces independent of
            # both the exact paid count and which first-stop delay was used.
            # A native absorbing surface has that same independence before
            # the cap when every later complete weather snapshot agrees.
            return horizon + 1, True, 0.0
        return paid_stops, stopped_first, fit_delay

    def allowed(
        offset, compound, left, dry, damp, used, candidate, before, update_index,
        age, expiry,
    ):
        critical = critical_at(update_index, compound)
        limit = dry if before.track_wetness < .08 and before.rain_intensity < .15 else damp
        return (not tire_slot_usable(age, expiry) or critical or (left > 0 and (
            TireCompound(compound) in (TireCompound.INTERMEDIATE, TireCompound.WET)
            or before.track_wetness > .3 or limit is None or limit > 0
        )) or (not legal(used) and not used & bits[TireCompound(candidate).value]))

    def reduced(value):
        return None if value is None else max(0, value - 1)

    @lru_cache(maxsize=None)
    def retained_tail(offset, compound, age, paid_stops, stopped_first, fit_delay):
        """Cost of the compulsory final stint at one fixed clock position.

        Once no paid stops remain, a legal current compound can be retained
        exactly when it is valid for every remaining surface.  The paid-stop
        count is then fixed, so this sum uses the same delayed surface at each
        lap and preserves both physical wear and the external cadence.
        """
        total = 0.0
        for index in range(offset, horizon):
            cancellation_checkpoint()
            update_index = updates(index, paid_stops, stopped_first, fit_delay)
            if critical_at(update_index, compound):
                return inf
            total += run(
                index, compound, age + index - offset,
                update_index,
            )
        return total

    @lru_cache(maxsize=None)
    def retainable(offset, compound, paid_stops, stopped_first, fit_delay):
        for index in range(offset, horizon):
            cancellation_checkpoint()
            if critical_at(updates(index, paid_stops, stopped_first, fit_delay), compound):
                return False
        return True

    def make_actions(state):
        (offset, compound, age, pool, left, dry, damp, used, paid_stops,
         stopped_first, fit_delay, expiry) = state
        before = surface(offset, paid_stops, stopped_first, fit_delay)
        current = TireCompound(compound)
        actions = []
        update_index = updates(offset, paid_stops, stopped_first, fit_delay)
        if (tire_slot_usable(age, expiry) and not critical_at(update_index, current.value)
                and (offset + 1 < horizon or legal(used | bits[current.value]))):
            next_paid, next_stopped, next_delay = canonical_clock_state(
                offset + 1, paid_stops, stopped_first, fit_delay,
            )
            actions.append([
                (offset + 1, current.value, age + 1, pool, left, dry, damp,
                 canonical_used(used | bits[current.value]),
                 next_paid, next_stopped, next_delay, expiry),
                run(offset, current, age,
                    update_index),
                None,
            ])
        previous = None
        for index, candidate in enumerate(pool):
            cancellation_checkpoint()
            if candidate == previous:
                continue
            previous = candidate
            target, target_age, target_expiry = candidate
            if not tire_slot_usable(target_age, target_expiry) or critical_at(update_index, target):
                continue
            if offset + 1 == horizon and not legal(used | bits[target]):
                continue
            if not allowed(offset, compound, left, dry, damp, used, target,
                           before, update_index, age, expiry):
                continue
            after_paid = paid_stops + 1
            after_updates = updates(
                offset, after_paid, stopped_first, fit_delay,
            )
            exchanged = exchange_tire_slots(pool, index, (compound, age, expiry))
            fit_cost = tire_warmup_seconds(warmup, target) if warmup else 0.0
            next_paid, next_stopped, next_delay = canonical_clock_state(
                offset + 1, after_paid, stopped_first, fit_delay + fit_cost,
            )
            actions.append([
                (offset + 1, target, target_age + 1, exchanged,
                 max(0, left - 1), reduced(dry), reduced(damp),
                 canonical_used(used | bits[target]),
                 next_paid, next_stopped, next_delay, target_expiry),
                green_stop + run(offset, target, target_age, after_updates,
                                 fitted=bool(warmup)),
                None,
            ])

        def action_key(action):
            if native_clock:
                action[0] = _canonical_inventory_state(
                    action[0], horizon, require_compound_rule, dead_stock[action[0][0]])
            elif native:
                action[0] = _expired_inventory_state(action[0])
            bound = completion_bound(action[0])
            action[2] = bound
            return action[1] + bound

        actions.sort(key=action_key)
        return actions

    solve_cache, excluded = {}, {}
    fresh_bound = None
    fresh_bound_factory = None

    def terminal(state):
        (offset, compound, age, _pool, left, _dry, _damp, used,
         paid_stops, stopped_first, fit_delay, expiry) = state
        if offset >= horizon:
            return ProjectedControlCost(0, 0.) if legal(used) else retired
        if (left == 0 and legal(used) and (expiry < 0 or expiry - age >= horizon - offset)
                and retainable(offset, compound, paid_stops, stopped_first, fit_delay)):
            return ProjectedControlCost(horizon - offset, retained_tail(
                offset, compound, age, paid_stops, stopped_first, fit_delay))
        return None

    def solve(initial):
        """Evaluate the finite-pool strategy DAG without recursion."""
        if native_clock:
            initial = _canonical_inventory_state(
                initial, horizon, require_compound_rule, dead_stock[initial[0]])
        elif native:
            initial = _expired_inventory_state(initial)
        if native_clock and forecast_context is not None:
            return _bounded_inventory_suffix(initial, solve_cache, excluded, make_actions, terminal)
        if initial in solve_cache:
            return solve_cache[initial]
        frames = [[initial, None, 0, retired]]
        while frames:
            cancellation_checkpoint()
            state, actions, index, best = frames[-1]
            if state in solve_cache:
                frames.pop()
                continue
            (offset, compound, age, pool, left, dry, damp, used,
             paid_stops, stopped_first, fit_delay, expiry) = state
            if actions is None:
                if offset >= horizon:
                    # Keep the terminal value in the frame so the common
                    # completion path can add its incoming edge.
                    frames[-1][1:] = [[], 0,
                                     ProjectedControlCost(0, 0.) if legal(used) else retired]
                    continue
                if (left == 0 and legal(used) and (expiry < 0 or expiry - age >= horizon - offset)
                        and retainable(offset, compound, paid_stops, stopped_first,
                                       fit_delay)):
                    frames[-1][1:] = [[], 0, ProjectedControlCost(
                        horizon - offset, retained_tail(
                            offset, compound, age, paid_stops, stopped_first, fit_delay))]
                    continue
                actions = make_actions(state)
                frames[-1][1:] = [actions, 0, retired]
                continue
            if index < len(actions):
                child, edge, bound = actions[index]
                frames[-1][2] += 1
                if child in solve_cache:
                    option = solve_cache[child].prepend_lap(edge)
                    if option.rank > frames[-1][3].rank:
                        frames[-1][3] = option
                elif not best.finished or nextafter(edge + bound, -inf) < best.seconds:
                    frames.append([child, None, 0, retired])
                continue
            solve_cache[state] = best
            frames.pop()
            if frames:
                parent = frames[-1]
                child, edge, _bound = parent[1][parent[2] - 1]
                option = solve_cache[state].prepend_lap(edge)
                if option.rank > parent[3].rank:
                    parent[3] = option
        return solve_cache[initial]

    current_id = inventory.current_set_id
    current = inventory.sets.get(current_id)
    if current is not None and current.remaining_laps is not None:
        inventory.current_remaining_laps(tire_age)
    current_expiry = tire_set_slot(current)[2] if current is not None else -1
    usable_current = (current is not None and current_id not in inventory.unavailable_ids
                      and tire_slot_usable(tire_age, current_expiry))
    stock = tuple(inventory.replacements())
    pool = tuple(sorted(tire_set_slot(item) for item in stock))

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
    last_change = 0
    last_surface = forecast_json(weather) if native_clock else None
    previous = {compound: critical_at(0, compound.value)
                for compound in TireCompound}
    for update in range(1, weather_clock.max_updates + 1):
        cancellation_checkpoint()
        if native_clock:
            snapshot = forecast_json(projected_surface(update))
            if snapshot != last_surface:
                last_change = update
            last_surface = snapshot
        for compound in TireCompound:
            critical = critical_at(update, compound.value)
            critical_changes += critical and not previous[compound]
            previous[compound] = critical
    if native_clock:
        absorbing_update = last_change
    rule_stops = 0 if legal(used_mask) else 2 - (used_mask & 7).bit_count()
    max_paid_stops = remaining_stops + rule_stops + 1 + critical_changes
    if any(item.remaining_laps is not None for item in inventory.sets.values()):
        max_paid_stops = horizon  # Usage expiry can compel one paid fit per own lap.
    clock_surfaces = _inventory_clock_surfaces(
        horizon, weather_clock, warmup, max_paid_stops, native=native_clock)

    @lru_cache(maxsize=None)
    def lower_running(offset, compound, age):
        return min(
            run(offset, compound, age, update)
            for update in clock_surfaces[offset]
        )

    if native_clock:
        # Include every before/after-service clock allowed by the relaxation.
        # Removing a compound requires it to remain critical throughout this
        # entire superset, not just along the currently cheapest schedule.
        relaxed_critical = {
            compound.value: tuple(all(critical_at(update, compound.value)
                                      for update in clock_surfaces[offset])
                                  for offset in range(horizon))
            for compound in TireCompound
        }
        dead_stock = _dead_inventory_compounds(relaxed_critical)
        if forecast_context is not None and not warmup:
            shared_services = None
            if (shared_laps is not None and horizon <= 100 and safety_car is None
                    and current_lap_time_modifier == 1. and active_aero_enabled and gaps is None):
                # Unlimited fresh service ignores physical pool, allowance
                # and rule histories. Its weather observations and native
                # physics remain exact; the initial retained tyre stays local.
                key = (current_lap, horizon, forecast_json(weather), forecast_context,
                       _green_weather_clock_key(weather_clock, warmup), green_stop)
                shared_services = control_relaxation_memo(shared_laps, key)
            fresh_bound = _fresh_inventory_completion_bound(
                horizon, tuple(compound.value for compound in TireCompound),
                lambda offset, compound, age, clock: run(
                    offset, compound, age, updates(offset, *clock)),
                lambda offset, compound, clock: not critical_at(updates(offset, *clock), compound),
                lambda clock: (clock[0] + 1, clock[1]),
                lambda offset, clock: canonical_clock_state(offset, *clock, 0.)[:2],
                green_stop, solved=shared_services)
        elif forecast_context is not None and warmup and horizon <= 100:
            def build_fitting_bound():
                shared_services = None
                if (shared_laps is not None and safety_car is None
                        and current_lap_time_modifier == 1. and active_aero_enabled
                        and gaps is None):
                    # Each fit widens future delay by the smallest/largest fee,
                    # independently of compound choice. Its first outlap sees
                    # the pre-fee surface. Retained sets preserve actual delay.
                    key = ("fit", current_lap, horizon, forecast_json(weather), forecast_context,
                           weather_clock, green_stop, tuple(sorted(warmup.items())))
                    shared_services = control_relaxation_memo(shared_laps, key)
                return _fitting_inventory_completion_bound(
                    horizon, tuple(compound.value for compound in TireCompound),
                    run, updates, critical_at, canonical_clock_state, warmup, green_stop,
                    solved=shared_services)

            fresh_bound_factory = build_fitting_bound

    initial_ages = tuple((item.compound.value,
                          tire_age if item.id == inventory.current_set_id else item.age)
                         for item in inventory.sets.values()
                         if item.id not in inventory.unavailable_ids)
    lower_critical = {compound.value: (False,) * horizon for compound in TireCompound}

    @lru_cache(maxsize=1)
    def lower_bounds():
        common = None
        if shared_laps is not None:
            common = control_wear_bound(driver, car, track, physical_total_laps, current_lap)
        value = _conserved_wear_lower_bounds(
            horizon, initial_ages, lower_critical, lower_running,
        )
        if common is not None:
            # The shared stock relaxation ignores weather chronology. Keep
            # the suffix's stronger clock/set-age bound when it can prune
            # paths that the original field-wide relaxation still admits.
            return tuple(max(shared, local) for shared, local in zip(common, value, strict=True))
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
        nonlocal fresh_bound
        (offset, compound, age, _pool, _left, _dry, _damp, used, paid,
         stopped, fit_delay, expiry) = state
        compliant = legal(used)
        if offset >= horizon:
            return 0.0 if compliant else inf
        if (fresh_bound is None and fresh_bound_factory is not None
                and len(solve_cache) + len(excluded) > horizon * horizon):
            # Short/simple suffixes finish more cheaply with the existing
            # bound. Pay for service relaxation only once the physical search
            # has accumulated more states than this horizon's quadratic budget.
            fresh_bound = fresh_bound_factory()
        fresh_clock = (paid, stopped, fit_delay) if warmup else (paid, stopped)
        if not tire_slot_usable(age, expiry):
            value = nextafter(green_stop + lower_bounds()[offset], -inf)
            return (max(value, fresh_bound(offset, compound, age, fresh_clock, expiry))
                    if fresh_bound is not None else value)
        base_age = age - offset
        key = compound, base_age, paid, stopped, fit_delay, compliant
        if key not in completion_rows:
            completion_rows[key] = [horizon, [None] * horizon + [0.0 if compliant else inf]]
        first, row = completion_rows[key]
        for index in range(first - 1, offset - 1, -1):
            cancellation_checkpoint()
            best = green_stop + lower_bounds()[index]
            update_index = updates(index, paid, stopped, fit_delay)
            if not critical_at(update_index, compound):
                stay = run(index, compound, base_age + index, update_index)
                best = min(best, stay + row[index + 1])
            row[index] = nextafter(best, -inf)
        completion_rows[key][0] = min(first, offset)
        value = max(lower_bounds()[offset], row[offset])
        if fresh_bound is not None:
            value = max(value, fresh_bound(offset, compound, age, fresh_clock, expiry))
        return value

    def initial_cost(item, available, charge, consume, first_kind, paid_stops,
                     stopped_first, age_override=None, fitted=False):
        compound = item.compound.value
        age = item.age if age_override is None else age_override
        if (critical_at(updates(0, paid_stops, stopped_first), compound)
                or horizon == 1 and not legal(used_mask | bits[compound])):
            return invalid
        after_updates = updates(0, paid_stops, stopped_first)
        if consume:
            after_updates = updates(0, paid_stops + 1, True)
        next_stopped = stopped_first or bool(consume)
        fit_cost = tire_warmup_seconds(warmup, compound) if fitted and warmup else 0.0
        next_paid, next_stopped, next_delay = canonical_clock_state(
            1, paid_stops + consume, next_stopped, fit_cost,
        )
        state = (1, compound, age + 1, available,
                 max(0, remaining_stops - consume),
                 reduced(remaining_dry_stops) if consume else remaining_dry_stops,
                 reduced(remaining_damp_stops) if consume else remaining_damp_stops,
                 canonical_used(used_mask | bits[compound]), next_paid, next_stopped,
                 next_delay, tire_set_slot(item)[2])
        return solve(state).prepend_lap(charge + run(
            0, compound, age, after_updates, first_kind, fitted=fitted))

    wait = invalid
    if usable_current and (free_fit or not force_stop):
        wait = initial_cost(current, pool, 0.0, 0, 0, 0, False, tire_age,
                            fitted=current_fit_pending)

    best, selected = invalid, None
    choices = ((current,) if free_fit and usable_current else ()) + stock
    for item in choices:
        cancellation_checkpoint()
        if item.id == current_id:
            cost = wait
        else:
            if not free_fit and not force_stop and usable_current and not allowed(
                0, current.compound.value, remaining_stops, remaining_dry_stops,
                remaining_damp_stops, used_mask, item.compound.value,
                surface(0, 0, False), updates(0, 0, False),
                tire_age, current_expiry,
            ):
                continue
            candidate = tire_set_slot(item)
            index = pool.index(candidate)
            available = pool[:index] + pool[index + 1:]
            if usable_current:
                available = tuple(sorted(available + (tire_set_slot(current, tire_age),)))
            consume = 0 if free_fit else 1
            charge = 0.0 if free_fit else current_stop
            # A free fit still starts in the same occupied lane as the
            # retained branch.  It consumes no pit time, but it must keep
            # the observed traffic-gap semantics for its first lap.
            cost = initial_cost(item, tuple(available), charge, consume,
                                1 if not free_fit else 0, 0, False,
                                fitted=(item.id != current_id))
        if cost.rank > best.rank:
            best, selected = cost, item
    return InventoryDecision.from_continuations(
        best, wait, selected.id if selected else None, selected.compound if selected else None)


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
        cancellation_checkpoint()
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
            cancellation_checkpoint()
            residuals[slot] = min(residuals[slot], nextafter(cost - minimum, -inf))
        extra = 0.
        for value in nsmallest(horizon - offset, residuals):
            cancellation_checkpoint()
            extra = nextafter(extra + value, -inf)
        lower[offset] = max(baseline, nextafter(baseline + extra, -inf))
    return tuple(lower)


def plan_inventory_strategy(
    driver, car, track, weather, inventory, current_lap, *, tire_age=0,
    remaining_stops=3, remaining_dry_stops=None, remaining_damp_stops=None,
    used_compounds=(), pit_lane_factor=1., additional_current_stop_cost=0.,
    current_lap_time_modifier=1., active_aero_enabled=True, physical_total_laps=None,
    weather_intervals=None, current_traffic_gaps=None, force_stop=False, free_fit=False,
    require_compound_rule=True, weather_clock=None, tire_warmup=None,
    current_fit_pending=False, forecast_context=None, safety_car=None, control_context=None,
):
    """Rank legal finishes, accepted distance, then deterministic elapsed time.

    The anonymous future pool retains compound, age, usage expiry and multiplicity. Only
    interchangeable usable IDs are merged; an exhausted active slot no longer
    distinguishes otherwise identical suffixes. Local memoization is confined to this
    call, so every model, surface and cadence is intrinsically in its context.
    A usable control field prices the known neutralized prefix on a stable
    dry surface, then returns to this same physical-pool green search. Other
    surface and free-fit paths retain their existing weather-clock costs.
    Incomplete continuations keep infinite finishing costs and separately
    retain their accepted laps and time through the last legal crossing.
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
    if type(current_fit_pending) is not bool:
        raise ValueError("current_fit_pending must be boolean")
    tire_warmup = validate_tire_warmup(tire_warmup)
    horizon = track.total_laps - current_lap + 1
    intervals = normalize_weather_intervals(horizon, weather_intervals, weather=weather,
                                            forecast_context=forecast_context)
    forecast_context = getattr(intervals, "context", forecast_context)
    _validate_weather_clock(weather_clock, horizon)
    if control_context is not None and type(control_context) is not StrategyControlContext:
        raise ValueError("control_context must be a StrategyControlContext")
    # The field prices every known controlled crossing. A stable dry surface
    # has no additional branch-dependent weather state; transitional weather
    # and free refits retain the established surface-clock planner.
    controlled = (control_context is not None and not free_fit and not control_context.paid_fit
                  and weather.track_wetness == weather.rain_intensity == 0.
                  and forecast_context is None)
    if controlled:
        car, track = car.model_copy(deep=True), track.model_copy(deep=True)
    gaps = normalize_current_traffic_gaps(current_traffic_gaps)
    if safety_car is not None:
        gaps = safety_car.traffic_gaps
    driver = driver.model_copy(deep=True)
    simulator = LapSimulator(np.random.default_rng(0))
    bits = {compound.value: 1 << index if index < 3 else 8
            for index, compound in enumerate(TireCompound)}
    mask = 0
    for compound in used_compounds:
        mask |= bits[TireCompound(compound).value]
    if (usable_weather_control(control_context, weather_clock) and not free_fit
            and not controlled):
        current = inventory.sets.get(inventory.current_set_id)
        tire = TIRE_COMPOUNDS[TireCompound.MEDIUM if current is None else current.compound]
        result = plan_controlled_weather(
            driver, car, track, weather, tire, tire_age, current_lap, remaining_stops,
            control_context=control_context, physical_total_laps=physical,
            used_compounds=used_compounds, remaining_dry_stops=remaining_dry_stops,
            remaining_damp_stops=remaining_damp_stops, inventory=inventory,
            force_stop=force_stop, require_compound_rule=require_compound_rule,
            tire_warmup=tire_warmup, current_fit_pending=current_fit_pending,
            forecast_context=forecast_context)
        return InventoryDecision.from_continuations(
            result.pit, result.wait, result.set_id, result.compound)
    if weather_clock is not None and not controlled:
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
            tire_warmup=tire_warmup, current_fit_pending=current_fit_pending,
            forecast_context=forecast_context,
            safety_car=safety_car,
        )
    prepared_lap_time = simulator.prepare_deterministic_lap_time(driver, car, track, physical)
    native = prepared_lap_time is not None and native_physics(driver, car, track, weather)
    shared_laps = control_lap_memo(driver, car, track, physical) if native else None
    surfaces = tuple(projected_surfaces(weather, horizon, intervals))
    service = expected_stationary_time(car)
    green_stop = track.pit_lane_delta + service
    current_stop = (track.pit_lane_delta * pit_lane_factor + expected_stationary_time(car)
                    + additional_current_stop_cost)
    invalid = ProjectedControlCost(-1, inf)
    retired = ProjectedControlCost(0, 0., False)

    def legal(used):
        return not require_compound_rule or bool(used & 8) or (used & 7).bit_count() >= 2

    critical = {c.value: tuple(s.tire_mismatch(c) == "critical" for s in surfaces)
                for c in TireCompound}
    dead_stock = (_dead_inventory_compounds(critical) if native and not controlled else
                  (frozenset(),) * (horizon + 1))

    def allowed(offset, compound, left, dry, damp, used, candidate, age, expiry):
        limit = dry if (surfaces[offset].track_wetness < .08
                        and surfaces[offset].rain_intensity < .15) else damp
        return (not tire_slot_usable(age, expiry) or critical[compound][offset]
                or (left > 0 and (compound in ("wet", "intermediate")
                    or surfaces[offset].track_wetness > .3 or limit is None or limit > 0))
                or (not legal(used) and not used & bits[candidate]))

    def reduced(value):
        return None if value is None else max(0, value - 1)

    @lru_cache(maxsize=None)
    def running(offset, compound, age, first_kind=None):
        first = first_kind is not None
        tire = TIRE_COMPOUNDS[TireCompound(compound)]
        aero_enabled = active_aero_enabled if first else True
        gap = gaps[first_kind] if first and gaps is not None else None
        if shared_laps is not None:
            value = memoized_control_lap(shared_laps, prepared_lap_time, tire, surfaces[offset],
                                         current_lap + offset, age, gap, aero_enabled)
        elif prepared_lap_time is None:
            value = isolated_strategy_lap(
                simulator, driver, car, track, tire, surfaces[offset],
                current_lap + offset, physical, tire_age=age,
                active_aero_enabled=aero_enabled, gap_to_car_ahead=gap,
            )
        else:
            value = prepared_lap_time(
                tire, surfaces[offset], current_lap + offset, age, gap, aero_enabled,
            )
        return (current_running_time(value, current_lap_time_modifier, safety_car,
                                     stopped=first_kind == 1) if first else value)

    def run(offset, compound, age, first_kind=None, fitted=False):
        value = running(offset, compound, age, first_kind)
        fee = tire_warmup_seconds(tire_warmup, compound) if fitted and tire_warmup else 0.
        return (current_fitted_time(value, fee, safety_car, stopped=first_kind == 1)
                if first_kind is not None else value + fee)

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
        return exchange_tire_slots(pool, index, current)

    solved, excluded = {}, {}
    fresh_bound = None
    if native and forecast_context is not None:
        shared_services = None
        if (shared_laps is not None and horizon <= 100 and safety_car is None
                and current_lap_time_modifier == 1. and active_aero_enabled and gaps is None):
            # The leading candidate supplies its own weather cadence. Only
            # fresh future service ignores the physical pool; retained wear,
            # usage expiry and executable suffix costs remain local.
            key = ("own", current_lap, horizon, forecast_json(weather), intervals, green_stop,
                   tuple(sorted(tire_warmup.items())))
            shared_services = control_relaxation_memo(shared_laps, key)
        fresh_bound = _fresh_inventory_completion_bound(
            horizon, tuple(compound.value for compound in TireCompound),
            lambda offset, compound, age, clock: run(offset, compound, age),
            lambda offset, compound, clock: not critical[compound][offset],
            lambda clock: clock, lambda offset, clock: clock, green_stop, solved=shared_services,
            fitted_running=(lambda offset, compound, age, clock:
                            run(offset, compound, age, fitted=True)) if tire_warmup else None)

    @lru_cache(maxsize=None)
    def retained_tail(offset, compound, age):
        # Different exhausted pools can leave the same compulsory final
        # stint. Its cost depends only on this call's lap, compound and age.
        # Keep the original reverse summation order for identical rounding.
        total = 0.
        for number in range(horizon - 1, offset - 1, -1):
            cancellation_checkpoint()
            total = run(number, compound, age + number - offset) + total
        return total

    completion_rows = {}

    def completion_bound(offset, compound, age, used, expiry):
        """Retain actual wear until the first future service, then relax stock.

        A completion either keeps this set through the finish, when legal, or
        pays a green pit entry before switching. After that first service the
        conserved-wear bound ignores eligibility, availability and later pit
        charges. This remains optimistic without assuming monotone tyre pace.
        """
        compliant = legal(used)
        if offset >= horizon:
            return 0.0 if compliant else inf
        if not tire_slot_usable(age, expiry):
            return nextafter(green_stop + lower_bounds()[offset], -inf)
        base_age = age - offset
        key = compound, base_age, compliant
        if key not in completion_rows:
            completion_rows[key] = [horizon, [None] * horizon + [0.0 if compliant else inf]]
        first, row = completion_rows[key]
        for index in range(first - 1, offset - 1, -1):
            cancellation_checkpoint()
            best = green_stop + lower_bounds()[index]
            if not critical[compound][index]:
                best = min(best, run(index, compound, base_age + index) + row[index + 1])
            row[index] = nextafter(best, -inf)
        completion_rows[key][0] = min(first, offset)
        value = max(lower_bounds()[offset], row[offset])
        if fresh_bound is not None:
            value = max(value, fresh_bound(offset, compound, age, None, expiry))
        return value

    def frame(state):
        offset, compound, age, pool, left, dry, damp, used, expiry = state
        if offset == horizon:
            return ProjectedControlCost(0, 0.) if legal(used) else retired
        if (left == 0 and legal(used) and not any(critical[compound][offset:])
                and (expiry < 0 or expiry - age >= horizon - offset)):
            return ProjectedControlCost(horizon - offset, retained_tail(offset, compound, age))
        best = retired
        if (tire_slot_usable(age, expiry) and not critical[compound][offset]
                and (offset + 1 < horizon or legal(used | bits[compound]))):
            cost = run(offset, compound, age)
            child = (offset + 1, compound, age + 1, pool, left, dry, damp,
                     used | bits[compound], expiry)
            best = (yield child).prepend_lap(cost)
        previous = None
        for index, candidate in enumerate(pool):
            cancellation_checkpoint()
            if candidate == previous:
                continue
            previous = candidate
            target, target_age, target_expiry = candidate
            if (not tire_slot_usable(target_age, target_expiry) or critical[target][offset]
                    or not allowed(offset, compound, left, dry, damp, used,
                                   target, age, expiry)):
                continue
            if offset + 1 == horizon and not legal(used | bits[target]):
                continue
            cost = green_stop + run(offset, target, target_age,
                                    fitted=bool(tire_warmup))
            if best.finished:
                bound = completion_bound(
                    offset + 1, target, target_age + 1, used | bits[target], target_expiry,
                )
                if controlled and native:
                    bound = max(bound, unlimited_dry_bound(
                        offset + 1, target, target_age + 1, max(0, left - 1),
                        used | bits[target]))
                if nextafter(cost + bound, -inf) >= best.seconds:
                    continue
            # A full-distance time bound cannot discard a longer retirement.
            # Sorting a replacement pool is only needed for admitted branches.
            child = (offset + 1, target, target_age + 1,
                     exchange(pool, index, (compound, age, expiry)), max(0, left - 1),
                     reduced(dry), reduced(damp), used | bits[target], target_expiry)
            option = (yield child).prepend_lap(cost)
            if option.rank > best.rank:
                best = option
        return best

    def terminal(state):
        offset, compound, age, _pool, left, _dry, _damp, used, expiry = state
        if offset == horizon:
            return ProjectedControlCost(0, 0.) if legal(used) else retired
        if (left == 0 and legal(used) and not any(critical[compound][offset:])
                and (expiry < 0 or expiry - age >= horizon - offset)):
            return ProjectedControlCost(horizon - offset, retained_tail(offset, compound, age))
        return None

    def make_actions(state):
        offset, compound, age, pool, left, dry, damp, used, expiry = state
        actions = []
        if (tire_slot_usable(age, expiry) and not critical[compound][offset]
                and (offset + 1 < horizon or legal(used | bits[compound]))):
            actions.append([
                (offset + 1, compound, age + 1, pool, left, dry, damp,
                 used | bits[compound], expiry), run(offset, compound, age), None,
            ])
        previous = None
        for index, candidate in enumerate(pool):
            cancellation_checkpoint()
            if candidate == previous:
                continue
            previous = candidate
            target, target_age, target_expiry = candidate
            if (not tire_slot_usable(target_age, target_expiry) or critical[target][offset]
                    or not allowed(offset, compound, left, dry, damp, used,
                                   target, age, expiry)
                    or offset + 1 == horizon and not legal(used | bits[target])):
                continue
            actions.append([
                (offset + 1, target, target_age + 1,
                 exchange(pool, index, (compound, age, expiry)), max(0, left - 1),
                 reduced(dry), reduced(damp), used | bits[target], target_expiry),
                green_stop + run(offset, target, target_age, fitted=bool(tire_warmup)), None,
            ])
        for action in actions:
            child = _canonical_inventory_state(
                action[0], horizon, require_compound_rule, dead_stock[action[0][0]])
            action[0] = child
            bound = completion_bound(child[0], child[1], child[2], child[7], child[-1])
            if controlled:
                bound = max(bound, unlimited_dry_bound(child[0], child[1], child[2],
                                                       child[4], child[7]))
            action[2] = bound
        return actions

    def solve(initial):
        if native:
            initial = _canonical_inventory_state(
                initial, horizon, require_compound_rule, dead_stock[initial[0]])
            if forecast_context is not None:
                return _bounded_inventory_suffix(initial, solved, excluded, make_actions, terminal)
        if initial in solved:
            return solved[initial]
        stack = [(initial, frame(initial))]
        value = None
        while stack:
            cancellation_checkpoint()
            state, generator = stack[-1]
            try:
                child = generator.send(value)
            except StopIteration as result:
                value = result.value
                solved[state] = value
                stack.pop()
                continue
            if native:
                child = _canonical_inventory_state(
                    child, horizon, require_compound_rule, dead_stock[child[0]])
            if child in solved:
                value = solved[child]
            else:
                stack.append((child, frame(child)))
                value = None
        return value

    current_id = inventory.current_set_id
    current = inventory.sets.get(current_id)
    if current is not None and current.remaining_laps is not None:
        inventory.current_remaining_laps(tire_age)
    current_expiry = tire_set_slot(current)[2] if current is not None else -1
    usage_limited = any(item.remaining_laps is not None for item in inventory.sets.values())
    usable_current = (current is not None and current_id not in inventory.unavailable_ids
                      and tire_slot_usable(tire_age, current_expiry))
    stock = tuple(inventory.replacements())
    pool = tuple(sorted(tire_set_slot(item) for item in stock))

    if controlled:
        root = control_context.new_field()
        if ((type(root) is ObservedStandardField
             and (root.lap != current_lap or root.now != control_context.now))
                or (type(root) is not ObservedStandardField
                    and (root.timeline.states[root.identifier].completed_laps + 1 != current_lap
                         or root.timeline._clock.scheduled_laps != physical))):
            raise ValueError("control_context must match the current lap and physical distance")
        memo, lap_costs, control_bound_rows, unlimited_bounds, retained_rows = {}, {}, {}, {}, {}
        unlimited_tables = None
        uniform_prefix = (len(root.rows) == 1 or not root.safety_car
                          if type(root) is ObservedStandardField else len(root.free_paces) == 1)

        def unlimited_dry_bound(offset, compound, age, left, used):
            """Native wear cannot make a set faster than a fresh replacement.

            Relax stock ages, availability and compound obligations, allowing
            an extra correction stop when needed. The result bounds physical
            green costs, never supplies an executable inventory schedule.
            """
            nonlocal unlimited_tables
            if usage_limited:
                # This relaxation caps paid fits by elective stops. Usage
                # expiry can require more visits, so that cap is not a bound.
                return 0.
            if offset == horizon:
                return 0.
            budget = min(left + int(not legal(used)), horizon - offset)
            if budget > 3:
                return 0.
            key = offset, compound, age, budget
            if key in unlimited_bounds:
                return unlimited_bounds[key]
            if unlimited_tables is None:
                clean = driver.model_copy(deep=True)
                clean.reset_race_state()
                clean.id = clean.name = clean.team_id = "projection"
                package = car.model_copy(deep=True)
                package.team_id = package.team_name = "projection"
                models = forecast_json(clean), forecast_json(package), forecast_json(track)
                sets = tuple(forecast_json(TIRE_COMPOUNDS[choice]) for choice in SLICKS)
                scale = simulator.weather_pace_multiplier(driver, car, surfaces[0])
                profile = tuple(sorted((tire_warmup or {}).items()))
                unlimited_tables = _floor_tables(models, sets, physical, scale, profile)
            costs, prefixes = unlimited_tables
            row_key = offset, compound, age
            if row_key not in retained_rows:
                retained_rows[row_key] = np.cumsum([
                    running(index, compound, age + index - offset)
                    for index in range(offset, horizon)])
            old = retained_rows[row_key]
            number = current_lap + offset
            best = float(old[-1])
            if budget and offset + 1 < horizon:
                best = min(best, float(np.min(
                    old[:-1] + costs[budget, 7, number + 1:track.total_laps + 1])))
            if budget:
                for index in range(3):
                    prefix = prefixes[number, index]
                    value = float(prefix[-1])
                    if budget > 1 and offset + 1 < horizon:
                        value = min(value, float(np.min(
                            prefix[:-1] + costs[budget - 1, 7,
                                                number + 1:track.total_laps + 1])))
                    best = min(best, value + track.pit_lane_delta + service)
            # Cumulative arrays and recursive physical-set sums group their
            # arithmetic differently. Keep the relaxation below near ties.
            value = max(0., nextafter(best - 4 * (horizon - offset + 2) * ulp(best), -inf))
            unlimited_bounds[key] = value
            return value

        def bound_running(offset, compound, age):
            multiplier = (root.running_modifier if uniform_prefix
                          and offset < root.intervals_left else 1.)
            return running(offset, compound, age) * multiplier

        @lru_cache(maxsize=1)
        def controlled_lower_bounds():
            if not uniform_prefix:
                return lower_bounds()
            return _conserved_wear_lower_bounds(horizon, initial_ages, critical, bound_running)

        def controlled_completion_bound(field, state):
            offset, compound, age, _, left, dry, _, used, _expiry = state
            compliant = legal(used)
            if offset == horizon:
                return 0. if compliant else inf
            if not usage_limited and compliant and (left == 0 or dry == 0):
                return sum(bound_running(index, compound, age + index - offset)
                           for index in range(offset, horizon))
            factor = .55 if field.controlled and field.safety_car else (
                .75 if field.controlled else 1.)
            minimum_stop = track.pit_lane_delta * factor + service
            key = compound, age - offset, compliant, minimum_stop
            if key not in control_bound_rows:
                control_bound_rows[key] = [horizon, [None] * horizon
                                          + [0. if compliant else inf]]
            first, row = control_bound_rows[key]
            lower = controlled_lower_bounds()
            for index in range(first - 1, offset - 1, -1):
                cancellation_checkpoint()
                value = minimum_stop + lower[index]
                if not critical[compound][index]:
                    value = min(value, bound_running(index, compound, age - offset + index)
                                + row[index + 1])
                row[index] = nextafter(value, -inf)
            control_bound_rows[key][0] = min(first, offset)
            budget = min(left + int(not compliant), horizon - offset)
            relaxed = unlimited_dry_bound(offset, compound, age, left, used)
            # A real controlled stop cannot save more than this lane loss
            # relative to a green stop. Running and fitting stay separate.
            relaxed -= budget * track.pit_lane_delta * (1. - factor)
            if uniform_prefix:
                controlled_laps = max(0, min(root.intervals_left, horizon) - offset)
                relaxed += (controlled_laps * (root.running_modifier - 1.)
                            * minimum_lap_time(track))
            return max(lower[offset], row[offset], relaxed)

        def controlled_running(field, offset, compound, age):
            gap = field.gap_ahead(
                field.free_paces[field.identifier] * field.running_modifier
                if type(field) is not ObservedStandardField else 1.)
            aero = not field.controlled
            key = offset, compound, age, gap, aero
            if native and key in lap_costs:
                return lap_costs[key]
            tire = TIRE_COMPOUNDS[TireCompound(compound)]
            if prepared_lap_time is not None:
                value = prepared_lap_time(tire, surfaces[offset], current_lap + offset,
                                          age, gap, aero)
            else:
                clean = driver.model_copy(deep=True)
                clean.current_tire_laps = age
                value = simulator.calculate_lap_time(
                    clean, car.model_copy(deep=True), track.model_copy(deep=True),
                    tire.model_copy(deep=True), surfaces[offset].model_copy(deep=True),
                    current_lap + offset, physical, sample_variation=False,
                    active_aero_enabled=aero, gap_to_car_ahead=gap)
            if native:
                lap_costs[key] = value
            return value

        def controlled_action(field, state, *, fitted=False, first=False, cutoff=None):
            cancellation_checkpoint()
            offset, compound, age, available, left, dry, damp, used, expiry = state
            branch = field.fork()
            factor = .55 if field.controlled and field.safety_car else (
                .75 if field.controlled else 1.)
            delay = (control_context.current_stop_delay if first else
                     track.pit_lane_delta * factor + service)
            branch.enter(delay if fitted else None)
            fee = (tire_warmup_seconds(tire_warmup, compound) if tire_warmup
                   and (fitted or first and current_fit_pending) else 0.)
            branch.cross(controlled_running(branch, offset, compound, age), fee)
            if (offset + 1 == horizon or branch.finished) and not legal(used | bits[compound]):
                return retired
            child = (offset + 1, compound, age + 1, available,
                     max(0, left - int(fitted)), reduced(dry) if fitted else dry,
                     reduced(damp) if fitted else damp, used | bits[compound], expiry)
            if (native and root.running_modifier >= 1. and cutoff is not None
                    and cutoff.finished and cutoff.laps == horizon - offset):
                # Native dirty air and disabled aero cannot beat clean-air
                # green running; SC catch-up never runs below free pace.
                # Relaxed stock/stop bounds remain optimistic over every
                # full-distance path. A shorter flagged path already loses
                # on distance, so it cannot invalidate this time cutoff.
                optimistic = (branch.now - field.now
                              + controlled_completion_bound(branch, child))
                rounding = 4 * (horizon - offset + 2) * ulp(
                    max(branch.now, field.now + cutoff.seconds))
                if optimistic > cutoff.seconds + rounding:
                    return invalid
            suffix = controlled_future(branch, child)
            return suffix.prepend_lap(branch.now - field.now)

        def controlled_future(field, state):
            cancellation_checkpoint()
            offset, compound, age, available, left, dry, damp, used, expiry = state
            if offset == horizon or field.finished:
                return ProjectedControlCost(0, 0.) if legal(used) else retired
            if not field.projection_required:
                return solve(state)
            key = (observed_control_key(field), _canonical_inventory_state(
                state, horizon, require_compound_rule)) if native else None
            if native and key in memo:
                return memo[key]
            best = (controlled_action(field, state) if tire_slot_usable(age, expiry)
                    and not critical[compound][offset] else retired)
            previous = None
            for index, candidate in enumerate(available):
                cancellation_checkpoint()
                if candidate == previous:
                    continue
                previous = candidate
                target, target_age, target_expiry = candidate
                if (not tire_slot_usable(target_age, target_expiry) or critical[target][offset]
                        or not allowed(offset, compound, left, dry, damp, used,
                                       target, age, expiry)):
                    continue
                child = (offset, target, target_age,
                         exchange(available, index, (compound, age, expiry)),
                         left, dry, damp, used, target_expiry)
                option = controlled_action(field, child, fitted=True, cutoff=best)
                if option.rank > best.rank:
                    best = option
            if native:
                memo[key] = best
            return best

        wait = invalid
        if usable_current and not force_stop and not critical[current.compound.value][0]:
            wait = controlled_action(
                root, (0, current.compound.value, tire_age, pool, remaining_stops,
                       remaining_dry_stops, remaining_damp_stops, mask, current_expiry), first=True)
        best, selected = invalid, None
        for item in stock:
            cancellation_checkpoint()
            if critical[item.compound.value][0] or (not force_stop and usable_current
                    and not allowed(0, current.compound.value, remaining_stops,
                                    remaining_dry_stops, remaining_damp_stops,
                                    mask, item.compound.value, tire_age, current_expiry)):
                continue
            candidate = tire_set_slot(item)
            index = pool.index(candidate)
            available = pool[:index] + pool[index + 1:]
            if usable_current:
                available = tuple(sorted(available + (tire_set_slot(current, tire_age),)))
            value = controlled_action(
                root, (0, item.compound.value, item.age, available, remaining_stops,
                       remaining_dry_stops, remaining_damp_stops, mask, candidate[2]),
                fitted=True, first=True,
                cutoff=best)
            if value.laps > 0 and value.rank > best.rank:
                best, selected = value, item
        return InventoryDecision.from_continuations(
            best, wait, selected.id if selected else None, selected.compound if selected else None)

    def initial_cost(item, age, available, charge, consume, kind, fitted=False):
        compound = item.compound.value
        if critical[compound][0] or horizon == 1 and not legal(mask | bits[compound]):
            return invalid
        state = (1, compound, age + 1, available,
                 max(0, remaining_stops - consume),
                 reduced(remaining_dry_stops) if consume else remaining_dry_stops,
                 reduced(remaining_damp_stops) if consume else remaining_damp_stops,
                 mask | bits[compound], tire_set_slot(item)[2])
        return solve(state).prepend_lap(charge + run(0, compound, age, kind, fitted=fitted))

    wait = invalid
    if usable_current and (free_fit or not force_stop):
        wait = initial_cost(current, tire_age, pool, 0., 0, 0,
                            fitted=current_fit_pending)
    best, selected = invalid, None
    choices = ((current,) if free_fit and usable_current else ()) + stock
    for item in choices:
        cancellation_checkpoint()
        if item.id == current_id:
            cost = wait
        else:
            if not free_fit and not force_stop and usable_current and not allowed(
                0, current.compound.value, remaining_stops, remaining_dry_stops,
                remaining_damp_stops, mask, item.compound.value,
                tire_age, current_expiry,
            ):
                continue
            candidate = tire_set_slot(item)
            index = pool.index(candidate)
            available = pool[:index] + pool[index + 1:]
            if usable_current:
                available = tuple(sorted(available + (tire_set_slot(current, tire_age),)))
            cost = initial_cost(item, item.age, available, 0. if free_fit else current_stop,
                                0 if free_fit else 1, 0 if free_fit else 1,
                                fitted=(item.id != current_id))
        if cost.rank > best.rank:
            best, selected = cost, item
    return InventoryDecision.from_continuations(
        best, wait, selected.id if selected else None, selected.compound if selected else None)

register_forecast_helpers(globals(), ("project_next_surface",))
register_forecast_helpers(globals(), ("current_running_time", "current_fitted_time"))
register_forecast_helpers(globals(), (
    "ObservedStandardField", "StrategyControlContext", "ProjectedControlCost",
    "StrategyWeatherClock", "observed_control_key", "native_physics",
    "_floor_tables", "forecast_json", "minimum_lap_time",
    "_inventory_clock_surfaces",
    "isolated_strategy_lap", "plan_controlled_weather", "usable_weather_control",
    "control_lap_memo", "memoized_control_lap", "control_wear_bound",
    "control_relaxation_memo", "_green_weather_clock_key",
    "exchange_tire_slots", "tire_set_slot", "tire_slot_usable",
    "_expired_inventory_state",
    "_canonical_inventory_state", "_dead_inventory_compounds",
    "_bounded_inventory_suffix",
    "_fresh_inventory_completion_bound",
    "_fitting_inventory_completion_bound",
))
